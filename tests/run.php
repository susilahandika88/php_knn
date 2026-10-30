<?php

declare(strict_types=1);

/**
 * Compares the PHP KNeighborsClassifier with scikit-learn results stored in
 * tests/fixtures/cases.json.
 *
 *     php tests/run.php            # all cases
 *     php tests/run.php iris       # only cases whose name contains "iris"
 */

require_once __DIR__ . '/../autoload.php';

use MachineLearning\Knn\Exceptions\NotFittedError;
use MachineLearning\Knn\KNeighborsClassifier;

const DIST_TOL = 1e-7;
const PROBA_TOL = 1e-9;

$fixture = json_decode((string) file_get_contents(__DIR__ . '/fixtures/cases.json'), true, 512, JSON_THROW_ON_ERROR);
$filter = $argv[1] ?? null;

$passed = 0;
$failed = 0;
$failures = [];

function check(string $case, string $what, bool $ok, string $detail = ''): void
{
    global $passed, $failed, $failures;
    if ($ok) {
        $passed++;
        return;
    }
    $failed++;
    $failures[] = "FAIL [{$case}] {$what}" . ($detail !== '' ? " -- {$detail}" : '');
}

function approx(mixed $a, mixed $b, float $tol): bool
{
    if (is_array($a) || is_array($b)) {
        if (!is_array($a) || !is_array($b) || count($a) !== count($b)) {
            return false;
        }
        $a = array_values($a);
        $b = array_values($b);
        foreach ($a as $i => $v) {
            if (!approx($v, $b[$i], $tol)) {
                return false;
            }
        }
        return true;
    }
    if ($b === 'inf') {
        $b = INF;
    }
    if ($a === null || $b === null) {
        return $a === $b;
    }
    $a = (float) $a;
    $b = (float) $b;
    if (is_nan($a) || is_nan($b)) {
        return is_nan($a) && is_nan($b);
    }
    if (is_infinite($a) || is_infinite($b)) {
        return $a === $b;
    }
    return abs($a - $b) <= $tol + $tol * abs($b);
}

function sameLabels(mixed $a, mixed $b): bool
{
    if (is_array($a) || is_array($b)) {
        if (!is_array($a) || !is_array($b) || count($a) !== count($b)) {
            return false;
        }
        $a = array_values($a);
        $b = array_values($b);
        foreach ($a as $i => $v) {
            if (!sameLabels($v, $b[$i])) {
                return false;
            }
        }
        return true;
    }
    if (is_string($a) || is_string($b)) {
        return $a === $b;
    }
    return (float) $a === (float) $b;
}

/** sklearn prints sets in a random order; sort the items inside "str among {...}". */
function normalizeSets(string $message): string
{
    return (string) preg_replace_callback('/among \{([^}]*)\}/', static function (array $m): string {
        $items = explode(', ', $m[1]);
        sort($items);
        return 'among {' . implode(', ', $items) . '}';
    }, $message);
}

function fixParams(array $params): array
{
    if (($params['p'] ?? null) === 'inf') {
        $params['p'] = INF;
    }
    return $params;
}

function fixX(mixed $x): mixed
{
    if (is_array($x)) {
        return array_map('fixX', $x);
    }
    return match ($x) {
        '__nan__' => NAN,
        'inf' => INF,
        default => $x,
    };
}

/** @return array{0: callable|null, 1: callable|null} weights callable, metric callable */
function callables(array $case): array
{
    $weights = null;
    $metric = null;
    $wc = $case['weights_callable'] ?? null;
    if ($wc === 'inverse_square') {
        $weights = static fn (array $d): array => array_map(static fn (array $row): array => array_map(static fn (float $x): float => 1.0 / ($x * $x), $row), $d);
    } elseif ($wc === 'zero') {
        $weights = static fn (array $d): array => array_map(static fn (array $row): array => array_fill(0, count($row), 0.0), $d);
    }
    if (($case['metric_callable'] ?? null) === 'l1') {
        $metric = static function (array $a, array $b): float {
            $s = 0.0;
            foreach ($a as $i => $x) {
                $s += abs($x - $b[$i]);
            }
            return $s;
        };
    }
    return [$weights, $metric];
}

function build(array $case): KNeighborsClassifier
{
    $params = fixParams($case['params']);
    [$weights, $metric] = callables($case);
    if ($weights !== null) {
        $params['weights'] = $weights;
    }
    if ($metric !== null) {
        $params['metric'] = $metric;
    }
    return new KNeighborsClassifier(...$params);
}

/** Runs $fn collecting E_USER_WARNING messages. @return array{0: mixed, 1: array<int,string>} */
function withWarnings(callable $fn): array
{
    $messages = [];
    set_error_handler(static function (int $no, string $msg) use (&$messages): bool {
        $messages[] = $msg;
        return true;
    }, E_USER_WARNING);
    try {
        $result = $fn();
    } finally {
        restore_error_handler();
    }
    return [$result, $messages];
}

// ---------------------------------------------------------------- regular cases
foreach ($fixture['cases'] as $case) {
    $name = $case['name'];
    if ($filter !== null && stripos($name, $filter) === false) {
        continue;
    }
    $e = $case['expected'];
    $ties = $case['ties'];

    try {
        $model = build($case);
        [, $warnings] = withWarnings(static fn () => $model->fit($case['X'], $case['y']));
    } catch (Throwable $t) {
        check($name, 'fit', false, get_class($t) . ': ' . $t->getMessage());
        continue;
    }

    // warnings sklearn raises that this port reproduces
    foreach ($case['warnings'] as $w) {
        foreach (['Mind that for 0 < p < 1', 'Parameter p is found in metric_params', 'A column-vector y'] as $prefix) {
            if (str_starts_with($w, $prefix)) {
                check($name, "warning '{$prefix}'", (bool) array_filter($warnings, static fn (string $m): bool => str_starts_with($m, $prefix)));
            }
        }
    }

    if (isset($e['classes'])) {
        check($name, 'classes_', sameLabels($model->classes_, $e['classes']), json_encode($model->classes_) . ' vs ' . json_encode($e['classes']));
        check($name, 'outputs_2d_', $model->outputs_2d_ === $e['outputs_2d']);
        check($name, '_fit_method', $model->_fit_method === $e['fit_method'], "{$model->_fit_method} vs {$e['fit_method']}");
        $effMetric = is_string($model->effective_metric_) ? $model->effective_metric_ : 'callable';
        check($name, 'effective_metric_', $effMetric === $e['effective_metric'], "{$effMetric} vs {$e['effective_metric']}");
        check($name, 'effective_metric_params_', approx($model->effective_metric_params_, $e['effective_metric_params'], 1e-12), json_encode($model->effective_metric_params_) . ' vs ' . json_encode($e['effective_metric_params']));
        check($name, 'n_features_in_', $model->n_features_in_ === $e['n_features_in']);
        check($name, 'n_samples_fit_', $model->n_samples_fit_ === $e['n_samples_fit']);
    } elseif (isset($e['fit_method'])) {
        check($name, '_fit_method', $model->_fit_method === $e['fit_method'], "{$model->_fit_method} vs {$e['fit_method']}");
    }

    $q = $case['queries'];
    try {
        if (isset($e['kneighbors_dist'])) {
            [$dist, $ind] = $model->kneighbors($q);
            check($name, 'kneighbors distances', approx($dist, $e['kneighbors_dist'], DIST_TOL));
            if (!$ties) {
                check($name, 'kneighbors indices', $ind === $e['kneighbors_ind'], 'first row ' . json_encode($ind[0]) . ' vs ' . json_encode($e['kneighbors_ind'][0]));
            }
        }

        if (isset($e['train_dist'])) {
            [$d2, $i2] = $model->kneighbors();
            check($name, 'kneighbors(None) distances', approx($d2, $e['train_dist'], DIST_TOL));
            if (!$ties) {
                check($name, 'kneighbors(None) indices', $i2 === $e['train_ind']);
            }
        }

        $pred = $model->predict($q);
        $proba = $model->predictProba($q);
        if ($case['compare_predict'] ?? !$ties) {
            check($name, 'predict', sameLabels($pred, $e['predict']), 'first ' . json_encode(array_slice($pred, 0, 5)) . ' vs ' . json_encode(array_slice($e['predict'], 0, 5)));
            if (isset($e['predict_proba'])) {
                check($name, 'predict_proba', approx($proba, $e['predict_proba'], PROBA_TOL * 1e3));
            }
        }

        if (isset($e['graph_conn'])) {
            $g = $model->kneighbors_graph($q);
            if (!$ties) {
                check($name, 'kneighbors_graph indices', $g['indices'] === $e['graph_conn']['indices']);
            }
            check($name, 'kneighbors_graph indptr', $g['indptr'] === $e['graph_conn']['indptr']);
            check($name, 'kneighbors_graph shape', $g['shape'] === $e['graph_conn']['shape']);
            $gd = $model->kneighborsGraph($q, null, 'distance');
            // csr drops explicit zeros when sorted; compare as a sum of all distances
            check($name, 'kneighbors_graph distance', abs(array_sum($gd['data']) - array_sum($e['graph_dist'])) < 1e-6);
        }

        if (isset($e['score']) && ($case['compare_predict'] ?? !$ties)) {
            check($name, 'score', approx($model->score($q, $e['score_input']), $e['score'], 1e-12));
            if (isset($e['score_weighted'])) {
                check($name, 'score(sample_weight)', approx($model->score($q, $e['score_input'], $e['score_weight_input']), $e['score_weighted'], 1e-12));
            }
        }
    } catch (Throwable $t) {
        check($name, 'query', false, get_class($t) . ': ' . $t->getMessage() . ' @' . basename($t->getFile()) . ':' . $t->getLine());
    }
}

// ---------------------------------------------------------------- error cases
foreach ($fixture['errors'] as $case) {
    $name = 'error: ' . $case['name'];
    if ($filter !== null && stripos($name, $filter) === false) {
        continue;
    }
    $X = fixX($case['X']);
    $y = $case['y'];
    $params = fixParams($case['params']);
    [$weights] = callables($case);
    if ($weights !== null) {
        $params['weights'] = $weights;
    }
    $phase = $case['phase'] ?? 'fit';
    $message = null;
    try {
        $m = new KNeighborsClassifier(...$params);
        switch ($phase) {
            case 'predict_unfitted':
                $m->predict($X);
                break;
            case 'kneighbors_unfitted':
                $m->kneighbors($X);
                break;
            case 'predict':
                $m->fit($X, $y)->predict($case['queries'] ?? $X);
                break;
            case 'kneighbors0':
                $m->fit($X, $y)->kneighbors($X, 0);
                break;
            case 'kneighbors5':
                $m->fit($X, $y)->kneighbors($X, 5);
                break;
            case 'kneighbors_train4':
                $m->fit($X, $y)->kneighbors(null, 4);
                break;
            case 'graph_bad_mode':
                $m->fit($X, $y)->kneighborsGraph(null, null, 'x');
                break;
            default:
                $m->fit($X, $y);
        }
        check($name, 'raises', false, 'no exception');
    } catch (Throwable $t) {
        $message = $t->getMessage();
        if (isset($case['expect_prefix'])) {
            check($name, 'message prefix', str_starts_with($message, $case['expect_prefix']), $message);
        } else {
            check($name, 'message', normalizeSets($message) === normalizeSets($case['message']), "got [{$message}] expected [{$case['message']}]");
        }
        if ($case['type'] === 'NotFittedError') {
            check($name, 'exception class', $t instanceof NotFittedError);
        } else {
            check($name, 'exception class', $t instanceof InvalidArgumentException, get_class($t));
        }
    }
}

foreach ($failures as $line) {
    echo $line, "\n";
}
echo "\n{$passed} checks passed, {$failed} failed (scikit-learn {$fixture['sklearn_version']} fixtures)\n";
exit($failed === 0 ? 0 : 1);

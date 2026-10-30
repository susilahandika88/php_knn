<?php

declare(strict_types=1);

/** API behaviour that is specific to the PHP port: aliases, parameters, warnings. Run: php tests/api.php */

require_once __DIR__ . '/../autoload.php';

use MachineLearning\Knn\Exceptions\InvalidParameterError;
use MachineLearning\Knn\Exceptions\NotFittedError;
use MachineLearning\Knn\KNeighborsClassifier;

$failed = 0;
$total = 0;

function ok(string $what, bool $cond): void
{
    global $failed, $total;
    $total++;
    if (!$cond) {
        $failed++;
        echo "FAIL {$what}\n";
    }
}

function throws(string $what, string $class, callable $fn, ?string $contains = null): void
{
    try {
        $fn();
        ok("{$what} (no exception)", false);
    } catch (Throwable $t) {
        ok("{$what} ({$class}, got " . get_class($t) . ': ' . $t->getMessage() . ')', $t instanceof $class && ($contains === null || str_contains($t->getMessage(), $contains)));
    }
}

$X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [5.0, 5.0], [5.0, 6.0], [6.0, 5.0]];
$y = ['a', 'a', 'a', 'b', 'b', 'b'];

// defaults match sklearn
$knn = new KNeighborsClassifier();
ok('default params', $knn->getParams() === [
    'algorithm' => 'auto', 'leaf_size' => 30, 'metric' => 'minkowski', 'metric_params' => null,
    'n_jobs' => null, 'n_neighbors' => 5, 'p' => 2, 'weights' => 'uniform',
]);

// fit returns $this, snake_case aliases, named arguments
$knn = new KNeighborsClassifier(n_neighbors: 3, weights: 'distance');
ok('fit returns self', $knn->fit($X, $y) === $knn);
ok('predict', $knn->predict([[0.5, 0.5], [5.5, 5.5]]) === ['a', 'b']);
ok('predict_proba alias', $knn->predict_proba([[0.5, 0.5]]) === $knn->predictProba([[0.5, 0.5]]));
ok('kneighbors_graph alias', $knn->kneighbors_graph([[0.5, 0.5]])['shape'] === [1, 6]);
ok('get_params alias', $knn->get_params()['n_neighbors'] === 3);
ok('classes_', $knn->classes_ === ['a', 'b']);
ok('n_features_in_', $knn->n_features_in_ === 2);
ok('score', $knn->score($X, $y) === 1.0);

// set_params
$knn->set_params(['n_neighbors' => 1, 'weights' => 'uniform']);
ok('set_params applied', $knn->n_neighbors === 1 && $knn->weights === 'uniform');
throws('set_params unknown name', InvalidArgumentException::class, fn () => $knn->setParams(['foo' => 1]), "Invalid parameter 'foo'");
throws('unknown method', BadMethodCallException::class, fn () => $knn->nope());

// parameters are validated at fit, with sklearn's wording
throws('bad n_neighbors', InvalidParameterError::class, fn () => (new KNeighborsClassifier(0))->fit($X, $y), "The 'n_neighbors' parameter");
throws('float n_neighbors', InvalidParameterError::class, fn () => (new KNeighborsClassifier(2.5))->fit($X, $y), 'Got 2.5 instead');
throws('not fitted', NotFittedError::class, fn () => (new KNeighborsClassifier())->predict($X), 'is not fitted yet');

// refitting replaces the previous model
$knn = new KNeighborsClassifier(n_neighbors: 1);
$knn->fit($X, $y)->fit([[0.0], [10.0]], [1, 2]);
ok('refit', $knn->n_features_in_ === 1 && $knn->classes_ === [1, 2] && $knn->predict([[9.0]]) === [2]);

// algorithm selection like sklearn's auto
$m = (new KNeighborsClassifier(n_neighbors: 1))->fit($X, $y);
ok('auto small data -> kd_tree', $m->_fit_method === 'kd_tree');
$m = (new KNeighborsClassifier(n_neighbors: 3))->fit($X, $y);
ok('auto k >= n/2 -> brute', $m->_fit_method === 'brute');
$m = (new KNeighborsClassifier(n_neighbors: 1, metric: 'canberra'))->fit($X, $y);
ok('auto canberra -> ball_tree', $m->_fit_method === 'ball_tree');
$m = (new KNeighborsClassifier(n_neighbors: 1, metric: 'cosine'))->fit($X, $y);
ok('auto cosine -> brute', $m->_fit_method === 'brute');

// warnings are raised as E_USER_WARNING
$seen = [];
set_error_handler(static function (int $no, string $msg) use (&$seen): bool {
    $seen[] = $msg;
    return true;
}, E_USER_WARNING);
(new KNeighborsClassifier(n_neighbors: 1, p: 0.5))->fit($X, $y);
(new KNeighborsClassifier(n_neighbors: 1))->fit($X, [[0], [0], [0], [1], [1], [1]]);
restore_error_handler();
ok('p<1 warning', (bool) array_filter($seen, static fn (string $m): bool => str_starts_with($m, 'Mind that for 0 < p < 1')));
ok('column-vector warning', (bool) array_filter($seen, static fn (string $m): bool => str_starts_with($m, 'A column-vector y')));

// callable metric and callable weights
$m = new KNeighborsClassifier(n_neighbors: 2, metric: static fn (array $a, array $b): float => abs($a[0] - $b[0]) + abs($a[1] - $b[1]), weights: static fn (array $d): array => array_map(static fn (array $r): array => array_fill(0, count($r), 1.0), $d));
$m->fit($X, $y);
ok('callable metric + weights', $m->predict([[0.2, 0.2]]) === ['a']);

echo "{$total} checks, {$failed} failed\n";
exit($failed === 0 ? 0 : 1);

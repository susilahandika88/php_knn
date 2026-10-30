<?php

declare(strict_types=1);

namespace MachineLearning\Knn;

use BadMethodCallException;
use InvalidArgumentException;
use MachineLearning\Knn\Exceptions\InvalidParameterError;
use MachineLearning\Knn\Exceptions\NotFittedError;
use MachineLearning\Knn\Metrics\DistanceMetric;
use MachineLearning\Knn\Support\Validation;
use MachineLearning\Knn\Trees\BallTree;
use MachineLearning\Knn\Trees\KDTree;
use MachineLearning\Knn\Trees\NeighborTree;

/**
 * Classifier implementing the k-nearest neighbors vote, a from-scratch PHP port of
 * sklearn.neighbors.KNeighborsClassifier (checked against scikit-learn 1.7).
 *
 * Constructor arguments and public attributes use sklearn's names, so named arguments
 * read the same as in Python:
 *
 *     $knn = new KNeighborsClassifier(n_neighbors: 3, weights: 'distance');
 *     $knn->fit($X, $y);
 *     $knn->predict($queries);
 *     $knn->predict_proba($queries);
 *
 * Both camelCase (predictProba) and sklearn's snake_case (predict_proba) method names work.
 *
 * Differences from scikit-learn: no sparse input, no NaN-aware metrics (nan_euclidean) and no
 * parallelism (n_jobs is accepted and ignored). Neighbours at exactly equal distance are ordered
 * by training index.
 */
final class KNeighborsClassifier
{
    private const PARAMS = ['algorithm', 'leaf_size', 'metric', 'metric_params', 'n_jobs', 'n_neighbors', 'p', 'weights'];

    private const METRICS = [
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean',
        'hamming', 'haversine', 'infinity', 'jaccard', 'l1', 'l2', 'mahalanobis', 'manhattan', 'minkowski',
        'p', 'precomputed', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
        'sqeuclidean', 'yule',
    ];

    // ---- hyper-parameters (same names as sklearn) ----
    // (typed as mixed so a wrong value reaches validateParams() and gets sklearn's error message)
    public mixed $n_neighbors;
    /** @var string|callable|null 'uniform', 'distance', a callable taking the distance matrix, or null */
    public mixed $weights;
    public mixed $algorithm;
    public mixed $leaf_size;
    public mixed $p;
    /** @var string|callable */
    public mixed $metric;
    public mixed $metric_params;
    public mixed $n_jobs;

    // ---- fitted attributes ----
    /** @var array<int, mixed>|array<int, array<int, mixed>>|null sorted labels; one list per output when outputs_2d_ */
    public ?array $classes_ = null;
    public ?bool $outputs_2d_ = null;
    public ?int $n_features_in_ = null;
    public ?int $n_samples_fit_ = null;
    /** @var string|callable|null */
    public mixed $effective_metric_ = null;
    public ?array $effective_metric_params_ = null;
    /** Search method actually used: 'brute', 'kd_tree' or 'ball_tree'. */
    public ?string $_fit_method = null;

    /** @var array<int, array<int, float>>|null */
    private ?array $fitX = null;
    /** @var array<int, array<int, int>>|null class index per sample, one column per output */
    private ?array $encodedY = null;
    private ?NeighborTree $tree = null;
    /** @var callable|null */
    private $distanceFn = null;

    public function __construct(
        mixed $n_neighbors = 5,
        mixed $weights = 'uniform',
        mixed $algorithm = 'auto',
        mixed $leaf_size = 30,
        mixed $p = 2,
        mixed $metric = 'minkowski',
        mixed $metric_params = null,
        mixed $n_jobs = null,
    ) {
        $this->n_neighbors = $n_neighbors;
        $this->weights = $weights;
        $this->algorithm = $algorithm;
        $this->leaf_size = $leaf_size;
        $this->p = $p;
        $this->metric = $metric;
        $this->metric_params = $metric_params;
        $this->n_jobs = $n_jobs;
    }

    /** snake_case aliases (predict_proba, kneighbors_graph, get_params, set_params). */
    public function __call(string $name, array $arguments): mixed
    {
        $camel = lcfirst(str_replace('_', '', ucwords($name, '_')));
        if ($camel !== $name && method_exists($this, $camel)) {
            return $this->{$camel}(...$arguments);
        }
        throw new BadMethodCallException('Call to undefined method ' . self::class . "::{$name}()");
    }

    // ------------------------------------------------------------------ parameters

    /** @return array<string, mixed> */
    public function getParams(bool $deep = true): array
    {
        $out = [];
        foreach (self::PARAMS as $name) {
            $out[$name] = $this->{$name};
        }
        return $out;
    }

    /** @param array<string, mixed> $params */
    public function setParams(array $params): self
    {
        foreach ($params as $name => $value) {
            if (!in_array($name, self::PARAMS, true)) {
                throw new InvalidArgumentException(
                    "Invalid parameter '{$name}' for estimator KNeighborsClassifier. Valid parameters are: ['"
                    . implode("', '", self::PARAMS) . "']."
                );
            }
            $this->{$name} = $value;
        }
        return $this;
    }

    private function validateParams(): void
    {
        $got = static fn (mixed $v): string => Validation::repr($v);

        if (!is_string($this->algorithm) || !in_array($this->algorithm, ['auto', 'kd_tree', 'ball_tree', 'brute'], true)) {
            throw new InvalidParameterError("The 'algorithm' parameter of KNeighborsClassifier must be a str among {'auto', 'kd_tree', 'ball_tree', 'brute'}. Got " . $got($this->algorithm) . ' instead.');
        }
        if (!is_int($this->leaf_size) || $this->leaf_size < 1) {
            throw new InvalidParameterError("The 'leaf_size' parameter of KNeighborsClassifier must be an int in the range [1, inf). Got " . $got($this->leaf_size) . ' instead.');
        }
        $metricOk = is_string($this->metric) ? in_array($this->metric, array_merge(self::METRICS, ['nan_euclidean']), true) : is_callable($this->metric);
        if (!$metricOk) {
            throw new InvalidParameterError(
                "The 'metric' parameter of KNeighborsClassifier must be a str among {'" . implode("', '", self::METRICS)
                . "'}, or a callable. Got " . $got($this->metric) . ' instead.'
            );
        }
        if ($this->metric_params !== null && !is_array($this->metric_params)) {
            throw new InvalidParameterError("The 'metric_params' parameter of KNeighborsClassifier must be an instance of 'dict' or None. Got " . $got($this->metric_params) . ' instead.');
        }
        if ($this->n_jobs !== null && !is_int($this->n_jobs)) {
            throw new InvalidParameterError("The 'n_jobs' parameter of KNeighborsClassifier must be an int or None. Got " . $got($this->n_jobs) . ' instead.');
        }
        if ($this->n_neighbors !== null && (!is_int($this->n_neighbors) || $this->n_neighbors < 1)) {
            throw new InvalidParameterError("The 'n_neighbors' parameter of KNeighborsClassifier must be an int in the range [1, inf) or None. Got " . $got($this->n_neighbors) . ' instead.');
        }
        if ($this->p !== null && (!(is_int($this->p) || is_float($this->p)) || !($this->p > 0))) {
            throw new InvalidParameterError("The 'p' parameter of KNeighborsClassifier must be a float in the range (0.0, inf] or None. Got " . $got($this->p) . ' instead.');
        }
        $w = $this->weights;
        if (!($w === null || $w === 'uniform' || $w === 'distance' || (!is_string($w) && is_callable($w)))) {
            throw new InvalidParameterError("The 'weights' parameter of KNeighborsClassifier must be a str among {'uniform', 'distance'}, a callable or None. Got " . $got($w) . ' instead.');
        }
    }

    // ------------------------------------------------------------------ fit

    /**
     * @param array<int, array<int, int|float>> $X training samples, shape (n_samples, n_features)
     * @param array<int, mixed>|array<int, array<int, mixed>> $y labels, or one row of labels per sample for multi-output
     */
    public function fit(array $X, array $y): self
    {
        $this->validateParams();

        $precomputed = $this->metric === 'precomputed';
        $rows = Validation::checkArray($X, 'X', $precomputed);
        [$targets, $is2d, $singleColumn] = Validation::checkTarget($y, count($rows));

        if ($singleColumn) {
            trigger_error(
                'A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().',
                E_USER_WARNING
            );
        }
        $this->outputs_2d_ = $is2d && !$singleColumn;

        $columns = [];
        if ($is2d) {
            $width = count($targets[0]);
            for ($k = 0; $k < $width; $k++) {
                $columns[] = array_column($targets, $k);
            }
        } else {
            $columns[] = $targets;
        }
        $classes = [];
        $encoded = array_fill(0, count($rows), []);
        foreach ($columns as $column) {
            [$cls, $codes] = Validation::encodeLabels($column, $this->outputs_2d_);
            $classes[] = $cls;
            foreach ($codes as $i => $code) {
                $encoded[$i][] = $code;
            }
        }
        $this->classes_ = $this->outputs_2d_ ? $classes : $classes[0];
        $this->encodedY = $encoded;

        $this->checkAlgorithmMetric();

        $params = $this->metric_params ?? [];
        $effectiveP = $params['p'] ?? $this->p;
        $effParams = $params;
        if ($this->metric === 'minkowski') {
            $effParams['p'] = $effectiveP;
        }
        $this->effective_metric_ = $this->metric;
        if ($this->metric === 'minkowski') {
            $p = $effParams['p'] ?? 2;
            unset($effParams['p']);
            $w = $effParams['w'] ?? null;
            unset($effParams['w']);
            if ($p == 1 && $w === null) {
                $this->effective_metric_ = 'manhattan';
            } elseif ($p == 2 && $w === null) {
                $this->effective_metric_ = 'euclidean';
            } elseif (is_float($p) && is_infinite($p) && $w === null) {
                $this->effective_metric_ = 'chebyshev';
            } else {
                $effParams['p'] = $p;
                $effParams['w'] = $w;
            }
        }
        $this->effective_metric_params_ = $effParams;

        if ($precomputed) {
            if (count($rows) !== count($rows[0])) {
                throw new InvalidArgumentException('Precomputed matrix must be square. Input is a ' . count($rows) . 'x' . count($rows[0]) . ' matrix.');
            }
        }
        $this->n_features_in_ = count($rows[0]);
        $this->n_samples_fit_ = count($rows);
        $this->fitX = $rows;

        $this->selectFitMethod();
        $this->checkSemiMetric();
        $this->tree = null;
        $this->distanceFn = null;
        if (!$precomputed) {
            $this->distanceFn = $this->buildDistance();
        }

        if ($this->_fit_method === 'ball_tree') {
            $this->tree = new BallTree($rows, $this->leaf_size, $this->distanceFn);
        } elseif ($this->_fit_method === 'kd_tree') {
            $eff = $this->effective_metric_;
            $ep = $this->effective_metric_params_;
            if ($eff === 'minkowski' && ($ep['w'] ?? null) !== null) {
                throw new InvalidArgumentException(
                    "algorithm='kd_tree' is not valid for metric='minkowski' with a weight parameter 'w': try algorithm='ball_tree' or algorithm='brute' instead."
                );
            }
            [$kdP, $kdW] = $this->minkowskiExponent();
            $this->tree = new KDTree($rows, $this->leaf_size, $this->distanceFn, $kdP, $kdW);
        }
        return $this;
    }

    private function checkAlgorithmMetric(): void
    {
        $metric = $this->metric;
        if ($this->algorithm === 'auto') {
            $algCheck = $metric === 'precomputed'
                ? 'brute'
                : (!is_string($metric) || in_array($metric, DistanceMetric::BALL_TREE, true) ? 'ball_tree' : 'brute');
        } else {
            $algCheck = $this->algorithm;
        }

        if (!is_string($metric)) {
            if ($this->algorithm === 'kd_tree') {
                throw new InvalidArgumentException(
                    "kd_tree does not support callable metric 'callable'Function call overheadwill result in very poor performance."
                );
            }
        } elseif ($metric === 'nan_euclidean') {
            throw new InvalidArgumentException("Metric 'nan_euclidean' is not supported by this PHP port.");
        } elseif ($metric !== 'precomputed' && !DistanceMetric::isValidFor($algCheck, $metric)) {
            throw new InvalidArgumentException(
                "Metric '{$metric}' not valid. Use sorted(sklearn.neighbors.VALID_METRICS['{$algCheck}']) to get valid options. Metric can also be a callable function."
            );
        } elseif ($metric === 'precomputed' && $algCheck !== 'brute') {
            throw new InvalidArgumentException("Metric 'precomputed' not valid. Use sorted(sklearn.neighbors.VALID_METRICS['{$algCheck}']) to get valid options. Metric can also be a callable function.");
        }

        if ($this->metric_params !== null && array_key_exists('p', $this->metric_params) && $this->p !== null) {
            trigger_error('Parameter p is found in metric_params. The corresponding parameter from __init__ is ignored.', E_USER_WARNING);
        }
    }

    private function selectFitMethod(): void
    {
        $this->_fit_method = $this->algorithm;
        if ($this->_fit_method !== 'auto') {
            return;
        }
        $eff = $this->effective_metric_;
        $ep = $this->effective_metric_params_;
        $nFeat = $this->n_features_in_;
        $nSamples = $this->n_samples_fit_;

        if (
            $this->metric === 'precomputed'
            || $nFeat > 15
            || ($this->n_neighbors !== null && $this->n_neighbors >= intdiv($nSamples, 2))
        ) {
            $this->_fit_method = 'brute';
        } elseif ($eff === 'minkowski' && $ep['p'] < 1) {
            $this->_fit_method = 'brute';
        } elseif ($eff === 'minkowski' && ($ep['w'] ?? null) !== null) {
            $this->_fit_method = 'ball_tree';
        } elseif (is_string($eff) && in_array($eff, DistanceMetric::KD_TREE, true)) {
            $this->_fit_method = 'kd_tree';
        } elseif (!is_string($eff) || in_array($eff, DistanceMetric::BALL_TREE, true)) {
            $this->_fit_method = 'ball_tree';
        } else {
            $this->_fit_method = 'brute';
        }
    }

    private function checkSemiMetric(): void
    {
        if ($this->effective_metric_ === 'minkowski' && $this->effective_metric_params_['p'] < 1) {
            if ($this->_fit_method === 'brute') {
                trigger_error('Mind that for 0 < p < 1, Minkowski metrics are not distance metrics. Continuing the execution with `algorithm=\'brute\'`.', E_USER_WARNING);
            } else {
                throw new InvalidArgumentException(
                    "algorithm=\"{$this->_fit_method}\" does not support 0 < p < 1 for the Minkowski metric. To resolve this problem either set p >= 1 or algorithm=\"brute\"."
                );
            }
        }
    }

    /** @return callable(array<int,float>, array<int,float>): float */
    private function buildDistance(): callable
    {
        $eff = $this->effective_metric_;
        $params = $this->effective_metric_params_ ?? [];
        if (!is_string($eff)) {
            $fn = $eff;
            return $params === []
                ? static fn (array $a, array $b): float => (float) $fn($a, $b)
                : static fn (array $a, array $b): float => (float) $fn($a, $b, ...$params);
        }
        return DistanceMetric::resolve($eff, $params);
    }

    /** @return array{0: float, 1: array<int, float>|null} Minkowski exponent and weights for the KD-tree bounds */
    private function minkowskiExponent(): array
    {
        $ep = $this->effective_metric_params_ ?? [];
        return match ($this->effective_metric_) {
            'euclidean', 'l2' => [2.0, null],
            'manhattan', 'cityblock', 'l1' => [1.0, null],
            'chebyshev', 'infinity' => [INF, null],
            default => [(float) ($ep['p'] ?? 2), $ep['w'] ?? null],
        };
    }

    // ------------------------------------------------------------------ neighbours

    /**
     * Finds the k nearest neighbours of each query row.
     *
     * With $X === null the training points are queried and each point is excluded from its own result.
     *
     * @param array<int, array<int, int|float>>|null $X
     * @return array{0: array<int, array<int, float>>, 1: array<int, array<int, int>>}|array<int, array<int, int>>
     *         [distances, indices] (ascending distance), or only the indices when $return_distance is false
     */
    public function kneighbors(?array $X = null, ?int $n_neighbors = null, bool $return_distance = true): array
    {
        $this->checkIsFitted();

        if ($n_neighbors === null) {
            $n_neighbors = $this->n_neighbors;
        } elseif ($n_neighbors <= 0) {
            throw new InvalidArgumentException("Expected n_neighbors > 0. Got {$n_neighbors}");
        }
        if ($n_neighbors === null) {
            throw new InvalidArgumentException('n_neighbors must be set to query neighbours');
        }

        $queryIsTrain = $X === null;
        if ($queryIsTrain) {
            $queries = $this->fitX;
            $n_neighbors++;
        } else {
            $queries = $this->checkQuery($X);
        }

        $nFit = $this->n_samples_fit_;
        if ($n_neighbors > $nFit) {
            if ($queryIsTrain) {
                $n_neighbors--;
                $inequality = 'n_neighbors < n_samples_fit';
            } else {
                $inequality = 'n_neighbors <= n_samples_fit';
            }
            throw new InvalidArgumentException(
                "Expected {$inequality}, but n_neighbors = {$n_neighbors}, n_samples_fit = {$nFit}, n_samples = " . count($queries)
            );
        }

        $dists = [];
        $inds = [];
        foreach ($queries as $q) {
            [$d, $i] = $this->queryOne($q, $n_neighbors);
            $dists[] = $d;
            $inds[] = $i;
        }

        if ($queryIsTrain) {
            // drop each sample from its own neighbour list; if it is missing (many duplicates), drop the first
            $keep = $n_neighbors - 1;
            foreach ($inds as $r => $row) {
                $pos = array_search($r, $row, true);
                if ($pos === false) {
                    $pos = 0;
                }
                array_splice($inds[$r], $pos, 1);
                array_splice($dists[$r], $pos, 1);
                $inds[$r] = array_slice($inds[$r], 0, $keep);
                $dists[$r] = array_slice($dists[$r], 0, $keep);
            }
        }
        return $return_distance ? [$dists, $inds] : $inds;
    }

    /**
     * Graph of k-neighbours in scipy CSR layout.
     *
     * @param array<int, array<int, int|float>>|null $X
     * @return array{data: array<int, float>, indices: array<int, int>, indptr: array<int, int>, shape: array{0:int,1:int}}
     */
    public function kneighborsGraph(?array $X = null, ?int $n_neighbors = null, string $mode = 'connectivity'): array
    {
        $this->checkIsFitted();
        $n_neighbors ??= $this->n_neighbors;

        if ($mode === 'connectivity') {
            $ind = $this->kneighbors($X, $n_neighbors, false);
            $data = array_fill(0, count($ind) * $n_neighbors, 1.0);
        } elseif ($mode === 'distance') {
            [$dist, $ind] = $this->kneighbors($X, $n_neighbors, true);
            $data = array_merge(...$dist);
        } else {
            throw new InvalidArgumentException("Unsupported mode, must be one of \"connectivity\", or \"distance\" but got \"{$mode}\" instead");
        }
        $nQueries = count($ind);
        return [
            'data' => $data,
            'indices' => array_merge(...$ind),
            'indptr' => range(0, $nQueries * $n_neighbors, $n_neighbors),
            'shape' => [$nQueries, $this->n_samples_fit_],
        ];
    }

    /**
     * @param array<int, float> $q
     * @return array{0: array<int, float>, 1: array<int, int>}
     */
    private function queryOne(array $q, int $k): array
    {
        if ($this->tree !== null) {
            return $this->tree->query($q, $k);
        }

        $bestD = [];
        $bestI = [];
        $worst = INF;
        if ($this->metric === 'precomputed') {
            foreach ($q as $j => $d) {
                if ($d <= $worst) {
                    $this->insert($bestD, $bestI, $k, $d, $j);
                    $worst = count($bestD) < $k ? INF : $bestD[$k - 1];
                }
            }
            return [$bestD, $bestI];
        }

        $fit = $this->fitX;
        if ($this->effective_metric_ === 'euclidean') {
            // compare squared distances, take the square root only for the winners
            foreach ($fit as $j => $row) {
                $s = 0.0;
                foreach ($q as $f => $x) {
                    $d = $x - $row[$f];
                    $s += $d * $d;
                }
                if ($s <= $worst) {
                    $this->insert($bestD, $bestI, $k, $s, $j);
                    $worst = count($bestD) < $k ? INF : $bestD[$k - 1];
                }
            }
            return [array_map('sqrt', $bestD), $bestI];
        }

        $fn = $this->distanceFn;
        foreach ($fit as $j => $row) {
            $d = $fn($q, $row);
            if ($d <= $worst || is_nan($d)) {
                $this->insert($bestD, $bestI, $k, $d, $j);
                $worst = count($bestD) < $k ? INF : $bestD[$k - 1];
            }
        }
        return [$bestD, $bestI];
    }

    /**
     * @param array<int, float> $bestD
     * @param array<int, int> $bestI
     */
    private function insert(array &$bestD, array &$bestI, int $k, float $dist, int $idx): void
    {
        $n = count($bestD);
        if ($n === $k) {
            if ($dist > $bestD[$n - 1] || ($dist == $bestD[$n - 1] && $idx > $bestI[$n - 1])) {
                return;
            }
            array_pop($bestD);
            array_pop($bestI);
            $n--;
        }
        $pos = $n;
        while ($pos > 0 && ($bestD[$pos - 1] > $dist || ($bestD[$pos - 1] == $dist && $bestI[$pos - 1] > $idx))) {
            $pos--;
        }
        array_splice($bestD, $pos, 0, [$dist]);
        array_splice($bestI, $pos, 0, [$idx]);
    }

    // ------------------------------------------------------------------ predict

    /**
     * Predicts the class label of each query row.
     *
     * @param array<int, array<int, int|float>> $X
     * @return array<int, mixed>|array<int, array<int, mixed>> labels (one list of labels per sample when multi-output)
     */
    public function predict(array $X): array
    {
        $this->checkIsFitted();
        [$dist, $ind, $weights] = $this->neighboursAndWeights($X, true);

        $nOut = $this->outputs_2d_ ? count($this->classes_) : 1;
        $pred = [];
        foreach ($ind as $r => $neighbours) {
            $row = [];
            for ($k = 0; $k < $nOut; $k++) {
                $classes = $this->outputs_2d_ ? $this->classes_[$k] : $this->classes_;
                $votes = [];
                foreach ($neighbours as $c => $idx) {
                    $label = $this->encodedY[$idx][$k];
                    $votes[$label] = ($votes[$label] ?? 0.0) + ($weights === null ? 1.0 : $weights[$r][$c]);
                }
                // first class with the highest vote wins, so ties go to the smallest class
                ksort($votes);
                $best = null;
                $bestCount = 0.0;
                foreach ($votes as $label => $count) {
                    if ($count > $bestCount) {
                        $best = $label;
                        $bestCount = $count;
                    }
                }
                $row[] = $classes[$best ?? 0];
            }
            $pred[] = $this->outputs_2d_ ? $row : $row[0];
        }
        return $pred;
    }

    /**
     * Class membership probabilities of each query row.
     *
     * @param array<int, array<int, int|float>> $X
     * @return array<int, array<int, float>>|array<int, array<int, array<int, float>>> one row of probabilities per sample,
     *         or one such matrix per output when multi-output; columns follow classes_
     */
    public function predictProba(array $X): array
    {
        $this->checkIsFitted();
        [$dist, $ind, $weights] = $this->neighboursAndWeights($X, false);

        $nOut = $this->outputs_2d_ ? count($this->classes_) : 1;
        $all = [];
        for ($k = 0; $k < $nOut; $k++) {
            $nClasses = count($this->outputs_2d_ ? $this->classes_[$k] : $this->classes_);
            $proba = [];
            foreach ($ind as $r => $neighbours) {
                $row = array_fill(0, $nClasses, 0.0);
                foreach ($neighbours as $c => $idx) {
                    $row[$this->encodedY[$idx][$k]] += $weights === null ? 1.0 : $weights[$r][$c];
                }
                $sum = array_sum($row);
                foreach ($row as $j => $v) {
                    $row[$j] = $sum == 0.0 ? NAN : $v / $sum;
                }
                $proba[] = $row;
            }
            $all[] = $proba;
        }
        return $this->outputs_2d_ ? $all : $all[0];
    }

    /**
     * Mean accuracy on the given test data and labels (subset accuracy for multi-output).
     *
     * @param array<int, array<int, int|float>> $X
     * @param array<int, mixed> $y
     * @param array<int, int|float>|null $sample_weight
     */
    public function score(array $X, array $y, ?array $sample_weight = null): float
    {
        $pred = $this->predict($X);
        if (count($pred) !== count($y)) {
            throw new InvalidArgumentException('Found input variables with inconsistent numbers of samples: [' . count($y) . ', ' . count($pred) . ']');
        }
        $y = array_values($y);
        $w = $sample_weight === null ? array_fill(0, count($y), 1.0) : array_values($sample_weight);
        $hit = 0.0;
        $total = 0.0;
        foreach ($pred as $i => $p) {
            $total += $w[$i];
            if (self::sameLabel($p, $y[$i])) {
                $hit += $w[$i];
            }
        }
        return $total == 0.0 ? NAN : $hit / $total;
    }

    /**
     * @param array<int, array<int, int|float>> $X
     * @return array{0: array<int, array<int, float>>|null, 1: array<int, array<int, int>>, 2: array<int, array<int, float>>|null}
     */
    private function neighboursAndWeights(array $X, bool $forPredict): array
    {
        $weightsParam = $this->weights;
        if ($weightsParam === 'distance' || (!is_string($weightsParam) && is_callable($weightsParam))) {
            [$dist, $ind] = $this->kneighbors($X);
        } else {
            $dist = null;
            $ind = $this->kneighbors($X, null, false);
        }

        $weights = $dist === null ? null : $this->getWeights($dist);
        if ($weights !== null) {
            foreach ($weights as $row) {
                $allZero = true;
                foreach ($row as $w) {
                    if ($w != 0) {
                        $allZero = false;
                        break;
                    }
                }
                if ($allZero) {
                    throw new InvalidArgumentException(
                        "All neighbors of some sample is getting zero weights. Please modify 'weights' to avoid this case if you are using a user-defined function."
                    );
                }
            }
        }
        return [$dist, $ind, $weights];
    }

    /**
     * @param array<int, array<int, float>> $dist
     * @return array<int, array<int, float>>|null
     */
    private function getWeights(array $dist): ?array
    {
        if ($this->weights === null || $this->weights === 'uniform') {
            return null;
        }
        if ($this->weights === 'distance') {
            $out = [];
            foreach ($dist as $r => $row) {
                $hasZero = in_array(0.0, $row, true);
                $w = [];
                foreach ($row as $d) {
                    // a query sitting on a training point: those points get weight 1, the rest 0
                    $w[] = $hasZero ? ($d == 0.0 ? 1.0 : 0.0) : 1.0 / $d;
                }
                $out[$r] = $w;
            }
            return $out;
        }
        return ($this->weights)($dist);
    }

    // ------------------------------------------------------------------ checks

    private function checkIsFitted(): void
    {
        if ($this->_fit_method === null) {
            throw new NotFittedError(
                "This KNeighborsClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            );
        }
    }

    /**
     * @param array<int, array<int, int|float>> $X
     * @return array<int, array<int, float>>
     */
    private function checkQuery(array $X): array
    {
        $precomputed = $this->metric === 'precomputed';
        $rows = Validation::checkArray($X, 'X', $precomputed);
        $width = count($rows[0]);
        if ($width !== $this->n_features_in_) {
            throw new InvalidArgumentException("X has {$width} features, but KNeighborsClassifier is expecting {$this->n_features_in_} features as input.");
        }
        return $rows;
    }

    private static function sameLabel(mixed $a, mixed $b): bool
    {
        if (is_array($a) || is_array($b)) {
            if (!is_array($a) || !is_array($b) || count($a) !== count($b)) {
                return false;
            }
            foreach (array_values($a) as $i => $v) {
                if (!self::sameLabel($v, array_values($b)[$i])) {
                    return false;
                }
            }
            return true;
        }
        if (is_string($a) || is_string($b)) {
            return (string) $a === (string) $b;
        }
        return (float) $a === (float) $b;
    }
}

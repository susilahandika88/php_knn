<?php

declare(strict_types=1);

namespace MachineLearning\Knn\Metrics;

use InvalidArgumentException;

/**
 * Distance functions between two feature vectors (plain float arrays).
 *
 * resolve() returns a closure fn(array $a, array $b): float for a metric name
 * (or a user callable), after validating its parameters like sklearn's
 * DistanceMetric does.
 */
final class DistanceMetric
{
    /** Metrics the KD-tree supports (sklearn VALID_METRICS['kd_tree']). */
    public const KD_TREE = [
        'euclidean', 'l2', 'minkowski', 'p', 'manhattan', 'cityblock', 'l1', 'chebyshev', 'infinity',
    ];

    /** Metrics the Ball-tree supports (sklearn VALID_METRICS['ball_tree']). */
    public const BALL_TREE = [
        'euclidean', 'l2', 'minkowski', 'p', 'manhattan', 'cityblock', 'l1', 'chebyshev', 'infinity',
        'seuclidean', 'mahalanobis', 'hamming', 'canberra', 'braycurtis', 'jaccard', 'dice',
        'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'haversine',
    ];

    /** Metrics the brute-force search supports (this port). */
    public const BRUTE = [
        'euclidean', 'l2', 'minkowski', 'p', 'manhattan', 'cityblock', 'l1', 'chebyshev', 'infinity',
        'sqeuclidean', 'cosine', 'correlation', 'seuclidean', 'mahalanobis', 'hamming', 'canberra',
        'braycurtis', 'jaccard', 'dice', 'rogerstanimoto', 'russellrao', 'sokalmichener',
        'sokalsneath', 'haversine', 'yule',
    ];

    private function __construct()
    {
    }

    /**
     * @param string|callable $metric
     * @param array<string, mixed> $params metric_params (p, w, V, VI)
     * @return callable(array<int,float>, array<int,float>): float
     */
    public static function resolve(string|callable $metric, array $params = []): callable
    {
        if (!is_string($metric)) {
            return static fn (array $a, array $b): float => (float) $metric($a, $b);
        }

        switch ($metric) {
            case 'euclidean':
            case 'l2':
                return static function (array $a, array $b): float {
                    $s = 0.0;
                    foreach ($a as $i => $x) {
                        $d = $x - $b[$i];
                        $s += $d * $d;
                    }
                    return sqrt($s);
                };

            case 'sqeuclidean':
                return static function (array $a, array $b): float {
                    $s = 0.0;
                    foreach ($a as $i => $x) {
                        $d = $x - $b[$i];
                        $s += $d * $d;
                    }
                    return $s;
                };

            case 'manhattan':
            case 'cityblock':
            case 'l1':
                return static function (array $a, array $b): float {
                    $s = 0.0;
                    foreach ($a as $i => $x) {
                        $s += abs($x - $b[$i]);
                    }
                    return $s;
                };

            case 'chebyshev':
            case 'infinity':
                return static function (array $a, array $b): float {
                    $m = 0.0;
                    foreach ($a as $i => $x) {
                        $d = abs($x - $b[$i]);
                        if ($d > $m) {
                            $m = $d;
                        }
                    }
                    return $m;
                };

            case 'minkowski':
            case 'p':
                return self::minkowski($params);

            case 'cosine':
                return static function (array $a, array $b): float {
                    $dot = 0.0;
                    $na = 0.0;
                    $nb = 0.0;
                    foreach ($a as $i => $x) {
                        $y = $b[$i];
                        $dot += $x * $y;
                        $na += $x * $x;
                        $nb += $y * $y;
                    }
                    // zero vectors stay zero after normalisation, like sklearn's cosine_distances
                    if ($na == 0.0 || $nb == 0.0) {
                        return 1.0;
                    }
                    $d = 1.0 - $dot / (sqrt($na) * sqrt($nb));
                    return min(2.0, max(0.0, $d));
                };

            case 'correlation':
                return static function (array $a, array $b): float {
                    $n = count($a);
                    $ma = array_sum($a) / $n;
                    $mb = array_sum($b) / $n;
                    $dot = 0.0;
                    $na = 0.0;
                    $nb = 0.0;
                    foreach ($a as $i => $x) {
                        $u = $x - $ma;
                        $v = $b[$i] - $mb;
                        $dot += $u * $v;
                        $na += $u * $u;
                        $nb += $v * $v;
                    }
                    $den = sqrt($na) * sqrt($nb);
                    return $den == 0.0 ? NAN : 1.0 - $dot / $den;
                };

            case 'seuclidean':
                $v = self::vectorParam($params, 'V', 'seuclidean');
                return static function (array $a, array $b) use ($v): float {
                    $s = 0.0;
                    foreach ($a as $i => $x) {
                        $d = $x - $b[$i];
                        $s += $d * $d / $v[$i];
                    }
                    return sqrt($s);
                };

            case 'mahalanobis':
                $vi = $params['VI'] ?? $params['V'] ?? null;
                if (!is_array($vi)) {
                    throw new InvalidArgumentException("mahalanobis requires metric_params['VI'] (inverse covariance matrix)");
                }
                if (isset($params['V']) && !isset($params['VI'])) {
                    $vi = self::invert($params['V']);
                }
                return static function (array $a, array $b) use ($vi): float {
                    $n = count($a);
                    $d = [];
                    for ($i = 0; $i < $n; $i++) {
                        $d[$i] = $a[$i] - $b[$i];
                    }
                    $s = 0.0;
                    for ($i = 0; $i < $n; $i++) {
                        $row = 0.0;
                        for ($j = 0; $j < $n; $j++) {
                            $row += $vi[$i][$j] * $d[$j];
                        }
                        $s += $d[$i] * $row;
                    }
                    return sqrt(max(0.0, $s));
                };

            case 'hamming':
                return static function (array $a, array $b): float {
                    $c = 0;
                    foreach ($a as $i => $x) {
                        if ($x != $b[$i]) {
                            $c++;
                        }
                    }
                    return $c / count($a);
                };

            case 'canberra':
                return static function (array $a, array $b): float {
                    $s = 0.0;
                    foreach ($a as $i => $x) {
                        $den = abs($x) + abs($b[$i]);
                        if ($den != 0.0) {
                            $s += abs($x - $b[$i]) / $den;
                        }
                    }
                    return $s;
                };

            case 'braycurtis':
                return static function (array $a, array $b): float {
                    $num = 0.0;
                    $den = 0.0;
                    foreach ($a as $i => $x) {
                        $num += abs($x - $b[$i]);
                        $den += abs($x + $b[$i]);
                    }
                    return $den == 0.0 ? 0.0 : $num / $den;
                };

            case 'haversine':
                return static function (array $a, array $b): float {
                    if (count($a) !== 2) {
                        throw new InvalidArgumentException('haversine distance is only defined for 2 dimensions');
                    }
                    $sdlat = sin(($a[0] - $b[0]) / 2.0);
                    $sdlon = sin(($a[1] - $b[1]) / 2.0);
                    return 2.0 * asin(sqrt($sdlat * $sdlat + cos($a[0]) * cos($b[0]) * $sdlon * $sdlon));
                };

            case 'jaccard':
            case 'dice':
            case 'rogerstanimoto':
            case 'russellrao':
            case 'sokalmichener':
            case 'sokalsneath':
            case 'yule':
                return self::boolean($metric);
        }

        throw new InvalidArgumentException("Unrecognized metric '{$metric}'");
    }

    public static function isValidFor(string $algorithm, string|callable $metric): bool
    {
        if (!is_string($metric)) {
            return $algorithm === 'brute' || $algorithm === 'ball_tree';
        }
        return match ($algorithm) {
            'kd_tree' => in_array($metric, self::KD_TREE, true),
            'ball_tree' => in_array($metric, self::BALL_TREE, true),
            default => in_array($metric, self::BRUTE, true),
        };
    }

    /** @param array<string, mixed> $params */
    private static function minkowski(array $params): callable
    {
        $p = $params['p'] ?? 2;
        if (!is_int($p) && !is_float($p)) {
            throw new InvalidArgumentException("p must be a number, got " . gettype($p));
        }
        if (!($p > 0)) {
            throw new InvalidArgumentException('p must be greater than zero for minkowski metric');
        }
        $w = $params['w'] ?? null;
        if ($w !== null) {
            foreach ($w as $x) {
                if ($x < 0) {
                    throw new InvalidArgumentException('w cannot contain negative weights');
                }
            }
        }
        $p = (float) $p;
        if (is_infinite($p)) {
            return self::resolve('chebyshev');
        }
        return static function (array $a, array $b) use ($p, $w): float {
            $s = 0.0;
            foreach ($a as $i => $x) {
                $t = abs($x - $b[$i]) ** $p;
                $s += $w === null ? $t : $w[$i] * $t;
            }
            return $s ** (1.0 / $p);
        };
    }

    /** @param array<string, mixed> $params @return array<int, float> */
    private static function vectorParam(array $params, string $key, string $metric): array
    {
        if (!isset($params[$key]) || !is_array($params[$key])) {
            throw new InvalidArgumentException("{$metric} requires metric_params['{$key}']");
        }
        return array_values($params[$key]);
    }

    /** Boolean-vector dissimilarities (scipy definitions); nonzero means true. */
    private static function boolean(string $name): callable
    {
        return static function (array $a, array $b) use ($name): float {
            $ntt = $ntf = $nft = $nff = 0;
            foreach ($a as $i => $x) {
                $u = $x != 0;
                $v = $b[$i] != 0;
                if ($u && $v) {
                    $ntt++;
                } elseif ($u) {
                    $ntf++;
                } elseif ($v) {
                    $nft++;
                } else {
                    $nff++;
                }
            }
            $n = $ntt + $ntf + $nft + $nff;
            switch ($name) {
                case 'jaccard':
                    $den = $ntt + $ntf + $nft;
                    return $den == 0 ? 0.0 : ($ntf + $nft) / $den;
                case 'dice':
                    $den = 2 * $ntt + $ntf + $nft;
                    return $den == 0 ? 0.0 : ($ntf + $nft) / $den;
                case 'rogerstanimoto':
                    return 2.0 * ($ntf + $nft) / ($ntt + $nff + 2 * ($ntf + $nft));
                case 'russellrao':
                    return ($n - $ntt) / $n;
                case 'sokalmichener':
                    return 2.0 * ($ntf + $nft) / ($ntt + $nff + 2 * ($ntf + $nft));
                case 'yule':
                    $half = (float) ($ntf * $nft);
                    return $half == 0.0 ? 0.0 : 2.0 * $half / ($ntt * $nff + $half);
                default: // sokalsneath
                    $r = 2.0 * ($ntf + $nft);
                    return $r == 0.0 && $ntt == 0 ? 0.0 : $r / ($ntt + $r);
            }
        };
    }

    /** Gauss-Jordan inverse of a small square matrix. @param array<int, array<int, float>> $m @return array<int, array<int, float>> */
    private static function invert(array $m): array
    {
        $n = count($m);
        $a = [];
        for ($i = 0; $i < $n; $i++) {
            $a[$i] = array_merge(array_values($m[$i]), array_fill(0, $n, 0.0));
            $a[$i][$n + $i] = 1.0;
        }
        for ($c = 0; $c < $n; $c++) {
            $pivot = $c;
            for ($r = $c + 1; $r < $n; $r++) {
                if (abs($a[$r][$c]) > abs($a[$pivot][$c])) {
                    $pivot = $r;
                }
            }
            if ($a[$pivot][$c] == 0.0) {
                throw new InvalidArgumentException('Singular matrix for mahalanobis metric');
            }
            [$a[$c], $a[$pivot]] = [$a[$pivot], $a[$c]];
            $d = $a[$c][$c];
            for ($j = 0; $j < 2 * $n; $j++) {
                $a[$c][$j] /= $d;
            }
            for ($r = 0; $r < $n; $r++) {
                if ($r !== $c) {
                    $f = $a[$r][$c];
                    for ($j = 0; $j < 2 * $n; $j++) {
                        $a[$r][$j] -= $f * $a[$c][$j];
                    }
                }
            }
        }
        $inv = [];
        for ($i = 0; $i < $n; $i++) {
            $inv[$i] = array_slice($a[$i], $n);
        }
        return $inv;
    }
}

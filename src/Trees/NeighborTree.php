<?php

declare(strict_types=1);

namespace MachineLearning\Knn\Trees;

/**
 * Spatial index answering k-nearest-neighbour queries.
 *
 * Results are ordered by (distance, training index), so every algorithm
 * returns the same neighbours even when several points are equally far away.
 */
abstract class NeighborTree
{
    /**
     * @param array<int, array<int, float>> $data training rows
     * @param callable(array<int,float>, array<int,float>): float $distance
     */
    public function __construct(
        protected array $data,
        protected int $leafSize,
        protected $distance,
    ) {
    }

    /**
     * @param array<int, float> $query
     * @return array{0: array<int, float>, 1: array<int, int>} distances and indices, ascending
     */
    abstract public function query(array $query, int $k): array;

    /**
     * Keeps the k best (distance, index) pairs sorted ascending; returns the current k-th distance.
     *
     * @param array<int, float> $bestDist
     * @param array<int, int> $bestIdx
     */
    protected static function offer(array &$bestDist, array &$bestIdx, int $k, float $dist, int $idx): void
    {
        $n = count($bestDist);
        if ($n === $k) {
            $lastD = $bestDist[$n - 1];
            if ($dist > $lastD || ($dist == $lastD && $idx > $bestIdx[$n - 1])) {
                return;
            }
            array_pop($bestDist);
            array_pop($bestIdx);
            $n--;
        }
        $pos = $n;
        while ($pos > 0 && ($bestDist[$pos - 1] > $dist || ($bestDist[$pos - 1] == $dist && $bestIdx[$pos - 1] > $idx))) {
            $pos--;
        }
        array_splice($bestDist, $pos, 0, [$dist]);
        array_splice($bestIdx, $pos, 0, [$idx]);
    }

    /** @param array<int, float> $bestDist */
    protected static function worst(array $bestDist, int $k): float
    {
        return count($bestDist) < $k ? INF : $bestDist[$k - 1];
    }

    /**
     * Index of the feature with the largest spread among the rows in $idx[$start..$end).
     *
     * @param array<int, array<int, float>> $data
     * @param array<int, int> $idx
     */
    protected static function widestDimension(array $data, array $idx, int $start, int $end): int
    {
        $d = count($data[$idx[$start]]);
        $best = 0;
        $bestSpread = -1.0;
        for ($j = 0; $j < $d; $j++) {
            $lo = INF;
            $hi = -INF;
            for ($i = $start; $i < $end; $i++) {
                $v = $data[$idx[$i]][$j];
                if ($v < $lo) {
                    $lo = $v;
                }
                if ($v > $hi) {
                    $hi = $v;
                }
            }
            if ($hi - $lo > $bestSpread) {
                $bestSpread = $hi - $lo;
                $best = $j;
            }
        }
        return $best;
    }

    /**
     * Sorts $idx[$start..$end) by feature $dim (ties by index, so the split is deterministic).
     *
     * @param array<int, array<int, float>> $data
     * @param array<int, int> $idx
     */
    protected static function sortRange(array $data, array &$idx, int $start, int $end, int $dim): void
    {
        $slice = array_slice($idx, $start, $end - $start);
        usort($slice, static function (int $a, int $b) use ($data, $dim): int {
            return [$data[$a][$dim], $a] <=> [$data[$b][$dim], $b];
        });
        array_splice($idx, $start, $end - $start, $slice);
    }
}

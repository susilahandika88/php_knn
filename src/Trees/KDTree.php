<?php

declare(strict_types=1);

namespace MachineLearning\Knn\Trees;

/**
 * KD-tree for Minkowski-family metrics: splits on the widest feature at the median
 * and prunes with the distance from the query to each node's bounding box.
 */
final class KDTree extends NeighborTree
{
    /** @var array<int, int> */
    private array $idx;

    /** @var array<int, array{0:int,1:int,2:int,3:int,4:array<int,float>,5:array<int,float>}> start, end, left, right, lo, hi (left = -1 for a leaf) */
    private array $nodes = [];

    /**
     * @param array<int, array<int, float>> $data
     * @param callable(array<int,float>, array<int,float>): float $distance
     * @param float $p Minkowski exponent (INF for Chebyshev)
     * @param array<int, float>|null $weights optional Minkowski weights
     */
    public function __construct(
        array $data,
        int $leafSize,
        $distance,
        private float $p = 2.0,
        private ?array $weights = null,
    ) {
        parent::__construct($data, $leafSize, $distance);
        $this->idx = array_keys($data);
        if ($data !== []) {
            $this->build(0, count($data));
        }
    }

    private function build(int $start, int $end): int
    {
        $d = count($this->data[$this->idx[$start]]);
        $lo = array_fill(0, $d, INF);
        $hi = array_fill(0, $d, -INF);
        for ($i = $start; $i < $end; $i++) {
            $row = $this->data[$this->idx[$i]];
            for ($j = 0; $j < $d; $j++) {
                if ($row[$j] < $lo[$j]) {
                    $lo[$j] = $row[$j];
                }
                if ($row[$j] > $hi[$j]) {
                    $hi[$j] = $row[$j];
                }
            }
        }

        $id = count($this->nodes);
        $this->nodes[$id] = [$start, $end, -1, -1, $lo, $hi];
        if ($end - $start <= $this->leafSize) {
            return $id;
        }

        $dim = self::widestDimension($this->data, $this->idx, $start, $end);
        self::sortRange($this->data, $this->idx, $start, $end, $dim);
        $mid = intdiv($start + $end, 2);
        $left = $this->build($start, $mid);
        $right = $this->build($mid, $end);
        $this->nodes[$id][2] = $left;
        $this->nodes[$id][3] = $right;
        return $id;
    }

    public function query(array $query, int $k): array
    {
        $bestDist = [];
        $bestIdx = [];
        if ($this->nodes !== []) {
            $this->search(0, $query, $k, $bestDist, $bestIdx);
        }
        return [$bestDist, $bestIdx];
    }

    /**
     * @param array<int, float> $q
     * @param array<int, float> $bestDist
     * @param array<int, int> $bestIdx
     */
    private function search(int $id, array $q, int $k, array &$bestDist, array &$bestIdx): void
    {
        [$start, $end, $left, $right, $lo, $hi] = $this->nodes[$id];

        $worst = self::worst($bestDist, $k);
        if ($worst < INF && $this->boxDistance($q, $lo, $hi) > $worst + 1e-12 * max(1.0, $worst)) {
            return;
        }

        if ($left < 0) {
            for ($i = $start; $i < $end; $i++) {
                $j = $this->idx[$i];
                self::offer($bestDist, $bestIdx, $k, ($this->distance)($q, $this->data[$j]), $j);
            }
            return;
        }

        $dl = $this->boxDistance($q, $this->nodes[$left][4], $this->nodes[$left][5]);
        $dr = $this->boxDistance($q, $this->nodes[$right][4], $this->nodes[$right][5]);
        if ($dl <= $dr) {
            $this->search($left, $q, $k, $bestDist, $bestIdx);
            $this->search($right, $q, $k, $bestDist, $bestIdx);
        } else {
            $this->search($right, $q, $k, $bestDist, $bestIdx);
            $this->search($left, $q, $k, $bestDist, $bestIdx);
        }
    }

    /**
     * Lower bound of the distance from $q to any point inside the box [lo, hi].
     *
     * @param array<int, float> $q
     * @param array<int, float> $lo
     * @param array<int, float> $hi
     */
    private function boxDistance(array $q, array $lo, array $hi): float
    {
        $p = $this->p;
        $w = $this->weights;
        $acc = 0.0;
        foreach ($q as $j => $x) {
            $gap = 0.0;
            if ($x < $lo[$j]) {
                $gap = $lo[$j] - $x;
            } elseif ($x > $hi[$j]) {
                $gap = $x - $hi[$j];
            }
            if ($gap == 0.0) {
                continue;
            }
            if (is_infinite($p)) {
                if ($gap > $acc) {
                    $acc = $gap;
                }
            } elseif ($p == 2.0) {
                $acc += ($w === null ? 1.0 : $w[$j]) * $gap * $gap;
            } elseif ($p == 1.0) {
                $acc += ($w === null ? 1.0 : $w[$j]) * $gap;
            } else {
                $acc += ($w === null ? 1.0 : $w[$j]) * ($gap ** $p);
            }
        }
        if (is_infinite($p) || $p == 1.0) {
            return $acc;
        }
        return $p == 2.0 ? sqrt($acc) : $acc ** (1.0 / $p);
    }
}

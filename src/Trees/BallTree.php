<?php

declare(strict_types=1);

namespace MachineLearning\Knn\Trees;

/**
 * Ball-tree: every node is a ball (centroid + radius). Pruning uses the triangle
 * inequality, so it works with any true metric.
 */
final class BallTree extends NeighborTree
{
    /** @var array<int, int> */
    private array $idx;

    /** @var array<int, array{0:int,1:int,2:int,3:int,4:array<int,float>,5:float}> start, end, left, right, centroid, radius */
    private array $nodes = [];

    /**
     * @param array<int, array<int, float>> $data
     * @param callable(array<int,float>, array<int,float>): float $distance
     */
    public function __construct(array $data, int $leafSize, $distance)
    {
        parent::__construct($data, $leafSize, $distance);
        $this->idx = array_keys($data);
        if ($data !== []) {
            $this->build(0, count($data));
        }
    }

    private function build(int $start, int $end): int
    {
        $d = count($this->data[$this->idx[$start]]);
        $centroid = array_fill(0, $d, 0.0);
        for ($i = $start; $i < $end; $i++) {
            $row = $this->data[$this->idx[$i]];
            for ($j = 0; $j < $d; $j++) {
                $centroid[$j] += $row[$j];
            }
        }
        $count = $end - $start;
        for ($j = 0; $j < $d; $j++) {
            $centroid[$j] /= $count;
        }
        $radius = 0.0;
        for ($i = $start; $i < $end; $i++) {
            $r = ($this->distance)($centroid, $this->data[$this->idx[$i]]);
            if ($r > $radius) {
                $radius = $r;
            }
        }

        $id = count($this->nodes);
        $this->nodes[$id] = [$start, $end, -1, -1, $centroid, $radius];
        if ($count <= $this->leafSize) {
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
        [$start, $end, $left, $right, $centroid, $radius] = $this->nodes[$id];

        $dc = ($this->distance)($q, $centroid);
        $worst = self::worst($bestDist, $k);
        if ($worst < INF && max(0.0, $dc - $radius) > $worst + 1e-12 * max(1.0, $worst)) {
            return;
        }

        if ($left < 0) {
            for ($i = $start; $i < $end; $i++) {
                $j = $this->idx[$i];
                self::offer($bestDist, $bestIdx, $k, ($this->distance)($q, $this->data[$j]), $j);
            }
            return;
        }

        $dl = ($this->distance)($q, $this->nodes[$left][4]) - $this->nodes[$left][5];
        $dr = ($this->distance)($q, $this->nodes[$right][4]) - $this->nodes[$right][5];
        if ($dl <= $dr) {
            $this->search($left, $q, $k, $bestDist, $bestIdx);
            $this->search($right, $q, $k, $bestDist, $bestIdx);
        } else {
            $this->search($right, $q, $k, $bestDist, $bestIdx);
            $this->search($left, $q, $k, $bestDist, $bestIdx);
        }
    }
}

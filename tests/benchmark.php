<?php

declare(strict_types=1);

/** Rough speed check. Run: php tests/benchmark.php [n_train] [n_query] [n_features] */

require_once __DIR__ . '/../autoload.php';

use MachineLearning\Knn\KNeighborsClassifier;

$n = (int) ($argv[1] ?? 5000);
$q = (int) ($argv[2] ?? 500);
$d = (int) ($argv[3] ?? 8);

mt_srand(1);
$gauss = static function (): float {
    return sqrt(-2.0 * log(mt_rand(1, mt_getrandmax()) / mt_getrandmax())) * cos(2.0 * M_PI * mt_rand() / mt_getrandmax());
};
$X = [];
$y = [];
for ($i = 0; $i < $n; $i++) {
    $row = [];
    for ($j = 0; $j < $d; $j++) {
        $row[] = $gauss();
    }
    $X[] = $row;
    $y[] = $row[0] + $row[1] > 0 ? 1 : 0;
}
$Q = [];
for ($i = 0; $i < $q; $i++) {
    $row = [];
    for ($j = 0; $j < $d; $j++) {
        $row[] = $gauss();
    }
    $Q[] = $row;
}

printf("n_train=%d n_query=%d n_features=%d\n", $n, $q, $d);
foreach (['brute', 'kd_tree', 'ball_tree'] as $algo) {
    $t0 = microtime(true);
    $knn = new KNeighborsClassifier(n_neighbors: 5, algorithm: $algo);
    $knn->fit($X, $y);
    $t1 = microtime(true);
    $pred = $knn->predict($Q);
    $t2 = microtime(true);
    printf("%-10s fit %7.3fs  predict %7.3fs  (%d predictions, checksum %d)\n", $algo, $t1 - $t0, $t2 - $t1, count($pred), array_sum($pred));
}

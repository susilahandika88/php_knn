# KNeighborsClassifier untuk PHP

Implementasi algoritma K-Nearest Neighbors (klasifikasi) murni PHP, tanpa dependensi.
Butuh PHP 8.1 atau lebih baru.

## Instalasi

Salin folder `src/` dan `autoload.php` ke proyek Anda, lalu:

```php
require_once __DIR__ . '/autoload.php';
```

Atau lewat Composer dengan PSR-4: `"MachineLearning\\Knn\\": "src/"`.

## Mulai Cepat

```php
use MachineLearning\Knn\KNeighborsClassifier;

$X = [[0, 0], [0, 1], [1, 0], [5, 5], [5, 6], [6, 5]];
$y = ['a', 'a', 'a', 'b', 'b', 'b'];

$knn = new KNeighborsClassifier(n_neighbors: 3, weights: 'distance');
$knn->fit($X, $y);

$knn->predict([[0.5, 0.5], [5.5, 5.5]]);          // ['a', 'b']
$knn->predict_proba([[0.5, 0.5]]);                 // [[1.0, 0.0]]  (kolom mengikuti $knn->classes_)
[$dist, $ind] = $knn->kneighbors([[0.5, 0.5]]);    // jarak dan indeks tetangga
$knn->score($X, $y);                               // akurasi
```

Argumen konstruktor memakai named arguments, mis. `new KNeighborsClassifier(n_neighbors: 7, p: 1)`.
Setiap metode bisa dipanggil dengan gaya camelCase (`predictProba`) atau snake_case (`predict_proba`).

## Parameter Konstruktor

| Parameter | Default | Keterangan |
|---|---|---|
| `n_neighbors` | `5` | Jumlah tetangga terdekat yang dipakai. |
| `weights` | `'uniform'` | `'uniform'`, `'distance'` (bobot 1/jarak), atau callable yang menerima array jarak dan mengembalikan array bobot. |
| `algorithm` | `'auto'` | `'auto'`, `'brute'`, `'kd_tree'`, atau `'ball_tree'`. |
| `leaf_size` | `30` | Ukuran daun untuk `kd_tree` / `ball_tree`. |
| `p` | `2` | Pangkat metrik Minkowski (`1` = Manhattan, `2` = Euclidean). |
| `metric` | `'minkowski'` | Nama metrik, `'precomputed'`, atau callable `fn(array $a, array $b): float`. |
| `metric_params` | `null` | Parameter tambahan metrik, mis. `['V' => [...]]`, `['VI' => [...]]`, `['w' => [...]]`. |
| `n_jobs` | `null` | Diterima tetapi diabaikan (tidak ada paralelisme). |

## Metode

| Metode | Keterangan |
|---|---|
| `fit($X, $y)` | Melatih model. `$X` array 2D, `$y` array 1D (atau 2D untuk multi-output). Mengembalikan `$this`. |
| `predict($X)` | Mengembalikan label prediksi untuk tiap baris `$X`. |
| `predictProba($X)` | Probabilitas tiap kelas per baris; urutan kolom mengikuti `$knn->classes_`. |
| `kneighbors($X, $n_neighbors, $return_distance)` | Mengembalikan `[jarak, indeks]` tetangga terdekat. Jika `$X = null`, yang ditanyai adalah data latih dan tiap titik melewatkan dirinya sendiri. Dengan `$return_distance = false`, hanya indeks yang dikembalikan. |
| `kneighborsGraph($X, $n_neighbors, $mode)` | Graf tetangga (`mode`: `'connectivity'` atau `'distance'`) berupa array `data`, `indices`, `indptr`, `shape` (layout CSR). |
| `score($X, $y, $sample_weight)` | Akurasi rata-rata pada data dan label yang diberikan. |
| `getParams()` / `setParams([...])` | Membaca / mengubah parameter. `setParams` mengembalikan `$this`. |

## Atribut Setelah `fit`

Properti publik berikut terisi setelah `fit()`:

| Atribut | Isi |
|---|---|
| `classes_` | Daftar label kelas yang unik. |
| `n_features_in_` | Jumlah fitur pada data latih. |
| `n_samples_fit_` | Jumlah sampel pada data latih. |
| `effective_metric_` | Metrik yang benar-benar dipakai. |
| `effective_metric_params_` | Parameter metrik yang dipakai. |
| `outputs_2d_` | `true` jika `y` berbentuk 2D (multi-output). |
| `_fit_method` | Algoritma yang terpilih (`brute`, `kd_tree`, atau `ball_tree`). |

## Contoh

### Memilih metrik dan algoritma

```php
$knn = new KNeighborsClassifier(
    n_neighbors: 5,
    metric: 'manhattan',
    algorithm: 'kd_tree',
    leaf_size: 20,
);
$knn->fit($X, $y);
```

### Metrik kustom

```php
$knn = new KNeighborsClassifier(
    n_neighbors: 3,
    metric: fn(array $a, array $b): float => abs($a[0] - $b[0]),
);
```

### Bobot kustom

```php
$knn = new KNeighborsClassifier(
    weights: fn(array $dist): array => array_map(fn($d) => 1 / (1 + $d ** 2), $dist),
);
```

### Multi-output

```php
$y = [['a', 1], ['a', 1], ['b', 2], ['b', 2]];   // tiap baris: [label1, label2]
$knn->fit($X, $y);
$knn->predict($X);                               // tiap baris hasil berisi 2 label
```

### Mengubah parameter

```php
$knn->setParams(['n_neighbors' => 7, 'weights' => 'distance'])->fit($X, $y);
$knn->getParams();
```

### Graf tetangga

```php
$g = $knn->kneighborsGraph($X, 2, 'distance');
// $g['data'], $g['indices'], $g['indptr'], $g['shape']
```

## Metrik yang Didukung

`euclidean`/`l2`, `manhattan`/`cityblock`/`l1`, `chebyshev`/`infinity`, `minkowski` (dengan `p` dan bobot `w`;
`0 < p < 1` hanya untuk `brute`), `sqeuclidean`, `cosine`, `correlation`, `seuclidean` (`V`),
`mahalanobis` (`VI`), `hamming`, `canberra`, `braycurtis`, `haversine`, `jaccard`, `dice`,
`rogerstanimoto`, `russellrao`, `sokalmichener`, `sokalsneath`, `yule`, `precomputed`, dan callable.

Tipe label yang didukung: int, float (bilangan bulat), string, dan bool.

## Error dan Peringatan

| Situasi | Yang terjadi |
|---|---|
| Parameter tidak valid | `InvalidParameterError` |
| Memanggil `predict` dll. sebelum `fit` | `NotFittedError` |
| Bentuk data tidak valid (jumlah fitur beda, `n_neighbors` > jumlah sampel, dsb.) | `InvalidArgumentException` |
| Peringatan non-fatal | `E_USER_WARNING` |

Kelas exception ada di namespace `MachineLearning\Knn\Exceptions`.

## Batasan

- Tidak ada input sparse, metrik `nan_euclidean`, dan paralelisme.
- Tetangga dengan jarak persis sama diurutkan menurut indeks data latih.
- Model tidak bisa di-pickle; tidak ada fungsi simpan/muat.
- Fitur perlu diskalakan sendiri bila perlu.

## Tes

```
php tests/run.php        # tes utama
php tests/run.php iris   # hanya kasus yang namanya memuat "iris"
php tests/api.php        # perilaku API (alias, parameter, peringatan)
php tests/benchmark.php  # kecepatan kasar
```

## Struktur

```
src/KNeighborsClassifier.php     kelas utama
src/Metrics/DistanceMetric.php   fungsi jarak
src/Trees/                       KDTree, BallTree
src/Support/Validation.php       validasi input dan encoding label
src/Exceptions/                  NotFittedError, InvalidParameterError
```

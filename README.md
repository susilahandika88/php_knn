# KNeighborsClassifier untuk PHP

Port `sklearn.neighbors.KNeighborsClassifier` yang ditulis dari nol di PHP (tanpa library).
Perilakunya dicocokkan dengan scikit-learn **1.7.2** lewat 1218 pengecekan otomatis
(`tests/run.php`). Butuh PHP 8.1 atau lebih baru, tanpa dependensi.

## Pemakaian

```php
require_once __DIR__ . '/autoload.php';   // atau composer: PSR-4 "MachineLearning\\Knn\\" => src/

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

Nama argumen dan atribut sama dengan sklearn, jadi `new KNeighborsClassifier(n_neighbors: 7, p: 1)`
terbaca seperti di Python. Setiap metode bisa dipanggil camelCase (`predictProba`) atau snake_case
(`predict_proba`).

| sklearn | PHP |
|---|---|
| `fit(X, y)` | `fit($X, $y)` (mengembalikan `$this`) |
| `predict(X)` | `predict($X)` |
| `predict_proba(X)` | `predictProba($X)` / `predict_proba($X)` |
| `kneighbors(X, n_neighbors, return_distance)` | `kneighbors($X, $n_neighbors, $return_distance)`; `$X = null` menanyai data latih dan melewatkan dirinya sendiri |
| `kneighbors_graph(X, n_neighbors, mode)` | `kneighborsGraph(...)`, hasil berupa array `data/indices/indptr/shape` (layout CSR scipy) |
| `score(X, y, sample_weight)` | `score($X, $y, $sample_weight)` |
| `get_params()` / `set_params()` | `getParams()` / `setParams([...])` |
| `classes_`, `n_features_in_`, `n_samples_fit_`, `effective_metric_`, `effective_metric_params_`, `outputs_2d_` | properti publik dengan nama yang sama; `_fit_method` menunjukkan algoritma yang dipakai |

Parameter: `n_neighbors=5`, `weights='uniform'|'distance'|callable`, `algorithm='auto'|'brute'|'kd_tree'|'ball_tree'`,
`leaf_size=30`, `p=2`, `metric='minkowski'`, `metric_params=null`, `n_jobs=null` (diterima, diabaikan).

Yang didukung:

- Label bertipe int, float (bilangan bulat), string, atau bool; `y` 2D untuk multi-output.
- Metrik: euclidean/l2, manhattan/cityblock/l1, chebyshev/infinity, minkowski (`p`, bobot `w`, termasuk `0 < p < 1`
  untuk brute), sqeuclidean, cosine, correlation, seuclidean (`V`), mahalanobis (`VI`), hamming, canberra,
  braycurtis, haversine, jaccard, dice, rogerstanimoto, russellrao, sokalmichener, sokalsneath, yule,
  `precomputed`, dan callable `fn(array $a, array $b): float`.
- `kd_tree`, `ball_tree`, dan `brute` ditulis sendiri; `algorithm='auto'` memilih seperti sklearn.
- Pesan error dan peringatan memakai kalimat sklearn (`InvalidParameterError`, `NotFittedError`,
  `InvalidArgumentException`; peringatan lewat `E_USER_WARNING`).

## Perbedaan dengan sklearn

- Tidak ada input sparse, metrik `nan_euclidean`, dan paralelisme.
- Tetangga yang jaraknya persis sama diurutkan menurut indeks data latih (sklearn tidak menjamin urutan itu),
  sehingga pada data dengan banyak jarak kembar hasilnya bisa berbeda pada titik batas ke-k.
- Jarak dihitung langsung, tanpa trik ekspansi `‖x‖² − 2x·y + ‖y‖²` milik sklearn; selisih numeriknya
  sekitar 1e-8 untuk titik yang sangat dekat.
- Model tidak bisa di-pickle; tidak ada fungsi simpan/muat.
- Fitur perlu diskalakan sendiri bila perlu (sklearn pun tidak melakukannya).

## Tes

```
php tests/run.php        # bandingkan dengan hasil scikit-learn (tests/fixtures/cases.json)
php tests/run.php iris   # hanya kasus yang namanya memuat "iris"
php tests/api.php        # perilaku khusus port PHP (alias, parameter, peringatan)
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

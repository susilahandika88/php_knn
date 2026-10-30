<?php

declare(strict_types=1);

namespace MachineLearning\Knn\Support;

use InvalidArgumentException;

/**
 * Input checking and label encoding, with the same messages as scikit-learn's
 * check_array / check_X_y / np.unique(return_inverse=True).
 */
final class Validation
{
    private const NAN_HELP = "\nKNeighborsClassifier does not accept missing values encoded as NaN natively. "
        . 'For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. '
        . 'Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. '
        . 'See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: '
        . 'https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values';

    private function __construct()
    {
    }

    /** Python-style repr of a parameter value, for error messages. */
    public static function repr(mixed $v): string
    {
        return match (true) {
            $v === null => 'None',
            is_bool($v) => $v ? 'True' : 'False',
            is_int($v) => (string) $v,
            is_float($v) => is_infinite($v) ? ($v > 0 ? 'inf' : '-inf') : (is_nan($v) ? 'nan' : var_export($v, true)),
            is_string($v) => "'" . $v . "'",
            is_array($v) => json_encode($v) ?: 'array',
            default => 'callable',
        };
    }

    /**
     * Validates a 2D numeric array and returns it as a list of float rows.
     *
     * @return array<int, array<int, float>>
     */
    public static function checkArray(mixed $X, string $name = 'X', bool $nonNegative = false): array
    {
        if (!is_array($X)) {
            throw new InvalidArgumentException(
                "Expected 2D array, got scalar array instead:\narray=" . self::repr($X) . ".\n"
                . "Reshape your data either using array.reshape(-1, 1) if your data has a single feature "
                . "or array.reshape(1, -1) if it contains a single sample."
            );
        }
        $rows = array_values($X);
        if ($rows === [] || !is_array($rows[0])) {
            $shown = implode(' ', array_map(static fn ($v): string => is_scalar($v) ? self::floatText($v) : '?', $rows));
            throw new InvalidArgumentException(
                "Expected 2D array, got 1D array instead:\narray=[{$shown}].\n"
                . "Reshape your data either using array.reshape(-1, 1) if your data has a single feature "
                . "or array.reshape(1, -1) if it contains a single sample."
            );
        }

        $out = [];
        $width = null;
        foreach ($rows as $row) {
            if (!is_array($row)) {
                throw new InvalidArgumentException(
                    'setting an array element with a sequence. The requested array has an inhomogeneous shape.'
                );
            }
            $row = array_values($row);
            $width ??= count($row);
            if (count($row) !== $width) {
                throw new InvalidArgumentException(
                    'setting an array element with a sequence. The requested array has an inhomogeneous shape.'
                );
            }
            foreach ($row as $j => $v) {
                if (is_int($v) || is_bool($v)) {
                    $row[$j] = (float) $v;
                } elseif (!is_float($v)) {
                    throw new InvalidArgumentException("could not convert " . gettype($v) . " to float in {$name}");
                }
            }
            $out[] = $row;
        }

        $n = count($out);
        if ($n < 1) {
            throw new InvalidArgumentException("Found array with 0 sample(s) (shape=(0, {$width})) while a minimum of 1 is required by KNeighborsClassifier.");
        }
        if ($width < 1) {
            throw new InvalidArgumentException("Found array with 0 feature(s) (shape=({$n}, 0)) while a minimum of 1 is required by KNeighborsClassifier.");
        }
        foreach ($out as $row) {
            foreach ($row as $v) {
                if (is_nan($v)) {
                    throw new InvalidArgumentException("Input {$name} contains NaN." . self::NAN_HELP);
                }
                if (is_infinite($v)) {
                    throw new InvalidArgumentException("Input {$name} contains infinity or a value too large for dtype('float64').");
                }
                if ($nonNegative && $v < 0.0) {
                    throw new InvalidArgumentException("Negative values in data passed to {$name}.");
                }
            }
        }
        return $out;
    }

    /**
     * Validates y: a list of scalars (single output) or a list of equal-length lists (multi-output).
     *
     * @return array{0: array<int, mixed>, 1: bool, 2: bool} rows, is 2D, is a single-column 2D array
     */
    public static function checkTarget(mixed $y, int $nSamples): array
    {
        if (!is_array($y)) {
            throw new InvalidArgumentException('y must be an array of labels');
        }
        $y = array_values($y);
        if (count($y) !== $nSamples) {
            throw new InvalidArgumentException("Found input variables with inconsistent numbers of samples: [{$nSamples}, " . count($y) . ']');
        }
        $is2d = $y !== [] && is_array($y[0]);
        $width = null;
        foreach ($y as $i => $v) {
            if ($is2d) {
                if (!is_array($v)) {
                    throw new InvalidArgumentException('setting an array element with a sequence. The requested array has an inhomogeneous shape.');
                }
                $v = array_values($v);
                $y[$i] = $v;
                $width ??= count($v);
                if (count($v) !== $width) {
                    throw new InvalidArgumentException('setting an array element with a sequence. The requested array has an inhomogeneous shape.');
                }
                foreach ($v as $cell) {
                    self::checkLabel($cell);
                }
            } else {
                self::checkLabel($v);
            }
        }
        if ($is2d) {
            $y = self::commonDtype($y);
        }
        return [$y, $is2d, $is2d && $width === 1];
    }

    /**
     * numpy stores a 2D label array with one dtype: strings win over numbers, floats over ints.
     *
     * @param array<int, array<int, mixed>> $rows
     * @return array<int, array<int, mixed>>
     */
    private static function commonDtype(array $rows): array
    {
        $hasString = false;
        $hasOther = false;
        $hasFloat = false;
        $hasNumber = false;
        foreach ($rows as $row) {
            foreach ($row as $v) {
                if (is_string($v)) {
                    $hasString = true;
                } else {
                    $hasOther = true;
                    if (is_float($v)) {
                        $hasFloat = true;
                    }
                    if (is_int($v) || is_float($v)) {
                        $hasNumber = true;
                    }
                }
            }
        }
        if ($hasString && $hasOther) {
            foreach ($rows as $i => $row) {
                foreach ($row as $j => $v) {
                    $rows[$i][$j] = is_bool($v) ? ($v ? 'True' : 'False') : (string) $v;
                }
            }
        } elseif ($hasFloat) {
            foreach ($rows as $i => $row) {
                foreach ($row as $j => $v) {
                    $rows[$i][$j] = (float) $v;
                }
            }
        } elseif ($hasNumber) {
            foreach ($rows as $i => $row) {
                foreach ($row as $j => $v) {
                    $rows[$i][$j] = (int) $v;
                }
            }
        }
        return $rows;
    }

    private static function checkLabel(mixed $v): void
    {
        if (is_float($v) && is_nan($v)) {
            throw new InvalidArgumentException('Input y contains NaN.');
        }
        if (is_float($v) && is_infinite($v)) {
            throw new InvalidArgumentException("Input y contains infinity or a value too large for dtype('float64').");
        }
        if (!is_scalar($v)) {
            throw new InvalidArgumentException('y must contain scalar labels (int, float, string or bool)');
        }
    }

    /**
     * np.unique(values, return_inverse=True) plus check_classification_targets.
     *
     * @param array<int, mixed> $values
     * @return array{0: array<int, mixed>, 1: array<int, int>} sorted classes and, per sample, the class index
     */
    public static function encodeLabels(array $values, bool $multiOutput): array
    {
        $hasString = false;
        $hasNonString = false;
        $hasFloat = false;
        $allBool = true;
        foreach ($values as $v) {
            if (is_string($v)) {
                $hasString = true;
            } else {
                $hasNonString = true;
            }
            if (is_float($v)) {
                $hasFloat = true;
                if ($v != floor($v)) {
                    $kind = $multiOutput ? 'continuous-multioutput' : 'continuous';
                    throw new InvalidArgumentException(
                        "Unknown label type: {$kind}. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values."
                    );
                }
            }
            if (!is_bool($v)) {
                $allBool = false;
            }
        }

        // numpy turns a mix of strings and numbers into strings
        $asString = $hasString && $hasNonString;
        $map = [];
        $reps = [];
        foreach ($values as $v) {
            if ($asString || is_string($v)) {
                $text = is_bool($v) ? ($v ? 'True' : 'False') : (string) $v;
                $key = 's:' . $text;
                $rep = $text;
                $sort = $text;
            } else {
                $num = (float) $v + 0.0;
                $key = 'n:' . $num;
                $rep = $hasFloat ? $num : ($allBool ? (bool) $v : (int) $v);
                $sort = $num;
            }
            if (!isset($map[$key])) {
                $map[$key] = true;
                $reps[$key] = [$sort, $rep];
            }
        }

        $entries = array_values($reps);
        usort($entries, static fn (array $a, array $b): int => is_string($a[0]) ? strcmp($a[0], $b[0]) : $a[0] <=> $b[0]);

        $classes = [];
        $index = [];
        foreach ($entries as $i => [$sort, $rep]) {
            $classes[] = $rep;
            $index[(is_string($sort) ? 's:' : 'n:') . $sort] = $i;
        }

        $codes = [];
        foreach ($values as $v) {
            if ($asString || is_string($v)) {
                $text = is_bool($v) ? ($v ? 'True' : 'False') : (string) $v;
                $codes[] = $index['s:' . $text];
            } else {
                $codes[] = $index['n:' . ((float) $v + 0.0)];
            }
        }
        return [$classes, $codes];
    }

    private static function floatText(mixed $v): string
    {
        if (is_float($v) || is_int($v) || is_bool($v)) {
            $f = (float) $v;
            return $f == floor($f) && abs($f) < 1e15 ? number_format($f, 0, '.', '') . '.' : rtrim(rtrim(sprintf('%.8f', $f), '0'), '.');
        }
        return (string) $v;
    }
}

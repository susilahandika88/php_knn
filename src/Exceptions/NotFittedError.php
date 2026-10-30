<?php

declare(strict_types=1);

namespace MachineLearning\Knn\Exceptions;

use LogicException;

/** Raised when predict/kneighbors are called before fit (sklearn's NotFittedError). */
final class NotFittedError extends LogicException
{
}

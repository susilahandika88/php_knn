<?php

declare(strict_types=1);

namespace MachineLearning\Knn\Exceptions;

use InvalidArgumentException;

/** Raised for an invalid constructor / set_params value (sklearn's InvalidParameterError). */
final class InvalidParameterError extends InvalidArgumentException
{
}

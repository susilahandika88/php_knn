<?php

declare(strict_types=1);

// Autoloader without Composer: require_once __DIR__ . '/autoload.php';
spl_autoload_register(static function (string $class): void {
    $prefix = 'MachineLearning\\Knn\\';
    if (strncmp($class, $prefix, strlen($prefix)) !== 0) {
        return;
    }
    $relative = substr($class, strlen($prefix));
    $file = __DIR__ . '/src/' . str_replace('\\', '/', $relative) . '.php';
    if (is_file($file)) {
        require $file;
    }
});

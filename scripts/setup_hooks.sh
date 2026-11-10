#!/bin/sh
# Activa los hooks versionados en .githooks/ (una vez por clon).
set -e
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath .githooks
chmod +x .githooks/*
echo "Hooks activos: el pre-commit bloqueará material privado del hardware."

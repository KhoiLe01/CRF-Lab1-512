#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LATEX_DIR="$ROOT_DIR/latex"
BUILD_DIR="$ROOT_DIR/build"

mkdir -p "$BUILD_DIR"

(
  cd "$LATEX_DIR"
  latexmk -C -outdir="$BUILD_DIR" Lab_1.tex
  latexmk -pdf -interaction=nonstopmode -outdir="$BUILD_DIR" Lab_1.tex
)

echo "Built PDF: $BUILD_DIR/Lab_1.pdf"

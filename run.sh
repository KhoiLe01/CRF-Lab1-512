#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LATEX_DIR="$ROOT_DIR/latex"
BUILD_DIR="$ROOT_DIR/build"

if ! command -v latexmk >/dev/null 2>&1; then
  echo "Error: latexmk was not found on PATH." >&2
  echo "Install a LaTeX distribution that provides latexmk (for example TeX Live or MacTeX)." >&2
  echo "On Windows, run this script from a Unix-like shell such as WSL or Git Bash after installing the LaTeX toolchain." >&2
  exit 127
fi

mkdir -p "$BUILD_DIR"

(
  cd "$LATEX_DIR"
  /Library/TeX/texbin/latexmk -C -outdir="$BUILD_DIR" Lab_1.tex
  /Library/TeX/texbin/latexmk -pdf -interaction=nonstopmode -outdir="$BUILD_DIR" Lab_1.tex
)

echo "Built PDF: $BUILD_DIR/Lab_1.pdf"

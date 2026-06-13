#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOXYFILE="${ROOT_DIR}/docs/Doxyfile"
OUTPUT_DIR="${ROOT_DIR}/build/docs"
HTML_INDEX="${OUTPUT_DIR}/html/index.html"

if ! command -v doxygen >/dev/null 2>&1; then
  echo "error: doxygen is required but was not found in PATH." >&2
  echo "       Ubuntu: sudo apt-get install -y doxygen graphviz" >&2
  echo "       macOS:  brew install doxygen graphviz" >&2
  exit 1
fi

if ! command -v dot >/dev/null 2>&1; then
  echo "warning: graphviz 'dot' was not found in PATH; diagrams may be unavailable." >&2
fi

cd "${ROOT_DIR}"

rm -rf "${OUTPUT_DIR}"
doxygen "${DOXYFILE}"

if [ ! -f "${HTML_INDEX}" ]; then
  echo "error: documentation generation did not produce ${HTML_INDEX}" >&2
  exit 1
fi

echo "Documentation generated at: ${HTML_INDEX}"

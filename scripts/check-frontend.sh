#!/usr/bin/env bash
# Run frontend code quality checks using Prettier.
# Usage: ./scripts/check-frontend.sh [--fix]

set -e

FRONTEND_DIR="frontend"
FIX=false

for arg in "$@"; do
  case $arg in
    --fix) FIX=true ;;
  esac
done

if ! command -v npx &>/dev/null; then
  echo "Error: npx not found. Install Node.js to run frontend quality checks."
  exit 1
fi

if [ "$FIX" = true ]; then
  echo "Formatting frontend files with Prettier..."
  npx prettier --write "$FRONTEND_DIR/"
  echo "Done."
else
  echo "Checking frontend formatting with Prettier..."
  npx prettier --check "$FRONTEND_DIR/"
  echo "All frontend files are properly formatted."
fi

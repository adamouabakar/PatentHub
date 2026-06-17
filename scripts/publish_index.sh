#!/usr/bin/env bash
# Package LanceDB index and publish to GitHub Release.
# Usage: ./scripts/publish_index.sh [tag] [index_path]
set -euo pipefail

TAG="${1:-index-latest}"
INDEX_PATH="${2:-./data/patents.lance}"
REPO="${GITHUB_REPOSITORY:-}"
OUTPUT_DIR="${RUNNER_TEMP:-/tmp}/patenthub-release"
ARCHIVE_NAME="patents-index.tar.gz"

if [[ ! -d "$INDEX_PATH" ]]; then
  echo "Error: index not found at $INDEX_PATH" >&2
  echo "Run: python -m ingest.build_index --limit N" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
tar -czf "$OUTPUT_DIR/$ARCHIVE_NAME" -C "$(dirname "$INDEX_PATH")" "$(basename "$INDEX_PATH")"

echo "Created $OUTPUT_DIR/$ARCHIVE_NAME ($(du -h "$OUTPUT_DIR/$ARCHIVE_NAME" | cut -f1))"

if [[ -z "$REPO" ]]; then
  echo "GITHUB_REPOSITORY not set — archive ready at $OUTPUT_DIR/$ARCHIVE_NAME"
  exit 0
fi

if ! command -v gh &>/dev/null; then
  echo "gh CLI not found — upload manually" >&2
  exit 1
fi

gh release upload "$TAG" "$OUTPUT_DIR/$ARCHIVE_NAME" --clobber
echo "Uploaded to release $TAG on $REPO"
#!/usr/bin/env bash
# Push the repository to the remote GPU box.
#
# Only source + data cross the wire (node_modules, git history and trained
# models stay off). The remote node_modules/ directory is kept between runs, so
# the tfjs-node-gpu native install only happens once.
#   ./scripts/sync.sh
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=scripts/_env.sh
. "$(dirname "$0")/_env.sh"
load_env

REMOTE_HOST="${REMOTE_HOST:?set REMOTE_HOST in .env}"
REMOTE_USER="${REMOTE_USER:-$USER}"
REMOTE_DIR="${REMOTE_DIR:-openhps-tf}"
TARGET="${REMOTE_USER}@${REMOTE_HOST}"

echo "==> syncing repo to ${TARGET}:${REMOTE_DIR}"

ssh "$TARGET" "mkdir -p ${REMOTE_DIR}"

# --delete keeps the box in sync with local edits; excluded paths (node_modules,
# trained models, run logs) are preserved on the box.
rsync -az --delete --info=progress2 \
  --exclude '.git/' \
  --exclude 'node_modules/' \
  --exclude 'dist/' \
  --exclude 'docs/out/' \
  --exclude 'coverage/' \
  --exclude '.env' \
  --exclude 'training/runs/' \
  --exclude 'test/data/model' \
  --exclude 'test/data/model2' \
  --exclude 'test/data/model_*/' \
  --exclude 'accuracy_results.json' \
  ./ "${TARGET}:${REMOTE_DIR}/"

echo "==> synced to ${TARGET}:${REMOTE_DIR}"

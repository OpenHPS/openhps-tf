#!/bin/bash
set -euo pipefail
cd /opt/app
# The container runs as an arbitrary --user; HOME may not be set, so pin the npm
# cache to a container-local, writable tmpdir instead of /.npm.
npm install --unsafe-perm --cache /tmp/.npm-cache
EPOCHS="${EPOCHS:-250}" MODEL_DIR="${MODEL_DIR:-test/data/model}" \
BATCH_SIZE="${BATCH_SIZE:-256}" PATIENCE="${PATIENCE:-20}" \
ENSEMBLE="${ENSEMBLE:-1}" \
npm run test -- --grep "data.openhps.accuracy"

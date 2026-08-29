#!/usr/bin/env bash
# Run the accuracy-model training on the remote GPU box and pull the artefacts
# back.
#
#   ./scripts/sync.sh          # first push the repo + data
#   ./scripts/train_remote.sh  # EPOCHS=400 ./scripts/train_remote.sh
#
# Training runs detached (`docker run -d`), so a dropped SSH connection does not
# kill it. The script tails the container log until the run finishes; Ctrl-C
# detaches without stopping the remote job. Reattach with:
#   ssh HOST 'docker logs -f <container>'
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=scripts/_env.sh
. "$(dirname "$0")/_env.sh"
load_env

REMOTE_HOST="${REMOTE_HOST:?set REMOTE_HOST in .env}"
REMOTE_USER="${REMOTE_USER:-$USER}"
REMOTE_DIR="${REMOTE_DIR:-openhps-tf}"
REMOTE_IMAGE="${REMOTE_IMAGE:-openhps-tf-train:1.0}"
REMOTE_GPUS="${REMOTE_GPUS:-2}"
EPOCHS="${EPOCHS:-250}"
MODEL_DIR="${MODEL_DIR:-test/data/model}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PATIENCE="${PATIENCE:-20}"
ENSEMBLE="${ENSEMBLE:-1}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d-%H%M%S)}"
TARGET="${REMOTE_USER}@${REMOTE_HOST}"
CONTAINER="openhps-tf-train-${RUN_NAME}"

echo "==> run ${RUN_NAME} on ${TARGET} (gpus=${REMOTE_GPUS}, epochs=${EPOCHS}, batch=${BATCH_SIZE}, patience=${PATIENCE}, ensemble=${ENSEMBLE}, model=${MODEL_DIR})"

ssh "$TARGET" "mkdir -p ${REMOTE_DIR}/training/runs/${RUN_NAME}"
REMOTE_ABS="$(ssh "$TARGET" "cd ${REMOTE_DIR} && pwd")"
echo "==> remote path: ${REMOTE_ABS}"

# Detached so a dropped SSH connection does not kill training. --user keeps
# artefacts owned by you rather than root, so rsync back works. node_modules
# lives in the bind mount (kept between runs by sync.sh's excludes), so the
# native tfjs-node-gpu install happens only once.
ssh "$TARGET" "docker run -d \
  --name ${CONTAINER} \
  --gpus 'device=${REMOTE_GPUS}' \
  --user \$(id -u):\$(id -g) \
  -v ${REMOTE_ABS}:/opt/app \
  -w /opt/app \
  -e EPOCHS=${EPOCHS} \
  -e MODEL_DIR=${MODEL_DIR} \
  -e BATCH_SIZE=${BATCH_SIZE} \
  -e PATIENCE=${PATIENCE} \
  -e ENSEMBLE=${ENSEMBLE} \
  ${REMOTE_IMAGE} \
  bash docker/docker-entrypoint.sh" >/dev/null

echo "==> container ${CONTAINER} started; tailing (Ctrl-C detaches)"
ssh "$TARGET" "docker logs -f ${CONTAINER}" 2>&1 | tee "/tmp/${CONTAINER}.log" || true

echo "==> waiting for container to exit..."
# docker logs -f can return early (Ctrl-C / dropped SSH); docker wait blocks
# until the container has actually finished and returns its real exit code.
STATUS="$(ssh "$TARGET" "docker wait ${CONTAINER}" 2>/dev/null || echo "?")"
echo "==> exit code: ${STATUS}"

echo "==> pulling artefacts"
mkdir -p "training/runs/${RUN_NAME}"
ssh "$TARGET" "mkdir -p ${REMOTE_DIR}/${MODEL_DIR}" 2>/dev/null || true
rsync -az "${TARGET}:${REMOTE_DIR}/${MODEL_DIR}/" "${MODEL_DIR}/" || true
# Ensemble members (modelDir_1, modelDir_2, ...).
for i in $(seq 1 $((ENSEMBLE - 1))); do
    ssh "$TARGET" "mkdir -p ${REMOTE_DIR}/${MODEL_DIR}_${i}" 2>/dev/null || true
    rsync -az "${TARGET}:${REMOTE_DIR}/${MODEL_DIR}_${i}/" "${MODEL_DIR}_${i}/" || true
done
rsync -az "${TARGET}:${REMOTE_DIR}/accuracy_results.json" "./accuracy_results.json" 2>/dev/null || true
cp "/tmp/${CONTAINER}.log" "training/runs/${RUN_NAME}/train.log" 2>/dev/null || true

# Keep the box tidy; the logs are already on disk locally.
ssh "$TARGET" "docker rm ${CONTAINER}" >/dev/null 2>&1 || true

echo "==> local model: ${MODEL_DIR}/ (log: training/runs/${RUN_NAME}/train.log)"
echo "==> local heatmap source: accuracy_results.json"

exit "${STATUS:-0}"

#!/usr/bin/env bash
# Provision the remote GPU box. Run once per box; idempotent.
#
#   ./scripts/sync.sh        # first push the repo (needed for the GPU check)
#   ./scripts/remote_setup.sh
#
# Reads REMOTE_* from .env. Uses Docker rather than a virtualenv: the host
# needs only Docker + nvidia-container-toolkit + an NVIDIA driver -- no system
# CUDA. The training image is the repo's own docker/Dockerfile
# (nvidia/cuda:11.2.2 + Node 20), built here.
#
# tfjs-node-gpu is pinned to libtensorflow-gpu 2.9.1 (CUDA 11.2, kernels up to
# sm_86). Blackwell cards (RTX 5060 Ti, sm_120) cannot run it and must not be
# selected via REMOTE_GPUS.
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
TARGET="${REMOTE_USER}@${REMOTE_HOST}"
REMOTE_ABS="$(ssh "$TARGET" "cd ${REMOTE_DIR} && pwd")"

echo "==> ${TARGET}:${REMOTE_ABS}  (image=${REMOTE_IMAGE}, gpus=${REMOTE_GPUS})"

echo
echo "--- host prerequisites ---"
ssh "$TARGET" bash -s <<'REMOTE'
set -euo pipefail
command -v docker >/dev/null || { echo "!! docker not installed on the host"; exit 1; }
docker_info="$(docker info 2>/dev/null || true)"
if [[ "$docker_info" != *nvidia* ]]; then
  echo "!! docker has no 'nvidia' runtime. Install nvidia-container-toolkit:"
  echo "   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
  exit 1
fi
command -v nvidia-smi >/dev/null || { echo "!! nvidia-smi not found; install the NVIDIA driver"; exit 1; }
echo "docker: $(docker --version)"
echo
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used,utilization.gpu --format=csv
REMOTE

echo
echo "--- compute capability vs tfjs-node-gpu ---"
# tfjs-node-gpu ships libtensorflow-gpu 2.9.1 (CUDA 11.2): SASS/PTX up to sm_86
# (Ampere). sm_120 (Blackwell) requires CUDA 12.8+, which tfjs cannot use.
ssh "$TARGET" 'nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader' \
  | while IFS=, read -r idx cap; do
      cap_num="$(echo "$cap" | tr -d ' ')"
      major="${cap_num%%.*}"
      if [ "${major:-0}" -ge 12 ]; then
        echo "  GPU${idx} (sm_${cap_num}) — NOT supported by tfjs-node-gpu (Blackwell needs CUDA 12.8+). Do not pin this card."
      elif [ "${major:-0}" -ge 8 ]; then
        echo "  GPU${idx} (sm_${cap_num}) — supported by tfjs-node-gpu."
      else
        echo "  GPU${idx} (sm_${cap_num}) — too old for tfjs-node-gpu."
      fi
    done
echo "  Pin the supported card with REMOTE_GPUS in .env (srv3: 2 = RTX 3060 Ti)."

echo
echo "==> uploading docker build context"
ssh "$TARGET" "mkdir -p ${REMOTE_DIR}/docker"
rsync -az docker/ "${TARGET}:${REMOTE_DIR}/docker/"

echo "==> building ${REMOTE_IMAGE} (first run pulls the CUDA base image, ~5GB)"
ssh "$TARGET" "cd ${REMOTE_DIR}/docker && docker build -t ${REMOTE_IMAGE} \
  --build-arg NUM_THREADS=8 \
  --build-arg NODEJS_MAJOR_VERSION=20 \
  --build-arg UBUNTU_VERSION=22.04 \
  --build-arg CUDA_VERSION=11.8.0 ."

echo
echo "==> verifying tfjs-node-gpu actually computes on GPU ${REMOTE_GPUS}"
# A device merely being *listed* is not proof: an unsupported arch shows up and
# then fails at kernel launch. docker/verify-gpu.js runs a real matmul.
ssh "$TARGET" "docker run --rm \
  --gpus 'device=${REMOTE_GPUS}' \
  --user \$(id -u):\$(id -g) \
  -v ${REMOTE_ABS}:/opt/app \
  -w /opt/app \
  ${REMOTE_IMAGE} \
  bash -c 'npm install --unsafe-perm --cache /tmp/.npm-cache >/dev/null 2>&1 && node docker/verify-gpu.js'"

echo
echo "==> done. Next: ./scripts/sync.sh && ./scripts/train_remote.sh"

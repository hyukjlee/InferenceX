#!/usr/bin/env bash
# Self-hosted docker launcher for the standalone m15-17 MI355X node
# (smci355-ccs-aus-m15-17.cs-aus.dcgpu, 8x MI355X gfx950, ~3.0 TiB DRAM).
#
# m15-17 is NOT part of the mia1 slurm/enroot cluster -- it runs plain docker,
# so launch_mi355x-amds.sh's salloc/enroot/pyxis path does not apply. This
# launcher mirrors that script's single-node recipe-path resolution and env
# contract, but executes the recipe inside `docker run` with the InferenceX
# checkout bind-mounted at /workspace.
#
# Env parity with the slurm path matters: there, `srun --export=ALL` hands the
# container the entire benchmark-tmpl.yml env block. Docker has no equivalent,
# so every variable in that block is forwarded explicitly below. Keep the two in
# sync when benchmark-tmpl.yml gains a variable.
#
# Host paths (pre-created once, see the runner bring-up notes):
#   /data/models       HF hub cache (amd/Kimi-K2.7-Code-MXFP4 staged, ~514 GB)
#   /data/aiperf-cache aiperf trace mmap cache  (= /it-share/aiperf-cache/)
#   /data/vllm-cache   torch.compile cache      (= VLLM_CACHE_ROOT on amds)
set -uo pipefail
set -x

# Per-runner port offset (last char of runner name), same scheme as amds.
PORT_OFFSET="${RUNNER_NAME: -1}"
[[ "$PORT_OFFSET" =~ ^[0-9]$ ]] || PORT_OFFSET=0
export PORT=$(( 8888 + PORT_OFFSET ))

# Node-local caches. Container-side paths match the amds slurm mounts so the
# recipe and aiperf see identical locations on both runners.
HOST_HF_CACHE="${M1517_HF_CACHE:-/data/models}"
CONTAINER_HF_CACHE="${HF_HUB_CACHE:-/mnt/hf_hub_cache/}"
HOST_AIPERF_CACHE="${M1517_AIPERF_CACHE:-/data/aiperf-cache}"
HOST_VLLM_CACHE="${M1517_VLLM_CACHE:-/data/vllm-cache}"
mkdir -p "$HOST_AIPERF_CACHE" "$HOST_VLLM_CACHE" 2>/dev/null || true

FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "atom" ]] && printf '_atom' || printf '')
SPEC_SUFFIX=$([[ "${SPEC_DECODING:-none}" == "mtp" ]] && printf '_mtp' || printf '')

# Recipe-path resolution mirrors launch_mi355x-amds.sh.
SCRIPT_BASE="${EXP_NAME%%_*}_${PRECISION}_mi355x"
SCRIPT_FW="benchmarks/single_node/${SCENARIO_SUBDIR:-fixed_seq_len/}${SCRIPT_BASE}_${FRAMEWORK}${SPEC_SUFFIX}.sh"
SCRIPT_FALLBACK="benchmarks/single_node/${SCENARIO_SUBDIR:-fixed_seq_len/}${SCRIPT_BASE}${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh"
if [[ -f "$SCRIPT_FW" ]]; then
    BENCHMARK_SCRIPT="$SCRIPT_FW"
else
    BENCHMARK_SCRIPT="$SCRIPT_FALLBACK"
fi
echo "[m1517docker] recipe: $BENCHMARK_SCRIPT"

# GPU selection: first $TP devices. m15-17 is a shared node -- other tenants'
# containers run without an explicit device list, so this only reserves our
# slice by convention, not by enforcement.
DEVLIST=""
for i in $(seq 0 $((TP-1))); do DEVLIST="${DEVLIST}${i},"; done
DEVLIST="${DEVLIST%,}"

CONTAINER="m1517_${RUNNER_NAME}_$$"

# Pre-clean stale containers holding our GPUs.
docker rm -f "$CONTAINER" 2>/dev/null || true

# Ensure image is present (pull if missing; some are local-only builds).
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker pull "$IMAGE" || { echo "[m1517docker] docker pull failed for $IMAGE" >&2; exit 1; }
fi

# The recipe writes RESULT_DIR (=/workspace/results) and the top-level result
# json under /workspace; both are the bind-mounted checkout, so artifacts land
# in $GITHUB_WORKSPACE for the upload steps.
docker run --rm --name "$CONTAINER" \
    --device /dev/kfd --device /dev/dri \
    --ipc=host --shm-size=0 \
    --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    -e ROCR_VISIBLE_DEVICES="$DEVLIST" \
    -e HIP_VISIBLE_DEVICES="$DEVLIST" \
    -e HF_HUB_CACHE="$CONTAINER_HF_CACHE" \
    -e HF_HOME="$CONTAINER_HF_CACHE" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e PORT="$PORT" \
    -e RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}" \
    -e MODEL="$MODEL" \
    -e MODEL_PREFIX="${MODEL_PREFIX:-}" \
    -e EXP_NAME="${EXP_NAME:-}" \
    -e PRECISION="${PRECISION:-}" \
    -e FRAMEWORK="${FRAMEWORK:-}" \
    -e IMAGE="${IMAGE:-}" \
    -e TP="$TP" \
    -e PP_SIZE="${PP_SIZE:-1}" \
    -e DCP_SIZE="${DCP_SIZE:-1}" \
    -e PCP_SIZE="${PCP_SIZE:-1}" \
    -e EP_SIZE="${EP_SIZE:-1}" \
    -e DP_ATTENTION="${DP_ATTENTION:-false}" \
    -e CONC="$CONC" \
    -e ISL="${ISL:-0}" \
    -e OSL="${OSL:-0}" \
    -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-0}" \
    -e SPEC_DECODING="${SPEC_DECODING:-none}" \
    -e DISAGG="${DISAGG:-false}" \
    -e SCENARIO_TYPE="${SCENARIO_TYPE:-}" \
    -e SCENARIO_SUBDIR="${SCENARIO_SUBDIR:-}" \
    -e IS_AGENTIC="${IS_AGENTIC:-0}" \
    -e KV_OFFLOADING="${KV_OFFLOADING:-}" \
    -e KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-}" \
    -e KV_OFFLOAD_BACKEND_METADATA="${KV_OFFLOAD_BACKEND_METADATA:-}" \
    -e ROUTER_METADATA="${ROUTER_METADATA:-}" \
    -e KV_P2P_TRANSFER="${KV_P2P_TRANSFER:-}" \
    -e TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-0}" \
    -e DURATION="${DURATION:-3600}" \
    -e RUN_EVAL="${RUN_EVAL:-false}" \
    -e EVAL_ONLY="${EVAL_ONLY:-false}" \
    -e EVAL_LIMIT="${EVAL_LIMIT:-}" \
    -e SWEBENCH_GEN_MODE="${SWEBENCH_GEN_MODE:-}" \
    -e SWEBENCH_USE_MODAL="${SWEBENCH_USE_MODAL:-true}" \
    -e MODAL_TOKEN_ID="${MODAL_TOKEN_ID:-}" \
    -e MODAL_TOKEN_SECRET="${MODAL_TOKEN_SECRET:-}" \
    -e AIPERF_FAILED_REQUEST_THRESHOLD="${AIPERF_FAILED_REQUEST_THRESHOLD:-0.10}" \
    -e AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache \
    -e RESULT_DIR="${RESULT_DIR:-/workspace/results}" \
    -e RESULT_FILENAME="${RESULT_FILENAME:-}" \
    -e RUNNER_TYPE="${RUNNER_TYPE:-}" \
    -e VLLM_CACHE_ROOT=/vllm_cache \
    -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
    -e PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}" \
    -e PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/inferencex-pycache}" \
    -e PYTHONHASHSEED=0 \
    -v "$GITHUB_WORKSPACE":/workspace \
    -v "$HOST_HF_CACHE":"$CONTAINER_HF_CACHE" \
    -v "$HOST_AIPERF_CACHE":/aiperf_mmap_cache \
    -v "$HOST_VLLM_CACHE":/vllm_cache \
    -w /workspace \
    --entrypoint bash \
    "$IMAGE" \
    "$BENCHMARK_SCRIPT"
RC=$?

# The recipe runs as root inside the container and writes results/ (and any
# other artifacts) into the bind-mounted workspace as root. The next job's
# actions/checkout `clean: true` runs as the (non-root) runner user and would
# fail to remove them (EACCES). Chown the workspace back to the host UID/GID
# via a throwaway root container so the checkout can always clean it.
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
docker run --rm -v "$GITHUB_WORKSPACE":/workspace --entrypoint chown \
    "$IMAGE" -R "${HOST_UID}:${HOST_GID}" /workspace 2>/dev/null || \
    echo "[m1517docker] WARN: workspace chown failed; next checkout may need manual cleanup"

echo "[m1517docker] recipe exit=$RC"
exit $RC

#!/usr/bin/env bash
# Stage-1 smoke test: prove DiffusionGemma serves, fits VRAM, and tool-calls
# before standing up the full stack. Two modes:
#
#  A) Launch locally (full-VM / SSH box with docker + nvidia runtime):
#       HF_TOKEN=hf_xxx bash deploy/try_vllm.sh
#     Runs vLLM standalone (it owns the whole card), then tests it.
#
#  B) Test a remote endpoint (e.g. a RunPod pod already running the vLLM image):
#       VLLM_ENDPOINT=https://<pod-id>-8000.proxy.runpod.net/v1 bash deploy/try_vllm.sh
#     Skips launching; just runs the client checks against that URL.
#
# First run downloads ~18GB (NVFP4), so mode A takes several minutes.
set -euo pipefail

MODEL="${VLLM_MODEL:-RedHatAI/diffusiongemma-26B-A4B-it-NVFP4}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:gemma}"
PORT="${VLLM_PORT:-8000}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
GPU_FRAC="${VLLM_GPU_FRACTION:-0.85}"
NAME="diffusiongemma-smoke"
LOG="/tmp/diffusiongemma-vllm.log"

ENDPOINT="${VLLM_ENDPOINT:-}"
if [ -n "$ENDPOINT" ]; then
  LAUNCH=0
  BASE="${ENDPOINT%/}"
  echo "[try_vllm] client mode — testing existing endpoint $BASE"
else
  LAUNCH=1
  BASE="http://localhost:${PORT}/v1"
  [ -z "${HF_TOKEN:-}" ] && echo "WARNING: HF_TOKEN unset — the gated Gemma repo will fail to download." >&2
fi

cleanup() { [ "$LAUNCH" = 1 ] && docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

if [ "$LAUNCH" = 1 ]; then
  echo "[try_vllm] launching $MODEL on $IMAGE (gpu_frac=$GPU_FRAC, max_len=$MAX_LEN)"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" --ipc=host --gpus all \
    -p "${PORT}:8000" \
    -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "$IMAGE" \
      --model "$MODEL" \
      --max-model-len "$MAX_LEN" \
      --max-num-seqs "${VLLM_MAX_NUM_SEQS:-4}" \
      --gpu-memory-utilization "$GPU_FRAC" \
      --generation-config vllm \
      --hf-overrides '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}' \
      --diffusion-config '{"canvas_length": 256}' \
      --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
      --host 0.0.0.0 --port 8000 >/dev/null
  echo "[try_vllm] streaming server log to $LOG — waiting for readiness…"
  docker logs -f "$NAME" >"$LOG" 2>&1 &
fi

# Wait up to ~20 min for the OpenAI server (download + load can be slow).
for _ in $(seq 1 240); do
  if curl -fsS -m 3 "$BASE/models" >/dev/null 2>&1; then break; fi
  if [ "$LAUNCH" = 1 ] && ! docker ps -q -f name="$NAME" | grep -q .; then
    echo "[try_vllm] container exited early — last log lines:" >&2; tail -n 40 "$LOG" >&2; exit 1
  fi
  sleep 5
done
curl -fsS -m 3 "$BASE/models" >/dev/null 2>&1 || { echo "[try_vllm] server never became ready" >&2; exit 1; }
echo "[try_vllm] server is up."

echo; echo "=== /v1/models ==="
curl -fsS "$BASE/models" | python3 -m json.tool

echo; echo "=== plain completion ==="
curl -fsS "$BASE/chat/completions" -H 'Content-Type: application/json' -d "$(python3 - "$MODEL" <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[1], "max_tokens": 200,
  "messages": [{"role": "user", "content": "In two sentences, what is a guardianship petition?"}]}))
PY
)" | python3 -c "import sys,json; m=json.load(sys.stdin)['choices'][0]['message']; print(m.get('content') or '(no content)')"

echo; echo "=== tool-calling test (must emit a tool_call) ==="
curl -fsS "$BASE/chat/completions" -H 'Content-Type: application/json' -d "$(python3 - "$MODEL" <<'PY'
import json, sys
print(json.dumps({
  "model": sys.argv[1], "max_tokens": 256, "tool_choice": "auto",
  "messages": [{"role": "user", "content": "What's the weather in Bentonville, Arkansas? Use the tool."}],
  "tools": [{"type": "function", "function": {
    "name": "get_weather", "description": "Get current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
}))
PY
)" | python3 -c "
import sys,json
m=json.load(sys.stdin)['choices'][0]['message']
tc=m.get('tool_calls') or []
if tc:
    f=tc[0]['function']; print('TOOL CALL OK ->', f['name'], f.get('arguments'))
else:
    print('NO TOOL CALL. content:', (m.get('content') or '')[:300]); sys.exit(2)
"

if [ "$LAUNCH" = 1 ]; then
  echo; echo "=== GPU memory ==="
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>/dev/null || echo "(nvidia-smi unavailable)"
fi

echo; echo "[try_vllm] PASS — model serves, completes, and tool-calls."
[ "$LAUNCH" = 1 ] && echo "[try_vllm] full server log: $LOG"

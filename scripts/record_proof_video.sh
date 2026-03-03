#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# record_proof_video.sh
#
# Runs a REAL E2E test against LiteLLM proxy with:
#   - Bedrock Claude Opus 4.6
#   - Thinking enabled (budget_tokens=5000)
#   - WebSearch tool (Perplexity backend)
#   - A current-events prompt (US-Iran war news)
#
# Captures all terminal output and generates a ~60s video showing:
#   1. The request (prompt clearly visible)
#   2. Proxy logs showing websearch interception
#   3. The full response text
#   4. Verdict
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${WK_DIR}/artifacts/proof-${TS}"
mkdir -p "${OUT}"

PYTHON="${PYTHON_BIN:-$(command -v python3.13 || command -v python3)}"
export PYTHONPATH="${WK_DIR}"

LOG="${OUT}/session.log"
PROXY_LOG="${OUT}/proxy.log"
PROXY_CFG="${OUT}/proxy_config.yaml"
RESP_FILE="${OUT}/response.json"

# ── Proxy Config ──
cat > "${PROXY_CFG}" <<YAML
model_list:
  - model_name: claude-opus-4-6
    litellm_params:
      model: bedrock/us.anthropic.claude-opus-4-6-v1
search_tools:
  - search_tool_name: "my-perplexity-search"
    litellm_params:
      search_provider: "perplexity"
      api_key: "os.environ/PERPLEXITY_API_KEY"
litellm_settings:
  callbacks: ["websearch_interception"]
  websearch_interception_params:
    enabled_providers: ["bedrock"]
    search_tool_name: "my-perplexity-search"
YAML

# ── Helpers ──
PORT=14099
cleanup() {
  if [ -n "${PROXY_PID:-}" ]; then
    kill "$PROXY_PID" 2>/dev/null || true
    wait "$PROXY_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

tee_log() { tee -a "$LOG"; }

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0: Start proxy (silent — not part of recording)
# ══════════════════════════════════════════════════════════════════════════════
echo "Starting LiteLLM proxy on port ${PORT}..." | tee_log
cd "${WK_DIR}"
${PYTHON} -m litellm.proxy.proxy_cli --config "${PROXY_CFG}" \
  --port "${PORT}" --detailed_debug --num_workers 1 \
  > "${PROXY_LOG}" 2>&1 &
PROXY_PID=$!

# Wait for proxy readiness
for i in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "Proxy ready (PID=${PROXY_PID})" | tee_log
    break
  fi
  [ "$i" -eq 30 ] && { echo "Proxy failed to start"; cat "${PROXY_LOG}" | tail -20; exit 1; }
  sleep 1
done

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Capture the real test
# ══════════════════════════════════════════════════════════════════════════════
# This is what gets recorded into the video
SESSION_OUTPUT="${OUT}/test_output.txt"

{
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════════╗"
  echo "║  LiteLLM REAL E2E PROOF — WebSearch + Thinking Fix                 ║"
  echo "║  PR: giulio-leone/litellm#28                                       ║"
  echo "║  Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')                                    ║"
  echo "╚══════════════════════════════════════════════════════════════════════╝"
  echo ""
  echo "Test Configuration:"
  echo "  Model:    bedrock/us.anthropic.claude-opus-4-6-v1 (Bedrock Claude Opus 4.6)"
  echo "  Thinking: enabled, budget_tokens=5000"
  echo "  Search:   Perplexity AI (real web search)"
  echo "  Proxy:    LiteLLM with fix applied"
  echo ""
  echo "────────────────────────────────────────────────────────────────────────"
  echo "STEP 1: Sending request to LiteLLM proxy..."
  echo "────────────────────────────────────────────────────────────────────────"
  echo ""
  echo '$ curl -s http://127.0.0.1:'"${PORT}"'/v1/messages \'
  echo '    -H "content-type: application/json" \'
  echo '    -H "anthropic-version: 2023-06-01" \'
  echo '    -d '"'"'{'
  echo '      "model": "claude-opus-4-6",'
  echo '      "max_tokens": 8192,'
  echo '      "thinking": {"type": "enabled", "budget_tokens": 5000},'
  echo '      "tools": [{"name": "WebSearch", ...}],'
  echo '      "messages": [{"role": "user", "content":'
  echo '        "Use WebSearch to find the latest news about the'
  echo '         US-Iran situation in 2025. Summarize the key'
  echo '         developments and provide source URLs."'
  echo '      }]'
  echo "    }'"
  echo ""
  echo "Waiting for response (real API call to Bedrock + Perplexity)..."
  echo ""

  # ── ACTUAL CURL CALL ──
  HTTP_CODE=$(curl -s -o "${RESP_FILE}" -w "%{http_code}" \
    "http://127.0.0.1:${PORT}/v1/messages" \
    -H "content-type: application/json" \
    -H "anthropic-version: 2023-06-01" \
    --max-time 300 \
    -d '{
      "model": "claude-opus-4-6",
      "max_tokens": 8192,
      "thinking": {"type": "enabled", "budget_tokens": 5000},
      "tools": [{"name": "WebSearch", "description": "Search the web for current information", "input_schema": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}}, "required": ["query"]}}],
      "messages": [{"role": "user", "content": "Use WebSearch to find the latest news about the US-Iran situation in 2025. Summarize the key developments and provide source URLs."}]
    }')

  echo "────────────────────────────────────────────────────────────────────────"
  echo "STEP 2: Response received! HTTP ${HTTP_CODE}"
  echo "────────────────────────────────────────────────────────────────────────"
  echo ""

  # Parse response
  STOP_REASON=$(python3 -c "import json; r=json.load(open('${RESP_FILE}')); print(r.get('stop_reason','ERROR'))" 2>/dev/null || echo "PARSE_ERROR")
  MODEL=$(python3 -c "import json; r=json.load(open('${RESP_FILE}')); print(r.get('model','unknown'))" 2>/dev/null || echo "unknown")

  echo "  stop_reason: ${STOP_REASON}"
  echo "  model:       ${MODEL}"
  echo ""

  # Show response text
  echo "────────────────────────────────────────────────────────────────────────"
  echo "STEP 3: Response content"
  echo "────────────────────────────────────────────────────────────────────────"
  echo ""
  python3 -c "
import json, textwrap
r = json.load(open('${RESP_FILE}'))
for block in r.get('content', []):
    if block.get('type') == 'text':
        text = block['text']
        # Show first 1500 chars
        for line in text[:1500].split('\n'):
            print('  ' + line)
        if len(text) > 1500:
            print('  ...[truncated]...')
        break
" 2>/dev/null || echo "  [Error parsing response]"
  echo ""

  # Show proxy log evidence
  echo "────────────────────────────────────────────────────────────────────────"
  echo "STEP 4: Proxy log evidence (websearch interception)"
  echo "────────────────────────────────────────────────────────────────────────"
  echo ""
  grep "WebSearchInterception:" "${PROXY_LOG}" | while read -r line; do
    ts="${line%%  *}"
    msg="${line#*WebSearchInterception: }"
    echo "  [LOG] ${msg}"
  done
  echo ""

  # Verdict
  echo "────────────────────────────────────────────────────────────────────────"
  echo "VERDICT"
  echo "────────────────────────────────────────────────────────────────────────"
  echo ""
  
  PASS=true
  if [ "${HTTP_CODE}" = "200" ]; then
    echo "  ✅ HTTP 200 (no 400 error — thinking constraint fix works)"
  else
    echo "  ❌ HTTP ${HTTP_CODE} — UNEXPECTED"
    PASS=false
  fi

  if [ "${STOP_REASON}" = "end_turn" ]; then
    echo "  ✅ stop_reason=end_turn (websearch interception completed full loop)"
  else
    echo "  ❌ stop_reason=${STOP_REASON} — UNEXPECTED"
    PASS=false
  fi

  INTERCEPT_COUNT=$(grep -c "WebSearchInterception: Detected" "${PROXY_LOG}" 2>/dev/null || echo "0")
  if [ "${INTERCEPT_COUNT}" -gt 0 ]; then
    echo "  ✅ WebSearch interception fired (${INTERCEPT_COUNT} time(s))"
  else
    echo "  ❌ WebSearch interception did NOT fire"
    PASS=false
  fi

  AGENTIC_COUNT=$(grep -c "Executing agentic loop" "${PROXY_LOG}" 2>/dev/null || echo "0")
  if [ "${AGENTIC_COUNT}" -gt 0 ]; then
    echo "  ✅ Agentic loop executed (${AGENTIC_COUNT} iteration(s))"
  else
    echo "  ❌ Agentic loop did NOT execute"
    PASS=false
  fi

  echo ""
  if [ "${PASS}" = true ]; then
    echo "  ╔═══════════════════════════════════════════╗"
    echo "  ║   ✅  ALL CHECKS PASSED — BUGS FIXED     ║"
    echo "  ╚═══════════════════════════════════════════╝"
  else
    echo "  ╔═══════════════════════════════════════════╗"
    echo "  ║   ❌  SOME CHECKS FAILED                 ║"
    echo "  ╚═══════════════════════════════════════════╝"
  fi
  echo ""

} 2>&1 | tee "${SESSION_OUTPUT}"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Generate video from captured output
# ══════════════════════════════════════════════════════════════════════════════
echo "Generating proof video..."

python3 << PYEOF
import textwrap, os, subprocess, math
from PIL import Image, ImageDraw, ImageFont

SESSION = "${SESSION_OUTPUT}"
FRAME_DIR = "${OUT}/video_frames"
VIDEO_OUT = "${OUT}/proof_video.mp4"
os.makedirs(FRAME_DIR, exist_ok=True)

W, H = 1920, 1080
BG = (15, 15, 25)
GREEN = (0, 255, 100)
WHITE = (210, 210, 210)
YELLOW = (255, 220, 50)
CYAN = (0, 200, 255)
RED = (255, 60, 60)

try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 20)
except:
    FONT = ImageFont.load_default()

LINES_PER_FRAME = 42
FPS = 5
LINES_PER_SEC = 4  # scroll speed

# Read session output
with open(SESSION) as f:
    raw_lines = f.read().split("\n")

# Wrap long lines
lines = []
for l in raw_lines:
    if len(l) > 95:
        for wl in textwrap.wrap(l, width=95):
            lines.append(wl)
    else:
        lines.append(l)

def color_for_line(line):
    ls = line.strip()
    if ls.startswith("✅") or ls.startswith("╔") or ls.startswith("╚") or ls.startswith("║"):
        return GREEN
    if ls.startswith("❌"):
        return RED
    if ls.startswith("$") or ls.startswith("STEP") or ls.startswith("VERDICT"):
        return YELLOW
    if ls.startswith("[LOG]"):
        return CYAN
    if "────" in ls or "═══" in ls:
        return (80, 80, 100)
    return WHITE

def render_frame(visible_lines, frame_idx):
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    y = 20
    for line in visible_lines:
        color = color_for_line(line)
        draw.text((30, y), line, fill=color, font=FONT)
        y += 24
    path = os.path.join(FRAME_DIR, f"f_{frame_idx:05d}.png")
    img.save(path)

total_lines = len(lines)
# Calculate frames needed: scroll through all lines
# Each frame scrolls LINES_PER_SEC/FPS lines
scroll_step = max(1, LINES_PER_SEC / FPS)
# Add pause frames at beginning and end
PAUSE_START = FPS * 2   # 2s pause at start
PAUSE_END = FPS * 3     # 3s pause at end

frame_idx = 0

# Pause at start (show first screen)
for _ in range(PAUSE_START):
    visible = lines[0:LINES_PER_FRAME]
    render_frame(visible, frame_idx)
    frame_idx += 1

# Scroll through content
scroll_pos = 0.0
while int(scroll_pos) + LINES_PER_FRAME < total_lines:
    start = int(scroll_pos)
    visible = lines[start:start + LINES_PER_FRAME]
    render_frame(visible, frame_idx)
    frame_idx += 1
    scroll_pos += scroll_step

# Pause at end (show last screen)
last_start = max(0, total_lines - LINES_PER_FRAME)
for _ in range(PAUSE_END):
    visible = lines[last_start:last_start + LINES_PER_FRAME]
    render_frame(visible, frame_idx)
    frame_idx += 1

duration = frame_idx / FPS
print(f"Rendered {frame_idx} frames → {duration:.0f}s at {FPS}fps")

# ffmpeg
cmd = [
    "ffmpeg", "-y",
    "-framerate", str(FPS),
    "-i", os.path.join(FRAME_DIR, "f_%05d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-preset", "fast", "-crf", "23",
    VIDEO_OUT
]
subprocess.run(cmd, capture_output=True)
sz = os.path.getsize(VIDEO_OUT) / (1024*1024)
print(f"Video: {VIDEO_OUT} ({sz:.1f}MB)")
PYEOF

echo ""
echo "═══════════════════════════════════════════════"
echo "Artifacts in: ${OUT}/"
ls -lh "${OUT}/proof_video.mp4" "${OUT}/session.log" "${OUT}/response.json" 2>/dev/null
echo "═══════════════════════════════════════════════"

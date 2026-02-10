#!/usr/bin/env bash
# =============================================================================
# LiteLLM Proxy Setup Script
#
# Usage:
#   git clone https://github.com/Counterweight-AI/litellm.git
#   cd litellm
#   ./setup.sh
#
# This script:
#   1. Checks for Python 3.10+
#   2. Creates a virtual environment
#   3. Installs LiteLLM with proxy support
#   4. Patches the proxy config for the current machine
#   5. Prompts for API keys and writes a .env file
#   6. Starts the proxy on port 4000
# =============================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$REPO_ROOT/litellm/proxy/proxy_config.yaml"
ROUTING_RULES="$REPO_ROOT/litellm/router_strategy/auto_router/routing_rules.yaml"
VENV_DIR="$REPO_ROOT/.venv"
ENV_FILE="$REPO_ROOT/.env"
MODELS_FILE="$REPO_ROOT/models.yaml"

MIN_PY_MINOR=10  # Python 3.10+ required (mcp, python-multipart, polars need it)
MAX_PY_MINOR=13  # Python 3.14+ breaks uvloop; cap at 3.13

# ---------- helpers ----------------------------------------------------------

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# Cross-platform sed -i (GNU vs BSD/macOS)
sedi() {
    if sed --version &>/dev/null 2>&1; then
        sed -i "$@"
    else
        sed -i '' "$@"
    fi
}

# ---------- 1. Python check -------------------------------------------------

echo ""
echo -e "${BOLD}=== LiteLLM Proxy Setup ===${NC}"
echo ""

info "Checking for Python 3.${MIN_PY_MINOR}–3.${MAX_PY_MINOR}..."

# Search order: versioned binaries first (most precise), then homebrew/pyenv,
# then generic names. This avoids macOS system 3.9 and bleeding-edge 3.14+.
PYTHON=""
for candidate in \
    python3.13 python3.12 python3.11 python3.10 \
    /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.10 \
    /opt/homebrew/bin/python3 \
    /usr/local/bin/python3 \
    "$HOME/.pyenv/shims/python3" \
    python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PY_MAJOR=$("$candidate" -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo 0)
        PY_MINOR=$("$candidate" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 0)
        if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge "$MIN_PY_MINOR" ] && [ "$PY_MINOR" -le "$MAX_PY_MINOR" ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo ""
    fail "Python 3.${MIN_PY_MINOR}–3.${MAX_PY_MINOR} is required but not found.

  - Python < 3.10 is missing required packages (mcp, python-multipart).
  - Python 3.14+ is not yet supported (uvloop incompatibility).

  Install a supported version:
    macOS:   brew install python@3.12
    Ubuntu:  sudo apt install python3.12 python3.12-venv
    Any:     https://www.python.org/downloads/

  Then re-run: ./setup.sh"
fi

PY_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
ok "Found $PYTHON ($PY_VERSION)"

# ---------- 2. Virtual environment ------------------------------------------

info "Setting up virtual environment..."

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Created virtual environment at .venv/"
else
    ok "Virtual environment already exists at .venv/"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ---------- 3. Install -------------------------------------------------------

info "Upgrading pip..."
pip install --upgrade pip --quiet

info "Installing LiteLLM with proxy extras (this may take a minute)..."
pip install -e "$REPO_ROOT[proxy]" 2>&1 | tail -5

# Verify the install
if ! command -v litellm &>/dev/null; then
    fail "litellm CLI not found after install. Check the output above for errors."
fi

LITELLM_VERSION=$(python -c 'from importlib.metadata import version; print(version("litellm"))' 2>/dev/null || echo "unknown")
ok "Installed litellm $LITELLM_VERSION"

# ---------- 4. Patch proxy config -------------------------------------------

info "Configuring proxy config for this machine..."

if [ ! -f "$CONFIG_FILE" ]; then
    fail "Proxy config not found at $CONFIG_FILE"
fi

if [ ! -f "$ROUTING_RULES" ]; then
    fail "Routing rules not found at $ROUTING_RULES"
fi

# Fix the auto_router_config_path to point to this clone's absolute path
sedi "s|auto_router_config_path:.*|auto_router_config_path: \"$ROUTING_RULES\"|" "$CONFIG_FILE"
ok "Updated auto_router_config_path -> $ROUTING_RULES"

# Disable mcp_semantic_tool_filter (requires optional semantic-router package)
sedi "s|enabled: true|enabled: false|" "$CONFIG_FILE"
ok "Disabled mcp_semantic_tool_filter (enable after: pip install 'litellm[semantic-router]')"

# ---------- 5. API keys (.env) ----------------------------------------------

# Initialize all key variables (may be populated by prompts or existing .env)
GOOGLE_KEY="" OPENAI_KEY="" ANTHROPIC_KEY=""
DEEPSEEK_KEY="" MOONSHOT_KEY="" ZAI_KEY="" XAI_KEY="" MINIMAX_KEY=""

if [ -f "$ENV_FILE" ]; then
    ok ".env already exists — skipping API key prompts"
    # Load existing keys so tier suggestions can use them
    _getkey() { awk -F= "/^$1=/{print \$2}" "$ENV_FILE" | head -1; }
    GOOGLE_KEY=$(_getkey GOOGLE_API_KEY)
    OPENAI_KEY=$(_getkey OPENAI_API_KEY)
    ANTHROPIC_KEY=$(_getkey ANTHROPIC_API_KEY)
    DEEPSEEK_KEY=$(_getkey DEEPSEEK_API_KEY)
    MOONSHOT_KEY=$(_getkey MOONSHOT_API_KEY)
    ZAI_KEY=$(_getkey ZAI_API_KEY)
    XAI_KEY=$(_getkey XAI_API_KEY)
    MINIMAX_KEY=$(_getkey MINIMAX_API_KEY)
else
    echo ""
    echo -e "${BOLD}API Key Setup${NC}"
    echo "Enter the keys you have; press Enter to skip any you don't need."
    echo ""

    # --- Core providers (always asked) ---
    read -rp "  Google API Key    (for Gemini — used by auto-router) : " GOOGLE_KEY
    read -rp "  OpenAI API Key    (for GPT models)                   : " OPENAI_KEY
    read -rp "  Anthropic API Key (for Claude via API)               : " ANTHROPIC_KEY

    # --- Optional providers ---
    echo ""
    echo -e "${BOLD}Optional providers:${NC}"
    echo "  1) DeepSeek"
    echo "  2) Kimi (Moonshot)"
    echo "  3) GLM (Zhipu/ZAI)"
    echo "  4) Grok (xAI)"
    echo "  5) MiniMax"
    echo ""
    read -rp "  Enter numbers to configure (e.g. 1 3 5), or press Enter to skip: " EXTRA_CHOICES

    for choice in $EXTRA_CHOICES; do
        case "$choice" in
            1) read -rp "  DeepSeek API Key   : " DEEPSEEK_KEY ;;
            2) read -rp "  Kimi API Key       : " MOONSHOT_KEY ;;
            3) read -rp "  GLM (ZAI) API Key  : " ZAI_KEY ;;
            4) read -rp "  Grok (xAI) API Key : " XAI_KEY ;;
            5) read -rp "  MiniMax API Key    : " MINIMAX_KEY ;;
            *) warn "Unknown option: $choice (skipping)" ;;
        esac
    done

    echo ""

    ALL_EMPTY=true
    for _k in "$GOOGLE_KEY" "$OPENAI_KEY" "$ANTHROPIC_KEY" \
              "$DEEPSEEK_KEY" "$MOONSHOT_KEY" "$ZAI_KEY" "$XAI_KEY" "$MINIMAX_KEY"; do
        if [ -n "$_k" ]; then ALL_EMPTY=false; break; fi
    done
    if $ALL_EMPTY; then
        warn "No API keys provided. The proxy will start but model calls will fail"
        warn "until you add keys to $ENV_FILE"
    fi

    cat > "$ENV_FILE" <<EOF
# LiteLLM API Keys — edit as needed
OPENAI_API_KEY=${OPENAI_KEY:-}
GOOGLE_API_KEY=${GOOGLE_KEY:-}
ANTHROPIC_API_KEY=${ANTHROPIC_KEY:-}
DEEPSEEK_API_KEY=${DEEPSEEK_KEY:-}
MOONSHOT_API_KEY=${MOONSHOT_KEY:-}
ZAI_API_KEY=${ZAI_KEY:-}
XAI_API_KEY=${XAI_KEY:-}
MINIMAX_API_KEY=${MINIMAX_KEY:-}
EOF

    ok "Created .env with your API keys"
fi

# ---------- 5b. Suggest auto-router tier models ------------------------------

# Collect providers that have keys
AVAILABLE=""
[ -n "$GOOGLE_KEY" ]    && AVAILABLE="$AVAILABLE google"
[ -n "$OPENAI_KEY" ]    && AVAILABLE="$AVAILABLE openai"
[ -n "$ANTHROPIC_KEY" ] && AVAILABLE="$AVAILABLE anthropic"
[ -n "$DEEPSEEK_KEY" ]  && AVAILABLE="$AVAILABLE deepseek"
[ -n "$MOONSHOT_KEY" ]  && AVAILABLE="$AVAILABLE moonshot"
[ -n "$ZAI_KEY" ]       && AVAILABLE="$AVAILABLE zai"
[ -n "$XAI_KEY" ]       && AVAILABLE="$AVAILABLE xai"
[ -n "$MINIMAX_KEY" ]   && AVAILABLE="$AVAILABLE minimax"

# Pick best available model per tier from models.yaml
eval "$(MODELS_FILE="$MODELS_FILE" AVAILABLE="$AVAILABLE" python << 'PYEOF'
import yaml, os
providers = set(os.environ.get("AVAILABLE", "").split())
with open(os.environ["MODELS_FILE"]) as f:
    cfg = yaml.safe_load(f)
for tier in ["low", "mid", "top"]:
    found = False
    for c in cfg["tier_candidates"].get(tier, []):
        if c["provider"] in providers:
            m, d, cost = c["model"], c["display"], c["cost"]
            safe_cost = cost.replace("$", "\\$")
            print(f'{tier.upper()}_MODEL="{m}"')
            print(f'{tier.upper()}_DISPLAY="{d} ({safe_cost})"')
            found = True
            break
    if not found:
        print(f'{tier.upper()}_MODEL=""')
        print(f'{tier.upper()}_DISPLAY=""')
PYEOF
)"

if [ -n "$LOW_MODEL" ] || [ -n "$MID_MODEL" ] || [ -n "$TOP_MODEL" ]; then
    echo ""
    echo -e "${BOLD}Auto-Router Tier Suggestions${NC}  (based on your API keys)"
    echo ""
    [ -n "$LOW_MODEL" ]  && echo -e "  low : ${GREEN}$LOW_DISPLAY${NC}  ->  $LOW_MODEL"
    [ -n "$MID_MODEL" ]  && echo -e "  mid : ${YELLOW}$MID_DISPLAY${NC}  ->  $MID_MODEL"
    [ -n "$TOP_MODEL" ]  && echo -e "  top : ${CYAN}$TOP_DISPLAY${NC}  ->  $TOP_MODEL"
    echo ""
    read -rp "  Apply these to routing_rules.yaml? [Y/n] " APPLY_TIERS
    if [ -z "$APPLY_TIERS" ] || [[ "$APPLY_TIERS" =~ ^[Yy] ]]; then
        [ -n "$LOW_MODEL" ] && sedi "/^  low:/,/^  [a-z]/{s|model:.*|model: \"$LOW_MODEL\"|;}" "$ROUTING_RULES"
        [ -n "$MID_MODEL" ] && sedi "/^  mid:/,/^  [a-z]/{s|model:.*|model: \"$MID_MODEL\"|;}" "$ROUTING_RULES"
        [ -n "$TOP_MODEL" ] && sedi "/^  top:/,/^[a-z]/{s|model:.*|model: \"$TOP_MODEL\"|;}" "$ROUTING_RULES"
        ok "Updated routing_rules.yaml with tier models"
    else
        info "Skipped — you can edit routing_rules.yaml manually later"
    fi
fi

# ---------- 5c. Add provider models to proxy_config.yaml --------------------

# Collect model entries to add, then insert them all at once before "auto".
_NEW_MODELS_FILE=$(mktemp)
trap 'rm -f "$_NEW_MODELS_FILE"' EXIT

# Queue a model entry (written to temp file, inserted later in one pass).
add_model() {
    local model_name="$1" model_id="$2" api_key_env="$3"
    # Skip if already in the config
    if grep -q "model_name: $model_name" "$CONFIG_FILE" 2>/dev/null; then
        return 0
    fi
    cat >> "$_NEW_MODELS_FILE" <<ENTRY

  - model_name: $model_name
    litellm_params:
      model: $model_id
      api_key: os.environ/$api_key_env
ENTRY
    return 0
}

# Read provider models from models.yaml and queue entries for available providers
_PROVIDER_MODELS=$(MODELS_FILE="$MODELS_FILE" AVAILABLE="$AVAILABLE" python << 'PYEOF'
import yaml, os
providers = set(os.environ.get("AVAILABLE", "").split())
with open(os.environ["MODELS_FILE"]) as f:
    cfg = yaml.safe_load(f)
for name, info in cfg["provider_models"].items():
    if name in providers:
        for m in info["models"]:
            print(f'{m["name"]}|{m["id"]}|{info["key_env"]}')
PYEOF
)

if [ -n "$_PROVIDER_MODELS" ]; then
    while IFS='|' read -r _mname _mid _mkey; do
        add_model "$_mname" "$_mid" "$_mkey"
    done <<< "$_PROVIDER_MODELS"
fi

# Flush queued models into proxy_config.yaml
if [ -s "$_NEW_MODELS_FILE" ]; then
    # Convert empty YAML list to block-style so we can append entries
    if grep -q 'model_list: \[\]' "$CONFIG_FILE"; then
        sedi 's/model_list: \[\]/model_list:/' "$CONFIG_FILE"
    fi

    if grep -q '- model_name: auto' "$CONFIG_FILE"; then
        # Insert before the "auto" entry
        awk -v newfile="$_NEW_MODELS_FILE" '
            /^  - model_name: auto$/ {
                while ((getline line < newfile) > 0) print line
                close(newfile)
            }
            { print }
        ' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
    else
        # No auto entry — append after "model_list:" line
        awk -v newfile="$_NEW_MODELS_FILE" '
            /^model_list:/ {
                print
                while ((getline line < newfile) > 0) print line
                close(newfile)
                next
            }
            { print }
        ' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
    fi
    ok "Added model(s) to proxy_config.yaml for configured providers"
else
    info "No new models to add to proxy_config.yaml"
fi
rm -f "$_NEW_MODELS_FILE"

# Ensure the "auto" model entry exists (uses the auto-router)
if ! grep -q 'model_name: auto' "$CONFIG_FILE"; then
    # Pick a default model: use the mid tier if set, otherwise first available provider model
    AUTO_DEFAULT="${MID_MODEL:-${LOW_MODEL:-${TOP_MODEL:-}}}"
    if [ -z "$AUTO_DEFAULT" ] && [ -n "$_PROVIDER_MODELS" ]; then
        AUTO_DEFAULT=$(echo "$_PROVIDER_MODELS" | head -1 | cut -d'|' -f2)
    fi
    AUTO_DEFAULT="${AUTO_DEFAULT:-gemini/gemini-3-flash-preview}"

    # Convert empty YAML list if not already done
    if grep -q 'model_list: \[\]' "$CONFIG_FILE"; then
        sedi 's/model_list: \[\]/model_list:/' "$CONFIG_FILE"
    fi

    # Append the auto entry at the end of model_list (before the next top-level key)
    awk -v routing="$ROUTING_RULES" -v default_model="$AUTO_DEFAULT" '
        /^model_list:/ { in_models=1 }
        in_models && (/^[a-z]/ || /^#/) && !/^model_list:/ {
            printf "\n  - model_name: auto\n"
            printf "    litellm_params:\n"
            printf "      model: \"auto_router/auto_router_1\"\n"
            printf "      auto_router_config_path: \"%s\"\n", routing
            printf "      auto_router_default_model: \"%s\"\n", default_model
            printf "\n"
            in_models=0
        }
        { print }
    ' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
    ok "Added auto-router entry to proxy_config.yaml (default: $AUTO_DEFAULT)"
fi

# ---------- 5d. Register litellm provider with OpenClaw ---------------------

OPENCLAW_CONFIG="$HOME/.openclaw/openclaw.json"

if [ -f "$OPENCLAW_CONFIG" ]; then
    info "Registering litellm provider in OpenClaw config..."
    python << 'PYEOF'
import json, os

config_path = os.path.expanduser("~/.openclaw/openclaw.json")
with open(config_path) as f:
    config = json.load(f)

# Ensure models.providers path exists
config.setdefault("models", {})
config["models"].setdefault("providers", {})

config["models"]["providers"]["litellm"] = {
    "baseUrl": "http://127.0.0.1:4000/v1",
    "apiKey": "sk-1234",
    "api": "openai-completions",
    "models": [
        {
            "id": "auto",
            "name": "LiteLLM Auto",
            "reasoning": False,
            "input": ["text"],
            "cost": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0
            },
            "contextWindow": 128000,
            "maxTokens": 8192
        }
    ]
}

# Set litellm/auto as the primary model
config.setdefault("agents", {})
config["agents"].setdefault("defaults", {})
old_primary = config["agents"]["defaults"].get("model", {}).get("primary")
config["agents"]["defaults"]["model"] = {
    "primary": "litellm/auto",
}
if old_primary and old_primary != "litellm/auto":
    config["agents"]["defaults"]["model"]["fallbacks"] = [old_primary]

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
PYEOF
    ok "Added litellm provider with auto model to $OPENCLAW_CONFIG"
else
    info "OpenClaw not found (~/.openclaw/openclaw.json missing) — skipping provider registration"
fi

# ---------- 6. Start proxy ---------------------------------------------------

echo ""
echo -e "${BOLD}=== Starting LiteLLM Proxy ===${NC}"
echo ""
echo -e "  Config : ${CYAN}$CONFIG_FILE${NC}"
echo -e "  Port   : ${CYAN}4000${NC}"
echo -e "  Key    : ${CYAN}sk-1234${NC}  (set in config litellm_settings.master_key)"
echo ""
echo -e "${BOLD}Test it:${NC}"
echo ""
echo "  # Health check"
echo "  curl http://localhost:4000/health"
echo ""
echo "  # Chat completion via the auto-router"
echo '  curl http://localhost:4000/v1/chat/completions \'
echo '    -H "Content-Type: application/json" \'
echo '    -H "Authorization: Bearer sk-1234" \'
echo '    -d '"'"'{"model":"auto","messages":[{"role":"user","content":"Hello!"}]}'"'"
echo ""
echo -e "${GREEN}Starting...${NC}"
echo ""

# Load env vars
set -a
# shellcheck disable=SC1091
source "$ENV_FILE"
set +a

exec litellm --config "$CONFIG_FILE" --port 4000

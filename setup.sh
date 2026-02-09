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

if [ -f "$ENV_FILE" ]; then
    ok ".env already exists — skipping API key prompts"
else
    echo ""
    echo -e "${BOLD}API Key Setup${NC}"
    echo "The proxy config includes models from several providers."
    echo "Enter the keys you have; press Enter to skip any you don't need."
    echo ""

    read -rp "  Google API Key    (for Gemini — used by auto-router) : " GOOGLE_KEY
    read -rp "  OpenAI API Key    (for GPT models)                   : " OPENAI_KEY
    read -rp "  Anthropic API Key (for Claude via API)               : " ANTHROPIC_KEY
    echo ""

    if [ -z "$GOOGLE_KEY" ] && [ -z "$OPENAI_KEY" ] && [ -z "$ANTHROPIC_KEY" ]; then
        warn "No API keys provided. The proxy will start but model calls will fail"
        warn "until you add keys to $ENV_FILE"
    fi

    cat > "$ENV_FILE" <<EOF
# LiteLLM API Keys — edit as needed
OPENAI_API_KEY=${OPENAI_KEY:-}
GOOGLE_API_KEY=${GOOGLE_KEY:-}
ANTHROPIC_API_KEY=${ANTHROPIC_KEY:-}
EOF

    ok "Created .env with your API keys"
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

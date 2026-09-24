#!/usr/bin/env bash
set -euo pipefail

ulimit -c 0
export HISTIGNORE="*openssl*:*sk-*:*PASSWORD*:*MASTER_KEY*"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECRETS_DIR="${SCRIPT_DIR}/secrets"
FORCE_FIREWALL=false
ENABLE_PROXY=false
INSTALL_SYSTEMD=false
TARGET_PORT="4000"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $(date +'%Y-%m-%dT%H:%M:%S%z') - $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $(date +'%Y-%m-%dT%H:%M:%S%z') - $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date +'%Y-%m-%dT%H:%M:%S%z') - $1" >&2; }

cleanup_on_error() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Deployment failed unexpectedly with exit code $exit_code."
    fi
}
trap cleanup_on_error EXIT

cd "${SCRIPT_DIR}"

for arg in "$@"; do
  case $arg in
    --force-open-firewall)
      FORCE_FIREWALL=true
      shift
      ;;
    --enable-egress-proxy)
      ENABLE_PROXY=true
      shift
      ;;
    --install-systemd)
      INSTALL_SYSTEMD=true
      shift
      ;;
    *)
      ;;
  esac
done

log_info "Starting Production LiteLLM Infrastructure Deployment..."

if [ "$(id -u)" -ne 0 ]; then
    log_error "Root privileges are required to setup secrets, systemd, and UFW rules."
    exit 1
fi

REQUIRED_TOOLS=("docker" "openssl" "envsubst")
for tool in "${REQUIRED_TOOLS[@]}"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        log_error "Required dependency '$tool' is missing."
        exit 1
    fi
done

if ! docker compose version >/dev/null 2>&1; then
    log_error "Docker Compose V2 is missing."
    exit 1
fi

# Firewall Check
if command -v ufw >/dev/null 2>&1; then
    UFW_STATUS=$(ufw status | grep -i "status: active" || true)
    if [ -n "${UFW_STATUS}" ]; then
        PORT_ALLOWED=$(ufw status | grep -E "${TARGET_PORT}(/tcp)?.*ALLOW" || true)
        if [ -z "${PORT_ALLOWED}" ]; then
            if [ "${FORCE_FIREWALL}" = true ]; then
                log_info "Force flag enabled. Granting rule for port ${TARGET_PORT}/tcp..."
                ufw allow "${TARGET_PORT}/tcp" comment 'LiteLLM Proxy'
            else
                log_error "Port ${TARGET_PORT} is blocked by UFW firewall."
                log_error "Re-run with '--force-open-firewall' or run: ufw allow ${TARGET_PORT}/tcp"
                exit 1
            fi
        fi
    fi
fi

# Provision Secrets
mkdir -p -m 700 "${SECRETS_DIR}"

generate_secret_file() {
    local file_path="$1"
    local secret_content="$2"
    if [ ! -f "${file_path}" ]; then
        printf "%s" "${secret_content}" > "${file_path}"
        chmod 600 "${file_path}"
        chown 101:101 "${file_path}" 2>/dev/null || true
    fi
}

DB_PASS_RAW="$(openssl rand -hex 20)"
generate_secret_file "${SECRETS_DIR}/db_password.txt" "${DB_PASS_RAW}"
DB_PASS_STORED="$(cat "${SECRETS_DIR}/db_password.txt")"
DB_CONNECTION_URL="postgresql://llmproxy:${DB_PASS_STORED}@db:5432/litellm"

generate_secret_file "${SECRETS_DIR}/db_url.txt" "${DB_CONNECTION_URL}"
generate_secret_file "${SECRETS_DIR}/master_key.txt" "sk-$(openssl rand -hex 16)"
generate_secret_file "${SECRETS_DIR}/salt_key.txt" "sk-$(openssl rand -hex 16)"
unset DB_PASS_RAW DB_PASS_STORED DB_CONNECTION_URL

# Environment File
if [ ! -f ".env" ]; then
    cat > .env <<EOF
STORE_MODEL_IN_DB=True
LITELLM_LOG=ERROR
EOF
    chmod 644 .env
fi

# Template Rendering
if [ ! -f "compose.yaml.template" ]; then
    log_error "Template 'compose.yaml.template' missing."
    exit 1
fi

if [ "${ENABLE_PROXY}" = true ]; then
    log_info "Mode: Proxy-Filtered Egress via Squid"
    
    PROXY_ENV_MAPPING=$(cat << 'EOF'
HTTP_PROXY: "http://squid:3128"
      HTTPS_PROXY: "http://squid:3128"
      http_proxy: "http://squid:3128"
      https_proxy: "http://squid:3128"
      NO_PROXY: "localhost,127.0.0.1,db,litellm,litellm_db,squid,litellm_squid"
      no_proxy: "localhost,127.0.0.1,db,litellm,litellm_db,squid,litellm_squid"
EOF
)

    LITELLM_SQUID_DEPENDENCY=$(cat << 'EOF'
squid:
        condition: service_healthy
EOF
)

    SQUID_SERVICE_BLOCK=$(cat << 'EOF'
squid:
    image: sameersbn/squid:3.5.27-2
    restart: unless-stopped
    container_name: litellm_squid
    user: root
    entrypoint:
      - /bin/sh
      - -c
      - |
        mkdir -p /etc/squid /var/spool/squid /var/log/squid
        cat << 'SQUIDCONF' > /etc/squid/squid.conf
        acl SSL_ports port 443
        acl Safe_ports port 80
        acl Safe_ports port 443
        acl CONNECT method CONNECT
        http_access deny !Safe_ports
        http_access deny CONNECT !SSL_ports
        http_access allow all
        http_port 3128
        SQUIDCONF
        chown -R proxy:proxy /var/spool/squid /var/log/squid /etc/squid
        exec /sbin/entrypoint.sh
    networks:
      public-egress:
        aliases:
          - squid
    tmpfs:
      - /var/spool/squid:rw,noexec,nosuid,nodev,size=64m
      - /var/log/squid:rw,noexec,nosuid,nodev,size=16m
    healthcheck:
      test: ["CMD", "bash", "-c", "cat < /dev/null > /dev/tcp/127.0.0.1/3128"]
      interval: 5s
      timeout: 3s
      retries: 5
      start_period: 5s
EOF
)

    export PROXY_ENV_MAPPING
    export LITELLM_SQUID_DEPENDENCY
    export SQUID_SERVICE_BLOCK
else
    log_info "Mode: Option 1 (Direct Egress)"
    export PROXY_ENV_MAPPING="# egress proxy disabled"
    export LITELLM_SQUID_DEPENDENCY="# no squid dependency"
    export SQUID_SERVICE_BLOCK="# no squid service"
fi

# Inject variables into template
envsubst '${PROXY_ENV_MAPPING} ${LITELLM_SQUID_DEPENDENCY} ${SQUID_SERVICE_BLOCK}' < compose.yaml.template > compose.yaml

# Systemd Integration
SERVICE_FILE="/etc/systemd/system/litellm.service"
DOCKER_BIN="$(command -v docker)"

if [ "${INSTALL_SYSTEMD}" = true ]; then
    log_info "Installing and enabling Systemd service unit..."
    cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=LiteLLM Proxy Stack
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${SCRIPT_DIR}
ExecStart=${DOCKER_BIN} compose up -d
ExecStop=${DOCKER_BIN} compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable litellm.service --quiet
else
    log_info "Skipping Systemd service installation (flag --install-systemd not set)."
fi

log_info "Validating compose configuration..."
docker compose config --quiet

log_info "Downloading container images in advance..."
docker compose pull

log_info "Starting containers in detached mode..."
if [ "${INSTALL_SYSTEMD}" = true ]; then
    systemctl restart litellm.service
else
    docker compose up -d
fi

# Calculate expected service count based on target setup
EXPECTED_SERVICES=$(docker compose config --services | wc -l)

# Maximum allowed wait time for all services to report a healthy status
MAX_WAIT_SECONDS=180
WAIT_INTERVAL=5
ELAPSED=0
BAR_WIDTH=30

# Helper function to render the progress bar
draw_progress_bar() {
    local elapsed="$1"
    local max="$2"
    local healthy_count="$3"
    local expected="$4"
    
    local percentage=$(( elapsed * 100 / max ))
    if [ "$percentage" -gt 100 ]; then percentage=100; fi
    
    local filled=$(( percentage * BAR_WIDTH / 100 ))
    local empty=$(( BAR_WIDTH - filled ))
    
    # Build visual bar strings
    local bar_filled=$(printf "%${filled}s" "" | tr ' ' '#')
    local bar_empty=$(printf "%${empty}s" "" | tr ' ' '-')
    
    # Print bar over the current line (\r)
    printf "\r[INFO] Waiting: [%s%s] %3d%% (%ds/%ds) - Healthy services: %d/%d" \
        "${bar_filled}" "${bar_empty}" "${percentage}" "${elapsed}" "${max}" "${healthy_count}" "${expected}"
}

log_info "Waiting for all ${EXPECTED_SERVICES} services to pass healthchecks..."

until [ "$(docker compose ps --format json | grep -c '"Health":"healthy"' || true)" -eq "${EXPECTED_SERVICES}" ]; do
    CURRENT_HEALTHY=$(docker compose ps --format json | grep -c '"Health":"healthy"' || true)
    
    # Render progress bar
    draw_progress_bar "${ELAPSED}" "${MAX_WAIT_SECONDS}" "${CURRENT_HEALTHY}" "${EXPECTED_SERVICES}"

    # Abort if the total elapsed time exceeds the global timeout threshold
    if [ "$ELAPSED" -ge "$MAX_WAIT_SECONDS" ]; then
        echo ""
        log_error "Timeout: Services failed to become healthy within ${MAX_WAIT_SECONDS} seconds."
        docker compose ps
        exit 1
    fi

    # Fail fast if Docker explicitly marks any container as unhealthy after its start_period
    if docker compose ps --format json | grep -q '"Health":"unhealthy"'; then
        echo ""
        log_error "One or more containers failed their healthcheck."
        docker compose ps
        exit 1
    fi

    sleep "$WAIT_INTERVAL"
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

# Print completed state (100%) and move to a new line
draw_progress_bar "${ELAPSED}" "${MAX_WAIT_SECONDS}" "${EXPECTED_SERVICES}" "${EXPECTED_SERVICES}"
echo ""
log_info "All services are healthy!"

# Retrieve Master Key for display
LITELLM_MASTER_KEY="$(cat "${SECRETS_DIR}/master_key.txt")"

echo -e "\n${GREEN}=====================================================${NC}"
echo -e "${BOLD}${CYAN}          LITELLM SERVICE ACCESS DETAILS            ${NC}"
echo -e "${GREEN}=====================================================${NC}"
echo -e "  ${BOLD}URL:${NC}        http://127.0.0.1:${TARGET_PORT}/ui/login/"
echo -e "  ${BOLD}User:${NC}       admin (or Bearer Token / UI Auth)"
echo -e "  ${BOLD}Password:${NC}   ${YELLOW}${LITELLM_MASTER_KEY}${NC}"
echo -e "${GREEN}=====================================================${NC}\n"
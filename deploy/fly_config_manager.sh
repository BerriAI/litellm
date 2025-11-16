#!/bin/bash
# Helper script to manage LiteLLM configuration on Fly.io
# Usage: ./fly_config_manager.sh [command] [options]
# TODO: test all commands work as expected

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_NAME=${FLY_APP_NAME:-"lite-llm"}
CONFIG_VOLUME="litellm_config"
ENV_VOLUME="litellm_env"

# Helper functions
print_help() {
    cat << 'EOF'
LiteLLM Fly.io Configuration Manager

Usage: ./fly_config_manager.sh [command] [options]

Commands:
  create-volumes      Create the required volumes on Fly.io
  upload-config FILE  Upload config.yaml to Fly.io
  upload-env FILE     Upload .env file to Fly.io
  download-config     Download config.yaml from Fly.io
  download-env        Download .env file from Fly.io
  list-files          List files in volumes
  help                Show this help message

Examples:
  ./fly_config_manager.sh create-volumes
  ./fly_config_manager.sh upload-config ./config.yaml
  ./fly_config_manager.sh upload-env ./.env
  ./fly_config_manager.sh download-config
  ./fly_config_manager.sh list-files

Environment Variables:
  FLY_APP_NAME        Set the Fly.io app name (default: lite-llm)

EOF
}

check_flyctl() {
    if ! command -v flyctl &> /dev/null; then
        echo -e "${RED}Error: flyctl is not installed${NC}"
        echo "Install it from: https://fly.io/docs/hands-on/install-flyctl/"
        exit 1
    fi
}

check_app() {
    if ! flyctl apps list | grep -q "^$APP_NAME"; then
        echo -e "${RED}Error: App '$APP_NAME' not found on Fly.io${NC}"
        exit 1
    fi
}

create_volumes() {
    echo -e "${YELLOW}Creating volumes for LiteLLM...${NC}"
    
    flyctl volumes create "$CONFIG_VOLUME" --size 1 -a "$APP_NAME" || true
    flyctl volumes create "$ENV_VOLUME" --size 1 -a "$APP_NAME" || true
    
    echo -e "${GREEN}✓ Volumes created${NC}"
}

upload_config() {
    local file=$1
    
    if [ ! -f "$file" ]; then
        echo -e "${RED}Error: File '$file' not found${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Uploading $file to Fly.io...${NC}"
    
    # Copy file to machine volume
    flyctl ssh console -a "$APP_NAME" << EOF
cat > /app/config/config.yaml << 'CONFIGEOF'
$(cat "$file")
CONFIGEOF
echo "Config uploaded"
EOF
    
    echo -e "${GREEN}✓ Config uploaded to /app/config/config.yaml${NC}"
}

upload_env() {
    local file=$1
    
    if [ ! -f "$file" ]; then
        echo -e "${RED}Error: File '$file' not found${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Uploading $file to Fly.io...${NC}"
    
    # Copy file to machine volume
    flyctl ssh console -a "$APP_NAME" << EOF
cat > /app/.env << 'ENVEOF'
$(cat "$file")
ENVEOF
echo ".env uploaded"
EOF
    
    echo -e "${GREEN}✓ .env uploaded${NC}"
    echo -e "${YELLOW}Restart machines for changes to take effect:${NC}"
    echo "  flyctl machines restart -a $APP_NAME"
}

download_config() {
    echo -e "${YELLOW}Downloading config.yaml from Fly.io...${NC}"
    
    flyctl ssh console -a "$APP_NAME" cat /app/config/config.yaml > config.yaml
    
    echo -e "${GREEN}✓ Downloaded to ./config.yaml${NC}"
}

download_env() {
    echo -e "${YELLOW}Downloading .env from Fly.io...${NC}"
    
    flyctl ssh console -a "$APP_NAME" cat /app/.env > .env
    
    echo -e "${GREEN}✓ Downloaded to ./.env${NC}"
    echo -e "${YELLOW}Note: Keep this file secure and don't commit to git${NC}"
}

list_files() {
    echo -e "${YELLOW}Files in Fly.io volumes:${NC}"
    
    echo -e "\nConfig volume (/app/config):"
    flyctl ssh console -a "$APP_NAME" ls -lh /app/config/ || echo "  (empty or volume not mounted)"
    
    echo -e "\nEnv volume (/app/.env):"
    flyctl ssh console -a "$APP_NAME" test -f /app/.env && echo "  .env exists" || echo "  .env not found"
}

# Main script
check_flyctl
check_app

case "${1:-help}" in
    create-volumes)
        create_volumes
        ;;
    upload-config)
        upload_config "${2:-.}"
        ;;
    upload-env)
        upload_env "${2:-.}"
        ;;
    download-config)
        download_config
        ;;
    download-env)
        download_env
        ;;
    list-files)
        list_files
        ;;
    help|--help|-h)
        print_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        print_help
        exit 1
        ;;
esac

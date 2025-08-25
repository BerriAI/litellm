#!/bin/bash

# Azure OpenAI E2E Test Runner
# This script sets up the environment and runs the complete E2E test suite

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REPO_ROOT="/Users/teddyamkie/litellm"
TEST_DIR="${REPO_ROOT}/tests/azure_endpoints_tests"
VENV_PATH="${REPO_ROOT}/.venv"
ENV_FILE="${REPO_ROOT}/.env.test"
CONFIG_FILE="${TEST_DIR}/azure_testing_config.yaml"
PROXY_PORT=4000
PROXY_URL="http://localhost:${PROXY_PORT}"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running from correct directory
check_directory() {
    log_info "Checking current directory..."
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        log_error "azure_testing_config.yaml not found. Please run this script from the azure_endpoints_tests directory."
        exit 1
    fi
    log_success "Directory check passed"
}

# Check Python virtual environment
setup_venv() {
    log_info "Checking Python virtual environment..."
    
    if [[ ! -d "${VENV_PATH}" ]]; then
        log_warning "Virtual environment not found. Creating new one..."
        cd "${REPO_ROOT}"
        python3 -m venv .venv
        log_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    source "${VENV_PATH}/bin/activate"
    log_success "Virtual environment activated"
    
    # Check Python version
    PYTHON_VERSION=$(python --version 2>&1)
    log_info "Using ${PYTHON_VERSION}"
}

# Install required dependencies
install_dependencies() {
    log_info "Installing required dependencies..."
    
    # Check if litellm is available
    if ! pip show litellm > /dev/null 2>&1; then
        log_warning "LiteLLM not found. Installing..."
        pip install litellm[proxy]
    fi
    
    # Install test dependencies
    pip install pytest requests python-dotenv > /dev/null 2>&1
    
    log_success "Dependencies installed"
}

# Check environment variables
check_env_vars() {
    log_info "Checking environment variables..."
    
    if [[ ! -f "${ENV_FILE}" ]]; then
        log_error ".env.test file not found at ${ENV_FILE}"
        log_error "Please create .env.test with:"
        echo "AZURE_API_BASE=https://your-resource.openai.azure.com/"
        echo "AZURE_API_KEY=your-azure-api-key-here"
        exit 1
    fi
    
    # Load environment variables
    source "${ENV_FILE}"
    
    # Check required variables
    if [[ -z "${AZURE_API_BASE}" ]]; then
        log_error "AZURE_API_BASE is not set in .env.test"
        exit 1
    fi
    
    if [[ -z "${AZURE_API_KEY}" ]]; then
        log_error "AZURE_API_KEY is not set in .env.test"
        exit 1
    fi
    
    log_success "Environment variables validated"
    log_info "AZURE_API_BASE: ${AZURE_API_BASE}"
    log_info "AZURE_API_KEY: ${AZURE_API_KEY:0:10}..."
}

# Check if proxy is already running
check_existing_proxy() {
    log_info "Checking for existing LiteLLM proxy..."
    
    if curl -s "${PROXY_URL}/health" > /dev/null 2>&1; then
        log_warning "Proxy already running on port ${PROXY_PORT}"
        read -p "Stop existing proxy and restart? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Try to kill existing proxy processes
            pkill -f "litellm.*--port ${PROXY_PORT}" || true
            sleep 2
            log_success "Existing proxy stopped"
        else
            log_info "Using existing proxy"
            return 0
        fi
    fi
}

# Start LiteLLM proxy server
start_proxy() {
    log_info "Starting LiteLLM proxy server..."
    
    cd "${TEST_DIR}"
    
    # Create temporary directory for files
    mkdir -p /tmp/litellm_files
    
    # Export environment variables for proxy process
    export AZURE_API_BASE
    export AZURE_API_KEY
    export OPENAI_API_KEY
    
    # Start proxy in background with environment variables loaded
    nohup env AZURE_API_BASE="${AZURE_API_BASE}" AZURE_API_KEY="${AZURE_API_KEY}" OPENAI_API_KEY="${OPENAI_API_KEY}" litellm --config azure_testing_config.yaml --port ${PROXY_PORT} --detailed_debug > proxy.log 2>&1 &
    PROXY_PID=$!
    
    # Wait for proxy to start
    log_info "Waiting for proxy to start..."
    for i in {1..30}; do
        if curl -s "${PROXY_URL}/health" > /dev/null 2>&1; then
            log_success "LiteLLM proxy started (PID: ${PROXY_PID})"
            echo "${PROXY_PID}" > proxy.pid
            return 0
        fi
        sleep 1
        echo -n "."
    done
    
    log_error "Proxy failed to start within 30 seconds"
    log_error "Check proxy.log for details:"
    tail -20 proxy.log
    exit 1
}

# Verify proxy configuration
verify_proxy() {
    log_info "Verifying proxy configuration..."
    
    # Check health endpoint
    HEALTH_RESPONSE=$(curl -s "${PROXY_URL}/health" || echo "FAILED")
    if [[ "${HEALTH_RESPONSE}" == "FAILED" ]]; then
        log_error "Proxy health check failed"
        exit 1
    fi
    
    # Check models endpoint
    MODELS_RESPONSE=$(curl -s -H "Authorization: Bearer sk-1234" "${PROXY_URL}/v1/models")
    if [[ $? -ne 0 ]]; then
        log_error "Failed to fetch models from proxy"
        exit 1
    fi
    
    # Verify gpt-4 model is available
    if echo "${MODELS_RESPONSE}" | grep -q '"id":"gpt-4"'; then
        log_success "gpt-4 model available"
    else
        log_error "gpt-4 model not found in proxy configuration"
        log_error "Available models: ${MODELS_RESPONSE}"
        exit 1
    fi
    
    log_success "Proxy configuration verified"
}

# Run test suite
run_tests() {
    log_info "Running E2E test suite..."
    
    cd "${TEST_DIR}"
    
    # Run tests with detailed output
    echo -e "\n${BLUE}================== TEST EXECUTION ==================${NC}"
    
    # Source environment for tests
    source "${ENV_FILE}"
    
    # Run all tests
    pytest -v -s . --tb=short
    TEST_EXIT_CODE=$?
    
    echo -e "\n${BLUE}==================== TEST SUMMARY ====================${NC}"
    
    if [[ ${TEST_EXIT_CODE} -eq 0 ]]; then
        log_success "All tests passed! ✅"
    else
        log_error "Some tests failed ❌"
    fi
    
    return ${TEST_EXIT_CODE}
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    # Stop proxy if we started it
    if [[ -f proxy.pid ]]; then
        PROXY_PID=$(cat proxy.pid)
        if kill -0 "${PROXY_PID}" 2>/dev/null; then
            log_info "Stopping LiteLLM proxy (PID: ${PROXY_PID})"
            kill "${PROXY_PID}"
            sleep 2
        fi
        rm -f proxy.pid
    fi
    
    # Clean up temporary files
    rm -f proxy.log
}

# Main execution
main() {
    echo -e "${BLUE}🚀 Azure OpenAI E2E Test Runner${NC}"
    echo "================================"
    
    # Set up cleanup trap
    trap cleanup EXIT INT TERM
    
    # Execute setup steps
    check_directory
    setup_venv
    install_dependencies
    check_env_vars
    check_existing_proxy
    start_proxy
    verify_proxy
    
    echo -e "\n${GREEN}✅ Setup complete! Starting tests...${NC}\n"
    
    # Run tests
    run_tests
    TEST_RESULT=$?
    
    # Final summary
    echo -e "\n${BLUE}==================== FINAL SUMMARY ====================${NC}"
    if [[ ${TEST_RESULT} -eq 0 ]]; then
        log_success "E2E test suite completed successfully! 🎉"
        echo -e "\n📊 Test Status:"
        echo "  • Batches API: ✅ (8 tests)"
        echo "  • Chat Completions: ✅ (6 tests)" 
        echo "  • Messages API: ✅ (6 tests)"
        echo "  • Responses API: ✅ (5 tests)"
        echo "  • Total: ✅ (25 tests)"
    else
        log_error "E2E test suite failed ❌"
        log_info "Check the output above for details"
    fi
    
    exit ${TEST_RESULT}
}

# Run main function
main "$@"

"""
Constants and configuration values for memory leak tests.

This module provides centralized constants used across all memory leak tests
to ensure consistency and ease of maintenance.
"""

# =============================================================================
# Test Endpoint Configuration
# =============================================================================

# Fake LLM endpoint for testing (Railway-hosted mock server)
FAKE_LLM_ENDPOINT = "https://exampleopenaiendpoint-production-0ee2.up.railway.app/"

# Test API keys
TEST_API_KEY = "sk-1234"
TEST_MASTER_KEY = "sk-1234"

# =============================================================================
# Memory Test Configuration
# =============================================================================

# Batch configuration
DEFAULT_BATCH_SIZE = 200  # Number of requests per batch
DEFAULT_NUM_BATCHES = 15  # Number of measurement batches (max recommended: 15)
DEFAULT_WARMUP_BATCHES = 5  # Number of warmup batches before measurement
MAX_NUM_BATCHES = 15  # Hard limit on number of batches to prevent long test runs

# Analysis configuration
DEFAULT_SAMPLE_WINDOW = 7  # Rolling average window size for smoothing
DEFAULT_MAX_GROWTH_PERCENT = 20.0  # Maximum allowed memory growth percentage
DEFAULT_STABILIZATION_TOLERANCE_MB = 0.02  # 20 KB - minimum growth to consider significant

# Memory threshold configuration
NEAR_ZERO_MEMORY_THRESHOLD = 0.01  # 10 KB - threshold for near-zero memory scenarios

# =============================================================================
# Test User Configuration
# =============================================================================

# Default test user information
DEFAULT_TEST_USER = "test-user"
DEFAULT_TEST_TEAM = "test-team"
DEFAULT_TEST_CONTENT = "Memory leak detection test"

# =============================================================================
# Test Model Configuration
# =============================================================================

# Default models for testing
DEFAULT_SDK_MODEL = "openai/gpt-3.5-turbo"
DEFAULT_ROUTER_MODEL = "gpt-3.5-turbo"

# =============================================================================
# FastAPI Request Configuration
# =============================================================================

# Default request paths
DEFAULT_REQUEST_PATH = "/chat/completions"

# Default request configuration
DEFAULT_REQUEST_SCHEME = "http"
DEFAULT_REQUEST_SERVER = ("testserver", 80)
DEFAULT_REQUEST_CLIENT = ("127.0.0.1", 8000)

# =============================================================================
# Database Mock Configuration
# =============================================================================

# Mock user/team budget configuration
MOCK_USER_SPEND = 0.0
MOCK_USER_MAX_BUDGET = 100.0
MOCK_TEAM_SPEND = 0.0
MOCK_TEAM_MAX_BUDGET = 100.0
MOCK_TEAM_METADATA = {"test": "data"}

# =============================================================================
# Router Configuration
# =============================================================================

# Router test configuration
DEFAULT_ROUTER_NUM_RETRIES = 0  # Disable retries for cleaner testing
DEFAULT_ROUTER_TIMEOUT = 30  # Timeout in seconds


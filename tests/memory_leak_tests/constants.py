"""
Constants and configuration values for memory leak tests.

This module provides centralized constants used across all memory leak tests
to ensure consistency and ease of maintenance.

Constants are organized into sections:
1. Test Endpoint Configuration - API endpoints and keys for testing
2. Memory Test Execution Configuration - Batch sizes and test run parameters
3. Memory Analysis Configuration - Statistical analysis parameters
4. Memory Leak Detection Thresholds - Criteria for detecting memory leaks
5. Test Data Configuration - Default test users, models, and content
6. FastAPI Request Configuration - HTTP request parameters for auth testing
7. Database Mock Configuration - Mock data for database operations
8. Router Configuration - Router-specific test parameters
"""

# =============================================================================
# Test Endpoint Configuration
# =============================================================================

# Mock LLM endpoint for testing (Railway-hosted mock server that returns fake completions)
FAKE_LLM_ENDPOINT = "https://exampleopenaiendpoint-production-0ee2.up.railway.app/"

# Test API keys used for authentication in memory leak tests
TEST_API_KEY = "sk-1234"  # API key used in test requests

# =============================================================================
# Memory Test Execution Configuration
# =============================================================================
# Controls how memory leak tests are executed (batch sizes, number of iterations, etc.)

# Number of API requests to execute in each batch during testing
# Larger batches = more requests before measuring memory, smoother growth patterns
# Reduced from 200 to 100 for faster test feedback cycles
DEFAULT_BATCH_SIZE = 100

# Number of batches to run during the measurement phase
# More batches = longer test runtime but more confidence in leak detection
# Reduced from 15 to 10 for faster feedback while maintaining reliability
DEFAULT_NUM_BATCHES = 10

# Number of warmup batches to run before measurement begins
# Warmup allows caches and memory allocations to stabilize before measurement
# Reduced from 5 to 3 for faster test startup
DEFAULT_WARMUP_BATCHES = 3

# Hard limit on number of batches to prevent excessively long test runs
MAX_NUM_BATCHES = 15

# =============================================================================
# Memory Analysis Configuration
# =============================================================================
# Statistical parameters for analyzing memory measurements and detecting patterns

# Window size for calculating rolling average of memory measurements
# Smooths out noise in memory measurements for more accurate trend detection
# Reduced from 7 to 5 to match smaller batch count (10 batches)
DEFAULT_ROLLING_AVERAGE_WINDOW = 5

# Maximum allowed memory growth percentage for basic test pass/fail
# Used in get_memory_test_config() for general test configuration
# Tests exceeding this threshold may still pass if memory stabilizes
DEFAULT_TEST_MAX_GROWTH_PERCENT = 20.0

# Minimum memory growth (in MB) to consider significant for basic test pass/fail
# Growth smaller than this threshold is considered noise and ignored
DEFAULT_TEST_STABILIZATION_TOLERANCE_MB = 0.02  # 20 KB

# Number of initial and final samples to average when calculating overall growth
# Averaging reduces impact of outliers on growth measurements
DEFAULT_NUM_SAMPLES_FOR_GROWTH_ANALYSIS = 3

# Maximum coefficient of variation (CV) percentage for memory measurements
# CV = (std_dev / mean) * 100. Higher CV indicates noisy/unstable measurements
# Tests with CV > this threshold are skipped due to unreliable environment
DEFAULT_MAX_COEFFICIENT_VARIATION = 40.0

# =============================================================================
# Memory Leak Detection Thresholds
# =============================================================================
# Criteria used by detect_memory_leak() to determine if a memory leak exists

# Maximum allowed memory growth percentage before declaring a memory leak
# Higher threshold than DEFAULT_TEST_MAX_GROWTH_PERCENT for more tolerance
DEFAULT_LEAK_DETECTION_MAX_GROWTH_PERCENT = 25.0

# Minimum memory growth (in MB) to consider significant when detecting leaks
# Larger than DEFAULT_TEST_STABILIZATION_TOLERANCE_MB for coarser leak detection
DEFAULT_LEAK_DETECTION_STABILIZATION_TOLERANCE_MB = 0.05  # 50 KB

# Number of final samples to check for continuous upward growth pattern
# Used to detect leaks where memory keeps growing rather than stabilizing
DEFAULT_LEAK_DETECTION_TAIL_SAMPLES = 5

# Threshold for considering memory measurements as "near zero"
# Measurements below this are treated specially to avoid division by zero
NEAR_ZERO_MEMORY_THRESHOLD_MB = 0.01  # 10 KB

# =============================================================================
# Error-Induced Memory Leak Detection Configuration
# =============================================================================
# Parameters for detecting memory leaks specifically caused by error handling

# Minimum percent increase between batches to consider it a memory spike
# Used to identify sudden memory jumps that may be caused by errors
DEFAULT_ERROR_MEMORY_SPIKE_THRESHOLD_PERCENT = 50.0

# Maximum percent variation allowed for memory to be considered "stable"
# After an error spike, subsequent batches within this tolerance = stabilized
DEFAULT_ERROR_SPIKE_STABILIZATION_TOLERANCE_PERCENT = 5.0

# Minimum number of batches after a spike to verify memory has stabilized
# Need multiple stable batches to confirm leak pattern vs transient spike
DEFAULT_ERROR_SPIKE_MIN_STABLE_BATCHES = 2

# =============================================================================
# Test Data Configuration
# =============================================================================
# Default user identifiers, models, and content used in memory leak tests

# Default user ID used in test requests (appears in logs and tracking)
DEFAULT_TEST_USER = "test-user"

# Default team ID used in test requests (for team-based features)
DEFAULT_TEST_TEAM = "test-team"

# Default message content sent in test completion requests
# Simple message to minimize processing time and focus on memory patterns
DEFAULT_TEST_MESSAGE_CONTENT = "Memory leak detection test"

# Default model for SDK completion tests (litellm.completion, litellm.acompletion)
# Format: "provider/model" to test provider-specific implementations
DEFAULT_SDK_MODEL = "openai/gpt-3.5-turbo"

# Default model for Router completion tests
# Uses model name as defined in Router's model_list configuration
DEFAULT_ROUTER_MODEL = "gpt-3.5-turbo"

# =============================================================================
# FastAPI Request Configuration
# =============================================================================
# HTTP request parameters for testing FastAPI endpoints (authentication tests)

# API endpoint path for chat completions (standard OpenAI-compatible path)
DEFAULT_REQUEST_PATH = "/chat/completions"

# HTTP scheme for test requests (http for local testing)
DEFAULT_REQUEST_SCHEME = "http"

# Server address tuple (hostname, port) for FastAPI test requests
# "testserver" is FastAPI's test client default server name
DEFAULT_REQUEST_SERVER = ("testserver", 80)

# Client address tuple (IP, port) simulating the request origin
DEFAULT_REQUEST_CLIENT = ("127.0.0.1", 8000)

# =============================================================================
# Database Mock Configuration
# =============================================================================
# Mock data for database operations in authentication and budget tracking tests

# Mock current spend for test user (starts at 0 to avoid budget limits)
MOCK_USER_SPEND = 0.0

# Mock maximum budget for test user (high enough to not trigger limits)
MOCK_USER_MAX_BUDGET = 100.0

# Mock current spend for test team (starts at 0 to avoid budget limits)
MOCK_TEAM_SPEND = 0.0

# Mock maximum budget for test team (high enough to not trigger limits)
MOCK_TEAM_MAX_BUDGET = 100.0

# Mock metadata attached to test team (arbitrary test data)
MOCK_TEAM_METADATA = {"test": "data"}

# =============================================================================
# Router Configuration
# =============================================================================
# Configuration parameters specific to Router-based memory leak tests

# Number of retries for failed Router requests
# Set to 0 to avoid retry logic complicating memory measurements
DEFAULT_ROUTER_NUM_RETRIES = 0

# Timeout in seconds for Router completion requests
# Generous timeout to avoid false failures in CI environments
DEFAULT_ROUTER_TIMEOUT = 30


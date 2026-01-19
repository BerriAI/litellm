#!/usr/bin/env python3
"""
Reproducer for json_logs issue when enabled AFTER litellm is imported.

This simulates what happens when using litellm proxy with json_logs: true in config,
where the logging module is imported BEFORE json_logs is enabled.
"""

import sys

# DO NOT set JSON_LOGS environment variable - this simulates the proxy server startup
# where the env var is not set, and json_logs is enabled later via config

print("=" * 80)
print("Simulating litellm proxy startup with json_logs in config (not env var)")
print("=" * 80)
print()

# Import litellm (logging will be initialized with JSON_LOGS=False)
print("Step 1: Importing litellm (json_logs not yet enabled)...")
import litellm
from litellm._logging import verbose_logger, verbose_router_logger, verbose_proxy_logger

print(f"   json_logs status after import: {litellm.json_logs}")
print()

# Simulate some early logging before json_logs is enabled
print("Step 2: Some early logging (before json_logs enabled)...")
verbose_logger.info("Early INFO log before json_logs enabled")
verbose_logger.error("Early ERROR log before json_logs enabled")
print()

# Now enable json_logs like the proxy server does after reading config
print("Step 3: Enabling json_logs via config (like proxy_cli.py does)...")
litellm.json_logs = True
litellm._turn_on_json()
print(f"   json_logs status after _turn_on_json(): {litellm.json_logs}")
print()

# Test logging after json_logs is enabled
print("Step 4: Testing logs after json_logs enabled...")
print()

print("Test 1: Regular INFO log (should be JSON)")
verbose_logger.info("This is an INFO message after json_logs enabled")
print()

print("Test 2: Regular ERROR log (should be JSON)")
verbose_logger.error("This is an ERROR message after json_logs enabled")
print()

print("Test 3: Exception log with logger.exception() (should be JSON)")
try:
    raise ValueError("Test exception after json_logs enabled")
except Exception as e:
    verbose_logger.exception(f"Caught exception: {e}")
print()

print("Test 4: Simulating vertex_ai error")
try:
    import json
    json.loads("")  # Will raise JSONDecodeError
except Exception as e:
    verbose_logger.exception(
        f"Failed to load vertex credentials. Error: {str(e)}"
    )
print()

print("Test 5: Router error")
verbose_router_logger.error(
    "Could not identify azure model 'test-model'. Set azure 'base_model'..."
)
print()

# Test logging from other loggers that might not be configured
print("Test 6: Testing root logger (might not be configured)")
import logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
print()

print("Test 7: Testing a random third-party style logger")
try:
    raise RuntimeError("Third party error")
except Exception:
    logging.error("This is from root logger with exception", exc_info=True)
print()

print("=" * 80)
print("Analysis:")
print("- If all logs after Step 4 are single-line JSON: json_logs works correctly")
print("- If exceptions are multi-line plain text: BUG REPRODUCED")
print("- Check if root logger or third-party loggers are formatted correctly")
print("=" * 80)

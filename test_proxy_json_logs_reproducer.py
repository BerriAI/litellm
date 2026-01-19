#!/usr/bin/env python3
"""
Comprehensive reproducer for json_logs issue in proxy server context.

This simulates the EXACT proxy server startup sequence including:
1. Import litellm modules (without JSON_LOGS env var)
2. Load config with json_logs: true
3. Call _turn_on_json()
4. Trigger various types of errors that dmc reported
"""

import os
import sys

# Ensure JSON_LOGS is NOT set
if "JSON_LOGS" in os.environ:
    del os.environ["JSON_LOGS"]

print("="  * 80)
print("REPRODUCER FOR NON-JSON LOGS IN LITELLM PROXY")
print("=" * 80)
print()

# Step 1: Import litellm (simulates module loading at proxy startup)
print("Step 1: Importing litellm modules (json_logs not yet enabled)...")
print()

import litellm
from litellm._logging import verbose_logger, verbose_router_logger, verbose_proxy_logger

print(f"Initial json_logs status: {litellm.json_logs}")
print(f"verbose_logger handlers: {verbose_logger.handlers}")
if verbose_logger.handlers:
    print(f"verbose_logger formatter type: {type(verbose_logger.handlers[0].formatter)}")
print()

# Step 2: Simulate some early initialization logging
print("Step 2: Simulating early logging (before config is loaded)...")
verbose_logger.info("Early INFO message during proxy initialization")
verbose_logger.error("Early ERROR message during proxy initialization")
print()

# Step 3: Simulate config loading and enabling json_logs
print("Step 3: Loading config and enabling json_logs...")
print()

# This simulates what proxy_cli.py does at line 680-682
litellm.json_logs = True
litellm._turn_on_json()

print(f"After _turn_on_json():")
print(f"  json_logs status: {litellm.json_logs}")
print(f"  verbose_logger handlers: {verbose_logger.handlers}")
if verbose_logger.handlers:
    print(f"  verbose_logger formatter type: {type(verbose_logger.handlers[0].formatter)}")
    print(f"  verbose_logger propagate: {verbose_logger.propagate}")
print()

# Print the message that appears in dmc's logs
print("Using json logs. Setting log_config to None.")
print()

# Step 4: Simulate various errors that dmc reported
print("Step 4: Simulating errors that dmc reported...")
print()

# Test 1: Vertex AI credential error (like dmc's log)
print("Test 1: Vertex AI credential error")
try:
    import json as json_module
    json_module.loads("")  # Triggers JSONDecodeError
except Exception as e:
    verbose_logger.exception(
        f"Failed to load vertex credentials. Check to see if credentials containing partial/invalid information. Error: {str(e)}"
    )
print()

# Test 2: Router error (like dmc's log)
print("Test 2: Router error about Azure model")
verbose_router_logger.error(
    "Could not identify azure model 'text-embedding-3-small'. Set azure 'base_model' for accurate max tokens, cost tracking, etc.- https://docs.litellm.ai/docs/proxy/cost_tracking#spend-tracking-for-azure-openai-models"
)
print()

# Test 3: Database error (like dmc's log)
print("Test 3: Simulating database error")
try:
    raise Exception("Can't reach database server at `pglitellmstag-rw.hudson-trading.com`:`5432`\n\nPlease make sure your database server is running at `pglitellmstag-rw.hudson-trading.com`:`5432`.")
except Exception as e:
    verbose_proxy_logger.exception(
        "Failed to reset budget for endusers: " + str(e)
    )
print()

# Test 4: Nested exceptions
print("Test 4: Nested exceptions")
try:
    try:
        json_module.loads("")
    except Exception as inner:
        raise Exception("Unable to load vertex credentials from environment. Got=") from inner
except Exception as e:
    verbose_logger.exception(
        f"Failed to load vertex credentials. Check to see if credentials containing partial/invalid information. Error: {str(e)}"
    )
print()

# Test 5: Logging from a module-level logger (like health_check.py)
print("Test 5: Logging from a module logger (like health_check.py)")
import logging
module_logger = logging.getLogger("litellm.proxy.health_check")
try:
    raise ValueError("Health check failed for model")
except Exception as e:
    module_logger.exception(f"Health check error: {e}")
print()

# Test 6: Direct logging.error with exc_info
print("Test 6: Using logging.error with exc_info=True")
try:
    raise RuntimeError("Test error")
except Exception:
    logging.error("This is from root logger with exception", exc_info=True)
print()

# Test 7: Check if there are multiple handlers
print("Step 5: Checking for multiple handlers (could cause duplicates)...")
print(f"Root logger handlers: {logging.getLogger().handlers}")
print(f"verbose_logger handlers: {verbose_logger.handlers}")
print(f"verbose_router_logger handlers: {verbose_router_logger.handlers}")
print(f"verbose_proxy_logger handlers: {verbose_proxy_logger.handlers}")
print()

print("=" * 80)
print("EXPECTED BEHAVIOR:")
print("- All logs after 'Using json logs' should be single-line JSON")
print("- Exceptions should have stacktrace in JSON format")
print()
print("IF YOU SEE PLAIN TEXT LOGS LIKE:")
print("  '15:46:06 - LiteLLM:ERROR: file.py:123 - Error message'")
print("  'Traceback (most recent call last):'")
print("  '  ...'")
print("THEN THE BUG IS REPRODUCED!")
print("=" * 80)

#!/usr/bin/env python3
"""
Reproducer for json_logs issue with exceptions not being formatted as JSON.

This script demonstrates that when json_logs is enabled, regular log messages
are formatted as JSON, but exceptions/tracebacks are still printed as plain text.
"""

import os
import sys
import logging

# Set JSON_LOGS environment variable before importing litellm
os.environ["JSON_LOGS"] = "true"

# Import litellm modules
import litellm
from litellm._logging import verbose_logger, verbose_router_logger, verbose_proxy_logger

# Explicitly turn on json logs
litellm.json_logs = True
litellm._turn_on_json()

print("=" * 80)
print("Testing JSON Logs with litellm")
print("json_logs enabled:", litellm.json_logs)
print("=" * 80)
print()

# Test 1: Regular INFO log (should be JSON)
print("Test 1: Regular INFO log")
verbose_logger.info("This is a regular INFO message")
print()

# Test 2: Regular ERROR log without exception (should be JSON)
print("Test 2: Regular ERROR log without exception")
verbose_logger.error("This is a regular ERROR message without exception")
print()

# Test 3: ERROR log with exception using logger.error() (problematic case)
print("Test 3: ERROR log with logger.error() inside exception handler")
try:
    raise ValueError("This is a test exception")
except Exception as e:
    verbose_logger.error(f"Caught an exception: {e}")
print()

# Test 4: ERROR log with exception using logger.exception() (should include traceback)
print("Test 4: ERROR log with logger.exception() inside exception handler")
try:
    raise ValueError("This is another test exception")
except Exception as e:
    verbose_logger.exception(f"Caught an exception with .exception(): {e}")
print()

# Test 5: Simulate the vertex_ai error that dmc reported
print("Test 5: Simulating vertex_ai credential loading error")
try:
    # Simulate JSON parsing error like in vertex_llm_base.py
    import json
    json_obj = json.loads("")  # This will fail
except Exception as e:
    verbose_logger.exception(
        f"Failed to load vertex credentials. Check to see if credentials containing partial/invalid information. Error: {str(e)}"
    )
print()

# Test 6: Simulate router error that dmc reported
print("Test 6: Simulating router error")
verbose_router_logger.error(
    "Could not identify azure model 'text-embedding-3-small'. Set azure 'base_model' for accurate max tokens, cost tracking, etc."
)
print()

# Test 7: Multiple nested exceptions
print("Test 7: Multiple nested exceptions")
try:
    try:
        raise RuntimeError("Inner exception")
    except RuntimeError as inner_e:
        raise ValueError("Outer exception") from inner_e
except Exception as e:
    verbose_logger.exception(f"Nested exception occurred: {e}")
print()

print("=" * 80)
print("Tests completed. Check output above:")
print("- Regular logs should be single-line JSON")
print("- Exceptions should ALSO be single-line JSON with stacktrace field")
print("- If exceptions are multi-line plain text, the bug is reproduced")
print("=" * 80)

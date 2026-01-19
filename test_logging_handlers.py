#!/usr/bin/env python3
"""
Test to understand the logging handler hierarchy and potential issues.
"""

import logging
import sys

# Simulate the proxy startup WITHOUT JSON_LOGS env var
print("=" * 80)
print("Step 1: Check Python's default logging configuration")
print("=" * 80)

root = logging.getLogger()
print(f"Root logger: {root}")
print(f"Root logger level: {root.level} ({logging.getLevelName(root.level)})")
print(f"Root logger handlers: {root.handlers}")
print(f"Root logger propagate: {root.propagate}")
print()

# Now import litellm's logging
print("=" * 80)
print("Step 2: Import litellm (without JSON_LOGS env var)")
print("=" * 80)

import litellm
from litellm._logging import verbose_logger, verbose_router_logger, verbose_proxy_logger

print(f"litellm.json_logs: {litellm.json_logs}")
print()

print(f"verbose_logger: {verbose_logger}")
print(f"  - name: {verbose_logger.name}")
print(f"  - level: {verbose_logger.level} ({logging.getLevelName(verbose_logger.level)})")
print(f"  - handlers: {verbose_logger.handlers}")
print(f"  - propagate: {verbose_logger.propagate}")
print()

print(f"Root logger after litellm import:")
print(f"  - handlers: {root.handlers}")
print(f"  - propagate: {root.propagate}")
print()

# Create a child logger like health_check.py does
print("=" * 80)
print("Step 3: Create a child logger (like health_check.py)")
print("=" * 80)

health_logger = logging.getLogger("litellm.proxy.health_check")
print(f"health_logger: {health_logger}")
print(f"  - name: {health_logger.name}")
print(f"  - level: {health_logger.level} ({logging.getLevelName(health_logger.level)})")
print(f"  - handlers: {health_logger.handlers}")
print(f"  - propagate: {health_logger.propagate}")
print()

# Test logging BEFORE json_logs is enabled
print("=" * 80)
print("Step 4: Test logging BEFORE json_logs enabled")
print("=" * 80)

verbose_logger.error("Test error from verbose_logger BEFORE json_logs")
health_logger.error("Test error from health_logger BEFORE json_logs")
logging.error("Test error from root logger BEFORE json_logs")
print()

# Now enable json_logs
print("=" * 80)
print("Step 5: Enable json_logs (simulate config loading)")
print("=" * 80)

litellm.json_logs = True
litellm._turn_on_json()

print(f"litellm.json_logs: {litellm.json_logs}")
print()

print(f"verbose_logger after _turn_on_json():")
print(f"  - handlers: {verbose_logger.handlers}")
print(f"  - propagate: {verbose_logger.propagate}")
if verbose_logger.handlers:
    print(f"  - handler formatter: {verbose_logger.handlers[0].formatter}")
print()

print(f"Root logger after _turn_on_json():")
print(f"  - handlers: {root.handlers}")
print(f"  - propagate: {root.propagate}")
if root.handlers:
    print(f"  - handler formatter: {root.handlers[0].formatter}")
print()

print(f"health_logger after _turn_on_json():")
print(f"  - handlers: {health_logger.handlers}")
print(f"  - propagate: {health_logger.propagate}")
print()

# Test logging AFTER json_logs is enabled
print("=" * 80)
print("Step 6: Test logging AFTER json_logs enabled")
print("=" * 80)

print("\nTest from verbose_logger:")
verbose_logger.error("Test error from verbose_logger AFTER json_logs")

print("\nTest from health_logger (should propagate to root):")
health_logger.error("Test error from health_logger AFTER json_logs")

print("\nTest from root logger:")
logging.error("Test error from root logger AFTER json_logs")

print("\nTest exception from verbose_logger:")
try:
    raise ValueError("Test exception")
except Exception as e:
    verbose_logger.exception(f"Caught exception: {e}")

print("\nTest exception from health_logger:")
try:
    raise ValueError("Test exception from health logger")
except Exception as e:
    health_logger.exception(f"Caught exception: {e}")

print()
print("=" * 80)
print("Analysis:")
print("- Check if health_logger messages are formatted as JSON")
print("- Check if exceptions have stacktrace in JSON")
print("- If health_logger is NOT JSON, then propagation is broken")
print("=" * 80)

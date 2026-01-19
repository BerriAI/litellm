#!/usr/bin/env python3
"""
Reproducer that simulates EXACT proxy startup sequence.

This simulates what happens when:
1. Proxy starts without JSON_LOGS env var
2. Logging module is imported and initialized with default formatters
3. Config is loaded with json_logs: true
4. _turn_on_json() is called
5. Errors are logged

This should reproduce the bug if there's an issue with the initialization order.

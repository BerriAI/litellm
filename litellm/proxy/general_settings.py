"""
Lightweight shared storage for proxy general settings.

This module intentionally has no side effects so it can be safely imported
by other parts of the codebase (e.g., cold storage handlers) without causing
proxy_server to execute.

The proxy_server will mutate this dict in-place at runtime to reflect the
current configuration.
"""
from typing import Any, Dict

# Shared mutable dict for general settings
general_settings: Dict[str, Any] = {}

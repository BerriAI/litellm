"""
Opik payload builder namespace.

Public API:
    build_opik_payload - Main function to create Opik trace and span payloads
"""

from .api import build_opik_payload

__all__ = ["build_opik_payload"]

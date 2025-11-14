"""
Shared TLS context for realtime websocket clients.

Realtime handlers (OpenAI, Azure, custom providers, health checks) should all
reuse the same SSLContext to ensure consistent cipher / CA settings and to
avoid paying the cost of constructing a new context on every websocket dial.
"""

from __future__ import annotations

import ssl

__all__ = ["SHARED_REALTIME_SSL_CONTEXT"]


def _create_default_context() -> ssl.SSLContext:
    """
    Create the reusable SSL context for realtime websocket connections.

    The default settings already validate certificates and hostnames; keeping
    the helper lets us tweak things (custom CA bundle, client certs, etc.) in
    one place later without touching every caller.
    """

    context = ssl.create_default_context()
    return context


SHARED_REALTIME_SSL_CONTEXT: ssl.SSLContext = _create_default_context()


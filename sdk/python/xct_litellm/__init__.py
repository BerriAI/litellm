"""Python SDK for the xct-litellm capability provider.

Public surface — kept tiny on purpose:

    XctClient                 — single entry point
    XctError + subclasses     — typed errors
    PKCESession               — OAuth helper for server-side flows

See README.md for usage.
"""

from .client import XctClient
from .errors import (
    AuthError,
    CapabilityNotFoundError,
    RateLimitError,
    XctError,
)
from .pkce import PKCESession

__all__ = [
    "XctClient",
    "XctError",
    "AuthError",
    "RateLimitError",
    "CapabilityNotFoundError",
    "PKCESession",
]

__version__ = "0.1.0"

"""Shunt worker endpoints for LiteLLM Proxy."""

from litellm.proxy.shunt_endpoints.endpoints import router

__all__ = ["router"]  # mutable-ok: __all__ must be a list per Python convention

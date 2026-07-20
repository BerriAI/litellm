"""
Dashboard AI endpoints package.

Re-exports the router from endpoints module.
"""

from litellm.proxy.management_endpoints.dashboard_ai.endpoints import (  # noqa: F401
    router,
)

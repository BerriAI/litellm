"""
Policy endpoints package.

Re-exports everything from endpoints module so existing imports
like `from litellm.proxy.management_endpoints.policy_endpoints import router`
continue to work. Patch targets also resolve correctly since names
are imported directly into this namespace.
"""

from litellm.proxy.management_endpoints.policy_endpoints.endpoints import *  # noqa: F401, F403
from litellm.proxy.management_endpoints.policy_endpoints.endpoints import (
    router,
)

from litellm.proxy.managed_agents_endpoints.endpoints import router
from litellm.proxy.managed_agents_endpoints import (
    endpoints_agents,
)  # noqa: F401  registers /agents routes
from litellm.proxy.managed_agents_endpoints import (
    endpoints_passthrough,
)  # noqa: F401  registers passthrough routes
from litellm.proxy.managed_agents_endpoints import (
    endpoints_sessions,
)  # noqa: F401  registers /sessions routes

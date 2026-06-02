"""
Agent / Session / Run REST API surface — `/v2/agents`, `/v2/sessions`,
`/v2/sessions/{sid}/runs`.

The `/v1/agents` namespace is owned by the existing A2A registry
(``litellm/proxy/agent_endpoints/``). To avoid collision, all endpoints
in this module mount under ``/v2/``.

Exports the FastAPI routers consumed by ``proxy_server.py``:

  * ``agent_router``    — /v2/agents CRUD
  * ``session_router``  — /v2/sessions CRUD + followup + conversation
  * ``run_router``      — /v2/sessions/{sid}/runs (+ events SSE, cancel)
  * ``internal_router`` — /v2/sessions/{sid}/internal/* (daemon callbacks)
"""

from litellm.proxy.agent_session_endpoints.agent_endpoints import (
    router as agent_router,
)
from litellm.proxy.agent_session_endpoints.internal_endpoints import (
    router as internal_router,
)
from litellm.proxy.agent_session_endpoints.run_endpoints import (
    router as run_router,
)
from litellm.proxy.agent_session_endpoints.session_endpoints import (
    router as session_router,
)

__all__ = [
    "agent_router",
    "session_router",
    "run_router",
    "internal_router",
]

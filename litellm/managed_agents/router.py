"""Composite router for v2 managed agents.

Aggregates the endpoint sub-routers and exposes a single managed_agents_router
for inclusion in the main FastAPI app.
"""
from fastapi import APIRouter

from litellm.managed_agents.endpoints.agents import router as agents_router
from litellm.managed_agents.endpoints.events import router as events_router
from litellm.managed_agents.endpoints.messages import router as messages_router
from litellm.managed_agents.endpoints.sessions import router as sessions_router

managed_agents_router = APIRouter(tags=["managed_agents_v2"])

managed_agents_router.include_router(agents_router)
managed_agents_router.include_router(sessions_router)
managed_agents_router.include_router(messages_router)
managed_agents_router.include_router(events_router)

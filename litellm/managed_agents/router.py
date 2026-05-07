"""Composite router for v2 managed agents.

Aggregates the endpoint sub-routers and exposes a single managed_agents_router
for inclusion in the main FastAPI app.
"""
from fastapi import APIRouter

from litellm.managed_agents.endpoints.agents import router as agents_router

# TODO: enable when litellm/managed_agents/endpoints/sessions.py ships
# from litellm.managed_agents.endpoints.sessions import router as sessions_router
# TODO: enable when litellm/managed_agents/endpoints/messages.py ships
# from litellm.managed_agents.endpoints.messages import router as messages_router
# TODO: enable when litellm/managed_agents/endpoints/events.py ships
# from litellm.managed_agents.endpoints.events import router as events_router

managed_agents_router = APIRouter(tags=["managed_agents_v2"])

managed_agents_router.include_router(agents_router)
# TODO: enable when litellm/managed_agents/endpoints/sessions.py ships
# managed_agents_router.include_router(sessions_router)
# TODO: enable when litellm/managed_agents/endpoints/messages.py ships
# managed_agents_router.include_router(messages_router)
# TODO: enable when litellm/managed_agents/endpoints/events.py ships
# managed_agents_router.include_router(events_router)

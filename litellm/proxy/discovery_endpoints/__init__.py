from .agent_skills_endpoints import router as agent_skills_discovery_router
from .ui_discovery_endpoints import router as ui_discovery_endpoints_router

__all__ = ["agent_skills_discovery_router", "ui_discovery_endpoints_router"]

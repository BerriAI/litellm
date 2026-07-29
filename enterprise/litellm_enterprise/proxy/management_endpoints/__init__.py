from fastapi import APIRouter

from .internal_user_endpoints import router as internal_user_endpoints_router
from .project_endpoints import router as project_endpoints_router

management_endpoints_router = APIRouter()
management_endpoints_router.include_router(internal_user_endpoints_router)
management_endpoints_router.include_router(project_endpoints_router)

__all__ = ["management_endpoints_router"]

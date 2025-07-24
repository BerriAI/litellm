from fastapi import APIRouter
from fastapi.responses import Response
from litellm_enterprise.enterprise_callbacks.send_emails.endpoints import (
    router as email_events_router,
)

from .audit_logging_endpoints import router as audit_logging_router
from .guardrails.endpoints import router as guardrails_router
from .management_endpoints import management_endpoints_router
from .utils import _should_block_robots
from .vector_stores.endpoints import router as vector_stores_router

router = APIRouter()
router.include_router(vector_stores_router)
router.include_router(guardrails_router)
router.include_router(email_events_router)
router.include_router(audit_logging_router)
router.include_router(management_endpoints_router)


@router.get("/robots.txt")
async def get_robots():
    """
    Block all web crawlers from indexing the proxy server endpoints
    This is useful for ensuring that the API endpoints aren't indexed by search engines
    """
    if _should_block_robots():
        return Response(content="User-agent: *\nDisallow: /", media_type="text/plain")
    else:
        return Response(status_code=404)

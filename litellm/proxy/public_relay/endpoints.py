from fastapi import APIRouter

from litellm.proxy.public_relay.admin_endpoints import router as admin_router
from litellm.proxy.public_relay.auth_endpoints import router as auth_router
from litellm.proxy.public_relay.portal_endpoints import router as portal_router
from litellm.proxy.public_relay.public_endpoints import router as public_router

router = APIRouter()
router.include_router(public_router)
router.include_router(auth_router)
router.include_router(portal_router)
router.include_router(admin_router)

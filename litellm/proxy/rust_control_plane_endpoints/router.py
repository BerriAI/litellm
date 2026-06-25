"""Combined router for Python endpoints consumed by the Rust data plane."""

from fastapi import APIRouter

from litellm.proxy.rust_control_plane_endpoints.auth_endpoints import (
    router as auth_router,
)
from litellm.proxy.rust_control_plane_endpoints.logging_endpoints import (
    router as logging_router,
)

rust_control_plane_router = APIRouter()
rust_control_plane_router.include_router(auth_router)
rust_control_plane_router.include_router(logging_router)

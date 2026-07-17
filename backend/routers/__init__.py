"""Admin routes owned by the backend (control plane).

These routers are the source of truth for the admin surface: they are defined
here and imported by ``litellm.proxy.proxy_server`` (which mounts them when the
backend package is importable) and served directly by ``backend.main``. Each
route authenticates through the ``auth_v2`` ``AuthSecurity`` stored on
``app.state.auth_v2`` rather than the legacy ``user_api_key_auth`` dependency.
"""

from .teams import ADMIN_PREFIX
from .teams import router as admin_teams_router

admin_routers = (admin_teams_router,)

__all__ = ["ADMIN_PREFIX", "admin_routers", "admin_teams_router"]

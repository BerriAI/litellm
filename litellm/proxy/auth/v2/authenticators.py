import secrets
from typing import Any, List, Optional, Protocol, runtime_checkable

from fastapi import HTTPException, status


@runtime_checkable
class Authenticator(Protocol):
    def can_handle(self, api_key: Optional[str]) -> bool:
        ...

    async def authenticate(self, api_key: str, ctx: "AuthContext") -> Any:
        ...


class AuthContext:
    """Carries the proxy dependencies an authenticator needs to resolve identity."""

    def __init__(
        self,
        prisma_client: Any,
        user_api_key_cache: Any,
        proxy_logging_obj: Any,
        parent_otel_span: Any = None,
    ):
        self.prisma_client = prisma_client
        self.user_api_key_cache = user_api_key_cache
        self.proxy_logging_obj = proxy_logging_obj
        self.parent_otel_span = parent_otel_span


class MasterKeyAuthenticator:
    """Authenticates the configured master key as the proxy admin.

    Checked before the virtual-key node because the master key also looks like a
    ``sk-`` token but is not a row in the key table. The raw key never propagates
    downstream; a stable alias stands in for it.
    """

    def can_handle(self, api_key: Optional[str]) -> bool:
        from litellm.proxy.proxy_server import master_key

        if not isinstance(api_key, str) or not isinstance(master_key, str):
            return False
        try:
            return secrets.compare_digest(api_key, master_key)
        except Exception:
            return False

    async def authenticate(self, api_key: str, ctx: AuthContext) -> Any:
        from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS
        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.proxy_server import litellm_proxy_admin_name

        return UserAPIKeyAuth(
            api_key=LITELLM_PROXY_MASTER_KEY_ALIAS,
            user_id=litellm_proxy_admin_name,
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )


class VirtualKeyAuthenticator:
    """Resolves a ``sk-`` virtual key to its identity via the existing key store."""

    def can_handle(self, api_key: Optional[str]) -> bool:
        return isinstance(api_key, str) and api_key.startswith("sk-")

    async def authenticate(self, api_key: str, ctx: AuthContext) -> Any:
        from litellm.proxy._types import hash_token
        from litellm.proxy.auth.auth_checks import get_key_object

        return await get_key_object(
            hashed_token=hash_token(token=api_key),
            prisma_client=ctx.prisma_client,
            user_api_key_cache=ctx.user_api_key_cache,
            parent_otel_span=ctx.parent_otel_span,
            proxy_logging_obj=ctx.proxy_logging_obj,
        )


# Master key is matched first (exact compare), then virtual keys. authlib-backed
# JWT / OAuth2 nodes slot in next, implementing the same interface.
AUTHENTICATORS: List[Authenticator] = [
    MasterKeyAuthenticator(),
    VirtualKeyAuthenticator(),
]


async def authenticate(api_key: Optional[str], ctx: AuthContext) -> Any:
    """Dispatch by credential shape to the first authenticator that handles it."""
    for authenticator in AUTHENTICATORS:
        if authenticator.can_handle(api_key):
            return await authenticator.authenticate(api_key, ctx)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="auth_v2: no authenticator for the supplied credential",
    )

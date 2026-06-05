import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Protocol, runtime_checkable

from fastapi import HTTPException, status

from ..context import AuthMethod

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth


@dataclass(frozen=True)
class AuthResult:
    """The output of the authenticator chain: the resolved identity and how."""

    identity: "UserAPIKeyAuth"
    method: AuthMethod


@runtime_checkable
class Authenticator(Protocol):
    method: AuthMethod

    def can_handle(self, api_key: Optional[str]) -> bool: ...

    async def authenticate(self, api_key: str, ctx: "AuthContext") -> Any: ...


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

    method = AuthMethod.MASTER_KEY

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

    method = AuthMethod.VIRTUAL_KEY

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


def _load_jwt_settings() -> Any:
    from litellm.proxy.proxy_server import general_settings

    from .jwt_claims import JWTSettings

    cfg = (general_settings or {}).get("auth_v2_jwt") or {}
    jwks_uri = cfg.get("jwks_uri")
    if not jwks_uri:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="auth_v2: JWT auth received a token but no jwks_uri is configured",
        )
    return JWTSettings(
        jwks_uri=jwks_uri,
        issuer=cfg.get("issuer"),
        audience=cfg.get("audience"),
        user_id_claim=cfg.get("user_id_claim", "sub"),
        team_claim=cfg.get("team_claim"),
        role_claim=cfg.get("role_claim"),
        role_map=cfg.get("role_map") or {},
    )


def _jwt_is_configured() -> bool:
    from litellm.proxy.proxy_server import general_settings

    cfg = (general_settings or {}).get("auth_v2_jwt") or {}
    return bool(cfg.get("jwks_uri"))


class JWTAuthenticator:
    """Verifies a bearer JWT with authlib and maps its claims to an identity."""

    method = AuthMethod.JWT

    def can_handle(self, api_key: Optional[str]) -> bool:
        # Only claim JWT-shaped tokens when JWT auth is actually configured, so an
        # unconfigured deployment falls through to a clean 401 instead of a 500.
        if not (
            isinstance(api_key, str)
            and not api_key.startswith("sk-")
            and api_key.count(".") == 2
        ):
            return False
        return _jwt_is_configured()

    async def authenticate(self, api_key: str, ctx: AuthContext) -> Any:
        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

        from .jwt_claims import extract_identity
        from .jwt_verifier import JWKSProvider, JWTVerificationError, verify

        settings = _load_jwt_settings()
        key_set = await JWKSProvider(settings.jwks_uri).get_key_set()
        try:
            claims = verify(api_key, key_set, settings.issuer, settings.audience)
        except JWTVerificationError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"auth_v2: {e}",
            )

        identity = extract_identity(claims, settings)
        user_role = None
        if identity.role is not None:
            try:
                user_role = LitellmUserRoles(identity.role)
            except ValueError:
                user_role = None

        return UserAPIKeyAuth(
            user_id=identity.user_id,
            team_id=identity.team_id,
            user_role=user_role,
            jwt_claims=claims,
        )


def _load_introspection_settings() -> Any:
    from litellm.proxy.proxy_server import general_settings

    from .oauth2_introspection import IntrospectionSettings

    cfg = (general_settings or {}).get("auth_v2_oauth2") or {}
    endpoint = cfg.get("introspection_endpoint")
    if not endpoint:
        return None
    return IntrospectionSettings(
        endpoint=endpoint,
        client_id=cfg.get("client_id"),
        client_secret=cfg.get("client_secret"),
        user_id_claim=cfg.get("user_id_claim", "sub"),
        team_claim=cfg.get("team_claim"),
        scope_claim=cfg.get("scope_claim", "scope"),
        role_map=cfg.get("role_map") or {},
    )


class OAuth2IntrospectionAuthenticator:
    """Validates an opaque bearer token via an RFC 7662 introspection endpoint."""

    method = AuthMethod.OAUTH2

    def can_handle(self, api_key: Optional[str]) -> bool:
        if not isinstance(api_key, str) or not api_key:
            return False
        # Opaque token: not a virtual key, not a JWT. Only when introspection is
        # configured, so unconfigured deployments fall through to a clean 401.
        if api_key.startswith("sk-") or api_key.count(".") == 2:
            return False
        return _load_introspection_settings() is not None

    async def authenticate(self, api_key: str, ctx: AuthContext) -> Any:
        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

        from .oauth2_introspection import (
            OAuth2IntrospectionError,
            parse_introspection_response,
        )

        settings = _load_introspection_settings()
        data = await self._introspect(api_key, settings)
        try:
            identity = parse_introspection_response(data, settings)
        except OAuth2IntrospectionError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=f"auth_v2: {e}"
            )

        user_role = None
        if identity.role is not None:
            try:
                user_role = LitellmUserRoles(identity.role)
            except ValueError:
                user_role = None

        return UserAPIKeyAuth(
            user_id=identity.user_id,
            team_id=identity.team_id,
            user_role=user_role,
        )

    async def _introspect(self, token: str, settings: Any) -> Any:
        import base64

        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
        from litellm.types.llms.custom_http import httpxSpecialProvider

        # RFC 7662 client auth is HTTP Basic; the shared handler's post() takes
        # headers, not an auth tuple, so build the header explicitly.
        headers = {}
        if settings.client_id:
            credentials = f"{settings.client_id}:{settings.client_secret or ''}"
            headers["Authorization"] = (
                "Basic " + base64.b64encode(credentials.encode()).decode()
            )

        client = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)
        response = await client.post(
            settings.endpoint, data={"token": token}, headers=headers
        )
        return response.json()


# Master key first (exact compare), then virtual keys, then JWTs, then opaque
# tokens via introspection. Dispatch is by credential shape, so the chain is
# deterministic, not "ask everyone".
AUTHENTICATORS: List[Authenticator] = [
    MasterKeyAuthenticator(),
    VirtualKeyAuthenticator(),
    JWTAuthenticator(),
    OAuth2IntrospectionAuthenticator(),
]


async def authenticate(api_key: Optional[str], ctx: AuthContext) -> AuthResult:
    """Dispatch by credential shape to the first authenticator that handles it."""
    for authenticator in AUTHENTICATORS:
        # The isinstance narrowing is redundant with can_handle at runtime (every
        # can_handle requires a str) but makes the str guarantee explicit to the
        # type checker before dispatching.
        if authenticator.can_handle(api_key) and isinstance(api_key, str):
            identity = await authenticator.authenticate(api_key, ctx)
            return AuthResult(identity=identity, method=authenticator.method)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="auth_v2: no authenticator for the supplied credential",
    )

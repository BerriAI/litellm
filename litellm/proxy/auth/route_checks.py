from collections.abc import Collection, Iterable, Mapping
from typing import Final, cast

from fastapi import HTTPException, Request, status

from litellm._auth import routes
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    MODEL_LIST_ROUTES,
    UI_TEAM_ID,
    CommonProxyErrors,
    KeyManagementRoutes,
    LiteLLM_UserTable,
    LiteLLMRoutes,
    LitellmUserRoles,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth import route_policy_adapter

from .auth_checks_organization import (  # noqa: F401  # legacy module exports
    _user_is_org_admin,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
)
from .auth_checks_organization import (
    user_is_org_admin as user_is_org_admin,  # noqa: PLC0414  # preserve the public compatibility re-export
)

# Management write routes denied to PROXY_ADMIN_VIEW_ONLY. Adding a new write
# endpoint to a management router REQUIRES adding it here too — the surrounding
# check falls through to "allow" if the route is not matched, which previously
# let view-only admins call /team/block, /team/unblock, /key/bulk_update, etc.
_PROXY_ADMIN_VIEW_ONLY_BLOCKED_ROUTES: Final = frozenset(
    [
        # user
        "/user/new",
        "/management/v1/users/bulk",
        "/user/delete",
        "/management/v1/users/bulk_delete",
        "/user/bulk_update",
        # team
        "/team/new",
        "/management/v1/teams/{team_id}/members/bulk_delete",
        "/management/v1/teams/{team_id}/members/bulk_update",
        "/team/update",
        "/team/delete",
        "/team/block",
        "/team/unblock",
        "/team/permissions_update",
        "/team/permissions_bulk_update",
        # model
        "/model/new",
        "/model/update",
        "/model/delete",
        # JWT key mapping
        "/jwt/key/mapping/new",
        "/jwt/key/mapping/update",
        "/jwt/key/mapping/delete",
        # key management — keep in sync with KeyManagementRoutes write entries
        KeyManagementRoutes.KEY_GENERATE.value,
        KeyManagementRoutes.KEY_UPDATE.value,
        KeyManagementRoutes.KEY_DELETE.value,
        KeyManagementRoutes.KEY_REGENERATE.value,
        KeyManagementRoutes.KEY_GENERATE_SERVICE_ACCOUNT.value,
        KeyManagementRoutes.KEY_BLOCK.value,
        KeyManagementRoutes.KEY_UNBLOCK.value,
        KeyManagementRoutes.KEY_BULK_UPDATE.value,
        KeyManagementRoutes.TEAM_KEY_BULK_UPDATE.value,
    ]
)

# Suffixes for `/key/{key_id}/...` path-parameterized write routes that the
# enum templates with `{key_id}`. The blocklist above can't match templated
# paths directly because the request route carries the resolved key id.
_PROXY_ADMIN_VIEW_ONLY_BLOCKED_KEY_SUFFIXES: Final = ("/regenerate", "/reset_spend")

_LLM_API_ROUTE_GROUPS: Final = frozenset(("llm_api_routes", "llm_api_routes_without_mcp"))
_EU_JEV_DECISION_ROUTE: Final = "/eu_jev/v1/systemone"
_AUTH_ENFORCED_PASS_THROUGH_ROUTE_GROUPS: Final = frozenset(("openai_routes", *_LLM_API_ROUTE_GROUPS))

# Writes that only touch the caller's own credential row (BYOK connections and
# per-user MCP credentials). Method-scoped because the generic credential CRUD
# handlers share these paths: the route check cannot admit PATCH/DELETE on a
# name like "user_connections" without also opening the admin CRUD handlers.
_SELF_SCOPED_CREDENTIAL_ROUTES: Final = (
    ("GET", "/credentials/user_connections"),
    ("POST", "/credentials/{credential_name:path}/user_connection/start"),
    ("POST", "/credentials/{credential_name:path}/user_connection/poll"),
    ("DELETE", "/credentials/{credential_name:path}/user_connection"),
    ("POST", "/v1/mcp/server/oauth/{server_id}/register"),
    ("POST", "/v1/mcp/server/{server_id}/oauth-user-credential"),
    ("DELETE", "/v1/mcp/server/{server_id}/oauth-user-credential"),
    ("POST", "/v1/mcp/server/{server_id}/user-credential"),
    ("DELETE", "/v1/mcp/server/{server_id}/user-credential"),
    ("POST", "/v1/mcp/server/{server_id}/user-env-vars"),
    ("DELETE", "/v1/mcp/server/{server_id}/user-env-vars"),
)


_UNBILLED_ROUTES: Final[frozenset[str]] = frozenset(
    {
        *MODEL_LIST_ROUTES,
        "/utils/token_counter",
        "/utils/transform_request",
        "/responses/input_tokens",
        "/v1/responses/input_tokens",
        "/openai/v1/responses/input_tokens",
    }
)
_TOKEN_COUNTING_SEGMENTS: Final[frozenset[str]] = frozenset({"count_tokens", "count-tokens"})
_TOKEN_COUNTING_ACTION: Final = "countTokens"


class RouteChecks:
    @staticmethod
    def check_route_enabled(route: str) -> None:
        from litellm.proxy.auth.route_enable_adapter import check_route_enabled

        check_route_enabled(route)

    @staticmethod
    def should_call_route(
        route: str,
        valid_token: UserAPIKeyAuth,
        request: Request | None = None,
    ):
        """
        Check if management route is disabled and raise exception
        """
        RouteChecks.check_route_enabled(route)
        # Check if Virtual Key is allowed to call the route - Applies to all Roles
        RouteChecks.is_virtual_key_allowed_to_call_route(route=route, valid_token=valid_token, request=request)
        return True

    @staticmethod
    def is_virtual_key_allowed_to_call_route(
        route: str,
        valid_token: UserAPIKeyAuth,
        request: Request | None = None,
    ) -> bool:
        return route_policy_adapter.check_virtual_route(route, valid_token, request)

    @staticmethod
    def _mask_user_id(user_id: str) -> str:
        """
        Mask user_id to prevent leaking sensitive information in error messages

        Args:
            user_id (str): The user_id to mask

        Returns:
            str: Masked user_id showing only first 2 and last 2 characters
        """
        from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker

        if not user_id or len(user_id) <= 4:
            return "***"

        # Use SensitiveDataMasker with custom configuration for user_id
        masker: Final = SensitiveDataMasker(visible_prefix=6, visible_suffix=2, mask_char="*")

        return masker.mask_value(user_id)

    @staticmethod
    def _raise_admin_only_route_exception(
        user_obj: LiteLLM_UserTable | None,
        route: str,
    ) -> None:
        """
        Raise exception for routes that require proxy admin access

        Args:
            user_obj (Optional[LiteLLM_UserTable]): The user object
            route (str): The route being accessed

        Raises:
            ProxyException: 401 with user role and masked user_id information
        """
        user_role = "unknown"
        user_id = "unknown"
        if user_obj is not None:
            user_role = user_obj.user_role or "unknown"
            user_id = user_obj.user_id or "unknown"

        masked_user_id: Final = RouteChecks._mask_user_id(user_id)
        raise ProxyException(
            message=(
                "Authentication Error, Only proxy admin can be used to generate, delete, update info for new "
                f"keys/users/teams. Route={route}. Your role={user_role}. Your user_id={masked_user_id}"
            ),
            type=ProxyErrorTypes.auth_error,
            param=None,
            code=status.HTTP_401_UNAUTHORIZED,
        )

    @staticmethod
    def non_proxy_admin_allowed_routes_check(
        user_obj: LiteLLM_UserTable | None,
        _user_role: LitellmUserRoles | None,
        route: str,
        request: Request,
        valid_token: UserAPIKeyAuth,
        request_data: dict,
    ):
        return route_policy_adapter.check_non_admin_route(
            user_obj, _user_role, route, request, valid_token, request_data
        )

    @staticmethod
    def custom_admin_only_route_check(route: str):
        from litellm.proxy.proxy_server import general_settings, premium_user

        if "admin_only_routes" in general_settings:
            if premium_user is not True:
                verbose_proxy_logger.error(
                    "Trying to use 'admin_only_routes' this is an Enterprise only feature. %s",
                    CommonProxyErrors.not_premium_user.value,
                )
                return
            if route in general_settings["admin_only_routes"]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"user not allowed to access this route. Route={route} is an admin only route",
                )

    @staticmethod
    def is_allowed_on_proxy(route: str) -> bool:
        from litellm.proxy.proxy_server import general_settings, premium_user

        if "allowed_routes" not in general_settings:
            return True
        if premium_user is not True:
            verbose_proxy_logger.error(
                "Trying to set allowed_routes. This is an Enterprise feature. %s",
                CommonProxyErrors.not_premium_user.value,
            )
        return route in general_settings["allowed_routes"]

    @staticmethod
    def is_unbilled_route(route: str) -> bool:
        return route_policy_adapter.classify(route, "Unbilled")

    @staticmethod
    def is_llm_api_route(route: str) -> bool:
        return isinstance(route, str) and route_policy_adapter.classify(route, "Llm")

    @staticmethod
    def is_builtin_eu_jev_decision_route(route: str) -> bool:
        if route.rstrip("/") != _EU_JEV_DECISION_ROUTE:
            return False
        from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (  # noqa: PLC0415  # pass_through_endpoints imports the auth stack
            InitPassThroughEndpointHelpers,
        )

        return not InitPassThroughEndpointHelpers.has_configured_route_ignoring_trailing_slash(
            route=_EU_JEV_DECISION_ROUTE
        )

    @staticmethod
    def _is_get_mcp_server_discovery_route(route: str, request: Request | None) -> bool:
        return route_policy_adapter.classify(route, "Discovery", RouteChecks.get_request_method(request))

    @staticmethod
    def is_management_route(route: str) -> bool:
        """
        Check if route is a management route
        """
        return RouteChecks.check_route_access(route=route, allowed_routes=LiteLLMRoutes.management_routes.value)

    @staticmethod
    def is_info_route(route: str) -> bool:
        """
        Check if route is an info route
        """
        return route in LiteLLMRoutes.info_routes.value

    @staticmethod
    def is_dashboard_session_info_route(route: str, team_id: str | None) -> bool:
        """
        Check if an Admin UI session key is calling an info route the dashboard polls in the background
        """
        return team_id == UI_TEAM_ID and RouteChecks.is_info_route(route)

    @staticmethod
    def _is_azure_openai_route(route: str) -> bool:
        return isinstance(route, str) and route_policy_adapter.classify(route, "Azure")

    @staticmethod
    def _route_matches_pattern(route: str, pattern: str) -> bool:
        return isinstance(route, str) and routes.match_route(route, (pattern,), "Template")

    @staticmethod
    def is_wildcard_pattern(pattern: str) -> bool:
        """
        Check if pattern is a wildcard pattern
        """
        return pattern.endswith("*")

    _is_wildcard_pattern = is_wildcard_pattern

    @staticmethod
    def route_matches_wildcard_pattern(route: str, pattern: str) -> bool:
        return routes.match_route(route, (pattern,), "Wildcard")

    @staticmethod
    def _route_matches_allowed_route(route: str, allowed_route: str) -> bool:
        return routes.match_route(route, (allowed_route,), "Prefix")

    @staticmethod
    def check_route_access(route: str, allowed_routes: Collection[str]) -> bool:
        return routes.match_route(route, tuple(allowed_routes))

    @staticmethod
    def get_request_method(request: Request | None) -> str | None:
        if request is None:
            return None

        try:
            method: Final = cast(  # cast-ok: request.method is str at runtime; tests hand a MagicMock
                object, request.method
            )
        except (AttributeError, KeyError):
            return None
        if not isinstance(method, str):
            return None

        return method.upper()

    _get_request_method = get_request_method

    @staticmethod
    def is_self_scoped_credential_route(route: str, request: Request | None) -> bool:
        return route_policy_adapter.classify(route, "SelfCredential", RouteChecks.get_request_method(request))

    @staticmethod
    def is_auth_enforced_pass_through_route(route: str, method: str | None = None) -> bool:
        """
        True for config/DB pass-through endpoints registered with auth=true.

        These routes are injected into ``openai_routes`` for spend/budget hooks but
        must not inherit blanket ``openai_routes`` RBAC; access is gated by
        ``allowed_passthrough_routes`` on the key or team.
        """
        from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
            InitPassThroughEndpointHelpers,
        )

        route_info: Final = InitPassThroughEndpointHelpers.get_registered_pass_through_route(route=route, method=method)
        if route_info is None:
            return False
        return route_info.get("auth") is True

    @staticmethod
    def _auth_pass_through_denied_exception(route: str) -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Key/team not allowed to access passthrough route {route}. "
                "Configure `allowed_passthrough_routes` on the team or key."
            ),
        )

    @staticmethod
    def _route_matches_denied_route(route: str, denied_route: str) -> bool:
        return routes.match_route(route, (denied_route,), "Denied")

    @staticmethod
    def matching_denied_passthrough_route(
        route: str, metadata_sources: Iterable[Mapping[str, object] | None]
    ) -> str | None:
        return route_policy_adapter.denied_passthrough(route, metadata_sources)

    @staticmethod
    def passthrough_route_denied_exception(route: str, denied_route: str) -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Key/team denied access to passthrough route {route}. "
                f"Matched `{denied_route}` in `denied_passthrough_routes`."
            ),
        )

    @staticmethod
    def _raise_if_passthrough_route_denied(route: str, valid_token: UserAPIKeyAuth) -> None:
        denied_route: Final = RouteChecks.matching_denied_passthrough_route(
            route=route,
            metadata_sources=(valid_token.metadata, valid_token.team_metadata),
        )
        if denied_route is not None:
            raise RouteChecks.passthrough_route_denied_exception(route=route, denied_route=denied_route)

    @staticmethod
    def jwt_team_routes_grant_pass_through(route: str, team_allowed_routes: Collection[str]) -> bool:
        return routes.match_route(route, tuple(team_allowed_routes), "Jwt")

    @staticmethod
    def _jwt_team_allowed_routes(valid_token: UserAPIKeyAuth) -> Collection[str]:
        """``team_allowed_routes`` for team tokens built by JWT auth; JWT-mapped virtual keys stay key-scoped."""
        if valid_token.jwt_claims is None or valid_token.token is not None or valid_token.team_id is None:
            return ()

        from litellm.proxy.proxy_server import jwt_handler

        return jwt_handler.litellm_jwtauth.team_allowed_routes

    @staticmethod
    def _require_auth_pass_through_access(
        route: str,
        valid_token: UserAPIKeyAuth,
        jwt_team_allowed_routes: Collection[str] = (),
    ) -> None:
        decision: Final = routes.passthrough_route(
            routes.RouteContext(route),
            route_policy_adapter.passthrough_facts(route, valid_token, jwt_team_allowed_routes, include_denied=True),
            required=True,
        )
        route_policy_adapter.raise_route_denial(decision, route, token=valid_token)

    @staticmethod
    def check_passthrough_route_access(route: str, user_api_key_dict: UserAPIKeyAuth) -> bool:
        return routes.passthrough_route(
            routes.RouteContext(route), route_policy_adapter.passthrough_facts(route, user_api_key_dict)
        ).allowed

    @staticmethod
    def is_assistants_api_request(request: Request) -> bool:
        """
        Returns True if `thread` or `assistant` is in the request path

        Args:
            request (Request): The request object

        Returns:
            bool: True if `thread` or `assistant` is in the request path, False otherwise
        """
        # Inline import — auth_utils participates in a proxy import cycle.
        from .auth_utils import get_request_route  # noqa: PLC0415

        route: Final = get_request_route(request)
        if "thread" in route or "assistant" in route:
            return True
        return False

    _is_assistants_api_request = is_assistants_api_request

    @staticmethod
    def is_generate_content_route(route: str) -> bool:
        """
        Returns True if this is a google generateContent or streamGenerateContent route

        These routes from google allow passing key=api_key in the query params
        """
        if "generateContent" in route:
            return True
        if "streamGenerateContent" in route:
            return True
        return False

    # HTTP methods that are intrinsically read-only and therefore safe to
    # default-allow for PROXY_ADMIN_VIEW_ONLY. Anything else (POST/PUT/PATCH/
    # DELETE) is treated as a write attempt and goes through the explicit
    # write-allowlist below.
    _SAFE_HTTP_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

    # Explicit write routes that PROXY_ADMIN_VIEW_ONLY must NEVER call. The
    # role-principle is "no writes, ever" — the management_routes list is the
    # authoritative source for which non-llm routes are writes; we just need
    # to filter out the read endpoints (info / list) that share the prefix.
    # A cleaner approach is to denylist by HTTP verb (POST/PUT/PATCH/DELETE);
    # this block stays as a backstop in case a write is implemented as GET.
    _ADMIN_VIEWER_BLOCKED_WRITE_ROUTES = frozenset(
        [
            "/user/new",
            "/management/v1/users/bulk",
            "/user/delete",
            "/management/v1/users/bulk_delete",
            "/user/bulk_update",
            "/team/new",
            "/management/v1/teams/{team_id}/members/bulk_delete",
            "/management/v1/teams/{team_id}/members/bulk_update",
            "/team/update",
            "/team/delete",
            "/model/new",
            "/model/update",
            "/model/delete",
            "/key/generate",
            "/key/delete",
            "/key/update",
            "/key/regenerate",
            "/key/service-account/generate",
            "/key/block",
            "/key/unblock",
            "/team/key/bulk_update",
        ]
    )

    @staticmethod
    def _check_proxy_admin_viewer_access(
        route: str,
        _user_role: str,
        request_data: dict,
        request: Request | None = None,
    ) -> None:
        return route_policy_adapter.check_viewer_route(route, _user_role, request_data, request)

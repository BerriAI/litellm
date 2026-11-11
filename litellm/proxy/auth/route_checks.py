import itertools
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, NoReturn, cast

from fastapi import HTTPException, Request, status
from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    MODEL_LIST_ROUTES,
    UI_TEAM_ID,
    CommonProxyErrors,
    LiteLLM_UserTable,
    LiteLLMRoutes,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.rust_bridge._native import (
    RustDenial,
    auth_allowed_routes_grant,
    auth_first_denied_route,
    auth_jwt_team_routes_grant,
    auth_mask_user_id,
    auth_route_denial,
    auth_route_matches_denied,
    auth_route_matches_prefix,
    auth_route_matches_wildcard,
    auth_viewer_route_denial,
)

from .auth_checks_organization import (  # noqa: F401  # legacy module exports
    _user_is_org_admin,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    user_is_org_admin,
)

_LLM_API_ROUTE_GROUPS: Final = frozenset(("llm_api_routes", "llm_api_routes_without_mcp"))
_EU_JEV_DECISION_ROUTE: Final = "/eu_jev/v1/systemone"
_AUTH_ENFORCED_PASS_THROUGH_ROUTE_GROUPS: Final = frozenset(("openai_routes", *_LLM_API_ROUTE_GROUPS))

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


def _allowed_passthrough_routes(metadata: Mapping[str, object] | None) -> list[str] | None:
    return cast(  # cast-ok: management endpoints validate this metadata key as a list of route strings on write
        "list[str] | None", (metadata or {}).get("allowed_passthrough_routes")
    )


class RouteChecks:
    @staticmethod
    def check_route_enabled(route: str) -> None:
        try:
            from litellm_enterprise.proxy.auth.route_checks import EnterpriseRouteChecks

            EnterpriseRouteChecks.should_call_route(route=route)
        except HTTPException as e:
            raise e
        except Exception:
            pass

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
        """
        Raises Exception if Virtual Key is not allowed to call the route
        """

        # Only check if valid_token.allowed_routes is set and is a list with at least one item
        if valid_token.allowed_routes is None:
            return True
        if not isinstance(valid_token.allowed_routes, list):
            return True
        if len(valid_token.allowed_routes) == 0:
            return True

        denied_auth_enforced_pass_through_route = False

        # explicit check for allowed routes (exact match or prefix match)
        for allowed_route in valid_token.allowed_routes:
            if RouteChecks._route_matches_allowed_route(route=route, allowed_route=allowed_route):
                return True

        ## check if 'allowed_route' is a field name in LiteLLMRoutes
        if any(allowed_route in LiteLLMRoutes._member_names_ for allowed_route in valid_token.allowed_routes):
            for allowed_route in valid_token.allowed_routes:
                if allowed_route in LiteLLMRoutes._member_names_:
                    if RouteChecks.check_route_access(
                        route=route,
                        allowed_routes=LiteLLMRoutes[allowed_route].value,
                    ):
                        if (
                            allowed_route in _AUTH_ENFORCED_PASS_THROUGH_ROUTE_GROUPS
                            and RouteChecks.is_auth_enforced_pass_through_route(
                                route=route,
                                method=RouteChecks.get_request_method(request=request),
                            )
                        ):
                            if RouteChecks.check_passthrough_route_access(route=route, user_api_key_dict=valid_token):
                                return True
                            denied_auth_enforced_pass_through_route = True
                        else:
                            return True

                    ################################################
                    #  For llm_api_routes, also check registered pass-through endpoints
                    ################################################
                    if allowed_route in _LLM_API_ROUTE_GROUPS:
                        if route == "/auto_router/session" and RouteChecks.get_request_method(request) == "GET":
                            return True

                        from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
                            InitPassThroughEndpointHelpers,
                        )

                        if InitPassThroughEndpointHelpers.is_registered_pass_through_route(route=route):
                            if RouteChecks.is_auth_enforced_pass_through_route(
                                route=route,
                                method=RouteChecks.get_request_method(request=request),
                            ):
                                if RouteChecks.check_passthrough_route_access(
                                    route=route, user_api_key_dict=valid_token
                                ):
                                    return True
                                denied_auth_enforced_pass_through_route = True
                            else:
                                return True

                        # Method-aware carve-out: allow GET on the two
                        # read-only MCP-server discovery endpoints
                        # (`/v1/mcp/server` and `/v1/mcp/server/{server_id}`)
                        # so virtual keys with allowed_routes=["llm_api_routes"]
                        # can list/inspect MCP servers. The GET handlers in
                        # mcp_management_endpoints.py sanitize the response
                        # for restricted virtual keys (stripping url,
                        # headers, env, credentials). POST/PUT/DELETE on
                        # these paths are admin-only management writes and
                        # are intentionally not covered.
                        if allowed_route == "llm_api_routes" and RouteChecks._is_get_mcp_server_discovery_route(
                            route=route, request=request
                        ):
                            return True

                        # Agent registry CRUD moved from llm_api_routes into
                        # management_routes so DISABLE_LLM_API_ENDPOINTS stops
                        # blocking it. Keys configured with
                        # allowed_routes=["llm_api_routes"] before that split
                        # could reach these paths, so keep them reachable here;
                        # the handlers in agent_endpoints/endpoints.py still
                        # enforce proxy-admin on writes and scope reads by role.
                        if allowed_route == "llm_api_routes" and RouteChecks.check_route_access(
                            route=route,
                            allowed_routes=LiteLLMRoutes.agent_management_routes.value,
                        ):
                            return True

        # check if wildcard pattern is allowed
        for allowed_route in valid_token.allowed_routes:
            if RouteChecks.route_matches_wildcard_pattern(route=route, pattern=allowed_route):
                return True

        if denied_auth_enforced_pass_through_route:
            raise RouteChecks._auth_pass_through_denied_exception(route=route)

        if valid_token.metadata.get("password_reset_required") is True:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "This account's password must be changed before the session can be used: "
                    "it was either found in a known data breach or set by an admin. "
                    "Change it via POST /user/password/change (UI: /ui/change-password), then log in again."
                ),
            )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Virtual key is not allowed to call this route. Only allowed to call routes: {valid_token.allowed_routes}. Tried to call route: {route}",
        )

    @staticmethod
    def _mask_user_id(user_id: str) -> str:
        """The user id an admin-only denial shows: its first 6 and last 2 characters."""
        return auth_mask_user_id(user_id)

    @staticmethod
    def non_proxy_admin_allowed_routes_check(
        user_obj: LiteLLM_UserTable | None,
        _user_role: LitellmUserRoles | None,
        route: str,
        request: Request,
        valid_token: UserAPIKeyAuth,
        request_data: dict,
    ) -> None:
        """Whether a non proxy admin may call the route, decided by gateway-auth-core."""
        denial: Final = auth_route_denial(
            route_facts(route, request, request_data, valid_token, user_obj, user_role=_user_role),
            organization_id=_organization_id(request_data),
        )
        if denial is not None:
            _raise(denial)

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
        resource, _, action = route.rsplit("/", 1)[-1].partition(":")
        return route in _UNBILLED_ROUTES or resource in _TOKEN_COUNTING_SEGMENTS or action == _TOKEN_COUNTING_ACTION

    @staticmethod
    def is_llm_api_route(route: str) -> bool:
        """
        Helper to checks if provided route is an OpenAI route


        Returns:
            - True: if route is an OpenAI route
            - False: if route is not an OpenAI route
        """
        # Ensure route is a string before performing checks
        if not isinstance(route, str):
            return False

        if route in LiteLLMRoutes.openai_routes.value:
            return True

        if route in LiteLLMRoutes.anthropic_routes.value:
            return True

        if route in LiteLLMRoutes.google_routes.value:
            return True

        if RouteChecks.check_route_access(route=route, allowed_routes=LiteLLMRoutes.mcp_inference_routes.value):
            return True

        if RouteChecks.check_route_access(route=route, allowed_routes=LiteLLMRoutes.agent_inference_routes.value):
            return True

        if RouteChecks.check_route_access(route=route, allowed_routes=LiteLLMRoutes.litellm_native_routes.value):
            return True

        # fuzzy match routes like "/v1/threads/thread_49EIN5QF32s4mH20M7GFKdlZ"
        # Check for routes with placeholders or wildcard patterns
        for openai_route in LiteLLMRoutes.openai_routes.value:
            # Replace placeholders with regex pattern
            # placeholders are written as "/threads/{thread_id}"
            if "{" in openai_route:
                if RouteChecks._route_matches_pattern(route=route, pattern=openai_route):
                    return True
            # Check for wildcard patterns like "/containers/*"
            if RouteChecks.is_wildcard_pattern(pattern=openai_route):
                if RouteChecks.route_matches_wildcard_pattern(route=route, pattern=openai_route):
                    return True

        # Check for Google routes with placeholders like "/v1beta/models/{model_name}:generateContent"
        for google_route in LiteLLMRoutes.google_routes.value:
            if "{" in google_route:
                if RouteChecks._route_matches_pattern(route=route, pattern=google_route):
                    return True

        # Check for Anthropic routes with placeholders
        for anthropic_route in LiteLLMRoutes.anthropic_routes.value:
            if "{" in anthropic_route:
                if RouteChecks._route_matches_pattern(route=route, pattern=anthropic_route):
                    return True

        if RouteChecks._is_azure_openai_route(route=route):
            return True

        for _llm_passthrough_route in LiteLLMRoutes.mapped_pass_through_routes.value:
            if route == _llm_passthrough_route or route.startswith(_llm_passthrough_route + "/"):
                return True
        return RouteChecks.is_builtin_eu_jev_decision_route(route=route)

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
        """
        Returns True if `request` is a GET against one of the two read-only
        MCP-server discovery paths:

        - GET `/v1/mcp/server`               (list)
        - GET `/v1/mcp/server/{server_id}`   (single server, single segment)

        Multi-segment paths (`/v1/mcp/server/{id}/approve`, etc.) and any
        non-GET method return False, so admin-only management writes on the
        same path prefix are not reachable through this carve-out.
        """
        if request is None or request.method.upper() != "GET":
            return False
        if route == "/v1/mcp/server":
            return True
        prefix: Final = "/v1/mcp/server/"
        if not route.startswith(prefix):
            return False
        remainder: Final = route[len(prefix) :]
        return bool(remainder) and "/" not in remainder

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
        """
        Check if route is a route from AzureOpenAI SDK client

        eg.
        route='/openai/deployments/vertex_ai/gemini-1.5-flash/chat/completions'
        """
        # Ensure route is a string before attempting regex matching
        if not isinstance(route, str):
            return False
        # Add support for deployment and engine model paths
        deployment_pattern: Final = r"^/openai/deployments/[^/]+/[^/]+/chat/completions$"
        engine_pattern: Final = r"^/engines/[^/]+/chat/completions$"

        if re.match(deployment_pattern, route) or re.match(engine_pattern, route):
            return True
        return False

    @staticmethod
    def _route_matches_pattern(route: str, pattern: str) -> bool:
        """
        Check if route matches the pattern placed in proxy/_types.py

        Example:
        - pattern: "/threads/{thread_id}"
        - route: "/threads/thread_49EIN5QF32s4mH20M7GFKdlZ"
        - returns: True


        - pattern: "/key/{token_id}/regenerate"
        - route: "/key/regenerate/82akk800000000jjsk"
        - returns: False, pattern is "/key/{token_id}/regenerate"
        """
        # Ensure route is a string before attempting regex matching
        if not isinstance(route, str):
            return False

        def _placeholder_to_regex(match: re.Match) -> str:
            placeholder: Final = match.group(0).strip("{}")
            if not placeholder.endswith(":path"):
                return r"[^/]+"
            # A ":path" placeholder takes whatever the router's own path
            # converter takes, slashes and colons alike, so an id spelled with
            # either (or both) still matches the template it was mounted under.
            #
            # Unless the template puts a ":" literal of its own after the
            # placeholder: the Google routes end in ":generateContent" and
            # friends, and there the value has to stop before that suffix
            # rather than swallow it and match a different verb.
            #
            # "[\s\S]" rather than ".", because "." stops at a newline and the
            # path converter does not: a %0A anywhere in the value would leave
            # the route unmatched here while still reaching the handler, which
            # turns this gate into a bypass for the lists built on it.
            return r"[^:]+" if ":" in match.string[match.end() :] else r"[\s\S]+"

        pattern = re.sub(r"\{[^}]+\}", _placeholder_to_regex, pattern)
        # Anchor the pattern to match the entire string
        pattern = f"^{pattern}$"
        if re.match(pattern, route):
            return True
        return False

    @staticmethod
    def is_wildcard_pattern(pattern: str) -> bool:
        """
        Check if pattern is a wildcard pattern
        """
        return pattern.endswith("*")

    _is_wildcard_pattern = is_wildcard_pattern

    @staticmethod
    def route_matches_wildcard_pattern(route: str, pattern: str) -> bool:
        """A trailing `*` matches by prefix, anything else exactly."""
        return auth_route_matches_wildcard(route, pattern)

    @staticmethod
    def _route_matches_allowed_route(route: str, allowed_route: str) -> bool:
        """The route itself or anything under it, never a sibling that only shares the prefix."""
        return auth_route_matches_prefix(route, allowed_route)

    @staticmethod
    def check_route_access(route: str, allowed_routes: Collection[str]) -> bool:
        """
        Check if a route has access by checking both exact matches and patterns

        Args:
            route (str): The route to check
            allowed_routes (Sequence): Allowed routes/patterns

        Returns:
            bool: True if route is allowed, False otherwise
        """
        #########################################################
        # exact match route is in allowed_routes
        #########################################################
        if route in allowed_routes:
            return True

        #########################################################
        # wildcard match route is in allowed_routes
        # e.g calling /anthropic/v1/messages is allowed if allowed_routes has /anthropic/*
        #########################################################
        if any(
            RouteChecks.route_matches_wildcard_pattern(route=route, pattern=allowed_route)
            for allowed_route in allowed_routes
            if RouteChecks.is_wildcard_pattern(pattern=allowed_route)
        ):
            return True

        #########################################################
        # pattern match route is in allowed_routes
        # pattern: "/threads/{thread_id}"
        # route: "/threads/thread_49EIN5QF32s4mH20M7GFKdlZ"
        # returns: True
        #########################################################
        if any(  # Check pattern match
            RouteChecks._route_matches_pattern(route=route, pattern=allowed_route) for allowed_route in allowed_routes
        ):
            return True

        return False

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
        """A `/` entry denies every route, since every route sits under the root."""
        return auth_route_matches_denied(route, denied_route)

    @staticmethod
    def forwarded_pass_through_routes(route: str) -> tuple[str, ...]:
        from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (  # noqa: PLC0415  # pass-through endpoints import this module
            InitPassThroughEndpointHelpers,
        )

        return InitPassThroughEndpointHelpers.forwarded_routes(route)

    @staticmethod
    def matching_denied_passthrough_route(
        route: str, metadata_sources: Iterable[Mapping[str, object] | None]
    ) -> str | None:
        """
        First ``denied_passthrough_routes`` entry across ``metadata_sources`` that matches ``route``.
        Unlike the allowlist (key list, else team list), every source's deny list applies.
        """
        denied_routes: Final = list(
            itertools.chain.from_iterable(
                cast(  # cast-ok: management endpoints validate this metadata key as a list of route strings on write
                    "list[str]", (metadata or {}).get("denied_passthrough_routes") or []
                )
                for metadata in metadata_sources
            )
        )
        if not denied_routes:
            return None
        return auth_first_denied_route(list(RouteChecks.forwarded_pass_through_routes(route)), denied_routes)

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
        """
        Explicit paths and trailing-wildcard prefixes grant auth=true pass-through. Blanket grants never do:
        a named route group like ``openai_routes`` is only ever compared as a path, and an entry that names
        no path segment (``*``, ``/*``) is skipped.
        """
        return auth_jwt_team_routes_grant(route, list(team_allowed_routes))

    @staticmethod
    def _jwt_team_allowed_routes(valid_token: UserAPIKeyAuth) -> Collection[str]:
        """``team_allowed_routes`` for team tokens built by JWT auth; JWT-mapped virtual keys stay key-scoped."""
        if valid_token.jwt_claims is None or valid_token.token is not None or valid_token.team_id is None:
            return ()

        from litellm.proxy.proxy_server import litellm_jwtauth

        return litellm_jwtauth.team_allowed_routes

    @staticmethod
    def check_passthrough_route_access(route: str, user_api_key_dict: UserAPIKeyAuth) -> bool:
        """The key's ``allowed_passthrough_routes`` when it lists any, otherwise the team's, matched by prefix."""
        return auth_allowed_routes_grant(
            route,
            key=_allowed_passthrough_routes(user_api_key_dict.metadata),
            team=_allowed_passthrough_routes(user_api_key_dict.team_metadata),
        )

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

    @staticmethod
    def _check_proxy_admin_viewer_access(
        route: str,
        _user_role: str,
        request_data: dict,
        request: Request | None = None,
    ) -> None:
        """A proxy admin viewer reads everything a proxy admin reads and never writes, decided by gateway-auth-core."""
        denial: Final = auth_viewer_route_denial(route_facts(route, request, request_data, None, None))
        if denial is not None:
            _raise(denial)


@dataclass(frozen=True, slots=True)
class RouteFacts:
    """What gateway-auth-core's route checks read from the request, the user row and the token. python-bridge
    reads it by attribute."""

    route: str
    method: str | None
    route_template: str | None
    query_user_id: str | None
    body_keys: Sequence[str]
    organization_id: str | None
    organizations: Sequence[str] | None
    team_id: str | None
    is_llm_api_route: bool
    auth_enforced_pass_through: bool
    forwarded_routes: Sequence[str]
    admin_only_routes: Sequence[str]
    has_user: bool
    user_id: str | None
    user_role: str | None
    organization_memberships: Sequence[tuple[str, str | None]] | None
    has_token: bool
    token_user_id: str | None
    can_read_spend: bool
    is_team_service_account: bool
    allowed_routes: Sequence[str] | None
    key_passthrough_routes: Sequence[str] | None
    team_passthrough_routes: Sequence[str] | None
    denied_passthrough_routes: Sequence[str]
    jwt_team_allowed_routes: Sequence[str] | None


def _raise(denial: RustDenial) -> NoReturn:
    from litellm.proxy.auth.denial import (
        denial_error,  # noqa: PLC0415  # denial imports auth_checks, which imports this module
    )

    raise denial_error(denial)


_STRINGS: Final = TypeAdapter(tuple[str, ...])


def _strings(value: object) -> tuple[str, ...] | None:
    return _STRINGS.validate_python(value) if isinstance(value, list | tuple) else None


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def _organization_id(request_data: Mapping[str, object]) -> str | None:
    return _text(request_data.get("organization_id"))


def _admin_only_routes() -> tuple[str, ...]:
    from litellm.proxy.proxy_server import general_settings, premium_user  # noqa: PLC0415  # rebound on config load

    if "admin_only_routes" not in general_settings:
        return ()
    if premium_user is not True:
        verbose_proxy_logger.error(
            "Trying to use 'admin_only_routes' this is an Enterprise only feature. %s",
            CommonProxyErrors.not_premium_user.value,
        )
        return ()
    return tuple(general_settings["admin_only_routes"])


def _query_user_id(request: Request | None) -> str | None:
    if request is None:
        return None
    try:
        return request.query_params.get("user_id")
    except (AttributeError, KeyError):
        return None


def _role_text(role: object) -> str | None:
    if role is None:
        return None
    return role.value if isinstance(role, LitellmUserRoles) else str(role)


def route_facts(
    route: str,
    request: Request | None,
    request_data: Mapping[str, object],
    valid_token: UserAPIKeyAuth | None,
    user_obj: LiteLLM_UserTable | None,
    *,
    user_role: LitellmUserRoles | str | None = None,
) -> RouteFacts:
    """`user_role` stands in for the user row's role when a caller checks a role without a row."""
    from litellm.proxy.auth.auth_utils import (  # noqa: PLC0415  # auth_utils imports this module
        get_request_route_template,
    )

    method: Final = RouteChecks.get_request_method(request)
    auth_enforced: Final = RouteChecks.is_auth_enforced_pass_through_route(route=route, method=method)
    memberships: Final = None if user_obj is None else user_obj.organization_memberships
    role: Final = _role_text(user_role if user_obj is None else user_obj.user_role)
    team_id: Final = request_data.get("team_id")
    jwt_team_routes: Final = () if valid_token is None else RouteChecks._jwt_team_allowed_routes(valid_token)
    key_metadata: Final = {} if valid_token is None else valid_token.metadata or {}
    team_metadata: Final = {} if valid_token is None else valid_token.team_metadata or {}
    return RouteFacts(
        route=route,
        method=method,
        route_template=None if request is None else get_request_route_template(request),
        query_user_id=_query_user_id(request),
        body_keys=tuple(str(key) for key in request_data),
        organization_id=_organization_id(request_data),
        organizations=_strings(request_data.get("organizations")),
        team_id=team_id if isinstance(team_id, str) else None,
        is_llm_api_route=RouteChecks.is_llm_api_route(route=route),
        auth_enforced_pass_through=auth_enforced,
        forwarded_routes=RouteChecks.forwarded_pass_through_routes(route) if auth_enforced else (),
        admin_only_routes=_admin_only_routes(),
        has_user=user_obj is not None or role is not None,
        user_id=None if user_obj is None else user_obj.user_id,
        user_role=role,
        organization_memberships=None
        if memberships is None
        else tuple((membership.organization_id, membership.user_role) for membership in memberships),
        has_token=valid_token is not None,
        token_user_id=None if valid_token is None else valid_token.user_id,
        can_read_spend=valid_token is not None
        and "get_spend_routes" in (getattr(valid_token, "permissions", None) or []),
        is_team_service_account=valid_token is not None and valid_token.is_team_service_account is True,
        allowed_routes=None if valid_token is None else _strings(valid_token.allowed_routes),
        key_passthrough_routes=_strings(key_metadata.get("allowed_passthrough_routes")),
        team_passthrough_routes=_strings(team_metadata.get("allowed_passthrough_routes")),
        denied_passthrough_routes=tuple(
            route
            for metadata in (key_metadata, team_metadata)
            for route in _strings(metadata.get("denied_passthrough_routes"))
            or ()  # comprehension-ok: the key's deny list then the team's
        ),
        jwt_team_allowed_routes=tuple(jwt_team_routes) if jwt_team_routes else None,
    )

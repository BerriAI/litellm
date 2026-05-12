import re
from typing import List, Optional

from fastapi import HTTPException, Request, status

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    CommonProxyErrors,
    KeyManagementRoutes,
    LiteLLM_UserTable,
    LiteLLMRoutes,
    LitellmUserRoles,
    UserAPIKeyAuth,
)

from .auth_checks_organization import _user_is_org_admin

# Management write routes denied to PROXY_ADMIN_VIEW_ONLY. Adding a new write
# endpoint to a management router REQUIRES adding it here too — the surrounding
# check falls through to "allow" if the route is not matched, which previously
# let view-only admins call /team/block, /team/unblock, /key/bulk_update, etc.
_PROXY_ADMIN_VIEW_ONLY_BLOCKED_ROUTES = frozenset(
    [
        # user
        "/user/new",
        "/user/delete",
        "/user/bulk_update",
        # team
        "/team/new",
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
_PROXY_ADMIN_VIEW_ONLY_BLOCKED_KEY_SUFFIXES = ("/regenerate", "/reset_spend")


class RouteChecks:
    @staticmethod
    def should_call_route(route: str, valid_token: UserAPIKeyAuth):
        """
        Check if management route is disabled and raise exception
        """
        try:
            from litellm_enterprise.proxy.auth.route_checks import EnterpriseRouteChecks

            EnterpriseRouteChecks.should_call_route(route=route)
        except HTTPException as e:
            raise e
        except Exception:
            pass

        # Check if Virtual Key is allowed to call the route - Applies to all Roles
        RouteChecks.is_virtual_key_allowed_to_call_route(
            route=route, valid_token=valid_token
        )
        return True

    @staticmethod
    def is_virtual_key_allowed_to_call_route(
        route: str, valid_token: UserAPIKeyAuth
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

        # explicit check for allowed routes (exact match or prefix match)
        for allowed_route in valid_token.allowed_routes:
            if RouteChecks._route_matches_allowed_route(
                route=route, allowed_route=allowed_route
            ):
                return True

        ## check if 'allowed_route' is a field name in LiteLLMRoutes
        if any(
            allowed_route in LiteLLMRoutes._member_names_
            for allowed_route in valid_token.allowed_routes
        ):
            for allowed_route in valid_token.allowed_routes:
                if allowed_route in LiteLLMRoutes._member_names_:
                    if RouteChecks.check_route_access(
                        route=route,
                        allowed_routes=LiteLLMRoutes._member_map_[allowed_route].value,
                    ):
                        return True

                    ################################################
                    #  For llm_api_routes, also check registered pass-through endpoints
                    ################################################
                    if allowed_route == "llm_api_routes":
                        from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
                            InitPassThroughEndpointHelpers,
                        )

                        if InitPassThroughEndpointHelpers.is_registered_pass_through_route(
                            route=route
                        ):
                            return True

        # check if wildcard pattern is allowed
        for allowed_route in valid_token.allowed_routes:
            if RouteChecks._route_matches_wildcard_pattern(
                route=route, pattern=allowed_route
            ):
                return True

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Virtual key is not allowed to call this route. Only allowed to call routes: {valid_token.allowed_routes}. Tried to call route: {route}",
        )

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
        masker = SensitiveDataMasker(visible_prefix=6, visible_suffix=2, mask_char="*")

        return masker._mask_value(user_id)

    @staticmethod
    def _raise_admin_only_route_exception(
        user_obj: Optional[LiteLLM_UserTable],
        route: str,
    ) -> None:
        """
        Raise exception for routes that require proxy admin access

        Args:
            user_obj (Optional[LiteLLM_UserTable]): The user object
            route (str): The route being accessed

        Raises:
            Exception: With user role and masked user_id information
        """
        user_role = "unknown"
        user_id = "unknown"
        if user_obj is not None:
            user_role = user_obj.user_role or "unknown"
            user_id = user_obj.user_id or "unknown"

        masked_user_id = RouteChecks._mask_user_id(user_id)
        raise Exception(
            f"Only proxy admin can be used to generate, delete, update info for new keys/users/teams. Route={route}. Your role={user_role}. Your user_id={masked_user_id}"
        )

    @staticmethod
    def non_proxy_admin_allowed_routes_check(
        user_obj: Optional[LiteLLM_UserTable],
        _user_role: Optional[LitellmUserRoles],
        route: str,
        request: Request,
        valid_token: UserAPIKeyAuth,
        request_data: dict,
    ):
        """
        Checks if Non Proxy Admin User is allowed to access the route
        """

        # Check user has defined custom admin routes
        RouteChecks.custom_admin_only_route_check(
            route=route,
        )

        if RouteChecks.is_llm_api_route(route=route):
            pass
        elif RouteChecks.is_info_route(route=route):
            # check if user allowed to call an info route
            if route == "/key/info":
                # handled by function itself
                pass
            elif route == "/user/info":
                # check if user can access this route
                query_params = request.query_params
                user_id = query_params.get("user_id")
                verbose_proxy_logger.debug(
                    f"user_id: {user_id} & valid_token.user_id: {valid_token.user_id}"
                )
                if user_id and user_id != valid_token.user_id:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="key not allowed to access this user's info. user_id={}, key's user_id={}".format(
                            user_id, valid_token.user_id
                        ),
                    )
            elif route == "/v2/user/info":
                # handled by the endpoint itself (full RBAC in handler)
                pass
            elif route == "/model/info":
                # /model/info just shows models user has access to
                pass
            elif route == "/team/info":
                pass  # handled by function itself
        elif (
            route in LiteLLMRoutes.global_spend_tracking_routes.value
            and getattr(valid_token, "permissions", None) is not None
            and "get_spend_routes" in getattr(valid_token, "permissions", [])
        ):
            pass
        elif _user_role == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value:
            RouteChecks._check_proxy_admin_viewer_access(
                route=route,
                _user_role=_user_role,
                request_data=request_data,
                request=request,
            )
        elif (
            _user_role == LitellmUserRoles.INTERNAL_USER.value
            and RouteChecks.check_route_access(
                route=route, allowed_routes=LiteLLMRoutes.internal_user_routes.value
            )
        ):
            pass
        elif _user_is_org_admin(
            request_data=request_data, user_object=user_obj
        ) and RouteChecks.check_route_access(
            route=route, allowed_routes=LiteLLMRoutes.org_admin_allowed_routes.value
        ):
            pass
        elif (
            _user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value
            and RouteChecks.check_route_access(
                route=route,
                allowed_routes=LiteLLMRoutes.internal_user_view_only_routes.value,
            )
        ):
            pass
        elif RouteChecks.check_route_access(
            route=route, allowed_routes=LiteLLMRoutes.self_managed_routes.value
        ):  # routes that manage their own allowed/disallowed logic
            pass
        elif route.startswith("/v1/mcp/") or route.startswith("/mcp-rest/"):
            pass  # authN/authZ handled by api itself
        elif RouteChecks.check_passthrough_route_access(
            route=route, user_api_key_dict=valid_token
        ):
            pass
        elif valid_token.allowed_routes is not None:
            # check if route is in allowed_routes (exact match or prefix match)
            route_allowed = False
            for allowed_route in valid_token.allowed_routes:
                if RouteChecks._route_matches_allowed_route(
                    route=route, allowed_route=allowed_route
                ):
                    route_allowed = True
                    break

                if RouteChecks._route_matches_wildcard_pattern(
                    route=route, pattern=allowed_route
                ):
                    route_allowed = True
                    break

            if not route_allowed:
                RouteChecks._raise_admin_only_route_exception(
                    user_obj=user_obj, route=route
                )
        else:
            RouteChecks._raise_admin_only_route_exception(
                user_obj=user_obj, route=route
            )

    @staticmethod
    def custom_admin_only_route_check(route: str):
        from litellm.proxy.proxy_server import general_settings, premium_user

        if "admin_only_routes" in general_settings:
            if premium_user is not True:
                verbose_proxy_logger.error(
                    f"Trying to use 'admin_only_routes' this is an Enterprise only feature. {CommonProxyErrors.not_premium_user.value}"
                )
                return
            if route in general_settings["admin_only_routes"]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"user not allowed to access this route. Route={route} is an admin only route",
                )
        pass

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

        if RouteChecks.check_route_access(
            route=route, allowed_routes=LiteLLMRoutes.mcp_inference_routes.value
        ):
            return True

        if RouteChecks.check_route_access(
            route=route, allowed_routes=LiteLLMRoutes.agent_routes.value
        ):
            return True

        if route in LiteLLMRoutes.litellm_native_routes.value:
            return True

        # fuzzy match routes like "/v1/threads/thread_49EIN5QF32s4mH20M7GFKdlZ"
        # Check for routes with placeholders or wildcard patterns
        for openai_route in LiteLLMRoutes.openai_routes.value:
            # Replace placeholders with regex pattern
            # placeholders are written as "/threads/{thread_id}"
            if "{" in openai_route:
                if RouteChecks._route_matches_pattern(
                    route=route, pattern=openai_route
                ):
                    return True
            # Check for wildcard patterns like "/containers/*"
            if RouteChecks._is_wildcard_pattern(pattern=openai_route):
                if RouteChecks._route_matches_wildcard_pattern(
                    route=route, pattern=openai_route
                ):
                    return True

        # Check for Google routes with placeholders like "/v1beta/models/{model_name}:generateContent"
        for google_route in LiteLLMRoutes.google_routes.value:
            if "{" in google_route:
                if RouteChecks._route_matches_pattern(
                    route=route, pattern=google_route
                ):
                    return True

        # Check for Anthropic routes with placeholders
        for anthropic_route in LiteLLMRoutes.anthropic_routes.value:
            if "{" in anthropic_route:
                if RouteChecks._route_matches_pattern(
                    route=route, pattern=anthropic_route
                ):
                    return True

        if RouteChecks._is_azure_openai_route(route=route):
            return True

        for _llm_passthrough_route in LiteLLMRoutes.mapped_pass_through_routes.value:
            if _llm_passthrough_route in route:
                return True
        return False

    @staticmethod
    def is_management_route(route: str) -> bool:
        """
        Check if route is a management route
        """
        return RouteChecks.check_route_access(
            route=route, allowed_routes=LiteLLMRoutes.management_routes.value
        )

    @staticmethod
    def is_info_route(route: str) -> bool:
        """
        Check if route is an info route
        """
        return route in LiteLLMRoutes.info_routes.value

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
        deployment_pattern = r"^/openai/deployments/[^/]+/[^/]+/chat/completions$"
        engine_pattern = r"^/engines/[^/]+/chat/completions$"

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
            placeholder = match.group(0).strip("{}")
            if placeholder.endswith(":path"):
                # allow "/" in the placeholder value, but don't eat the route suffix after ":"
                return r"[^:]+"
            return r"[^/]+"

        pattern = re.sub(r"\{[^}]+\}", _placeholder_to_regex, pattern)
        # Anchor the pattern to match the entire string
        pattern = f"^{pattern}$"
        if re.match(pattern, route):
            return True
        return False

    @staticmethod
    def _is_wildcard_pattern(pattern: str) -> bool:
        """
        Check if pattern is a wildcard pattern
        """
        return pattern.endswith("*")

    @staticmethod
    def _route_matches_wildcard_pattern(route: str, pattern: str) -> bool:
        """
        Check if route matches the wildcard pattern

        eg.

        pattern: "/scim/v2/*"
        route: "/scim/v2/Users"
        - returns: True

        pattern: "/scim/v2/*"
        route: "/chat/completions"
        - returns: False


        pattern: "/scim/v2/*"
        route: "/scim/v2/Users/123"
        - returns: True

        """
        if pattern.endswith("*"):
            # Get the prefix (everything before the wildcard)
            prefix = pattern[:-1]
            return route.startswith(prefix)
        else:
            # If there's no wildcard, the pattern and route should match exactly
            return route == pattern

    @staticmethod
    def _route_matches_allowed_route(route: str, allowed_route: str) -> bool:
        """
        Check if route matches the allowed_route pattern.
        Supports both exact match and prefix match.

        Examples:
        - allowed_route="/fake-openai-proxy-6", route="/fake-openai-proxy-6" -> True (exact match)
        - allowed_route="/fake-openai-proxy-6", route="/fake-openai-proxy-6/v1/chat/completions" -> True (prefix match)
        - allowed_route="/fake-openai-proxy-6", route="/fake-openai-proxy-600" -> False (not a valid prefix)

        Args:
            route: The actual route being accessed
            allowed_route: The allowed route pattern

        Returns:
            bool: True if route matches (exact or prefix), False otherwise
        """
        # Exact match
        if route == allowed_route:
            return True
        # Prefix match - ensure we add "/" to prevent false matches like /fake-openai-proxy-600
        if route.startswith(allowed_route + "/"):
            return True
        return False

    @staticmethod
    def check_route_access(route: str, allowed_routes: List[str]) -> bool:
        """
        Check if a route has access by checking both exact matches and patterns

        Args:
            route (str): The route to check
            allowed_routes (list): List of allowed routes/patterns

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
        wildcard_allowed_routes = [
            route
            for route in allowed_routes
            if RouteChecks._is_wildcard_pattern(pattern=route)
        ]
        for allowed_route in wildcard_allowed_routes:
            if RouteChecks._route_matches_wildcard_pattern(
                route=route, pattern=allowed_route
            ):
                return True

        #########################################################
        # pattern match route is in allowed_routes
        # pattern: "/threads/{thread_id}"
        # route: "/threads/thread_49EIN5QF32s4mH20M7GFKdlZ"
        # returns: True
        #########################################################
        if any(  # Check pattern match
            RouteChecks._route_matches_pattern(route=route, pattern=allowed_route)
            for allowed_route in allowed_routes
        ):
            return True

        return False

    @staticmethod
    def check_passthrough_route_access(
        route: str, user_api_key_dict: UserAPIKeyAuth
    ) -> bool:
        """
        Check if route is a passthrough route.
        Supports both exact match and prefix match.
        """
        metadata = user_api_key_dict.metadata
        team_metadata = user_api_key_dict.team_metadata or {}
        if metadata is None and team_metadata is None:
            return False
        if (
            "allowed_passthrough_routes" not in metadata
            and "allowed_passthrough_routes" not in team_metadata
        ):
            return False
        if (
            metadata.get("allowed_passthrough_routes") is None
            and team_metadata.get("allowed_passthrough_routes") is None
        ):
            return False

        allowed_passthrough_routes = (
            metadata.get("allowed_passthrough_routes")
            or team_metadata.get("allowed_passthrough_routes")
            or []
        )

        # Check if route matches any allowed passthrough route (exact or prefix match)
        for allowed_route in allowed_passthrough_routes:
            if RouteChecks._route_matches_allowed_route(
                route=route, allowed_route=allowed_route
            ):
                return True

        return False

    @staticmethod
    def _is_assistants_api_request(request: Request) -> bool:
        """
        Returns True if `thread` or `assistant` is in the request path

        Args:
            request (Request): The request object

        Returns:
            bool: True if `thread` or `assistant` is in the request path, False otherwise
        """
        if "thread" in request.url.path or "assistant" in request.url.path:
            return True
        return False

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
            "/user/delete",
            "/user/bulk_update",
            "/team/new",
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
        request: Optional[Request] = None,
    ) -> None:
        """
        Check access for PROXY_ADMIN_VIEW_ONLY role.

        Admin Viewer follows a read-parity-with-Proxy-Admin rule: anything Proxy
        Admin can read/list/get, Admin Viewer can read/list/get. The only
        exclusions are cost-incurring inference routes (Playground, /chat/
        completions, etc.) and any state-mutating request.

        Implementation:
          1. LLM/inference routes → 403 (cost-incurring).
          2. Safe HTTP method (GET/HEAD/OPTIONS) → allow by default. This is
             the read-parity guarantee — every new GET endpoint added anywhere
             in the codebase is automatically readable by Admin Viewer
             without needing to remember to add it to an allowlist.
          3. Unsafe HTTP method (POST/PUT/PATCH/DELETE):
             - Allow `/user/update` only when restricted to user_email/password.
             - Block all explicit writes in `_ADMIN_VIEWER_BLOCKED_WRITE_ROUTES`.
             - Otherwise allow only if the route is in admin_viewer_routes /
               global_spend_tracking_routes (legacy explicit-allow set).
             - Else 403.
        """
        if RouteChecks.is_llm_api_route(route=route):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"user not allowed to access this OpenAI routes, role= {_user_role}",
            )

        # Check if this is a write operation on management routes
        if RouteChecks.check_route_access(
            route=route, allowed_routes=LiteLLMRoutes.management_routes.value
        ):
            # For management routes, only allow read operations or specific allowed updates
            if route == "/user/update":
                # Check the Request params are valid for PROXY_ADMIN_VIEW_ONLY
                if request_data is not None and isinstance(request_data, dict):
                    _params_updated = request_data.keys()
                    for param in _params_updated:
                        if param not in ["user_email", "password"]:
                            raise HTTPException(
                                status_code=status.HTTP_403_FORBIDDEN,
                                detail=f"user not allowed to access this route, role= {_user_role}. Trying to access: {route} and updating invalid param: {param}. only user_email and password can be updated",
                            )
            elif route in _PROXY_ADMIN_VIEW_ONLY_BLOCKED_ROUTES or (
                route.startswith("/key/")
                and route.endswith(_PROXY_ADMIN_VIEW_ONLY_BLOCKED_KEY_SUFFIXES)
            ):
                # Block write operations for PROXY_ADMIN_VIEW_ONLY
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"user not allowed to access this route, role= {_user_role}. Trying to access: {route}",
                )
            # Allow read operations on management routes (like /user/info, /team/info, /model/info)
        method = request.method.upper() if request is not None else "GET"
        is_safe_method = method in RouteChecks._SAFE_HTTP_METHODS

        # ── Safe HTTP method: default-allow ──────────────────────────────
        if is_safe_method:
            return

        # ── Unsafe HTTP method: explicit checks ──────────────────────────
        # Allow `/user/update` for self-service email / password change.
        if route == "/user/update":
            if request_data is not None and isinstance(request_data, dict):
                for param in request_data.keys():
                    if param not in ["user_email", "password"]:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=(
                                f"user not allowed to access this route, role= {_user_role}. "
                                f"Trying to access: {route} and updating invalid param: {param}. "
                                "only user_email and password can be updated"
                            ),
                        )
            return

        # Hard-block known write routes regardless of HTTP method (defensive
        # — these are POSTs in practice, but pinning them here protects
        # against future GET-shaped writes).
        if route in RouteChecks._ADMIN_VIEWER_BLOCKED_WRITE_ROUTES or (
            route.startswith("/key/") and route.endswith("/regenerate")
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"user not allowed to access this route, role= {_user_role}. Trying to access: {route}",
            )

        # Legacy explicit-allow sets (kept for routes that are POST but
        # semantically read-only, e.g. /spend/calculate). Both admin_viewer_routes
        # and global_spend_tracking_routes are reads/listings.
        if RouteChecks.check_route_access(
            route=route, allowed_routes=LiteLLMRoutes.admin_viewer_routes.value
        ):
            return
        if RouteChecks.check_route_access(
            route=route, allowed_routes=LiteLLMRoutes.global_spend_tracking_routes.value
        ):
            return

        # NOTE: We intentionally do NOT fall back to allowing all
        # `management_routes`. That set is a mix of reads (info/list — handled
        # via the safe-method branch above) and writes (`/team/block`,
        # `/team/permissions_update`, `/jwt/key/mapping/{new,update,delete}`,
        # `/key/bulk_update`, `/key/{id}/reset_spend`). A blanket allow would
        # let Admin Viewer POST these write endpoints — violating the
        # "no writes, ever" rule. Default-deny instead.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"user not allowed to access this route, role= {_user_role}. Trying to access: {route}",
        )

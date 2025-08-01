import re
from typing import List, Optional

from fastapi import HTTPException, Request, status

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    CommonProxyErrors,
    LiteLLM_UserTable,
    LiteLLMRoutes,
    LitellmUserRoles,
    UserAPIKeyAuth,
)

from .auth_checks_organization import _user_is_org_admin


class RouteChecks:
    @staticmethod
    def should_call_route(route: str, valid_token: UserAPIKeyAuth):
        """
        Check if management route is disabled and raise exception
        """
        try:
            from litellm_enterprise.proxy.auth.route_checks import EnterpriseRouteChecks

            EnterpriseRouteChecks.should_call_route(route=route)
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

        # explicit check for allowed routes
        if route in valid_token.allowed_routes:
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

        # check if wildcard pattern is allowed
        for allowed_route in valid_token.allowed_routes:
            if RouteChecks._route_matches_wildcard_pattern(
                route=route, pattern=allowed_route
            ):
                return True

        raise Exception(
            f"Virtual key is not allowed to call this route. Only allowed to call routes: {valid_token.allowed_routes}. Tried to call route: {route}"
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
        elif (
            route in LiteLLMRoutes.info_routes.value
        ):  # check if user allowed to call an info route
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

            if RouteChecks.is_llm_api_route(route=route):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"user not allowed to access this OpenAI routes, role= {_user_role}",
                )
            if RouteChecks.check_route_access(
                route=route, allowed_routes=LiteLLMRoutes.management_routes.value
            ):

                # the Admin Viewer is only allowed to call /user/update for their own user_id and can only update
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
                else:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"user not allowed to access this route, role= {_user_role}. Trying to access: {route}",
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"user not allowed to access this route, role= {_user_role}. Trying to access: {route}",
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
        elif route.startswith("/v1/mcp/"):
            pass  # authN/authZ handled by api itself
        else:
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

        if route in LiteLLMRoutes.openai_routes.value:
            return True

        if route in LiteLLMRoutes.anthropic_routes.value:
            return True

        if RouteChecks.check_route_access(
            route=route, allowed_routes=LiteLLMRoutes.mcp_routes.value
        ):
            return True

        # fuzzy match routes like "/v1/threads/thread_49EIN5QF32s4mH20M7GFKdlZ"
        # Check for routes with placeholders
        for openai_route in LiteLLMRoutes.openai_routes.value:
            # Replace placeholders with regex pattern
            # placeholders are written as "/threads/{thread_id}"
            if "{" in openai_route:
                if RouteChecks._route_matches_pattern(
                    route=route, pattern=openai_route
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
        return route in LiteLLMRoutes.management_routes.value

    @staticmethod
    def _is_azure_openai_route(route: str) -> bool:
        """
        Check if route is a route from AzureOpenAI SDK client

        eg.
        route='/openai/deployments/vertex_ai/gemini-1.5-flash/chat/completions'
        """
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
        pattern = re.sub(r"\{[^}]+\}", r"[^/]+", pattern)
        # Anchor the pattern to match the entire string
        pattern = f"^{pattern}$"
        if re.match(pattern, route):
            return True
        return False

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
    def check_route_access(route: str, allowed_routes: List[str]) -> bool:
        """
        Check if a route has access by checking both exact matches and patterns

        Args:
            route (str): The route to check
            allowed_routes (list): List of allowed routes/patterns

        Returns:
            bool: True if route is allowed, False otherwise
        """
        return route in allowed_routes or any(  # Check exact match
            RouteChecks._route_matches_pattern(route=route, pattern=allowed_route)
            for allowed_route in allowed_routes
        )  # Check pattern match

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

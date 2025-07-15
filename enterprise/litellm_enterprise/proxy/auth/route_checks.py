from fastapi import HTTPException, status


class EnterpriseRouteChecks:
    @staticmethod
    def is_llm_api_route_disabled() -> bool:
        """
        Check if llm api route is disabled
        """
        from litellm.secret_managers.main import get_secret_bool

        return get_secret_bool("DISABLE_LLM_API_ENDPOINTS") is True

    @staticmethod
    def is_management_routes_disabled() -> bool:
        """
        Check if management route is disabled
        """
        from litellm.secret_managers.main import get_secret_bool

        return get_secret_bool("DISABLE_ADMIN_ENDPOINTS") is True

    @staticmethod
    def should_call_route(route: str):
        """
        Check if management route is disabled and raise exception
        """
        from litellm.proxy.auth.route_checks import RouteChecks

        if (
            RouteChecks.is_management_route(route=route)
            and EnterpriseRouteChecks.is_management_routes_disabled()
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Management routes are disabled for this instance.",
            )
        elif (
            RouteChecks.is_llm_api_route(route=route)
            and EnterpriseRouteChecks.is_llm_api_route_disabled()
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="LLM API routes are disabled for this instance.",
            )

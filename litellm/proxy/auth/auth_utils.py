from litellm._logging import verbose_proxy_logger


def route_in_additonal_public_routes(current_route: str):
    """
    Helper to check if the user defined public_routes on config.yaml

    Parameters:
    - current_route: str - the route the user is trying to call

    Returns:
    - bool - True if the route is defined in public_routes
    - bool - False if the route is not defined in public_routes


    In order to use this the litellm config.yaml should have the following in general_settings:

    ```yaml
    general_settings:
        master_key: sk-1234
        public_routes: ["LiteLLMRoutes.public_routes", "/spend/calculate"]
    ```
    """

    # check if user is premium_user - if not do nothing
    from litellm.proxy._types import LiteLLMRoutes
    from litellm.proxy.proxy_server import general_settings, premium_user

    try:
        if premium_user is not True:
            return False
        # check if this is defined on the config
        if general_settings is None:
            return False

        routes_defined = general_settings.get("public_routes", [])
        if current_route in routes_defined:
            return True

        return False
    except Exception as e:
        verbose_proxy_logger.error(f"route_in_additonal_public_routes: {str(e)}")
        return False

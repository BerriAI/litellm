from typing import Any, Dict

from fastapi import Request

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth


async def handle_oauth2_proxy_request(request: Request) -> UserAPIKeyAuth:
    """
    Handle request from oauth2 proxy.
    """
    from litellm.proxy.proxy_server import general_settings

    verbose_proxy_logger.debug("Handling oauth2 proxy request")
    # Define the OAuth2 config mappings
    oauth2_config_mappings: Dict[str, str] = (
        general_settings.get("oauth2_config_mappings") or {}
    )
    verbose_proxy_logger.debug(f"Oauth2 config mappings: {oauth2_config_mappings}")

    if not oauth2_config_mappings:
        raise ValueError("Oauth2 config mappings not found in general_settings")
    # Initialize a dictionary to store the mapped values
    auth_data: Dict[str, Any] = {}

    # Extract values from headers based on the mappings
    for key, header in oauth2_config_mappings.items():
        value = request.headers.get(header)
        if value:
            # Convert max_budget to float if present
            if key == "max_budget":
                auth_data[key] = float(value)
            # Convert models to list if present
            elif key == "models":
                auth_data[key] = [model.strip() for model in value.split(",")]
            else:
                auth_data[key] = value
    verbose_proxy_logger.debug(
        "Auth data before creating UserAPIKeyAuth object: keys=%s",
        list(auth_data.keys()),
    )
    user_api_key_auth = UserAPIKeyAuth(**auth_data)
    verbose_proxy_logger.debug(
        "UserAPIKeyAuth object created with keys: %s",
        list(user_api_key_auth.__fields_set__),
    )
    # Create and return UserAPIKeyAuth object
    return user_api_key_auth

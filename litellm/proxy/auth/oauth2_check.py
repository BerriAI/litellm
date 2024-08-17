from litellm.proxy._types import UserAPIKeyAuth


async def check_oauth2_token(token: str) -> UserAPIKeyAuth:
    """
    Makes a request to the token info endpoint to validate the OAuth2 token.

    Args:
    token (str): The OAuth2 token to validate.

    Returns:
    Literal[True]: If the token is valid.

    Raises:
    ValueError: If the token is invalid, the request fails, or the token info endpoint is not set.
    """
    import os
    from typing import Literal

    import httpx

    from litellm._logging import verbose_proxy_logger
    from litellm.llms.custom_httpx.http_handler import _get_async_httpx_client
    from litellm.proxy._types import CommonProxyErrors
    from litellm.proxy.proxy_server import premium_user

    if premium_user is not True:
        raise ValueError(
            "Oauth2 token validation is only available for premium users"
            + CommonProxyErrors.not_premium_user.value
        )

    verbose_proxy_logger.debug("Oauth2 token validation for token=%s", token)
    # Get the token info endpoint from environment variable
    token_info_endpoint = os.getenv("OAUTH_TOKEN_INFO_ENDPOINT")
    user_id_field_name = os.environ.get("OAUTH_USER_ID_FIELD_NAME", "sub")
    user_role_field_name = os.environ.get("OAUTH_USER_ROLE_FIELD_NAME", "role")
    user_team_id_field_name = os.environ.get("OAUTH_USER_TEAM_ID_FIELD_NAME", "team_id")

    if not token_info_endpoint:
        raise ValueError("OAUTH_TOKEN_INFO_ENDPOINT environment variable is not set")

    client = _get_async_httpx_client()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        response = await client.get(token_info_endpoint, headers=headers)

        # if it's a bad token we expect it to raise an HTTPStatusError
        response.raise_for_status()

        # If we get here, the request was successful
        data = response.json()

        verbose_proxy_logger.debug(
            "Oauth2 token validation for token=%s, response from /token/info=%s",
            token,
            data,
        )

        # You might want to add additional checks here based on the response
        # For example, checking if the token is expired or has the correct scope
        user_id = data.get(user_id_field_name)
        user_team_id = data.get(user_team_id_field_name)
        user_role = data.get(user_role_field_name)

        return UserAPIKeyAuth(
            api_key=token,
            team_id=user_team_id,
            user_id=user_id,
            user_role=user_role,
        )
    except httpx.HTTPStatusError as e:
        # This will catch any 4xx or 5xx errors
        raise ValueError(f"Oauth 2.0 Token validation failed: {e}")
    except Exception as e:
        # This will catch any other errors (like network issues)
        raise ValueError(f"An error occurred during token validation: {e}")

import os
from typing import Literal

import httpx

from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_async_httpx_client,
    _get_httpx_client,
)


async def check_oauth2_token(token: str) -> Literal[True]:
    """
    Makes a request to the token info endpoint to validate the OAuth2 token.

    Args:
    token (str): The OAuth2 token to validate.

    Returns:
    Literal[True]: If the token is valid.

    Raises:
    ValueError: If the token is invalid, the request fails, or the token info endpoint is not set.
    """
    # Get the token info endpoint from environment variable
    token_info_endpoint = os.getenv("OAUTH_TOKEN_INFO_ENDPOINT")

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

        # You might want to add additional checks here based on the response
        # For example, checking if the token is expired or has the correct scope

        return True
    except httpx.HTTPStatusError as e:
        # This will catch any 4xx or 5xx errors
        raise ValueError(f"Token validation failed: {e}")
    except Exception as e:
        # This will catch any other errors (like network issues)
        raise ValueError(f"An error occurred during token validation: {e}")

import os
from typing import Optional, Union

import httpx

try:
    from litellm._version import version
except Exception:
    version = "0.0.0"

def get_default_headers() -> dict:
    """
    Get default headers for HTTP requests.

    - Default: `User-Agent: litellm/{version}`
    - Override: set `LITELLM_USER_AGENT` to fully override the header value.
    """
    user_agent = os.environ.get("LITELLM_USER_AGENT")
    if user_agent is not None:
        return {"User-Agent": user_agent}

    return {"User-Agent": f"litellm/{version}"}

class HTTPHandler:
    def __init__(self, concurrent_limit=1000):
        headers = get_default_headers()
        # Create a client with a connection pool
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=concurrent_limit,
                max_keepalive_connections=concurrent_limit,
            ),
            headers=headers,
        )

    async def close(self):
        # Close the client when you're done with it
        await self.client.aclose()

    async def get(
        self, url: str, params: Optional[dict] = None, headers: Optional[dict] = None
    ):
        response = await self.client.get(url, params=params, headers=headers)
        return response

    async def post(
        self,
        url: str,
        data: Optional[Union[dict, str]] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
    ):
        try:
            response = await self.client.post(
                url, data=data, params=params, headers=headers  # type: ignore
            )
            return response
        except Exception as e:
            raise e

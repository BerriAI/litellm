import os
from collections.abc import Mapping, Sequence
from typing import Final

import httpx

try:
    from litellm._version import version
except Exception:
    version = "0.0.0"


def get_default_headers() -> dict[str, str]:
    """
    Get default headers for HTTP requests.

    - Default: `User-Agent: litellm/{version}`
    - Override: set `LITELLM_USER_AGENT` to fully override the header value.
    """
    user_agent: Final = os.environ.get("LITELLM_USER_AGENT")
    if user_agent is not None:
        return {"User-Agent": user_agent}

    return {"User-Agent": f"litellm/{version}"}


class HTTPHandler:
    def __init__(self, concurrent_limit: int = 1000):
        headers: Final = get_default_headers()
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
        self,
        url: str,
        params: Mapping[str, str | int | float | bool | Sequence[str | int | float | bool | None] | None] | None = None,
        headers: Mapping[str, str] | None = None,
    ):
        response: Final = await self.client.get(url, params=params, headers=headers)
        return response

    async def post(
        self,
        url: str,
        data: Mapping[str, object] | str | None = None,
        params: Mapping[str, str | int | float | bool | Sequence[str | int | float | bool | None] | None] | None = None,
        headers: Mapping[str, str] | None = None,
    ):
        try:
            response: Final = await self.client.post(
                url,
                data=data,
                params=params,
                headers=headers,
            )
            return response
        except Exception as e:
            raise e

import httpx, asyncio
from typing import Optional, Union, Mapping, Any

# https://www.python-httpx.org/advanced/timeouts
_DEFAULT_TIMEOUT = httpx.Timeout(timeout=5.0, connect=5.0)


class AsyncHTTPHandler:
    def __init__(
        self, timeout: httpx.Timeout = _DEFAULT_TIMEOUT, concurrent_limit=1000
    ):
        # Create a client with a connection pool
        self.client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=concurrent_limit,
                max_keepalive_connections=concurrent_limit,
            ),
        )

    async def close(self):
        # Close the client when you're done with it
        await self.client.aclose()

    async def __aenter__(self):
        return self.client

    async def __aexit__(self):
        # close the client when exiting
        await self.client.aclose()

    async def get(
        self, url: str, params: Optional[dict] = None, headers: Optional[dict] = None
    ):
        response = await self.client.get(url, params=params, headers=headers)
        return response

    async def post(
        self,
        url: str,
        data: Optional[Union[dict, str]] = None,  # type: ignore
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
    ):
        response = await self.client.post(
            url,
            data=data,  # type: ignore
            params=params,
            headers=headers,
        )
        return response

    def __del__(self) -> None:
        try:
            asyncio.get_running_loop().create_task(self.close())
        except Exception:
            pass


class HTTPHandler:
    def __init__(self, concurrent_limit=1000):
        # Create a client with a connection pool
        self.client = httpx.Client(
            limits=httpx.Limits(
                max_connections=concurrent_limit,
                max_keepalive_connections=concurrent_limit,
            )
        )

    def close(self):
        # Close the client when you're done with it
        self.client.close()

    def get(
        self, url: str, params: Optional[dict] = None, headers: Optional[dict] = None
    ):
        response = self.client.get(url, params=params, headers=headers)
        return response

    def post(
        self,
        url: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
    ):
        response = self.client.post(url, data=data, params=params, headers=headers)
        return response

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

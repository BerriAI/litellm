import asyncio
from typing import Callable, Union

import httpx
from aiohttp.client import ClientSession
from httpx_aiohttp import AiohttpTransport

from litellm._logging import verbose_logger


class LiteLLMAiohttpTransport(AiohttpTransport):
    """
    LiteLLM wrapper around AiohttpTransport to handle %-encodings in URLs
    and event loop lifecycle issues in CI/CD environments

    Credit to: https://github.com/karpetrosyan/httpx-aiohttp for this implementation
    """

    def __init__(self, client: Union[ClientSession, Callable[[], ClientSession]]):
        self.client = client
        super().__init__(client=client)
        # Store the client factory for recreating sessions when needed
        if callable(client):
            self._client_factory = client

    def _get_valid_client_session(self) -> ClientSession:
        """
        Helper to get a valid ClientSession for the current event loop.

        This handles the case where the session was created in a different
        event loop that may have been closed (common in CI/CD environments).
        """
        from aiohttp.client import ClientSession

        # If we don't have a client or it's not a ClientSession, create one
        if not isinstance(self.client, ClientSession):
            if hasattr(self, "_client_factory") and callable(self._client_factory):
                self.client = self._client_factory()
            else:
                self.client = ClientSession()
            return self.client

        # Check if the existing session is still valid for the current event loop
        try:
            session_loop = getattr(self.client, "_loop", None)
            current_loop = asyncio.get_running_loop()

            # If session is from a different or closed loop, recreate it
            if (
                session_loop is None
                or session_loop != current_loop
                or session_loop.is_closed()
            ):
                # Clean up the old session
                try:
                    # Note: not awaiting close() here as it might be from a different loop
                    # The session will be garbage collected
                    pass
                except Exception as e:
                    verbose_logger.debug(f"Error closing old session: {e}")
                    pass

                # Create a new session in the current event loop
                if hasattr(self, "_client_factory") and callable(self._client_factory):
                    self.client = self._client_factory()
                else:
                    self.client = ClientSession()

        except (RuntimeError, AttributeError):
            # If we can't check the loop or session is invalid, recreate it
            if hasattr(self, "_client_factory") and callable(self._client_factory):
                self.client = self._client_factory()
            else:
                self.client = ClientSession()

        return self.client

    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        from aiohttp import ClientTimeout
        from httpx_aiohttp.transport import (
            AiohttpResponseStream,
            map_aiohttp_exceptions,
        )
        from yarl import URL as YarlURL

        timeout = request.extensions.get("timeout", {})
        sni_hostname = request.extensions.get("sni_hostname")

        # Use helper to ensure we have a valid session for the current event loop
        client_session = self._get_valid_client_session()

        with map_aiohttp_exceptions():
            try:
                data = request.content
            except httpx.RequestNotRead:
                data = request.stream  # type: ignore
                request.headers.pop("transfer-encoding", None)  # handled by aiohttp

            response = await client_session.request(
                method=request.method,
                url=YarlURL(str(request.url), encoded=True),
                headers=request.headers,
                data=data,
                allow_redirects=False,
                auto_decompress=False,
                compress=False,
                timeout=ClientTimeout(
                    sock_connect=timeout.get("connect"),
                    sock_read=timeout.get("read"),
                    connect=timeout.get("pool"),
                ),
                server_hostname=sni_hostname,
            ).__aenter__()

        return httpx.Response(
            status_code=response.status,
            headers=response.headers,
            content=AiohttpResponseStream(response),
            request=request,
        )

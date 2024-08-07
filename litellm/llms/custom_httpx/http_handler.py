import asyncio
import os
import traceback
from typing import Any, Mapping, Optional, Union

import httpx

import litellm

try:
    from litellm._version import version
except:
    version = "0.0.0"

headers = {
    "User-Agent": f"litellm/{version}",
}

# https://www.python-httpx.org/advanced/timeouts
_DEFAULT_TIMEOUT = httpx.Timeout(timeout=5.0, connect=5.0)


class AsyncHTTPHandler:
    def __init__(
        self,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        concurrent_limit=1000,
    ):
        self.timeout = timeout
        self.client = self.create_client(
            timeout=timeout, concurrent_limit=concurrent_limit
        )

    def create_client(
        self, timeout: Optional[Union[float, httpx.Timeout]], concurrent_limit: int
    ) -> httpx.AsyncClient:

        # Check if the HTTP_PROXY and HTTPS_PROXY environment variables are set and use them accordingly.
        ssl_verify = bool(os.getenv("SSL_VERIFY", litellm.ssl_verify))
        cert = os.getenv(
            "SSL_CERTIFICATE", litellm.ssl_certificate
        )  # /path/to/client.pem

        if timeout is None:
            timeout = _DEFAULT_TIMEOUT
        # Create a client with a connection pool

        return httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=concurrent_limit,
                max_keepalive_connections=concurrent_limit,
            ),
            verify=ssl_verify,
            cert=cert,
            headers=headers,
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
        json: Optional[dict] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        stream: bool = False,
    ):
        try:
            if timeout is None:
                timeout = self.timeout
            req = self.client.build_request(
                "POST", url, data=data, json=json, params=params, headers=headers, timeout=timeout  # type: ignore
            )
            response = await self.client.send(req, stream=stream)
            response.raise_for_status()
            return response
        except (httpx.RemoteProtocolError, httpx.ConnectError):
            # Retry the request with a new session if there is a connection error
            new_client = self.create_client(timeout=timeout, concurrent_limit=1)
            try:
                return await self.single_connection_post_request(
                    url=url,
                    client=new_client,
                    data=data,
                    json=json,
                    params=params,
                    headers=headers,
                    stream=stream,
                )
            finally:
                await new_client.aclose()
        except httpx.TimeoutException as e:
            headers = {}
            if hasattr(e, "response") and e.response is not None:
                for key, value in e.response.headers.items():
                    headers["response_headers-{}".format(key)] = value

            raise litellm.Timeout(
                message=f"Connection timed out after {timeout} seconds.",
                model="default-model-name",
                llm_provider="litellm-httpx-handler",
                headers=headers,
            )
        except httpx.HTTPStatusError as e:
            setattr(e, "status_code", e.response.status_code)
            if stream is True:
                setattr(e, "message", await e.response.aread())
            else:
                setattr(e, "message", e.response.text)
            raise e
        except Exception as e:
            raise e

    async def delete(
        self,
        url: str,
        data: Optional[Union[dict, str]] = None,  # type: ignore
        json: Optional[dict] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        stream: bool = False,
    ):
        try:
            if timeout is None:
                timeout = self.timeout
            req = self.client.build_request(
                "DELETE", url, data=data, json=json, params=params, headers=headers, timeout=timeout  # type: ignore
            )
            response = await self.client.send(req, stream=stream)
            response.raise_for_status()
            return response
        except (httpx.RemoteProtocolError, httpx.ConnectError):
            # Retry the request with a new session if there is a connection error
            new_client = self.create_client(timeout=timeout, concurrent_limit=1)
            try:
                return await self.single_connection_post_request(
                    url=url,
                    client=new_client,
                    data=data,
                    json=json,
                    params=params,
                    headers=headers,
                    stream=stream,
                )
            finally:
                await new_client.aclose()
        except httpx.HTTPStatusError as e:
            setattr(e, "status_code", e.response.status_code)
            if stream is True:
                setattr(e, "message", await e.response.aread())
            else:
                setattr(e, "message", e.response.text)
            raise e
        except Exception as e:
            raise e

    async def single_connection_post_request(
        self,
        url: str,
        client: httpx.AsyncClient,
        data: Optional[Union[dict, str]] = None,  # type: ignore
        json: Optional[dict] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        stream: bool = False,
    ):
        """
        Making POST request for a single connection client.

        Used for retrying connection client errors.
        """
        req = client.build_request(
            "POST", url, data=data, json=json, params=params, headers=headers  # type: ignore
        )
        response = await client.send(req, stream=stream)
        response.raise_for_status()
        return response

    def __del__(self) -> None:
        try:
            asyncio.get_running_loop().create_task(self.close())
        except Exception:
            pass


class HTTPHandler:
    def __init__(
        self,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        concurrent_limit=1000,
        client: Optional[httpx.Client] = None,
    ):
        if timeout is None:
            timeout = _DEFAULT_TIMEOUT

        # Check if the HTTP_PROXY and HTTPS_PROXY environment variables are set and use them accordingly.
        ssl_verify = bool(os.getenv("SSL_VERIFY", litellm.ssl_verify))
        cert = os.getenv(
            "SSL_CERTIFICATE", litellm.ssl_certificate
        )  # /path/to/client.pem

        if client is None:
            # Create a client with a connection pool
            self.client = httpx.Client(
                timeout=timeout,
                limits=httpx.Limits(
                    max_connections=concurrent_limit,
                    max_keepalive_connections=concurrent_limit,
                ),
                verify=ssl_verify,
                cert=cert,
                headers=headers,
            )
        else:
            self.client = client

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
        data: Optional[Union[dict, str]] = None,
        json: Optional[Union[dict, str]] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        stream: bool = False,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ):
        try:

            if timeout is not None:
                req = self.client.build_request(
                    "POST", url, data=data, json=json, params=params, headers=headers, timeout=timeout  # type: ignore
                )
            else:
                req = self.client.build_request(
                    "POST", url, data=data, json=json, params=params, headers=headers  # type: ignore
                )
            response = self.client.send(req, stream=stream)
            return response
        except httpx.TimeoutException:
            raise litellm.Timeout(
                message=f"Connection timed out after {timeout} seconds.",
                model="default-model-name",
                llm_provider="litellm-httpx-handler",
            )
        except Exception as e:
            raise e

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _get_async_httpx_client(params: Optional[dict] = None) -> AsyncHTTPHandler:
    """
    Retrieves the async HTTP client from the cache
    If not present, creates a new client

    Caches the new client and returns it.
    """
    _params_key_name = ""
    if params is not None:
        for key, value in params.items():
            try:
                _params_key_name += f"{key}_{value}"
            except Exception:
                pass

    _cache_key_name = "async_httpx_client" + _params_key_name
    if _cache_key_name in litellm.in_memory_llm_clients_cache:
        return litellm.in_memory_llm_clients_cache[_cache_key_name]

    if params is not None:
        _new_client = AsyncHTTPHandler(**params)
    else:
        _new_client = AsyncHTTPHandler(
            timeout=httpx.Timeout(timeout=600.0, connect=5.0)
        )
    litellm.in_memory_llm_clients_cache[_cache_key_name] = _new_client
    return _new_client


def _get_httpx_client(params: Optional[dict] = None) -> HTTPHandler:
    """
    Retrieves the HTTP client from the cache
    If not present, creates a new client

    Caches the new client and returns it.
    """
    _params_key_name = ""
    if params is not None:
        for key, value in params.items():
            try:
                _params_key_name += f"{key}_{value}"
            except Exception:
                pass

    _cache_key_name = "httpx_client" + _params_key_name
    if _cache_key_name in litellm.in_memory_llm_clients_cache:
        return litellm.in_memory_llm_clients_cache[_cache_key_name]

    if params is not None:
        _new_client = HTTPHandler(**params)
    else:
        _new_client = HTTPHandler(timeout=httpx.Timeout(timeout=600.0, connect=5.0))

    litellm.in_memory_llm_clients_cache[_cache_key_name] = _new_client
    return _new_client

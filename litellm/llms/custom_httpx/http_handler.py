import asyncio
import os
import ssl
import time
from typing import TYPE_CHECKING, Any, Callable, List, Mapping, Optional, Union

import httpx
from httpx import USE_CLIENT_DEFAULT, AsyncHTTPTransport, HTTPTransport

import litellm
from litellm.litellm_core_utils.logging_utils import track_llm_api_timing
from litellm.types.llms.custom_http import *

if TYPE_CHECKING:
    from litellm import LlmProviders
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObject,
    )
else:
    LlmProviders = Any
    LiteLLMLoggingObject = Any

try:
    from litellm._version import version
except Exception:
    version = "0.0.0"

headers = {
    "User-Agent": f"litellm/{version}",
}

# https://www.python-httpx.org/advanced/timeouts
_DEFAULT_TIMEOUT = httpx.Timeout(timeout=5.0, connect=5.0)
_DEFAULT_TTL_FOR_HTTPX_CLIENTS = 3600  # 1 hour, re-use the same httpx client for 1 hour


def mask_sensitive_info(error_message):
    # Find the start of the key parameter
    if isinstance(error_message, str):
        key_index = error_message.find("key=")
    else:
        return error_message

    # If key is found
    if key_index != -1:
        # Find the end of the key parameter (next & or end of string)
        next_param = error_message.find("&", key_index)

        if next_param == -1:
            # If no more parameters, mask until the end of the string
            masked_message = error_message[: key_index + 4] + "[REDACTED_API_KEY]"
        else:
            # Replace the key with redacted value, keeping other parameters
            masked_message = (
                error_message[: key_index + 4]
                + "[REDACTED_API_KEY]"
                + error_message[next_param:]
            )

        return masked_message

    return error_message


class MaskedHTTPStatusError(httpx.HTTPStatusError):
    def __init__(
        self, original_error, message: Optional[str] = None, text: Optional[str] = None
    ):
        # Create a new error with the masked URL
        masked_url = mask_sensitive_info(str(original_error.request.url))
        # Create a new error that looks like the original, but with a masked URL

        super().__init__(
            message=original_error.message,
            request=httpx.Request(
                method=original_error.request.method,
                url=masked_url,
                headers=original_error.request.headers,
                content=original_error.request.content,
            ),
            response=httpx.Response(
                status_code=original_error.response.status_code,
                content=original_error.response.content,
                headers=original_error.response.headers,
            ),
        )
        self.message = message
        self.text = text


class AsyncHTTPHandler:
    def __init__(
        self,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        event_hooks: Optional[Mapping[str, List[Callable[..., Any]]]] = None,
        concurrent_limit=1000,
        client_alias: Optional[str] = None,  # name for client in logs
        ssl_verify: Optional[VerifyTypes] = None,
    ):
        self.timeout = timeout
        self.event_hooks = event_hooks
        self.client = self.create_client(
            timeout=timeout,
            concurrent_limit=concurrent_limit,
            event_hooks=event_hooks,
            ssl_verify=ssl_verify,
        )
        self.client_alias = client_alias

    def create_client(
        self,
        timeout: Optional[Union[float, httpx.Timeout]],
        concurrent_limit: int,
        event_hooks: Optional[Mapping[str, List[Callable[..., Any]]]],
        ssl_verify: Optional[VerifyTypes] = None,
    ) -> httpx.AsyncClient:

        # SSL certificates (a.k.a CA bundle) used to verify the identity of requested hosts.
        # /path/to/certificate.pem
        if ssl_verify is None:
            ssl_verify = os.getenv("SSL_VERIFY", litellm.ssl_verify)

        ssl_security_level = os.getenv("SSL_SECURITY_LEVEL")

        # If ssl_verify is not False and we need a lower security level
        if (
            not ssl_verify
            and ssl_security_level
            and isinstance(ssl_security_level, str)
        ):
            # Create a custom SSL context with reduced security level
            custom_ssl_context = ssl.create_default_context()
            custom_ssl_context.set_ciphers(ssl_security_level)

            # If ssl_verify is a path to a CA bundle, load it into our custom context
            if isinstance(ssl_verify, str) and os.path.exists(ssl_verify):
                custom_ssl_context.load_verify_locations(cafile=ssl_verify)

            # Use our custom SSL context instead of the original ssl_verify value
            ssl_verify = custom_ssl_context

        # An SSL certificate used by the requested host to authenticate the client.
        # /path/to/client.pem
        cert = os.getenv("SSL_CERTIFICATE", litellm.ssl_certificate)

        if timeout is None:
            timeout = _DEFAULT_TIMEOUT
        # Create a client with a connection pool
        transport = self._create_async_transport()

        return httpx.AsyncClient(
            transport=transport,
            event_hooks=event_hooks,
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
        self,
        url: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        follow_redirects: Optional[bool] = None,
    ):
        # Set follow_redirects to UseClientDefault if None
        _follow_redirects = (
            follow_redirects if follow_redirects is not None else USE_CLIENT_DEFAULT
        )

        response = await self.client.get(
            url, params=params, headers=headers, follow_redirects=_follow_redirects  # type: ignore
        )
        return response

    @track_llm_api_timing()
    async def post(
        self,
        url: str,
        data: Optional[Union[dict, str]] = None,  # type: ignore
        json: Optional[dict] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        stream: bool = False,
        logging_obj: Optional[LiteLLMLoggingObject] = None,
    ):
        start_time = time.time()
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
            new_client = self.create_client(
                timeout=timeout, concurrent_limit=1, event_hooks=self.event_hooks
            )
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
            end_time = time.time()
            time_delta = round(end_time - start_time, 3)
            headers = {}
            error_response = getattr(e, "response", None)
            if error_response is not None:
                for key, value in error_response.headers.items():
                    headers["response_headers-{}".format(key)] = value

            raise litellm.Timeout(
                message=f"Connection timed out. Timeout passed={timeout}, time taken={time_delta} seconds",
                model="default-model-name",
                llm_provider="litellm-httpx-handler",
                headers=headers,
            )
        except httpx.HTTPStatusError as e:
            if stream is True:
                setattr(e, "message", await e.response.aread())
                setattr(e, "text", await e.response.aread())
            else:
                setattr(e, "message", mask_sensitive_info(e.response.text))
                setattr(e, "text", mask_sensitive_info(e.response.text))

            setattr(e, "status_code", e.response.status_code)

            raise e
        except Exception as e:
            raise e

    async def put(
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
                "PUT", url, data=data, json=json, params=params, headers=headers, timeout=timeout  # type: ignore
            )
            response = await self.client.send(req)
            response.raise_for_status()
            return response
        except (httpx.RemoteProtocolError, httpx.ConnectError):
            # Retry the request with a new session if there is a connection error
            new_client = self.create_client(
                timeout=timeout, concurrent_limit=1, event_hooks=self.event_hooks
            )
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
            error_response = getattr(e, "response", None)
            if error_response is not None:
                for key, value in error_response.headers.items():
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

    async def patch(
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
                "PATCH", url, data=data, json=json, params=params, headers=headers, timeout=timeout  # type: ignore
            )
            response = await self.client.send(req)
            response.raise_for_status()
            return response
        except (httpx.RemoteProtocolError, httpx.ConnectError):
            # Retry the request with a new session if there is a connection error
            new_client = self.create_client(
                timeout=timeout, concurrent_limit=1, event_hooks=self.event_hooks
            )
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
            error_response = getattr(e, "response", None)
            if error_response is not None:
                for key, value in error_response.headers.items():
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
            new_client = self.create_client(
                timeout=timeout, concurrent_limit=1, event_hooks=self.event_hooks
            )
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

    def _create_async_transport(self) -> Optional[AsyncHTTPTransport]:
        """
        Create an async transport with IPv4 only if litellm.force_ipv4 is True.
        Otherwise, return None.

        Some users have seen httpx ConnectionError when using ipv6 - forcing ipv4 resolves the issue for them
        """
        if litellm.force_ipv4:
            return AsyncHTTPTransport(local_address="0.0.0.0")
        else:
            return None


class HTTPHandler:
    def __init__(
        self,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        concurrent_limit=1000,
        client: Optional[httpx.Client] = None,
        ssl_verify: Optional[Union[bool, str]] = None,
    ):
        if timeout is None:
            timeout = _DEFAULT_TIMEOUT

        # SSL certificates (a.k.a CA bundle) used to verify the identity of requested hosts.
        # /path/to/certificate.pem

        if ssl_verify is None:
            ssl_verify = os.getenv("SSL_VERIFY", litellm.ssl_verify)

        # An SSL certificate used by the requested host to authenticate the client.
        # /path/to/client.pem
        cert = os.getenv("SSL_CERTIFICATE", litellm.ssl_certificate)

        if client is None:
            transport = self._create_sync_transport()

            # Create a client with a connection pool
            self.client = httpx.Client(
                transport=transport,
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
        self,
        url: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        follow_redirects: Optional[bool] = None,
    ):
        # Set follow_redirects to UseClientDefault if None
        _follow_redirects = (
            follow_redirects if follow_redirects is not None else USE_CLIENT_DEFAULT
        )

        response = self.client.get(
            url, params=params, headers=headers, follow_redirects=_follow_redirects  # type: ignore
        )
        return response

    def post(
        self,
        url: str,
        data: Optional[Union[dict, str]] = None,
        json: Optional[Union[dict, str, List]] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        stream: bool = False,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        files: Optional[dict] = None,
        content: Any = None,
        logging_obj: Optional[LiteLLMLoggingObject] = None,
    ):
        try:
            if timeout is not None:
                req = self.client.build_request(
                    "POST",
                    url,
                    data=data,  # type: ignore
                    json=json,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                    files=files,
                    content=content,  # type: ignore
                )
            else:
                req = self.client.build_request(
                    "POST", url, data=data, json=json, params=params, headers=headers, files=files, content=content  # type: ignore
                )
            response = self.client.send(req, stream=stream)
            response.raise_for_status()
            return response
        except httpx.TimeoutException:
            raise litellm.Timeout(
                message=f"Connection timed out after {timeout} seconds.",
                model="default-model-name",
                llm_provider="litellm-httpx-handler",
            )
        except httpx.HTTPStatusError as e:
            if stream is True:
                setattr(e, "message", mask_sensitive_info(e.response.read()))
                setattr(e, "text", mask_sensitive_info(e.response.read()))
            else:
                error_text = mask_sensitive_info(e.response.text)
                setattr(e, "message", error_text)
                setattr(e, "text", error_text)

            setattr(e, "status_code", e.response.status_code)

            raise e
        except Exception as e:
            raise e

    def patch(
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
                    "PATCH", url, data=data, json=json, params=params, headers=headers, timeout=timeout  # type: ignore
                )
            else:
                req = self.client.build_request(
                    "PATCH", url, data=data, json=json, params=params, headers=headers  # type: ignore
                )
            response = self.client.send(req, stream=stream)
            response.raise_for_status()
            return response
        except httpx.TimeoutException:
            raise litellm.Timeout(
                message=f"Connection timed out after {timeout} seconds.",
                model="default-model-name",
                llm_provider="litellm-httpx-handler",
            )
        except httpx.HTTPStatusError as e:

            if stream is True:
                setattr(e, "message", mask_sensitive_info(e.response.read()))
                setattr(e, "text", mask_sensitive_info(e.response.read()))
            else:
                error_text = mask_sensitive_info(e.response.text)
                setattr(e, "message", error_text)
                setattr(e, "text", error_text)

            setattr(e, "status_code", e.response.status_code)

            raise e
        except Exception as e:
            raise e

    def put(
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
                    "PUT", url, data=data, json=json, params=params, headers=headers, timeout=timeout  # type: ignore
                )
            else:
                req = self.client.build_request(
                    "PUT", url, data=data, json=json, params=params, headers=headers  # type: ignore
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

    def _create_sync_transport(self) -> Optional[HTTPTransport]:
        """
        Create an HTTP transport with IPv4 only if litellm.force_ipv4 is True.
        Otherwise, return None.

        Some users have seen httpx ConnectionError when using ipv6 - forcing ipv4 resolves the issue for them
        """
        if litellm.force_ipv4:
            return HTTPTransport(local_address="0.0.0.0")
        else:
            return None


def get_async_httpx_client(
    llm_provider: Union[LlmProviders, httpxSpecialProvider],
    params: Optional[dict] = None,
) -> AsyncHTTPHandler:
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

    _cache_key_name = "async_httpx_client" + _params_key_name + llm_provider
    _cached_client = litellm.in_memory_llm_clients_cache.get_cache(_cache_key_name)
    if _cached_client:
        return _cached_client

    if params is not None:
        _new_client = AsyncHTTPHandler(**params)
    else:
        _new_client = AsyncHTTPHandler(
            timeout=httpx.Timeout(timeout=600.0, connect=5.0)
        )

    litellm.in_memory_llm_clients_cache.set_cache(
        key=_cache_key_name,
        value=_new_client,
        ttl=_DEFAULT_TTL_FOR_HTTPX_CLIENTS,
    )
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

    _cached_client = litellm.in_memory_llm_clients_cache.get_cache(_cache_key_name)
    if _cached_client:
        return _cached_client

    if params is not None:
        _new_client = HTTPHandler(**params)
    else:
        _new_client = HTTPHandler(timeout=httpx.Timeout(timeout=600.0, connect=5.0))

    litellm.in_memory_llm_clients_cache.set_cache(
        key=_cache_key_name,
        value=_new_client,
        ttl=_DEFAULT_TTL_FOR_HTTPX_CLIENTS,
    )
    return _new_client

import asyncio
import os
import ssl
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Union

import httpx
from aiohttp import ClientSession, TCPConnector
from httpx import USE_CLIENT_DEFAULT, AsyncHTTPTransport, HTTPTransport
from httpx._types import RequestFiles

import litellm
from litellm._logging import verbose_logger
from litellm.constants import _DEFAULT_TTL_FOR_HTTPX_CLIENTS
from litellm.litellm_core_utils.logging_utils import track_llm_api_timing
from litellm.types.llms.custom_http import *

if TYPE_CHECKING:
    from litellm import LlmProviders
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObject,
    )
    from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport
else:
    LlmProviders = Any
    LiteLLMLoggingObject = Any
    LiteLLMAiohttpTransport = Any

try:
    from litellm._version import version
except Exception:
    version = "0.0.0"

headers = {
    "User-Agent": f"litellm/{version}",
}

# https://www.python-httpx.org/advanced/timeouts
_DEFAULT_TIMEOUT = httpx.Timeout(timeout=5.0, connect=5.0)


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

        transport = AsyncHTTPHandler._create_async_transport(
            ssl_context=ssl_verify if isinstance(ssl_verify, ssl.SSLContext) else None,
            ssl_verify=ssl_verify if isinstance(ssl_verify, bool) else None,
        )

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

        params = params or {}
        params.update(HTTPHandler.extract_query_params(url))

        response = await self.client.get(
            url, params=params, headers=headers, follow_redirects=_follow_redirects  # type: ignore
        )
        return response

    @track_llm_api_timing()
    async def post(
        self,
        url: str,
        data: Optional[Union[dict, str, bytes]] = None,  # type: ignore
        json: Optional[dict] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        stream: bool = False,
        logging_obj: Optional[LiteLLMLoggingObject] = None,
        files: Optional[RequestFiles] = None,
        content: Any = None,
    ):
        start_time = time.time()
        try:
            if timeout is None:
                timeout = self.timeout

            req = self.client.build_request(
                "POST",
                url,
                data=data,  # type: ignore
                json=json,
                params=params,
                headers=headers,
                timeout=timeout,
                files=files,
                content=content,
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
        data: Optional[Union[dict, str, bytes]] = None,  # type: ignore
        json: Optional[dict] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        stream: bool = False,
        content: Any = None,
    ):
        """
        Making POST request for a single connection client.

        Used for retrying connection client errors.
        """
        req = client.build_request(
            "POST", url, data=data, json=json, params=params, headers=headers, content=content  # type: ignore
        )
        response = await client.send(req, stream=stream)
        response.raise_for_status()
        return response

    def __del__(self) -> None:
        try:
            asyncio.get_running_loop().create_task(self.close())
        except Exception:
            pass

    @staticmethod
    def _create_async_transport(
        ssl_context: Optional[ssl.SSLContext] = None, ssl_verify: Optional[bool] = None
    ) -> Optional[Union[LiteLLMAiohttpTransport, AsyncHTTPTransport]]:
        """
        - Creates a transport for httpx.AsyncClient
            - if litellm.force_ipv4 is True, it will return AsyncHTTPTransport with local_address="0.0.0.0"
            - [Default] It will return AiohttpTransport
            - Users can opt out of using AiohttpTransport by setting litellm.use_aiohttp_transport to False


        Notes on this handler:
        - Why AiohttpTransport?
            - By default, we use AiohttpTransport since it offers much higher throughput and lower latency than httpx.

        - Why force ipv4?
            - Some users have seen httpx ConnectionError when using ipv6 - forcing ipv4 resolves the issue for them
        """
        #########################################################
        # AIOHTTP TRANSPORT is off by default
        #########################################################
        if AsyncHTTPHandler._should_use_aiohttp_transport():
            return AsyncHTTPHandler._create_aiohttp_transport(
                ssl_context=ssl_context, ssl_verify=ssl_verify
            )

        #########################################################
        # HTTPX TRANSPORT is used when aiohttp is not installed
        #########################################################
        return AsyncHTTPHandler._create_httpx_transport()

    @staticmethod
    def _should_use_aiohttp_transport() -> bool:
        """
        AiohttpTransport is the default transport for litellm.

        Httpx can be used by the following
            - litellm.disable_aiohttp_transport = True
            - os.getenv("DISABLE_AIOHTTP_TRANSPORT") = "True"
        """
        import os

        from litellm.secret_managers.main import str_to_bool

        #########################################################
        # Check if user disabled aiohttp transport
        ########################################################
        if (
            litellm.disable_aiohttp_transport is True
            or str_to_bool(os.getenv("DISABLE_AIOHTTP_TRANSPORT", "False")) is True
        ):
            return False

        #########################################################
        # Default: Use AiohttpTransport
        ########################################################
        verbose_logger.debug("Using AiohttpTransport...")
        return True

    @staticmethod
    def _get_ssl_connector_kwargs(
        ssl_verify: Optional[bool] = None,
        ssl_context: Optional[ssl.SSLContext] = None,
    ) -> Dict[str, Any]:
        """
        Helper method to get SSL connector initialization arguments for aiohttp TCPConnector.
        
        SSL Configuration Priority:
        1. If ssl_context is provided -> use the custom SSL context
        2. If ssl_verify is False -> disable SSL verification (ssl=False)
        3. If ssl_verify is True/None -> use default SSL context with certifi CA bundle

        Returns:
            Dict with appropriate SSL configuration for TCPConnector
        """
        connector_kwargs: Dict[str, Any] = {
            "local_addr": ("0.0.0.0", 0) if litellm.force_ipv4 else None,
        }
        
        if ssl_context is not None:
            # Priority 1: Use the provided custom SSL context
            connector_kwargs["ssl"] = ssl_context
        elif ssl_verify is False:
            # Priority 2: Explicitly disable SSL verification
            connector_kwargs["verify_ssl"] = False
        else:
            # Priority 3: Use our default SSL context with certifi CA bundle
            # This covers ssl_verify=True and ssl_verify=None cases
            connector_kwargs["ssl"] = AsyncHTTPHandler._get_ssl_context()
        
        return connector_kwargs

    @staticmethod
    def _create_aiohttp_transport(
        ssl_verify: Optional[bool] = None,
        ssl_context: Optional[ssl.SSLContext] = None,
    ) -> LiteLLMAiohttpTransport:
        """
        Creates an AiohttpTransport with RequestNotRead error handling

        Note: aiohttp TCPConnector ssl parameter accepts:
        - SSLContext: custom SSL context
        - False: disable SSL verification
        - True: use default SSL verification (equivalent to ssl.create_default_context())
        """
        from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport

        connector_kwargs = AsyncHTTPHandler._get_ssl_connector_kwargs(
            ssl_verify=ssl_verify, ssl_context=ssl_context
        )

        verbose_logger.debug("Creating AiohttpTransport...")
        return LiteLLMAiohttpTransport(
            client=lambda: ClientSession(
                connector=TCPConnector(**connector_kwargs)
            ),
        )
    

    @staticmethod
    def _get_ssl_context() -> ssl.SSLContext:
        """
        Get the SSL context for the AiohttpTransport
        """
        import certifi
        return ssl.create_default_context(
            cafile=certifi.where()
        )

    @staticmethod
    def _create_httpx_transport() -> Optional[AsyncHTTPTransport]:
        """
        Creates an AsyncHTTPTransport

        - If force_ipv4 is True, it will create an AsyncHTTPTransport with local_address set to "0.0.0.0"
        - [Default] If force_ipv4 is False, it will return None
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
        params = params or {}
        params.update(self.extract_query_params(url))

        response = self.client.get(
            url, params=params, headers=headers, follow_redirects=_follow_redirects  # type: ignore
        )

        return response

    @staticmethod
    def extract_query_params(url: str) -> Dict[str, str]:
        """
        Parse a URL’s query-string into a dict.

        :param url: full URL, e.g. "https://.../path?foo=1&bar=2"
        :return: {"foo": "1", "bar": "2"}
        """
        from urllib.parse import parse_qsl, urlsplit

        parts = urlsplit(url)
        return dict(parse_qsl(parts.query))

    def post(
        self,
        url: str,
        data: Optional[Union[dict, str, bytes]] = None,
        json: Optional[Union[dict, str, List]] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        stream: bool = False,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        files: Optional[Union[dict, RequestFiles]] = None,
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

    def delete(
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
            if timeout is not None:
                req = self.client.build_request(
                    "DELETE", url, data=data, json=json, params=params, headers=headers, timeout=timeout  # type: ignore
                )
            else:
                req = self.client.build_request(
                    "DELETE", url, data=data, json=json, params=params, headers=headers  # type: ignore
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

import asyncio
import os
import ssl
import sys
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import certifi
import httpx
from aiohttp import ClientSession, TCPConnector
from httpx import USE_CLIENT_DEFAULT, AsyncHTTPTransport, HTTPTransport
from httpx._types import RequestFiles

import litellm
from litellm._logging import verbose_logger
from litellm.constants import (
    _DEFAULT_TTL_FOR_HTTPX_CLIENTS,
    AIOHTTP_CONNECTOR_LIMIT,
    AIOHTTP_CONNECTOR_LIMIT_PER_HOST,
    AIOHTTP_KEEPALIVE_TIMEOUT,
    AIOHTTP_TTL_DNS_CACHE,
    DEFAULT_SSL_CIPHERS,
)
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


def _prepare_request_data_and_content(
    data: Optional[Union[dict, str, bytes]] = None,
    content: Any = None,
) -> Tuple[Optional[Union[dict, Mapping]], Any]:
    """
    Helper function to route data/content parameters correctly for httpx requests

    This prevents httpx DeprecationWarnings that cause memory leaks.

    Background:
    - httpx shows a DeprecationWarning when you pass bytes/str to `data=`
    - It wants you to use `content=` instead for bytes/str
    - The warning itself leaks memory when triggered repeatedly

    Solution:
    - Move bytes/str from `data=` to `content=` before calling build_request
    - Keep dicts in `data=` (that's still the correct parameter for dicts)

    Args:
        data: Request data (can be dict, str, or bytes)
        content: Request content (raw bytes/str)

    Returns:
        Tuple of (request_data, request_content) properly routed for httpx
    """
    request_data = None
    request_content = content

    if data is not None:
        if isinstance(data, (bytes, str)):
            # Bytes/strings belong in content= (only if not already provided)
            if content is None:
                request_content = data
        else:
            # dict/Mapping stays in data= parameter
            request_data = data

    return request_data, request_content


# Cache for SSL contexts to avoid creating duplicate contexts with the same configuration
# Key: tuple of (cafile, ssl_security_level, ssl_ecdh_curve)
# Value: ssl.SSLContext
_ssl_context_cache: Dict[
    Tuple[Optional[str], Optional[str], Optional[str]], ssl.SSLContext
] = {}


def _create_ssl_context(
    cafile: Optional[str],
    ssl_security_level: Optional[str],
    ssl_ecdh_curve: Optional[str],
) -> ssl.SSLContext:
    """
    Create an SSL context with the given configuration.
    This is separated from get_ssl_configuration to enable caching.
    """
    custom_ssl_context = ssl.create_default_context(cafile=cafile)

    # Optimize SSL handshake performance
    # Set minimum TLS version to 1.2 for better performance
    custom_ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2

    # Configure cipher suites for optimal performance
    if ssl_security_level and isinstance(ssl_security_level, str):
        # User provided custom cipher configuration (e.g., via SSL_SECURITY_LEVEL env var)
        custom_ssl_context.set_ciphers(ssl_security_level)
    else:
        # Use optimized cipher list that strongly prefers fast ciphers
        # but falls back to widely compatible ones
        custom_ssl_context.set_ciphers(DEFAULT_SSL_CIPHERS)

    # Configure ECDH curve for key exchange (e.g., to disable PQC and improve performance)
    # Set SSL_ECDH_CURVE env var or litellm.ssl_ecdh_curve to 'X25519' to disable PQC
    # Common valid curves: X25519, prime256v1, secp384r1, secp521r1
    if ssl_ecdh_curve and isinstance(ssl_ecdh_curve, str):
        try:
            custom_ssl_context.set_ecdh_curve(ssl_ecdh_curve)
            verbose_logger.debug(f"SSL ECDH curve set to: {ssl_ecdh_curve}")
        except AttributeError:
            verbose_logger.warning(
                f"SSL ECDH curve configuration not supported. "
                f"Python version: {sys.version.split()[0]}, OpenSSL version: {ssl.OPENSSL_VERSION}. "
                f"Requested curve: {ssl_ecdh_curve}. Continuing with default curves."
            )
        except ValueError as e:
            # Invalid curve name
            verbose_logger.warning(
                f"Invalid SSL ECDH curve name: '{ssl_ecdh_curve}'. {e}. "
                f"Common valid curves: X25519, prime256v1, secp384r1, secp521r1. "
                f"Continuing with default curves (including PQC)."
            )

    return custom_ssl_context


def get_ssl_configuration(
    ssl_verify: Optional[VerifyTypes] = None,
) -> Union[bool, str, ssl.SSLContext]:
    """
    Unified SSL configuration function that handles ssl_context and ssl_verify logic.

    SSL Configuration Priority:
    1. If ssl_verify is provided -> is a SSL context use the custom SSL context
    2. If ssl_verify is False -> disable SSL verification (ssl=False)
    3. If ssl_verify is a string -> use it as a path to CA bundle file
    4. If SSL_CERT_FILE environment variable is set and exists -> use it as CA bundle file
    5. Else will use default SSL context with certifi CA bundle

    If ssl_security_level is set, it will apply the security level to the SSL context.

    SSL contexts are cached to avoid creating duplicate contexts with the same configuration,
    which reduces memory allocation and improves performance.

    Args:
        ssl_verify: SSL verification setting. Can be:
            - None: Use default from environment/litellm settings
            - False: Disable SSL verification
            - True: Enable SSL verification
            - str: Path to CA bundle file

    Returns:
        Union[bool, str, ssl.SSLContext]: Appropriate SSL configuration
    """
    from litellm.secret_managers.main import str_to_bool

    if isinstance(ssl_verify, ssl.SSLContext):
        # If ssl_verify is already an SSLContext, return it directly
        return ssl_verify

    # Get ssl_verify from environment or litellm settings if not provided
    if ssl_verify is None:
        ssl_verify = os.getenv("SSL_VERIFY", litellm.ssl_verify)
        ssl_verify_bool = (
            str_to_bool(ssl_verify) if isinstance(ssl_verify, str) else ssl_verify
        )
        if ssl_verify_bool is not None:
            ssl_verify = ssl_verify_bool

    ssl_security_level = os.getenv("SSL_SECURITY_LEVEL", litellm.ssl_security_level)
    ssl_ecdh_curve = os.getenv("SSL_ECDH_CURVE", litellm.ssl_ecdh_curve)

    cafile = None
    if isinstance(ssl_verify, str) and os.path.exists(ssl_verify):
        cafile = ssl_verify
    if not cafile:
        ssl_cert_file = os.getenv("SSL_CERT_FILE")
        if ssl_cert_file and os.path.exists(ssl_cert_file):
            cafile = ssl_cert_file
        else:
            cafile = certifi.where()

    if ssl_verify is not False:
        # Create cache key from configuration parameters
        cache_key = (cafile, ssl_security_level, ssl_ecdh_curve)

        # Check if we have a cached SSL context for this configuration
        if cache_key not in _ssl_context_cache:
            _ssl_context_cache[cache_key] = _create_ssl_context(
                cafile=cafile,
                ssl_security_level=ssl_security_level,
                ssl_ecdh_curve=ssl_ecdh_curve,
            )

        # Return the cached SSL context
        return _ssl_context_cache[cache_key]

    return ssl_verify


_shared_realtime_ssl_context: Optional[Union[bool, str, ssl.SSLContext]] = None


def get_shared_realtime_ssl_context() -> Union[bool, str, ssl.SSLContext]:
    """
    Lazily create the SSL context reused by realtime websocket clients so we avoid
    import-order cycles during startup while keeping a single shared configuration.
    """
    global _shared_realtime_ssl_context
    if _shared_realtime_ssl_context is None:
        _shared_realtime_ssl_context = get_ssl_configuration()
    return _shared_realtime_ssl_context


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
        concurrent_limit=None,  # Kept for backward compatibility, but ignored (no limits)
        client_alias: Optional[str] = None,  # name for client in logs
        ssl_verify: Optional[VerifyTypes] = None,
        shared_session: Optional["ClientSession"] = None,
    ):
        self.timeout = timeout
        self.event_hooks = event_hooks
        self.client = self.create_client(
            timeout=timeout,
            event_hooks=event_hooks,
            ssl_verify=ssl_verify,
            shared_session=shared_session,
        )
        self.client_alias = client_alias

    def create_client(
        self,
        timeout: Optional[Union[float, httpx.Timeout]],
        event_hooks: Optional[Mapping[str, List[Callable[..., Any]]]],
        ssl_verify: Optional[VerifyTypes] = None,
        shared_session: Optional["ClientSession"] = None,
    ) -> httpx.AsyncClient:
        # Get unified SSL configuration
        ssl_config = get_ssl_configuration(ssl_verify)

        # An SSL certificate used by the requested host to authenticate the client.
        # /path/to/client.pem
        cert = os.getenv("SSL_CERTIFICATE", litellm.ssl_certificate)

        if timeout is None:
            timeout = _DEFAULT_TIMEOUT
        # Create a client with a connection pool

        transport = AsyncHTTPHandler._create_async_transport(
            ssl_context=ssl_config if isinstance(ssl_config, ssl.SSLContext) else None,
            ssl_verify=ssl_config if isinstance(ssl_config, bool) else None,
            shared_session=shared_session,
        )

        return httpx.AsyncClient(
            transport=transport,
            event_hooks=event_hooks,
            timeout=timeout,
            verify=ssl_config,
            cert=cert,
            headers=headers,
            follow_redirects=True,
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

            # Prepare data/content parameters to prevent httpx DeprecationWarning (memory leak fix)
            request_data, request_content = _prepare_request_data_and_content(
                data, content
            )

            req = self.client.build_request(
                "POST",
                url,
                data=request_data,
                json=json,
                params=params,
                headers=headers,
                timeout=timeout,
                files=files,
                content=request_content,
            )
            response = await self.client.send(req, stream=stream)
            response.raise_for_status()
            return response
        except (httpx.RemoteProtocolError, httpx.ConnectError):
            # Retry the request with a new session if there is a connection error
            new_client = self.create_client(
                timeout=timeout, event_hooks=self.event_hooks
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
        data: Optional[Union[dict, str, bytes]] = None,  # type: ignore
        json: Optional[dict] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        stream: bool = False,
        content: Any = None,
    ):
        try:
            if timeout is None:
                timeout = self.timeout

            # Prepare data/content parameters to prevent httpx DeprecationWarning (memory leak fix)
            request_data, request_content = _prepare_request_data_and_content(
                data, content
            )

            req = self.client.build_request(
                "PUT", url, data=request_data, json=json, params=params, headers=headers, timeout=timeout, content=request_content  # type: ignore
            )
            response = await self.client.send(req)
            response.raise_for_status()
            return response
        except (httpx.RemoteProtocolError, httpx.ConnectError):
            # Retry the request with a new session if there is a connection error
            new_client = self.create_client(
                timeout=timeout, event_hooks=self.event_hooks
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
        data: Optional[Union[dict, str, bytes]] = None,  # type: ignore
        json: Optional[dict] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        stream: bool = False,
        content: Any = None,
    ):
        try:
            if timeout is None:
                timeout = self.timeout

            # Prepare data/content parameters to prevent httpx DeprecationWarning (memory leak fix)
            request_data, request_content = _prepare_request_data_and_content(
                data, content
            )

            req = self.client.build_request(
                "PATCH", url, data=request_data, json=json, params=params, headers=headers, timeout=timeout, content=request_content  # type: ignore
            )
            response = await self.client.send(req)
            response.raise_for_status()
            return response
        except (httpx.RemoteProtocolError, httpx.ConnectError):
            # Retry the request with a new session if there is a connection error
            new_client = self.create_client(
                timeout=timeout, event_hooks=self.event_hooks
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
        data: Optional[Union[dict, str, bytes]] = None,  # type: ignore
        json: Optional[dict] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        stream: bool = False,
        content: Any = None,
    ):
        try:
            if timeout is None:
                timeout = self.timeout

            # Prepare data/content parameters to prevent httpx DeprecationWarning (memory leak fix)
            request_data, request_content = _prepare_request_data_and_content(
                data, content
            )

            req = self.client.build_request(
                "DELETE", url, data=request_data, json=json, params=params, headers=headers, timeout=timeout, content=request_content  # type: ignore
            )
            response = await self.client.send(req, stream=stream)
            response.raise_for_status()
            return response
        except (httpx.RemoteProtocolError, httpx.ConnectError):
            # Retry the request with a new session if there is a connection error
            new_client = self.create_client(
                timeout=timeout, event_hooks=self.event_hooks
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
        # Prepare data/content parameters to prevent httpx DeprecationWarning (memory leak fix)
        request_data, request_content = _prepare_request_data_and_content(data, content)

        req = client.build_request(
            "POST", url, data=request_data, json=json, params=params, headers=headers, content=request_content  # type: ignore
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
        ssl_context: Optional[ssl.SSLContext] = None,
        ssl_verify: Optional[bool] = None,
        shared_session: Optional["ClientSession"] = None,
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
                ssl_context=ssl_context,
                ssl_verify=ssl_verify,
                shared_session=shared_session,
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

        return connector_kwargs

    @staticmethod
    def _create_aiohttp_transport(
        ssl_verify: Optional[bool] = None,
        ssl_context: Optional[ssl.SSLContext] = None,
        shared_session: Optional["ClientSession"] = None,
    ) -> LiteLLMAiohttpTransport:
        """
        Creates an AiohttpTransport with RequestNotRead error handling

        Note: aiohttp TCPConnector ssl parameter accepts:
        - SSLContext: custom SSL context
        - False: disable SSL verification
        """
        from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport
        from litellm.secret_managers.main import str_to_bool

        connector_kwargs = AsyncHTTPHandler._get_ssl_connector_kwargs(
            ssl_verify=ssl_verify, ssl_context=ssl_context
        )
        #########################################################
        # Check if user enabled aiohttp trust env
        # use for HTTP_PROXY, HTTPS_PROXY, etc.
        ########################################################
        trust_env: bool = litellm.aiohttp_trust_env
        if str_to_bool(os.getenv("AIOHTTP_TRUST_ENV", "False")) is True:
            trust_env = True

        verbose_logger.debug("Creating AiohttpTransport...")

        # Use shared session if provided and valid
        if shared_session is not None and not shared_session.closed:
            verbose_logger.debug(
                f"SHARED SESSION: Reusing existing ClientSession (ID: {id(shared_session)})"
            )
            return LiteLLMAiohttpTransport(client=shared_session)

        # Create new session only if none provided or existing one is invalid
        verbose_logger.debug(
            "NEW SESSION: Creating new ClientSession (no shared session provided)"
        )
        transport_connector_kwargs = {
            "keepalive_timeout": AIOHTTP_KEEPALIVE_TIMEOUT,
            "ttl_dns_cache": AIOHTTP_TTL_DNS_CACHE,
            "enable_cleanup_closed": True,
            **connector_kwargs,
        }
        if AIOHTTP_CONNECTOR_LIMIT > 0:
            transport_connector_kwargs["limit"] = AIOHTTP_CONNECTOR_LIMIT
        if AIOHTTP_CONNECTOR_LIMIT_PER_HOST > 0:
            transport_connector_kwargs["limit_per_host"] = (
                AIOHTTP_CONNECTOR_LIMIT_PER_HOST
            )

        return LiteLLMAiohttpTransport(
            client=lambda: ClientSession(
                connector=TCPConnector(**transport_connector_kwargs),
                trust_env=trust_env,
            ),
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
        concurrent_limit=None,  # Kept for backward compatibility, but ignored (no limits)
        client: Optional[httpx.Client] = None,
        ssl_verify: Optional[Union[bool, str]] = None,
        disable_default_headers: Optional[
            bool
        ] = False,  # arize phoenix returns different API responses when user agent header in request
    ):
        if timeout is None:
            timeout = _DEFAULT_TIMEOUT

        # Get unified SSL configuration
        ssl_config = get_ssl_configuration(ssl_verify)

        # An SSL certificate used by the requested host to authenticate the client.
        # /path/to/client.pem
        cert = os.getenv("SSL_CERTIFICATE", litellm.ssl_certificate)

        if client is None:
            transport = self._create_sync_transport()

            # Create a client with a connection pool
            self.client = httpx.Client(
                transport=transport,
                timeout=timeout,
                verify=ssl_config,
                cert=cert,
                headers=headers if not disable_default_headers else None,
                follow_redirects=True,
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
            url,
            params=params,
            headers=headers,
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
            # Prepare data/content parameters to prevent httpx DeprecationWarning (memory leak fix)
            request_data, request_content = _prepare_request_data_and_content(
                data, content
            )

            if timeout is not None:
                req = self.client.build_request(
                    "POST",
                    url,
                    data=request_data,  # type: ignore
                    json=json,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                    files=files,
                    content=request_content,  # type: ignore
                )
            else:
                req = self.client.build_request(
                    "POST", url, data=request_data, json=json, params=params, headers=headers, files=files, content=request_content  # type: ignore
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
        data: Optional[Union[dict, str, bytes]] = None,
        json: Optional[Union[dict, str]] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        stream: bool = False,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        content: Any = None,
    ):
        try:
            # Prepare data/content parameters to prevent httpx DeprecationWarning (memory leak fix)
            request_data, request_content = _prepare_request_data_and_content(
                data, content
            )

            if timeout is not None:
                req = self.client.build_request(
                    "PATCH", url, data=request_data, json=json, params=params, headers=headers, timeout=timeout, content=request_content  # type: ignore
                )
            else:
                req = self.client.build_request(
                    "PATCH", url, data=request_data, json=json, params=params, headers=headers, content=request_content  # type: ignore
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
        data: Optional[Union[dict, str, bytes]] = None,
        json: Optional[Union[dict, str]] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        stream: bool = False,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        content: Any = None,
    ):
        try:
            # Prepare data/content parameters to prevent httpx DeprecationWarning (memory leak fix)
            request_data, request_content = _prepare_request_data_and_content(
                data, content
            )

            if timeout is not None:
                req = self.client.build_request(
                    "PUT", url, data=request_data, json=json, params=params, headers=headers, timeout=timeout, content=request_content  # type: ignore
                )
            else:
                req = self.client.build_request(
                    "PUT", url, data=request_data, json=json, params=params, headers=headers, content=request_content  # type: ignore
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
        data: Optional[Union[dict, str, bytes]] = None,  # type: ignore
        json: Optional[dict] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        stream: bool = False,
        content: Any = None,
    ):
        try:
            # Prepare data/content parameters to prevent httpx DeprecationWarning (memory leak fix)
            request_data, request_content = _prepare_request_data_and_content(
                data, content
            )

            if timeout is not None:
                req = self.client.build_request(
                    "DELETE", url, data=request_data, json=json, params=params, headers=headers, timeout=timeout, content=request_content  # type: ignore
                )
            else:
                req = self.client.build_request(
                    "DELETE", url, data=request_data, json=json, params=params, headers=headers, content=request_content  # type: ignore
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
            return getattr(litellm, "sync_transport", None)


def get_async_httpx_client(
    llm_provider: Union[LlmProviders, httpxSpecialProvider],
    params: Optional[dict] = None,
    shared_session: Optional["ClientSession"] = None,
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

    # Lazily initialize the global in-memory client cache to avoid relying on
    # litellm globals being fully populated during import time.
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is None:
        from litellm.caching.llm_caching_handler import LLMClientCache

        cache = LLMClientCache()
        setattr(litellm, "in_memory_llm_clients_cache", cache)

    _cached_client = cache.get_cache(_cache_key_name)
    if _cached_client:
        return _cached_client

    if params is not None:
        params["shared_session"] = shared_session
        _new_client = AsyncHTTPHandler(**params)
    else:
        _new_client = AsyncHTTPHandler(
            timeout=httpx.Timeout(timeout=600.0, connect=5.0),
            shared_session=shared_session,
        )

    cache.set_cache(
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

    # Lazily initialize the global in-memory client cache to avoid relying on
    # litellm globals being fully populated during import time.
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is None:
        from litellm.caching.llm_caching_handler import LLMClientCache

        cache = LLMClientCache()
        setattr(litellm, "in_memory_llm_clients_cache", cache)

    _cached_client = cache.get_cache(_cache_key_name)
    if _cached_client:
        return _cached_client

    if params is not None:
        _new_client = HTTPHandler(**params)
    else:
        _new_client = HTTPHandler(timeout=httpx.Timeout(timeout=600.0, connect=5.0))

    cache.set_cache(
        key=_cache_key_name,
        value=_new_client,
        ttl=_DEFAULT_TTL_FOR_HTTPX_CLIENTS,
    )
    return _new_client

import asyncio
import concurrent.futures
import inspect
import os
import socket
import ssl
import sys
import threading
import time
from collections.abc import AsyncIterable, Callable, Iterable, Mapping
from http.cookiejar import CookieJar, DefaultCookiePolicy
from typing import TYPE_CHECKING, Any, ClassVar, Final, Optional, TypeAlias, TypedDict

import certifi
import httpx
from aiohttp import ClientSession, DummyCookieJar, TCPConnector
from httpx import USE_CLIENT_DEFAULT, AsyncHTTPTransport, HTTPTransport
from httpx._types import RequestFiles

import litellm
from litellm._logging import verbose_logger
from litellm.constants import (
    _DEFAULT_TTL_FOR_HTTPX_CLIENTS,
    AIOHTTP_CONNECTOR_LIMIT,
    AIOHTTP_CONNECTOR_LIMIT_PER_HOST,
    AIOHTTP_KEEPALIVE_TIMEOUT,
    AIOHTTP_NEEDS_CLEANUP_CLOSED,
    AIOHTTP_SO_KEEPALIVE,
    AIOHTTP_TCP_KEEPCNT,
    AIOHTTP_TCP_KEEPIDLE,
    AIOHTTP_TCP_KEEPINTVL,
    AIOHTTP_TTL_DNS_CACHE,
    COMPLETION_HTTP_FALLBACK_SECONDS,
    DEFAULT_SSL_CIPHERS,
    HTTP_HANDLER_CONNECT_TIMEOUT_SECONDS,
)
from litellm.litellm_core_utils.logging_utils import track_llm_api_timing
from litellm.litellm_core_utils.request_timeout_resolver import (
    get_configured_request_timeout,
)
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


# aiohttp 3.10+ exposes a `socket_factory` kwarg on TCPConnector. Older
# versions don't — detect once and skip the keep-alive wiring there.
# https://docs.aiohttp.org/en/stable/client_reference.html#aiohttp.TCPConnector
_AIOHTTP_SUPPORTS_SOCKET_FACTORY: Final = "socket_factory" in inspect.signature(TCPConnector.__init__).parameters

_AddrInfo: TypeAlias = tuple[int | socket.AddressFamily, int | socket.SocketKind, int, str, tuple[object, ...]]

_RequestContent: TypeAlias = str | bytes | Iterable[bytes] | AsyncIterable[bytes]


class _TCPConnectorKwargs(TypedDict, total=False):
    local_addr: tuple[str, int] | None
    ssl: "ssl.SSLContext | bool"
    keepalive_timeout: float
    ttl_dns_cache: int
    enable_cleanup_closed: bool
    limit: int
    limit_per_host: int
    socket_factory: Callable[[_AddrInfo], socket.socket]


def _build_aiohttp_keepalive_socket_factory() -> Callable[[_AddrInfo], socket.socket] | None:
    """
    Build a socket_factory that enables SO_KEEPALIVE on aiohttp TCP sockets.

    Why: by default, aiohttp creates sockets without SO_KEEPALIVE, so the kernel
    sends nothing during a long idle TCP connection. NAT/LB hops (e.g. AWS NAT
    Gateway, 350s idle timeout) reap the flow well before slow provider
    responses (OpenAI/Azure: up to 600s) arrive. Enabling SO_KEEPALIVE makes
    the kernel emit TCP probes that reset the NAT idle timer.

    Returns None when AIOHTTP_SO_KEEPALIVE is disabled or aiohttp is too old.
    """
    if not AIOHTTP_SO_KEEPALIVE or not _AIOHTTP_SUPPORTS_SOCKET_FACTORY:
        return None

    def factory(addr_info: _AddrInfo) -> socket.socket:
        family, type_, proto = addr_info[0], addr_info[1], addr_info[2]
        sock: Final = socket.socket(family=family, type=type_, proto=proto)
        sock.setblocking(False)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        # Linux: TCP_KEEPIDLE is idle-before-first-probe.
        # macOS/Darwin: TCP_KEEPALIVE is the equivalent.
        if hasattr(socket, "TCP_KEEPIDLE"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, AIOHTTP_TCP_KEEPIDLE)
        elif hasattr(socket, "TCP_KEEPALIVE"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, AIOHTTP_TCP_KEEPIDLE)
        if hasattr(socket, "TCP_KEEPINTVL"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, AIOHTTP_TCP_KEEPINTVL)
        if hasattr(socket, "TCP_KEEPCNT"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, AIOHTTP_TCP_KEEPCNT)
        return sock

    return factory


def get_default_headers() -> dict:
    """
    Get default headers for HTTP requests.

    - Default: `User-Agent: litellm/{version}`
    - Override: set `LITELLM_USER_AGENT` to fully override the header value.
    """
    user_agent: Final = os.environ.get("LITELLM_USER_AGENT")
    if user_agent is not None:
        return {"User-Agent": user_agent}

    return {"User-Agent": f"litellm/{version}"}


# Initialize headers (User-Agent)
headers: Final = get_default_headers()

# https://www.python-httpx.org/advanced/timeouts
_DEFAULT_TIMEOUT: Final = httpx.Timeout(
    timeout=COMPLETION_HTTP_FALLBACK_SECONDS,
    connect=HTTP_HANDLER_CONNECT_TIMEOUT_SECONDS,
)


def _default_cached_client_timeout() -> httpx.Timeout:
    """Timeout for cached default httpx clients; honors an explicit litellm.request_timeout."""
    configured: Final = get_configured_request_timeout()
    if configured is None:
        return _DEFAULT_TIMEOUT
    return httpx.Timeout(timeout=configured, connect=HTTP_HANDLER_CONNECT_TIMEOUT_SECONDS)


_CLIENT_REFCOUNT_WHEN_HANDLER_IS_SOLE_REFERRER: Final = 2


def _handler_may_close_client(client_refcount: int, owns_client: bool) -> bool:
    """
    Whether a handler being finalized may close its client.

    Only when the handler built the client and is still its sole referrer. Finalization
    proves that nothing references the *handler*; it proves nothing about the client, which
    a cached handler may have handed to consumers that outlive it. Callers must read the
    refcount at the call site, since binding the client to a parameter would inflate it.
    """
    return owns_client and client_refcount <= _CLIENT_REFCOUNT_WHEN_HANDLER_IS_SOLE_REFERRER


def blocked_cookie_jar() -> CookieJar:
    """A jar that stores no response cookie and sends none, for httpx clients.

    LiteLLM's outbound clients are pooled and shared by every caller, so a cookie one
    upstream sets would be replayed to every other upstream on a matching domain.
    """
    return CookieJar(policy=DefaultCookiePolicy(allowed_domains=()))


_STREAMING_ERROR_BODY_READ_TIMEOUT_SECONDS: Final = 5.0
_STREAMING_ERROR_BODY_READ_EXECUTOR: Final = concurrent.futures.ThreadPoolExecutor(
    max_workers=50,
    thread_name_prefix="litellm-streaming-error-body-read",
)


def _prepare_request_data_and_content(
    data: dict | str | bytes | None = None,
    content: _RequestContent | None = None,
) -> tuple[dict | Mapping | None, _RequestContent | None]:
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
_ssl_context_cache: Final[dict[tuple[str | None, str | None, str | None], ssl.SSLContext]] = {}


def _create_ssl_context(
    cafile: str | None,
    ssl_security_level: str | None,
    ssl_ecdh_curve: str | None,
) -> ssl.SSLContext:
    """
    Create an SSL context with the given configuration.
    This is separated from get_ssl_configuration to enable caching.
    """
    custom_ssl_context: Final = ssl.create_default_context(cafile=cafile)

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
            verbose_logger.debug("SSL ECDH curve set to: %s", ssl_ecdh_curve)
        except AttributeError:
            verbose_logger.warning(
                "SSL ECDH curve configuration not supported. Python version: %s, OpenSSL version: %s. Requested curve: %s. Continuing with default curves.",
                sys.version.split()[0],
                ssl.OPENSSL_VERSION,
                ssl_ecdh_curve,
            )
        except ValueError as e:
            # Invalid curve name
            verbose_logger.warning(
                "Invalid SSL ECDH curve name: '%s'. %s. Common valid curves: X25519, prime256v1, secp384r1, secp521r1. Continuing with default curves (including PQC).",
                ssl_ecdh_curve,
                e,
            )

    return custom_ssl_context


def get_ssl_verify(
    ssl_verify: bool | str | None = None,
) -> bool | str:
    """
    Common utility to resolve the SSL verification setting.
    Prioritizes:
    1. Passed-in ssl_verify
    2. os.environ["SSL_VERIFY"]
    3. litellm.ssl_verify
    4. os.environ["SSL_CERT_FILE"] (if ssl_verify is True)

    Returns:
        Union[bool, str]: The resolved SSL verification setting (bool or path to CA bundle)
    """
    from litellm.secret_managers.main import str_to_bool

    if ssl_verify is None:
        ssl_verify = os.getenv("SSL_VERIFY", litellm.ssl_verify)

    # Convert string "False"/"True" to boolean if applicable
    if isinstance(ssl_verify, str):
        # If it's a file path, return it directly
        if os.path.exists(ssl_verify):
            return ssl_verify

        # Otherwise, check if it's a boolean string
        ssl_verify_bool: Final = str_to_bool(ssl_verify)
        if ssl_verify_bool is not None:
            ssl_verify = ssl_verify_bool

    # If SSL verification is enabled, check for SSL_CERT_FILE override
    if ssl_verify is True:
        ssl_cert_file: Final = os.getenv("SSL_CERT_FILE")
        if ssl_cert_file and os.path.exists(ssl_cert_file):
            return ssl_cert_file

    return ssl_verify if ssl_verify is not None else True


def get_ssl_configuration(
    ssl_verify: VerifyTypes | None = None,
) -> bool | str | ssl.SSLContext:
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
    if isinstance(ssl_verify, ssl.SSLContext):
        # If ssl_verify is already an SSLContext, return it directly
        return ssl_verify

    # Get resolved ssl_verify
    ssl_verify = get_ssl_verify(ssl_verify=ssl_verify)

    ssl_security_level: Final = os.getenv("SSL_SECURITY_LEVEL", litellm.ssl_security_level)
    ssl_ecdh_curve: Final = os.getenv("SSL_ECDH_CURVE", litellm.ssl_ecdh_curve)

    cafile = None
    if isinstance(ssl_verify, str) and os.path.exists(ssl_verify):
        cafile = ssl_verify
    if not cafile:
        ssl_cert_file: Final = os.getenv("SSL_CERT_FILE")
        if ssl_cert_file and os.path.exists(ssl_cert_file):
            cafile = ssl_cert_file
        else:
            cafile = certifi.where()

    if ssl_verify is not False:
        # Create cache key from configuration parameters
        cache_key: Final = (cafile, ssl_security_level, ssl_ecdh_curve)

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


_shared_realtime_ssl_context: bool | str | ssl.SSLContext | None = None


def get_shared_realtime_ssl_context() -> bool | str | ssl.SSLContext:
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
        key_index: Final = error_message.find("key=")
    else:
        return error_message

    # If key is found
    if key_index != -1:
        # Find the end of the key parameter (next & or end of string)
        next_param: Final = error_message.find("&", key_index)

        if next_param == -1:
            # If no more parameters, mask until the end of the string
            masked_message = error_message[: key_index + 4] + "[REDACTED_API_KEY]"
        else:
            # Replace the key with redacted value, keeping other parameters
            masked_message = error_message[: key_index + 4] + "[REDACTED_API_KEY]" + error_message[next_param:]

        return masked_message

    return error_message


def _safe_get_response_text(response: httpx.Response) -> str:
    """Safely read response text, falling back to empty string on decoding errors."""
    try:
        return response.text
    except Exception:
        return ""


async def _safe_aread_response(response: httpx.Response, timeout: float | None = None) -> bytes:
    """Safely read async response body, falling back to empty bytes on errors."""
    try:
        if timeout is not None:
            return await asyncio.wait_for(response.aread(), timeout=timeout)
        return await response.aread()
    except Exception:
        return b""


def _safe_read_response(response: httpx.Response, timeout: float | None = None) -> bytes:
    """Safely read sync response body, falling back to empty bytes on errors."""
    try:
        if timeout is not None:
            future: Final = _STREAMING_ERROR_BODY_READ_EXECUTOR.submit(response.read)
            try:
                return future.result(timeout=timeout)
            except Exception:
                response.close()
                return b""
        return response.read()
    except Exception:
        return b""


def _raise_masked_sync_error(e: httpx.HTTPStatusError, stream: bool) -> None:
    """Raise a MaskedHTTPStatusError for sync HTTP handlers."""
    if stream:
        try:
            _body: Final = mask_sensitive_info(
                _safe_read_response(
                    e.response,
                    timeout=_STREAMING_ERROR_BODY_READ_TIMEOUT_SECONDS,
                )
            )
            raise MaskedHTTPStatusError(e, message=_body, text=_body) from None
        finally:
            try:
                e.response.close()
            except Exception:
                pass
    _text: Final = mask_sensitive_info(_safe_get_response_text(e.response))
    raise MaskedHTTPStatusError(e, message=_text, text=_text) from None


async def _raise_masked_async_error(e: httpx.HTTPStatusError, stream: bool) -> None:
    """Raise a MaskedHTTPStatusError for async HTTP handlers."""
    if stream:
        try:
            _body: Final = mask_sensitive_info(
                await _safe_aread_response(
                    e.response,
                    timeout=_STREAMING_ERROR_BODY_READ_TIMEOUT_SECONDS,
                )
            )
            raise MaskedHTTPStatusError(e, message=_body, text=_body) from None
        finally:
            try:
                await e.response.aclose()
            except Exception:
                pass
    _text: Final = mask_sensitive_info(_safe_get_response_text(e.response))
    raise MaskedHTTPStatusError(e, message=_text, text=_text) from None


class MaskedHTTPStatusError(httpx.HTTPStatusError):
    def __init__(self, original_error, message: str | None = None, text: str | None = None):
        # Create a new error with the masked URL
        masked_url: Final = mask_sensitive_info(str(original_error.request.url))
        # Mask the original exception message too (it contains the full URL)
        masked_original_message: Final = mask_sensitive_info(str(original_error))

        # Safely access response content — decompression can fail (e.g. zlib error).
        # `.content` returns already-decoded bytes, so we must strip transport
        # encoding headers before rebuilding the Response (otherwise httpx will
        # try to decode the bytes a second time and raise DecodingError).
        try:
            response_content = original_error.response.content
        except Exception:
            response_content = b""

        response_headers: Final = {
            k: v
            for k, v in original_error.response.headers.items()
            if k.lower() not in ("content-encoding", "content-length")
        }

        try:
            request_content = original_error.request.content
        except httpx.RequestNotRead:
            request_content = b""

        masked_request: Final = httpx.Request(
            method=original_error.request.method,
            url=masked_url,
            headers=original_error.request.headers,
            content=request_content,
        )

        super().__init__(
            message=masked_original_message,
            request=masked_request,
            # Attach the masked request so `response.request` is set — otherwise
            # downstream code that inspects err.response.request (e.g.
            # exception_mapping_utils) hits `RuntimeError: .request not set`.
            response=httpx.Response(
                status_code=original_error.response.status_code,
                content=response_content,
                headers=response_headers,
                request=masked_request,
            ),
        )
        self.message = message
        self.text = text
        self.status_code = original_error.response.status_code


class AsyncHTTPHandler:
    def __init__(
        self,
        timeout: float | httpx.Timeout | None = None,
        event_hooks: Mapping[str, list[Callable[..., object]]] | None = None,
        concurrent_limit=None,  # Kept for backward compatibility, but ignored (no limits)
        client_alias: str | None = None,  # name for client in logs
        ssl_verify: VerifyTypes | None = None,
        shared_session: Optional["ClientSession"] = None,
    ):
        self.timeout = timeout
        self.event_hooks = event_hooks
        self.ssl_verify = ssl_verify
        self.shared_session = shared_session
        self._owns_client = True
        self._client = self.create_client(
            timeout=timeout,
            event_hooks=event_hooks,
            ssl_verify=ssl_verify,
            shared_session=shared_session,
        )
        self.client_alias = client_alias

    @property
    def client(self) -> httpx.AsyncClient:
        if self._owns_client and self._client.is_closed:
            self._client = self.create_client(
                timeout=self.timeout,
                event_hooks=self.event_hooks,
                ssl_verify=self.ssl_verify,
                shared_session=self.shared_session,
            )
        return self._client

    @client.setter
    def client(self, client: httpx.AsyncClient) -> None:
        self._client = client
        self._owns_client = False

    def create_client(
        self,
        timeout: float | httpx.Timeout | None,
        event_hooks: Mapping[str, list[Callable[..., object]]] | None,
        ssl_verify: VerifyTypes | None = None,
        shared_session: Optional["ClientSession"] = None,
    ) -> httpx.AsyncClient:
        # Get unified SSL configuration
        ssl_config: Final = get_ssl_configuration(ssl_verify)

        # An SSL certificate used by the requested host to authenticate the client.
        # /path/to/client.pem
        cert: Final = os.getenv("SSL_CERTIFICATE", litellm.ssl_certificate)

        if timeout is None:
            timeout = _DEFAULT_TIMEOUT
        # Create a client with a connection pool

        transport: Final = AsyncHTTPHandler._create_async_transport(
            ssl_context=ssl_config if isinstance(ssl_config, ssl.SSLContext) else None,
            ssl_verify=ssl_config if isinstance(ssl_config, bool) else None,
            shared_session=shared_session,
        )

        # Get default headers (User-Agent, overridable via LITELLM_USER_AGENT)
        default_headers: Final = get_default_headers()

        return httpx.AsyncClient(
            transport=transport,
            event_hooks=event_hooks,
            timeout=timeout,
            verify=ssl_config,
            cert=cert,
            headers=default_headers,
            cookies=blocked_cookie_jar(),
            follow_redirects=True,
        )

    async def close(self):
        # Close the client when you're done with it
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self):
        return self.client

    async def __aexit__(self):
        # close the client when exiting
        await self.close()

    async def get(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
        follow_redirects: bool | None = None,
        timeout: float | httpx.Timeout | None = None,
    ):
        # Set follow_redirects to UseClientDefault if None
        _follow_redirects: Final = follow_redirects if follow_redirects is not None else USE_CLIENT_DEFAULT

        params = params or {}
        params.update(HTTPHandler.extract_query_params(url))

        response: Final = await self.client.get(
            url,
            params=params,
            headers=headers,
            follow_redirects=_follow_redirects,
            timeout=timeout if timeout is not None else USE_CLIENT_DEFAULT,
        )
        return response

    @track_llm_api_timing()
    async def post(
        self,
        url: str,
        data: dict | str | bytes | None = None,
        json: dict | None = None,
        params: dict | None = None,
        headers: dict | None = None,
        timeout: float | httpx.Timeout | None = None,
        stream: bool = False,
        logging_obj: LiteLLMLoggingObject | None = None,
        files: RequestFiles | None = None,
        content: _RequestContent | None = None,
    ):
        start_time: Final = time.time()
        try:
            if timeout is None:
                timeout = self.timeout

            # Prepare data/content parameters to prevent httpx DeprecationWarning (memory leak fix)
            request_data, request_content = _prepare_request_data_and_content(data, content)

            req: Final = self.client.build_request(
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
            response: Final = await self.client.send(req, stream=stream)
            response.raise_for_status()
            return response
        except (httpx.RemoteProtocolError, httpx.ConnectError):
            # Retry the request with a new session if there is a connection error
            new_client: Final = self.create_client(timeout=timeout, event_hooks=self.event_hooks)
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
            end_time: Final = time.time()
            time_delta: Final = round(end_time - start_time, 3)
            headers = {}
            error_response: Final[httpx.Response | None] = getattr(e, "response", None)
            if error_response is not None:
                for key, value in error_response.headers.items():
                    headers[f"response_headers-{key}"] = value

            raise litellm.Timeout(
                message=f"Connection timed out. Timeout passed={timeout}, time taken={time_delta} seconds",
                model="default-model-name",
                llm_provider="litellm-httpx-handler",
                headers=headers,
            )
        except httpx.HTTPStatusError as e:
            await _raise_masked_async_error(e, stream)
        except Exception as e:
            raise e

    async def put(
        self,
        url: str,
        data: dict | str | bytes | None = None,
        json: dict | None = None,
        params: dict | None = None,
        headers: dict | None = None,
        timeout: float | httpx.Timeout | None = None,
        stream: bool = False,
        content: _RequestContent | None = None,
    ):
        try:
            if timeout is None:
                timeout = self.timeout

            # Prepare data/content parameters to prevent httpx DeprecationWarning (memory leak fix)
            request_data, request_content = _prepare_request_data_and_content(data, content)

            req: Final = self.client.build_request(
                "PUT",
                url,
                data=request_data,
                json=json,
                params=params,
                headers=headers,
                timeout=timeout,
                content=request_content,
            )
            response: Final = await self.client.send(req)
            response.raise_for_status()
            return response
        except (httpx.RemoteProtocolError, httpx.ConnectError):
            # Retry the request with a new session if there is a connection error
            new_client: Final = self.create_client(timeout=timeout, event_hooks=self.event_hooks)
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
            error_response: Final[httpx.Response | None] = getattr(e, "response", None)
            if error_response is not None:
                for key, value in error_response.headers.items():
                    headers[f"response_headers-{key}"] = value

            raise litellm.Timeout(
                message=f"Connection timed out after {timeout} seconds.",
                model="default-model-name",
                llm_provider="litellm-httpx-handler",
                headers=headers,
            )
        except httpx.HTTPStatusError as e:
            await _raise_masked_async_error(e, stream)
        except Exception as e:
            raise e

    async def patch(
        self,
        url: str,
        data: dict | str | bytes | None = None,
        json: dict | None = None,
        params: dict | None = None,
        headers: dict | None = None,
        timeout: float | httpx.Timeout | None = None,
        stream: bool = False,
        content: _RequestContent | None = None,
    ):
        try:
            if timeout is None:
                timeout = self.timeout

            # Prepare data/content parameters to prevent httpx DeprecationWarning (memory leak fix)
            request_data, request_content = _prepare_request_data_and_content(data, content)

            req: Final = self.client.build_request(
                "PATCH",
                url,
                data=request_data,
                json=json,
                params=params,
                headers=headers,
                timeout=timeout,
                content=request_content,
            )
            response: Final = await self.client.send(req)
            response.raise_for_status()
            return response
        except (httpx.RemoteProtocolError, httpx.ConnectError):
            # Retry the request with a new session if there is a connection error
            new_client: Final = self.create_client(timeout=timeout, event_hooks=self.event_hooks)
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
            error_response: Final[httpx.Response | None] = getattr(e, "response", None)
            if error_response is not None:
                for key, value in error_response.headers.items():
                    headers[f"response_headers-{key}"] = value

            raise litellm.Timeout(
                message=f"Connection timed out after {timeout} seconds.",
                model="default-model-name",
                llm_provider="litellm-httpx-handler",
                headers=headers,
            )
        except httpx.HTTPStatusError as e:
            await _raise_masked_async_error(e, stream)
        except Exception as e:
            raise e

    async def delete(
        self,
        url: str,
        data: dict | str | bytes | None = None,
        json: dict | None = None,
        params: dict | None = None,
        headers: dict | None = None,
        timeout: float | httpx.Timeout | None = None,
        stream: bool = False,
        content: _RequestContent | None = None,
    ):
        try:
            if timeout is None:
                timeout = self.timeout

            # Prepare data/content parameters to prevent httpx DeprecationWarning (memory leak fix)
            request_data, request_content = _prepare_request_data_and_content(data, content)

            req: Final = self.client.build_request(
                "DELETE",
                url,
                data=request_data,
                json=json,
                params=params,
                headers=headers,
                timeout=timeout,
                content=request_content,
            )
            response: Final = await self.client.send(req, stream=stream)
            response.raise_for_status()
            return response
        except (httpx.RemoteProtocolError, httpx.ConnectError):
            # Retry the request with a new session if there is a connection error
            new_client: Final = self.create_client(timeout=timeout, event_hooks=self.event_hooks)
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
            await _raise_masked_async_error(e, stream)
        except Exception as e:
            raise e

    async def single_connection_post_request(
        self,
        url: str,
        client: httpx.AsyncClient,
        data: dict | str | bytes | None = None,
        json: dict | None = None,
        params: dict | None = None,
        headers: dict | None = None,
        stream: bool = False,
        content: _RequestContent | None = None,
    ):
        """
        Making POST request for a single connection client.

        Used for retrying connection client errors.
        """
        # Prepare data/content parameters to prevent httpx DeprecationWarning (memory leak fix)
        request_data, request_content = _prepare_request_data_and_content(data, content)

        req: Final = client.build_request(
            "POST",
            url,
            data=request_data,
            json=json,
            params=params,
            headers=headers,
            content=request_content,
        )
        response: Final = await client.send(req, stream=stream)
        response.raise_for_status()
        return response

    # Strong references to finalizer-scheduled client-close tasks. A bare
    # create_task() result may be garbage-collected before it runs, leaving
    # the underlying aiohttp session unclosed ("Unclosed client session").
    # Mirrors LiteLLMAiohttpTransport._background_close_tasks.
    _finalizer_close_tasks: ClassVar[set["asyncio.Task[None]"]] = set()  # mutable-ok: strong refs for pending closes

    @classmethod
    def _on_finalizer_close_done(cls, task: "asyncio.Task[None]") -> None:
        cls._finalizer_close_tasks.discard(task)
        if task.cancelled():
            return
        exc: Final = task.exception()
        if exc is not None:
            verbose_logger.debug("Error closing client at finalization: %s", exc)

    def _aiohttp_session_bound_elsewhere(self, loop: asyncio.AbstractEventLoop) -> bool:
        """True when the wrapped aiohttp session is bound to a loop other than
        ``loop`` — awaiting ``aclose()`` here would touch that loop's internals."""
        from litellm.llms.custom_httpx.aiohttp_transport import (
            LiteLLMAiohttpTransport,
        )

        transport: Final = getattr(self._client, "_transport", None)
        if not isinstance(transport, LiteLLMAiohttpTransport):
            return False
        session: Final = transport.client
        if not isinstance(session, ClientSession) or session.closed:
            return False
        return getattr(session, "_loop", None) is not loop

    def _dispose_wrapped_aiohttp_session(self) -> None:
        """Dispose the wrapped aiohttp session when ``aclose()`` cannot run here.

        Finalization either has no running loop, or a loop the session is not
        bound to. Delegating to the transport's lifecycle-aware disposal picks
        the safe path per session state (async close on its own loop, threadsafe
        handoff to a loop running elsewhere, or the synchronous connector
        teardown that flips the flags ``ClientSession.__del__`` checks), so no
        "Unclosed client session" / "Unclosed connector" warnings fire at
        garbage collection.
        """
        from litellm.llms.custom_httpx.aiohttp_transport import (
            LiteLLMAiohttpTransport,
        )

        transport: Final = getattr(self._client, "_transport", None)
        if not isinstance(transport, LiteLLMAiohttpTransport):
            return
        # A shared session (e.g. the proxy's) is never this handler's to close.
        if not getattr(transport, "_owns_session", False):
            return
        session: Final = transport.client
        if isinstance(session, ClientSession) and not session.closed:
            transport._close_recycled_session(session)  # pyright: ignore[reportPrivateUsage]  # deliberate reuse of the transport's lifecycle-aware disposal; an async close can never run in this context

    def __del__(self) -> None:
        try:
            if not _handler_may_close_client(sys.getrefcount(self._client), self._owns_client):
                return
            try:
                loop: Final = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop at finalization time (worker threads after
                # their loop closed, interpreter/worker shutdown, GC in a
                # sync context). An async close can never run here.
                self._dispose_wrapped_aiohttp_session()
                return
            if self._aiohttp_session_bound_elsewhere(loop):
                # GC ran on a live loop (e.g. the app's) but the session
                # belongs to another, possibly dead, loop — awaiting aclose()
                # here is the cross-loop path the transport refuses.
                self._dispose_wrapped_aiohttp_session()
                return
            task: Final = loop.create_task(self._client.aclose())
            cls: Final = type(self)
            cls._finalizer_close_tasks.add(task)
            task.add_done_callback(cls._on_finalizer_close_done)
        except Exception:
            pass

    @staticmethod
    def _create_async_transport(
        ssl_context: ssl.SSLContext | None = None,
        ssl_verify: bool | None = None,
        shared_session: Optional["ClientSession"] = None,
    ) -> LiteLLMAiohttpTransport | AsyncHTTPTransport | None:
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
        ssl_verify: bool | None = None,
        ssl_context: ssl.SSLContext | None = None,
    ) -> _TCPConnectorKwargs:
        """
        Helper method to get SSL connector initialization arguments for aiohttp TCPConnector.

        SSL Configuration Priority:
        1. If ssl_context is provided -> use the custom SSL context
        2. If ssl_verify is False -> disable SSL verification (ssl=False)

        Returns:
            Dict with appropriate SSL configuration for TCPConnector
        """
        connector_kwargs: Final[_TCPConnectorKwargs] = {
            "local_addr": ("0.0.0.0", 0) if litellm.force_ipv4 else None,
        }

        if ssl_context is not None:
            # Priority 1: Use the provided custom SSL context
            connector_kwargs["ssl"] = ssl_context
        elif ssl_verify is False:
            # Priority 2: Explicitly disable SSL verification
            connector_kwargs["ssl"] = False

        return connector_kwargs

    @staticmethod
    def _create_aiohttp_transport(
        ssl_verify: bool | None = None,
        ssl_context: ssl.SSLContext | None = None,
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

        connector_kwargs = AsyncHTTPHandler._get_ssl_connector_kwargs(ssl_verify=ssl_verify, ssl_context=ssl_context)
        #########################################################
        # Check if user enabled aiohttp trust env
        # use for HTTP_PROXY, HTTPS_PROXY, etc.
        ########################################################
        trust_env: bool = litellm.aiohttp_trust_env
        if str_to_bool(os.getenv("AIOHTTP_TRUST_ENV", "False")) is True:
            trust_env = True

        #########################################################
        # Determine SSL config to pass to transport for per-request override
        # This ensures ssl_verify works even with shared sessions
        #########################################################
        ssl_for_transport: bool | ssl.SSLContext | None = None
        if ssl_context is not None:
            ssl_for_transport = ssl_context
        elif ssl_verify is False:
            ssl_for_transport = False

        verbose_logger.debug("Creating AiohttpTransport...")

        transport_connector_kwargs: Final[_TCPConnectorKwargs] = {
            "keepalive_timeout": AIOHTTP_KEEPALIVE_TIMEOUT,
            "ttl_dns_cache": AIOHTTP_TTL_DNS_CACHE,
            **connector_kwargs,
        }
        if AIOHTTP_NEEDS_CLEANUP_CLOSED:
            transport_connector_kwargs["enable_cleanup_closed"] = True
        if AIOHTTP_CONNECTOR_LIMIT > 0:
            transport_connector_kwargs["limit"] = AIOHTTP_CONNECTOR_LIMIT
        if AIOHTTP_CONNECTOR_LIMIT_PER_HOST > 0:
            transport_connector_kwargs["limit_per_host"] = AIOHTTP_CONNECTOR_LIMIT_PER_HOST
        # Returns None when SO_KEEPALIVE is disabled or aiohttp is too old to
        # accept socket_factory — version detection lives inside the builder.
        socket_factory: Final = _build_aiohttp_keepalive_socket_factory()
        if socket_factory is not None:
            transport_connector_kwargs["socket_factory"] = socket_factory

        def session_factory() -> ClientSession:
            return ClientSession(
                connector=TCPConnector(**transport_connector_kwargs),
                cookie_jar=DummyCookieJar(),
                trust_env=trust_env,
            )

        # Use shared session if provided and valid
        if shared_session is not None and not shared_session.closed:
            verbose_logger.debug("SHARED SESSION: Reusing existing ClientSession (ID: %s)", id(shared_session))
            return LiteLLMAiohttpTransport(
                client=shared_session,
                ssl_verify=ssl_for_transport,
                owns_session=False,
                session_factory=session_factory,
            )

        # Create new session only if none provided or existing one is invalid
        verbose_logger.debug("NEW SESSION: Creating new ClientSession (no shared session provided)")
        return LiteLLMAiohttpTransport(
            client=session_factory,
            ssl_verify=ssl_for_transport,
        )

    @staticmethod
    def _create_httpx_transport() -> AsyncHTTPTransport | None:
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
        timeout: float | httpx.Timeout | None = None,
        concurrent_limit=None,  # Kept for backward compatibility, but ignored (no limits)
        client: httpx.Client | None = None,
        ssl_verify: bool | str | None = None,
        disable_default_headers: bool
        | None = False,  # arize phoenix returns different API responses when user agent header in request
    ):
        self.timeout = timeout
        self.ssl_verify = ssl_verify
        self.disable_default_headers = disable_default_headers
        self._owns_client = client is None
        self._heal_lock = threading.Lock()
        self._client = self.create_client() if client is None else client

    def create_client(self) -> httpx.Client:
        # Get unified SSL configuration
        ssl_config: Final = get_ssl_configuration(self.ssl_verify)

        # An SSL certificate used by the requested host to authenticate the client.
        # /path/to/client.pem
        cert: Final = os.getenv("SSL_CERTIFICATE", litellm.ssl_certificate)

        # Get default headers (User-Agent, overridable via LITELLM_USER_AGENT)
        default_headers: Final = get_default_headers() if not self.disable_default_headers else None

        # Create a client with a connection pool
        return httpx.Client(
            transport=self._create_sync_transport(),
            timeout=self.timeout if self.timeout is not None else _DEFAULT_TIMEOUT,
            verify=ssl_config,
            cert=cert,
            headers=default_headers,
            cookies=blocked_cookie_jar(),
            follow_redirects=True,
        )

    @property
    def client(self) -> httpx.Client:
        if self._owns_client and self._client.is_closed:
            with self._heal_lock:
                if self._owns_client and self._client.is_closed:
                    self._client = self.create_client()
        return self._client

    @client.setter
    def client(self, client: httpx.Client) -> None:
        self._client = client
        self._owns_client = False

    def close(self):
        # Close the client when you're done with it
        if self._owns_client:
            self._client.close()

    def get(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
        follow_redirects: bool | None = None,
        timeout: float | httpx.Timeout | None = None,
    ):
        # Set follow_redirects to UseClientDefault if None
        _follow_redirects: Final = follow_redirects if follow_redirects is not None else USE_CLIENT_DEFAULT
        params = params or {}
        params.update(self.extract_query_params(url))

        response: Final = self.client.get(
            url,
            params=params,
            headers=headers,
            follow_redirects=_follow_redirects,
            timeout=timeout if timeout is not None else USE_CLIENT_DEFAULT,
        )

        return response

    @staticmethod
    def extract_query_params(url: str) -> dict[str, str]:
        """
        Parse a URL’s query-string into a dict.

        :param url: full URL, e.g. "https://.../path?foo=1&bar=2"
        :return: {"foo": "1", "bar": "2"}
        """
        from urllib.parse import parse_qsl, urlsplit

        parts: Final = urlsplit(url)
        return dict(parse_qsl(parts.query))

    def post(
        self,
        url: str,
        data: dict | str | bytes | None = None,
        json: dict | str | list | None = None,
        params: dict | None = None,
        headers: dict | None = None,
        stream: bool = False,
        timeout: float | httpx.Timeout | None = None,
        files: dict | RequestFiles | None = None,
        content: _RequestContent | None = None,
        logging_obj: LiteLLMLoggingObject | None = None,
    ):
        try:
            # Prepare data/content parameters to prevent httpx DeprecationWarning (memory leak fix)
            request_data, request_content = _prepare_request_data_and_content(data, content)

            if timeout is not None:
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
            else:
                req = self.client.build_request(
                    "POST",
                    url,
                    data=request_data,
                    json=json,
                    params=params,
                    headers=headers,
                    files=files,
                    content=request_content,
                )
            response: Final = self.client.send(req, stream=stream)
            response.raise_for_status()
            return response
        except httpx.TimeoutException:
            raise litellm.Timeout(
                message=f"Connection timed out after {timeout} seconds.",
                model="default-model-name",
                llm_provider="litellm-httpx-handler",
            )
        except httpx.HTTPStatusError as e:
            _raise_masked_sync_error(e, stream)
        except Exception as e:
            raise e

    def patch(
        self,
        url: str,
        data: dict | str | bytes | None = None,
        json: dict | str | None = None,
        params: dict | None = None,
        headers: dict | None = None,
        stream: bool = False,
        timeout: float | httpx.Timeout | None = None,
        content: _RequestContent | None = None,
    ):
        try:
            # Prepare data/content parameters to prevent httpx DeprecationWarning (memory leak fix)
            request_data, request_content = _prepare_request_data_and_content(data, content)

            if timeout is not None:
                req = self.client.build_request(
                    "PATCH",
                    url,
                    data=request_data,
                    json=json,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                    content=request_content,
                )
            else:
                req = self.client.build_request(
                    "PATCH",
                    url,
                    data=request_data,
                    json=json,
                    params=params,
                    headers=headers,
                    content=request_content,
                )
            response: Final = self.client.send(req, stream=stream)
            response.raise_for_status()
            return response
        except httpx.TimeoutException:
            raise litellm.Timeout(
                message=f"Connection timed out after {timeout} seconds.",
                model="default-model-name",
                llm_provider="litellm-httpx-handler",
            )
        except httpx.HTTPStatusError as e:
            _raise_masked_sync_error(e, stream)
        except Exception as e:
            raise e

    def put(
        self,
        url: str,
        data: dict | str | bytes | None = None,
        json: dict | str | None = None,
        params: dict | None = None,
        headers: dict | None = None,
        stream: bool = False,
        timeout: float | httpx.Timeout | None = None,
        content: _RequestContent | None = None,
    ):
        try:
            # Prepare data/content parameters to prevent httpx DeprecationWarning (memory leak fix)
            request_data, request_content = _prepare_request_data_and_content(data, content)

            if timeout is not None:
                req = self.client.build_request(
                    "PUT",
                    url,
                    data=request_data,
                    json=json,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                    content=request_content,
                )
            else:
                req = self.client.build_request(
                    "PUT",
                    url,
                    data=request_data,
                    json=json,
                    params=params,
                    headers=headers,
                    content=request_content,
                )
            response: Final = self.client.send(req, stream=stream)
            return response
        except httpx.TimeoutException:
            raise litellm.Timeout(
                message=f"Connection timed out after {timeout} seconds.",
                model="default-model-name",
                llm_provider="litellm-httpx-handler",
            )
        except httpx.HTTPStatusError as e:
            _raise_masked_sync_error(e, stream)
        except Exception as e:
            raise e

    def delete(
        self,
        url: str,
        data: dict | str | bytes | None = None,
        json: dict | None = None,
        params: dict | None = None,
        headers: dict | None = None,
        timeout: float | httpx.Timeout | None = None,
        stream: bool = False,
        content: _RequestContent | None = None,
    ):
        try:
            # Prepare data/content parameters to prevent httpx DeprecationWarning (memory leak fix)
            request_data, request_content = _prepare_request_data_and_content(data, content)

            if timeout is not None:
                req = self.client.build_request(
                    "DELETE",
                    url,
                    data=request_data,
                    json=json,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                    content=request_content,
                )
            else:
                req = self.client.build_request(
                    "DELETE",
                    url,
                    data=request_data,
                    json=json,
                    params=params,
                    headers=headers,
                    content=request_content,
                )
            response: Final = self.client.send(req, stream=stream)
            response.raise_for_status()
            return response
        except httpx.TimeoutException:
            raise litellm.Timeout(
                message=f"Connection timed out after {timeout} seconds.",
                model="default-model-name",
                llm_provider="litellm-httpx-handler",
            )
        except httpx.HTTPStatusError as e:
            _raise_masked_sync_error(e, stream)
        except Exception as e:
            raise e

    def __del__(self) -> None:
        try:
            if _handler_may_close_client(sys.getrefcount(self._client), self._owns_client):
                self._client.close()
        except Exception:
            pass

    def _create_sync_transport(self) -> HTTPTransport | None:
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
    llm_provider: LlmProviders | httpxSpecialProvider,
    params: dict | None = None,
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

    _cache_key_name: Final = "async_httpx_client" + _params_key_name + llm_provider

    # Lazily initialize the global in-memory client cache to avoid relying on
    # litellm globals being fully populated during import time.
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is None:
        from litellm.caching.llm_caching_handler import LLMClientCache

        cache = LLMClientCache()
        setattr(litellm, "in_memory_llm_clients_cache", cache)

    _cached_client: Final = cache.get_cache(_cache_key_name)
    if _cached_client:
        return _cached_client

    if params is not None:
        # Filter out params that are only used for cache key, not for AsyncHTTPHandler.__init__
        handler_params: Final = {k: v for k, v in params.items() if k != "disable_aiohttp_transport"}
        handler_params["shared_session"] = shared_session
        _new_client = AsyncHTTPHandler(**handler_params)
    else:
        _new_client = AsyncHTTPHandler(
            timeout=_default_cached_client_timeout(),
            shared_session=shared_session,
        )

    cache.set_cache(
        key=_cache_key_name,
        value=_new_client,
        ttl=_DEFAULT_TTL_FOR_HTTPX_CLIENTS,
        litellm_owned_client=True,
    )
    return _new_client


def _get_httpx_client(params: dict | None = None) -> HTTPHandler:
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

    _cache_key_name: Final = "httpx_client" + _params_key_name

    # Lazily initialize the global in-memory client cache to avoid relying on
    # litellm globals being fully populated during import time.
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is None:
        from litellm.caching.llm_caching_handler import LLMClientCache

        cache = LLMClientCache()
        setattr(litellm, "in_memory_llm_clients_cache", cache)

    _cached_client: Final = cache.get_cache(_cache_key_name)
    if _cached_client:
        return _cached_client

    if params is not None:
        # Filter out params that are only used for cache key, not for HTTPHandler.__init__
        handler_params: Final = {k: v for k, v in params.items() if k != "disable_aiohttp_transport"}
        _new_client = HTTPHandler(**handler_params)
    else:
        _new_client = HTTPHandler(timeout=_default_cached_client_timeout())

    cache.set_cache(
        key=_cache_key_name,
        value=_new_client,
        ttl=_DEFAULT_TTL_FOR_HTTPX_CLIENTS,
        litellm_owned_client=True,
    )
    return _new_client

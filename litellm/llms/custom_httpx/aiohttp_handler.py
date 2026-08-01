import asyncio
import ipaddress
import os
import socket
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Union, cast
from urllib.parse import urlparse

import aiohttp
import httpx  # type: ignore
from aiohttp import ClientSession, FormData
from aiohttp.abc import AbstractResolver

import litellm
import litellm.litellm_core_utils
import litellm.types
import litellm.types.utils
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.base_llm.image_variations.transformation import (
    BaseImageVariationConfig,
)
from litellm.constants import _DEFAULT_TTL_FOR_HTTPX_CLIENTS
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_ssl_configuration,
)
from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport
from litellm.types.llms.openai import FileTypes
from litellm.types.utils import HttpHandlerRequestFields, ImageResponse, LlmProviders
from litellm.utils import CustomStreamWrapper, ModelResponse, ProviderConfigManager

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

DEFAULT_TIMEOUT = 600

# CGNAT (100.64.0.0/10) is misclassified as global on Python < 3.11
_CGNAT = ipaddress.ip_network("100.64.0.0/10")


def _is_blocked_address(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if addr is not globally routable (private, loopback, reserved, etc.).

    Uses Python's built-in is_global to cover all IANA special-use ranges —
    including RFC 5737 documentation (192.0.2.0/24), RFC 2544 benchmarking
    (198.18.0.0/15), and class E (240.0.0.0/4) — without maintaining a manual list.
    CGNAT (100.64.0.0/10) is explicitly checked for Python < 3.11 compat.
    """
    # Unwrap IPv4-mapped IPv6 (::ffff:10.0.0.1 → 10.0.0.1)
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    # Python < 3.11 incorrectly marks CGNAT as global
    if isinstance(addr, ipaddress.IPv4Address) and addr in _CGNAT:
        return True
    return not addr.is_global or addr.is_multicast


def _assert_not_private_url(url: str) -> None:
    """Raise ValueError if url resolves to any private/reserved IP (SSRF protection).

    Validates all DNS answers, not just the first, to prevent A-record rotation attacks.

    On the sync path this is defence-in-depth for externally-supplied httpx clients.
    Internally-created clients use ``_SSRFGuardTransport``, which resolves, validates,
    and pins the IP at TCP-connect time — eliminating the DNS-rebinding TOCTOU window
    that a preflight-only check cannot close.

    Set ``litellm.allow_requests_to_internal_ips = True`` to disable this check
    for self-hosted / on-prem deployments where api_base is an internal address.
    """
    if litellm.allow_requests_to_internal_ips:
        return
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return
    try:
        answers = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return  # DNS failure — request will fail naturally
    for answer in answers:
        raw_ip = answer[4][0]
        try:
            addr = ipaddress.ip_address(raw_ip)
        except ValueError:
            continue
        if _is_blocked_address(addr):
            raise ValueError(
                f"api_base '{url}' resolves to a private/reserved IP address "
                f"({raw_ip}) which is not allowed (SSRF protection)"
            )


def _assert_not_private_ip_literal(url: str) -> None:
    """Raise ValueError if the url host is a private/reserved IP address literal.

    aiohttp's TCPConnector skips _SSRFGuardResolver when the host is already an
    IP address (no DNS lookup needed), so the resolver cannot block such URLs.
    This function fills that gap with a fast, non-blocking check (no DNS I/O).
    Hostname-based URLs are handled by _SSRFGuardResolver at connect time.

    Set ``litellm.allow_requests_to_internal_ips = True`` to disable this check.
    """
    if litellm.allow_requests_to_internal_ips:
        return
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return
    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        return  # Not an IP literal — resolver will validate at connect time
    if _is_blocked_address(addr):
        raise ValueError(
            f"api_base '{url}' contains a private/reserved IP address "
            f"({hostname}) which is not allowed (SSRF protection)"
        )


class _SSRFGuardResolver(AbstractResolver):
    """Custom aiohttp resolver that validates IPs at TCP-connection time.

    By hooking into aiohttp's resolver — used for every connection including
    redirect targets — this eliminates the DNS-rebinding TOCTOU window that
    a separate preflight check cannot close.  All DNS answers are validated,
    not just the first, to defend against A-record rotation.
    """

    async def resolve(
        self, host: str, port: int = 0, family: int = socket.AF_INET
    ) -> list:
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(
                host, port, family=family, type=socket.SOCK_STREAM
            )
        except socket.gaierror:
            raise  # Propagate so aiohttp wraps it in a ClientConnectorError
        if not litellm.allow_requests_to_internal_ips:
            for info in infos:
                raw_ip = info[4][0]
                try:
                    addr = ipaddress.ip_address(raw_ip)
                except ValueError:
                    continue
                if _is_blocked_address(addr):
                    raise ValueError(
                        f"Host '{host}' resolves to a private/reserved IP address "
                        f"({raw_ip}) which is not allowed (SSRF protection)"
                    )
        return [
            {
                "hostname": host,
                "host": info[4][0],
                "port": info[4][1] if len(info[4]) > 1 else port,
                "family": info[0],
                "proto": info[2],
                "flags": socket.AI_NUMERICHOST | socket.AI_NUMERICSERV,
            }
            for info in infos
        ]

    async def close(self) -> None:
        pass


class _SSRFGuardTransport(httpx.HTTPTransport):
    """Sync httpx transport that eliminates the DNS-rebinding TOCTOU gap.

    Mirrors ``_SSRFGuardResolver`` on the async path: resolves the hostname
    once, validates every returned IP, then pins the TCP connection to the
    first safe address by rewriting the request URL to an IP literal.
    httpcore skips DNS for IP-literal hosts, so the second resolution that
    creates the TOCTOU window never occurs.

    The original hostname is preserved in the ``Host`` header and the
    ``sni_hostname`` extension so that TLS certificate validation and
    virtual-host routing are unaffected.

    Set ``litellm.allow_requests_to_internal_ips = True`` to disable all
    SSRF checking.
    """

    def handle_request(self, request: httpx.Request) -> httpx.Response:  # type: ignore[override]
        if not litellm.allow_requests_to_internal_ips:
            host = request.url.host
            try:
                addr: Optional[ipaddress.IPv4Address | ipaddress.IPv6Address] = (
                    ipaddress.ip_address(host)
                )
            except ValueError:
                addr = None

            if addr is not None:
                # IP-literal host: validate inline — no DNS needed, no TOCTOU window.
                if _is_blocked_address(addr):
                    raise ValueError(
                        f"Host '{host}' is a private/reserved IP address "
                        f"which is not allowed (SSRF protection)"
                    )
            else:
                # Hostname path: resolve, validate all answers, then pin to prevent
                # a second DNS lookup (closes the DNS-rebinding TOCTOU window).
                port = request.url.port or (
                    443 if request.url.scheme == "https" else 80
                )
                try:
                    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
                except socket.gaierror:
                    raise

                for info in infos:
                    raw_ip = info[4][0]
                    try:
                        resolved = ipaddress.ip_address(raw_ip)
                    except ValueError:
                        continue
                    if _is_blocked_address(resolved):
                        raise ValueError(
                            f"Host '{host}' resolves to a private/reserved IP "
                            f"address ({raw_ip}) which is not allowed (SSRF protection)"
                        )

                if infos:
                    # Rewrite the URL to the first validated IP so that httpcore
                    # connects directly without a second DNS resolution — this closes
                    # the TOCTOU window entirely.
                    pinned_ip = infos[0][4][0]
                    pinned_url = request.url.copy_with(host=pinned_ip)

                    # Preserve the original hostname in the Host header (virtual
                    # hosting) and sni_hostname extension (TLS cert validation).
                    headers = [
                        (k, v) for k, v in request.headers.raw if k.lower() != b"host"
                    ]
                    default_port = 443 if request.url.scheme == "https" else 80
                    _port = request.url.port
                    host_header = (
                        f"{host}:{_port}"
                        if _port is not None and _port != default_port
                        else host
                    )
                    headers.append((b"host", host_header.encode("utf-8")))
                    extensions = {**request.extensions, "sni_hostname": host}
                    request = httpx.Request(
                        method=request.method,
                        url=pinned_url,
                        headers=headers,
                        content=request.stream,
                        extensions=extensions,
                    )

        return super().handle_request(request)


async def _on_ssrf_request_start(
    session: Any,
    trace_config_ctx: Any,
    params: Any,
) -> None:
    """Block IP-literal redirect targets before aiohttp makes the connection.

    _SSRFGuardResolver handles all hostname-based requests (including redirect
    hops) at DNS-resolution time.  This hook closes the gap where a server
    returns a Location header containing a bare IP literal — aiohttp skips DNS
    for those, so the resolver never fires.  Hostname-based URLs pass through
    untouched here.
    """
    _assert_not_private_ip_literal(str(params.url))


def _make_ssrf_trace_config() -> aiohttp.TraceConfig:
    """Return a TraceConfig that blocks IP-literal redirect targets."""
    cfg = aiohttp.TraceConfig()
    cfg.on_request_start.append(_on_ssrf_request_start)
    return cfg


_SSRF_SAFE_CLIENT_CACHE_KEY = "ssrf_safe_httpx_client"


def _get_ssrf_safe_sync_client() -> HTTPHandler:
    """Return a cached HTTPHandler whose transport validates and pins IPs at connect time.

    Mirrors the caching behaviour of ``_get_httpx_client`` (1-hour TTL via
    ``in_memory_llm_clients_cache``) so that persistent TCP/TLS connections are
    reused across requests — fixing the per-call pool regression that a naive
    ``HTTPHandler(transport=_SSRFGuardTransport())`` would introduce.

    SSL settings (``litellm.ssl_verify``, ``SSL_CERTIFICATE`` env-var) are
    forwarded to ``_SSRFGuardTransport`` so that callers using self-signed certs
    or mTLS are not silently broken.
    """
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is None:
        from litellm.caching.llm_caching_handler import LLMClientCache

        cache = LLMClientCache()
        setattr(litellm, "in_memory_llm_clients_cache", cache)

    _cached_client = cache.get_cache(_SSRF_SAFE_CLIENT_CACHE_KEY)
    if _cached_client:
        return _cached_client

    ssl_config = get_ssl_configuration(None)
    cert = os.getenv("SSL_CERTIFICATE", litellm.ssl_certificate)
    _new_client = HTTPHandler(
        transport=_SSRFGuardTransport(verify=ssl_config, cert=cert)
    )
    cache.set_cache(
        key=_SSRF_SAFE_CLIENT_CACHE_KEY,
        value=_new_client,
        ttl=_DEFAULT_TTL_FOR_HTTPX_CLIENTS,
    )
    return _new_client


class BaseLLMAIOHTTPHandler:
    def __init__(
        self,
        client_session: Optional[aiohttp.ClientSession] = None,
        transport: Optional[LiteLLMAiohttpTransport] = None,
        connector: Optional[aiohttp.BaseConnector] = None,
    ):
        self.client_session = client_session
        self._owns_session = client_session is None  # Track if we own the session for cleanup

        self.transport = transport
        self._owns_transport = transport is None  # Track if we own the transport for cleanup

        self.connector = connector
        self._owns_connector = connector is None  # Track if we own the connector for cleanup

    def _get_or_create_transport(self) -> Optional[LiteLLMAiohttpTransport]:
        """Get existing transport or create a new one if needed."""
        if self.transport:
            return self.transport

        # Create a transport using AsyncHTTPHandler's logic
        try:
            self.transport = AsyncHTTPHandler._create_aiohttp_transport()
            self._owns_transport = True
            return self.transport
        except Exception:
            # If transport creation fails, return None (will use direct session)
            return None

    def _get_connector(self) -> Optional[aiohttp.BaseConnector]:
        """Get or create a connector for the client session."""
        if self.connector:
            return self.connector
        elif self.transport and hasattr(self.transport, "client"):
            # Extract connector from transport if available
            client = self.transport.client
            if callable(client):
                # If client is a factory, we can't extract connector directly
                return None
            elif hasattr(client, "connector"):
                return client.connector
        return None

    def _create_client_session_with_transport(self) -> ClientSession:
        """Create a new client session using transport or connector configuration."""
        connector = self._get_connector()

        if self.transport and hasattr(self.transport, "_get_valid_client_session"):
            # Use transport's session creation if available
            session = self.transport._get_valid_client_session()
            return session
        elif connector:
            # Use provided connector
            session = aiohttp.ClientSession(connector=connector)
            return session
        else:
            # Default session creation — attach SSRF guard resolver so every
            # TCP connection (including redirect targets) is validated at the
            # network layer, eliminating the DNS-rebinding TOCTOU window.
            # The trace config adds a second layer: it blocks IP-literal
            # redirect targets that bypass the resolver (no DNS lookup needed).
            session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(resolver=_SSRFGuardResolver()),
                trace_configs=[_make_ssrf_trace_config()],
            )
            return session

    def _get_async_client_session(self, dynamic_client_session: Optional[ClientSession] = None) -> ClientSession:
        if dynamic_client_session:
            return dynamic_client_session
        elif self.client_session:
            return self.client_session
        else:
            # Create client session using transport/connector if available
            self.client_session = self._create_client_session_with_transport()
            self._owns_session = True  # We created this session, so we own it
            return self.client_session

    async def close(self):
        """Close the aiohttp client session and transport if we own them."""
        # Close client session if we own it
        if self.client_session and not self.client_session.closed and self._owns_session:
            await self.client_session.close()

        # Close transport if we own it
        if self.transport and self._owns_transport and hasattr(self.transport, "aclose"):
            try:
                await self.transport.aclose()
            except Exception:
                # Ignore errors during transport cleanup
                pass

    def __del__(self):
        """
        Cleanup: close aiohttp session on instance destruction.

        Provides defense-in-depth for issue #12443 - ensures cleanup happens
        even if atexit handler doesn't run (abnormal termination).
        """
        if self.client_session is not None and not self.client_session.closed and self._owns_session:
            try:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.close())
                except RuntimeError:
                    # No running loop — run cleanup in a temporary one.
                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(self.close())
                    finally:
                        loop.close()
            except Exception:
                pass

    async def _make_common_async_call(
        self,
        async_client_session: Optional[ClientSession],
        provider_config: BaseConfig,
        api_base: str,
        headers: dict,
        data: Optional[dict],
        timeout: Union[float, httpx.Timeout],
        litellm_params: dict,
        form_data: Optional[FormData] = None,
        stream: bool = False,
    ) -> aiohttp.ClientResponse:
        """Common implementation across stream + non-stream calls. Meant to ensure consistent error-handling."""
        max_retry_on_unprocessable_entity_error = provider_config.max_retry_on_unprocessable_entity_error

        response: Optional[aiohttp.ClientResponse] = None
        async_client_session = self._get_async_client_session(dynamic_client_session=async_client_session)

        # IP-literal URLs bypass _SSRFGuardResolver because aiohttp's TCPConnector
        # skips DNS resolution for hosts that are already IP addresses.  Check them
        # here with a fast, non-blocking parse (no socket I/O).  Hostname-based URLs
        # are handled by _SSRFGuardResolver at TCP-connect time, which also covers
        # redirect targets, eliminating the DNS-rebinding TOCTOU window.
        _assert_not_private_ip_literal(api_base)

        for i in range(max(max_retry_on_unprocessable_entity_error, 1)):
            try:
                response = await async_client_session.post(
                    url=api_base,
                    headers=headers,
                    json=data,
                    data=form_data,
                )
                if not response.ok:
                    response.raise_for_status()
            except aiohttp.ClientResponseError as e:
                setattr(e, "text", e.message)
                raise self._handle_error(e=e, provider_config=provider_config)
            except Exception as e:
                raise self._handle_error(e=e, provider_config=provider_config)
            break

        if response is None:
            raise provider_config.get_error_class(
                error_message="No response from the API",
                status_code=422,
                headers={},
            )

        return response

    def _make_common_sync_call(
        self,
        sync_httpx_client: HTTPHandler,
        provider_config: BaseConfig,
        api_base: str,
        headers: dict,
        data: dict,
        timeout: Optional[Union[float, httpx.Timeout]],
        litellm_params: dict,
        stream: bool = False,
        files: Optional[dict] = None,
        content: Any = None,
        params: Optional[dict] = None,
    ) -> httpx.Response:
        max_retry_on_unprocessable_entity_error = provider_config.max_retry_on_unprocessable_entity_error

        _assert_not_private_url(api_base)

        response: Optional[httpx.Response] = None

        for i in range(max(max_retry_on_unprocessable_entity_error, 1)):
            try:
                response = sync_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    data=data,  # do not json dump the data here. let the individual endpoint handle this.
                    timeout=timeout,
                    stream=stream,
                    files=files,
                    content=content,
                    params=params,
                )
            except httpx.HTTPStatusError as e:
                hit_max_retry = i + 1 == max_retry_on_unprocessable_entity_error
                should_retry = provider_config.should_retry_llm_api_inside_llm_translation_on_http_error(
                    e=e, litellm_params=litellm_params
                )
                if should_retry and not hit_max_retry:
                    data = provider_config.transform_request_on_unprocessable_entity_error(e=e, request_data=data)
                    continue
                else:
                    raise self._handle_error(e=e, provider_config=provider_config)
            except Exception as e:
                raise self._handle_error(e=e, provider_config=provider_config)
            break

        if response is None:
            raise provider_config.get_error_class(
                error_message="No response from the API",
                status_code=422,  # don't retry on this error
                headers={},
            )

        return response

    async def async_completion(
        self,
        custom_llm_provider: str,
        provider_config: BaseConfig,
        api_base: str,
        headers: dict,
        data: dict,
        timeout: Union[float, httpx.Timeout],
        model: str,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        client: Optional[ClientSession] = None,
    ):
        _response = await self._make_common_async_call(
            async_client_session=client,
            provider_config=provider_config,
            api_base=api_base,
            headers=headers,
            data=data,
            timeout=timeout,
            litellm_params=litellm_params,
            stream=False,
        )
        _transformed_response = await provider_config.transform_response(  # type: ignore
            model=model,
            raw_response=_response,  # type: ignore
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
        )
        return _transformed_response

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_llm_provider: str,
        model_response: ModelResponse,
        encoding,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        timeout: Union[float, httpx.Timeout],
        litellm_params: dict,
        acompletion: bool,
        stream: Optional[bool] = False,
        fake_stream: bool = False,
        api_key: Optional[str] = None,
        headers: Optional[dict] = {},
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler, ClientSession]] = None,
    ):
        provider_config = ProviderConfigManager.get_provider_chat_config(
            model=model, provider=litellm.LlmProviders(custom_llm_provider)
        )
        if provider_config is None:
            raise ValueError(f"Provider config not found for model: {model} and provider: {custom_llm_provider}")
        # get config from model, custom llm provider
        headers = provider_config.validate_environment(
            api_key=api_key,
            headers=headers or {},
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_base=api_base,
        )

        api_base = provider_config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
            stream=stream,
        )

        data = provider_config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        if acompletion is True:
            return self.async_completion(
                custom_llm_provider=custom_llm_provider,
                provider_config=provider_config,
                api_base=api_base,
                headers=headers,
                data=data,
                timeout=timeout,
                model=model,
                model_response=model_response,
                logging_obj=logging_obj,
                api_key=api_key,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                encoding=encoding,
                client=(client if client is not None and isinstance(client, ClientSession) else None),
            )

        if stream is True:
            if fake_stream is not True:
                data["stream"] = stream
            completion_stream, headers = self.make_sync_call(
                provider_config=provider_config,
                api_base=api_base,
                headers=headers,  # type: ignore
                data=data,
                model=model,
                messages=messages,
                logging_obj=logging_obj,
                timeout=timeout,
                fake_stream=fake_stream,
                client=(client if client is not None and isinstance(client, HTTPHandler) else None),
                litellm_params=litellm_params,
            )
            return CustomStreamWrapper(
                completion_stream=completion_stream,
                model=model,
                custom_llm_provider=custom_llm_provider,
                logging_obj=logging_obj,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_ssrf_safe_sync_client()
        else:
            sync_httpx_client = client

        response = self._make_common_sync_call(
            sync_httpx_client=sync_httpx_client,
            provider_config=provider_config,
            api_base=api_base,
            headers=headers,
            timeout=timeout,
            litellm_params=litellm_params,
            data=data,
        )
        return provider_config.transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
        )

    def make_sync_call(
        self,
        provider_config: BaseConfig,
        api_base: str,
        headers: dict,
        data: dict,
        model: str,
        messages: list,
        logging_obj,
        litellm_params: dict,
        timeout: Union[float, httpx.Timeout],
        fake_stream: bool = False,
        client: Optional[HTTPHandler] = None,
    ) -> Tuple[Any, dict]:
        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_ssrf_safe_sync_client()
        else:
            sync_httpx_client = client
        stream = True
        if fake_stream is True:
            stream = False

        response = self._make_common_sync_call(
            sync_httpx_client=sync_httpx_client,
            provider_config=provider_config,
            api_base=api_base,
            headers=headers,
            data=data,
            timeout=timeout,
            litellm_params=litellm_params,
            stream=stream,
        )

        if fake_stream is True:
            completion_stream = provider_config.get_model_response_iterator(
                streaming_response=response.json(), sync_stream=True
            )
        else:
            completion_stream = provider_config.get_model_response_iterator(
                streaming_response=response.iter_lines(), sync_stream=True
            )

        # LOGGING
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response="first stream response received",
            additional_args={"complete_input_dict": data},
        )

        return completion_stream, dict(response.headers)

    async def async_image_variations(
        self,
        client: Optional[ClientSession],
        provider_config: BaseImageVariationConfig,
        api_base: str,
        headers: dict,
        data: HttpHandlerRequestFields,
        timeout: float,
        litellm_params: dict,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str,
        model: Optional[str],
        image: FileTypes,
        optional_params: dict,
    ) -> ImageResponse:
        # create aiohttp form data if files in data
        form_data: Optional[FormData] = None
        if "files" in data and "data" in data:
            form_data = FormData()
            for k, v in data["files"].items():
                form_data.add_field(k, v[1], filename=v[0], content_type=v[2])

            for key, value in data["data"].items():
                form_data.add_field(key, value)

        _response = await self._make_common_async_call(
            async_client_session=client,
            provider_config=provider_config,
            api_base=api_base,
            headers=headers,
            data=None if form_data is not None else cast(dict, data),
            form_data=form_data,
            timeout=timeout,
            litellm_params=litellm_params,
            stream=False,
        )

        ## LOGGING
        logging_obj.post_call(
            api_key=api_key,
            original_response=_response.text,
            additional_args={
                "headers": headers,
                "api_base": api_base,
            },
        )

        ## RESPONSE OBJECT
        return await provider_config.async_transform_response_image_variation(
            model=model,
            model_response=model_response,
            raw_response=_response,
            logging_obj=logging_obj,
            request_data=cast(dict, data),
            image=image,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=None,
            api_key=api_key,
        )

    def image_variations(
        self,
        model_response: ImageResponse,
        api_key: str,
        model: Optional[str],
        image: FileTypes,
        timeout: float,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        litellm_params: dict,
        print_verbose: Optional[Callable] = None,
        api_base: Optional[str] = None,
        aimage_variation: bool = False,
        logger_fn=None,
        client=None,
        organization: Optional[str] = None,
        headers: Optional[dict] = None,
    ) -> ImageResponse:
        if model is None:
            raise ValueError("model is required for non-openai image variations")

        provider_config = ProviderConfigManager.get_provider_image_variation_config(
            model=model,  # openai defaults to dall-e-2
            provider=LlmProviders(custom_llm_provider),
        )

        if provider_config is None:
            raise ValueError(f"image variation provider not found: {custom_llm_provider}.")

        api_base = provider_config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
            stream=False,
        )

        headers = provider_config.validate_environment(
            api_key=api_key,
            headers=headers or {},
            model=model,
            messages=[{"role": "user", "content": "test"}],
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_base=api_base,
        )

        data = provider_config.transform_request_image_variation(
            model=model,
            image=image,
            optional_params=optional_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input="",
            api_key=api_key,
            additional_args={
                "headers": headers,
                "api_base": api_base,
                "complete_input_dict": data.copy(),
            },
        )

        if litellm_params.get("async_call", False):
            return self.async_image_variations(
                api_base=api_base,
                data=data,
                headers=headers,
                model_response=model_response,
                logging_obj=logging_obj,
                model=model,
                timeout=timeout,
                client=client,
                optional_params=optional_params,
                litellm_params=litellm_params,
                image=image,
                provider_config=provider_config,
            )  # type: ignore

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_ssrf_safe_sync_client()
        else:
            sync_httpx_client = client

        response = self._make_common_sync_call(
            sync_httpx_client=sync_httpx_client,
            provider_config=provider_config,
            api_base=api_base,
            headers=headers,
            timeout=timeout,
            litellm_params=litellm_params,
            stream=False,
            data=data.get("data") or {},
            files=data.get("files"),
            content=data.get("content"),
            params=data.get("params"),
        )

        ## LOGGING
        logging_obj.post_call(
            api_key=api_key,
            original_response=response.text,
            additional_args={
                "headers": headers,
                "api_base": api_base,
            },
        )

        ## RESPONSE OBJECT
        return provider_config.transform_response_image_variation(
            model=model,
            model_response=model_response,
            raw_response=response,
            logging_obj=logging_obj,
            request_data=cast(dict, data),
            image=image,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=None,
            api_key=api_key,
        )

    def _handle_error(self, e: Exception, provider_config: BaseConfig):
        status_code = getattr(e, "status_code", 500)
        error_headers = getattr(e, "headers", None)
        error_text = getattr(e, "text", str(e))
        error_response = getattr(e, "response", None)
        if error_headers is None and error_response:
            error_headers = getattr(error_response, "headers", None)
        if error_response and hasattr(error_response, "text"):
            error_text = getattr(error_response, "text", error_text)
        if error_headers:
            error_headers = dict(error_headers)
        else:
            error_headers = {}
        raise provider_config.get_error_class(
            error_message=error_text,
            status_code=status_code,
            headers=error_headers,
        )

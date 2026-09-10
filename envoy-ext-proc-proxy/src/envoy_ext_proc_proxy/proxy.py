"""Core proxy implementation using aiohttp."""

import asyncio
import logging
import ssl
from collections.abc import AsyncIterator

import aiohttp
import multidict
from aiohttp import web

from envoy_ext_proc_proxy.config import ProxyConfig
from envoy_ext_proc_proxy.http_utils import (
    REQUEST_ID_HEADER,
    classify_upstream_error,
    create_client_session,
    filter_request_headers,
    filter_response_headers,
    is_bodyless_response,
)
from envoy_ext_proc_proxy.session_registry import (
    EnvoyResponseBodyChunk,
    EnvoyResponseHeaders,
    ExtProcSession,
    SessionAbort,
    SessionRegistry,
    UpstreamRequestBodyChunk,
    UpstreamRequestHeaders,
)

logger = logging.getLogger("ext_proc_proxy.http")

CONFIG_KEY = web.AppKey("config", ProxyConfig)
UPSTREAM_SSL_KEY = web.AppKey("upstream_ssl_context", ssl.SSLContext)
CLIENT_SESSION_KEY = web.AppKey("client_session", aiohttp.ClientSession)
SESSION_REGISTRY_KEY = web.AppKey("session_registry", SessionRegistry)


async def stream_request_payload(request: web.Request) -> AsyncIterator[bytes]:
    """Asynchronously stream the request body chunks."""
    async for chunk in request.content.iter_any():
        yield chunk


def _extract_scheme(
    request: web.Request,
) -> tuple[str | None, web.Response | None]:
    """Extract and validate target scheme from X-Forwarded-Proto header."""
    x_proto = request.headers.get("X-Forwarded-Proto")
    if x_proto is not None:
        proto_clean = x_proto.strip().lower()
        if proto_clean in ("http", "https"):
            return proto_clean, None
        return None, web.Response(
            status=400,
            text=(f"Invalid X-Forwarded-Proto header value '{x_proto}'. Must be 'http' or 'https'."),
        )
    return "https", None


async def _forward_request_body_to_envoy(
    request: web.Request,
    session: ExtProcSession,
) -> None:
    try:
        content_iter = request.content.iter_any()
        try:
            prev_chunk = await anext(content_iter)
        except StopAsyncIteration:
            await session.put_request(UpstreamRequestBodyChunk(data=b"", is_last=True))
            return

        async for next_chunk in content_iter:
            if not await session.put_request(UpstreamRequestBodyChunk(data=prev_chunk, is_last=False)):
                return
            prev_chunk = next_chunk
        await session.put_request(UpstreamRequestBodyChunk(data=prev_chunk, is_last=True))
    except asyncio.CancelledError:
        raise
    except Exception as err:
        logger.warning("Error forwarding request body to Envoy for %s: %s", session.request_id, err)
        session.abort(f"Request body forwarding error: {err}")


async def _forward_envoy_body_to_client(
    session: ExtProcSession,
    response: web.StreamResponse,
) -> None:
    while True:
        item = await session.response_from_envoy_queue.get()
        if isinstance(item, SessionAbort):
            break
        if isinstance(item, EnvoyResponseBodyChunk):
            if item.data:
                await response.write(item.data)
            if item.is_last:
                break


async def handle_paired_proxy_request(
    request: web.Request,
    session: ExtProcSession,
) -> web.StreamResponse:
    """Handle request paired with an ext_proc session via X-Ai-Proxy-Request-Id."""
    session.is_paired = True
    target_scheme, err_response = _extract_scheme(request)
    if err_response is not None:
        return err_response

    logger.info("gRPC proxy request: %s %s://%s%s", request.method, target_scheme, request.host, request.path)

    outgoing_headers = filter_request_headers(request.headers)
    req_headers_list = list(outgoing_headers.items())
    prepared = False
    body_forward_task: asyncio.Task[None] | None = None

    try:
        if not await session.put_request(
            UpstreamRequestHeaders(
                method=request.method,
                path=str(request.rel_url),
                headers=req_headers_list,
                has_body=request.can_read_body,
                scheme=target_scheme,
            )
        ):
            return web.Response(
                status=502,
                text="502 Bad Gateway: Session aborted",
            )

        if request.can_read_body:
            body_forward_task = asyncio.create_task(_forward_request_body_to_envoy(request, session))

        first_item = await session.response_from_envoy_queue.get()
        if isinstance(first_item, SessionAbort):
            return web.Response(
                status=502,
                text=f"502 Bad Gateway: {first_item.reason}",
            )
        if not isinstance(first_item, EnvoyResponseHeaders):
            return web.Response(
                status=502,
                text="502 Bad Gateway: Unexpected response from ext_proc session",
            )

        resp_headers = multidict.CIMultiDict(first_item.headers)
        response = web.StreamResponse(
            status=first_item.status,
            headers=resp_headers,
        )
        await response.prepare(request)
        prepared = True

        if not first_item.is_empty_body:
            await _forward_envoy_body_to_client(session, response)

        await response.write_eof()

    except Exception as err:
        logger.warning("Error in paired proxy request for %s: %s", session.request_id, err)
        session.abort(f"HTTP proxy error: {err}")
        if not prepared:
            status_code, err_msg = classify_upstream_error(err)
            return web.Response(status=status_code, text=err_msg)
        raise
    finally:
        if body_forward_task is not None and not body_forward_task.done():
            body_forward_task.cancel()
            try:
                await body_forward_task
            except asyncio.CancelledError:
                pass
    return response


async def handle_proxy_request(request: web.Request) -> web.StreamResponse:
    """Handle incoming client HTTPS request and forward to target host."""
    # Check if request has X-Ai-Proxy-Request-Id header for paired session routing
    request_id = request.headers.get(REQUEST_ID_HEADER)
    if request_id is not None:
        session_registry: SessionRegistry | None = request.app.get(SESSION_REGISTRY_KEY)
        session = session_registry.get(request_id) if session_registry is not None else None
        if session is None:
            return web.Response(
                status=400,
                text=(f"400 Bad Request: Invalid or expired {REQUEST_ID_HEADER} '{request_id}'"),
            )
        return await handle_paired_proxy_request(request, session)

    logger.info("Direct request: %s %s://%s%s", request.method, request.scheme, request.host, request.path)
    # Standard proxying behavior when no X-Ai-Proxy-Request-Id is present.
    target_scheme, err_response = _extract_scheme(request)
    if err_response is not None:
        return err_response

    target_host = request.headers.get("Host") or request.host
    if not target_host:
        return web.Response(status=400, text="Missing Host header in request")

    # request.rel_url preserves path, query parameters, and fragments
    target_url = f"{target_scheme}://{target_host}{request.rel_url}"

    outgoing_headers = filter_request_headers(request.headers)

    body_data = None
    if request.can_read_body:
        body_data = stream_request_payload(request)

    session: aiohttp.ClientSession = request.app[CLIENT_SESSION_KEY]

    try:
        upstream_cm = session.request(
            method=request.method,
            url=target_url,
            headers=outgoing_headers,
            data=body_data,
            allow_redirects=False,
        )
        upstream_resp = await upstream_cm.__aenter__()
    except (aiohttp.ClientError, TimeoutError, ssl.SSLError, OSError) as err:
        status_code, err_msg = classify_upstream_error(err)
        logger.warning("Upstream request error for %s: %s", target_url, err)
        return web.Response(
            status=status_code,
            text=err_msg,
        )

    try:
        resp_headers = filter_response_headers(upstream_resp.headers)
        response = web.StreamResponse(
            status=upstream_resp.status,
            reason=upstream_resp.reason,
            headers=resp_headers,
        )

        await response.prepare(request)

        # Do not attempt to read body for HEAD requests or body-less status codes
        if not is_bodyless_response(request.method, upstream_resp.status):
            async for chunk in upstream_resp.content.iter_any():
                await response.write(chunk)

        await response.write_eof()
        return response
    finally:
        await upstream_cm.__aexit__(None, None, None)


async def client_session_cleanup_ctx(app: web.Application) -> AsyncIterator[None]:
    """Context manager for managing aiohttp.ClientSession lifecycle."""
    config: ProxyConfig = app[CONFIG_KEY]
    upstream_ssl_ctx = app.get(UPSTREAM_SSL_KEY)
    session = create_client_session(config=config, upstream_ssl_context=upstream_ssl_ctx)
    app[CLIENT_SESSION_KEY] = session

    yield

    await session.close()


def create_proxy_app(
    config: ProxyConfig | None = None,
    upstream_ssl_context: ssl.SSLContext | None = None,
    session_registry: SessionRegistry | None = None,
) -> web.Application:
    """Create and configure the proxy web.Application."""
    if config is None:
        config = ProxyConfig()

    app = web.Application()
    app[CONFIG_KEY] = config
    if upstream_ssl_context is not None:
        app[UPSTREAM_SSL_KEY] = upstream_ssl_context
    if session_registry is not None:
        app[SESSION_REGISTRY_KEY] = session_registry

    app.cleanup_ctx.append(client_session_cleanup_ctx)

    # Catch-all route to handle all HTTP methods and paths
    app.router.add_route("*", "/{path_info:.*}", handle_proxy_request)

    return app


async def run_proxy(
    config: ProxyConfig,
    ssl_context: ssl.SSLContext,
    session_registry: SessionRegistry,
) -> None:
    """Initialize and run the HTTPS proxy server."""
    app = create_proxy_app(config=config, session_registry=session_registry)
    runner = web.AppRunner(app, keepalive_timeout=config.keepalive_timeout)
    await runner.setup()

    site = web.TCPSite(
        runner,
        host=config.host,
        port=config.port,
        ssl_context=ssl_context,
    )
    await site.start()
    logger.info(
        "HTTPS proxy server listening on https://%s:%d (keepalive: %.1fs)",
        config.host,
        config.port,
        config.keepalive_timeout,
    )

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()

"""Envoy External Processing (ext_proc) gRPC server implementation using grpc.aio."""

import asyncio
import logging
import ssl
import uuid
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from urllib.parse import urlparse

import aiohttp
import grpc
import multidict
from envoy.service.ext_proc.v3 import (
    external_processor_pb2,
    external_processor_pb2_grpc,
)

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
    DEFAULT_QUEUE_MAXSIZE,
    EnvoyResponseBodyChunk,
    EnvoyResponseHeaders,
    ExtProcSession,
    SessionAbort,
    SessionRegistry,
    UpstreamRequestBodyChunk,
    UpstreamRequestHeaders,
)

logger = logging.getLogger("ext_proc_proxy.grpc")


@dataclass(frozen=True)
class RequestBodyChunk:
    """Request body chunk sent from Envoy to be streamed upstream."""

    data: bytes
    is_last: bool


@dataclass(frozen=True)
class UpstreamHeaders:
    """Upstream HTTP response status and filtered headers."""

    status: int
    headers: list[tuple[str, str]]
    is_empty_body: bool


@dataclass(frozen=True)
class UpstreamBodyChunk:
    """Upstream HTTP response body chunk."""

    data: bytes
    is_last: bool


@dataclass(frozen=True)
class UpstreamError:
    """Upstream error with status code and descriptive message."""

    status: int
    message: str


UpstreamResponseItem = UpstreamHeaders | UpstreamBodyChunk | UpstreamError


def _build_header_mutation(
    headers: list[tuple[str, str]],
    status_code: int | None = None,
    method: str | None = None,
    path: str | None = None,
    scheme: str | None = None,
    original_headers: Iterable[str] | None = None,
) -> external_processor_pb2.HeaderMutation:
    """Build a HeaderMutation protobuf message from key-value pairs."""
    mutation = external_processor_pb2.HeaderMutation()
    if status_code is not None:
        opt = mutation.set_headers.add()
        opt.header.key = ":status"
        opt.header.raw_value = str(status_code).encode("utf-8")

    if method is not None:
        opt = mutation.set_headers.add()
        opt.header.key = ":method"
        opt.header.raw_value = method.encode("utf-8")

    if path is not None:
        opt = mutation.set_headers.add()
        opt.header.key = ":path"
        opt.header.raw_value = path.encode("utf-8")

    if scheme is not None:
        opt = mutation.set_headers.add()
        opt.header.key = ":scheme"
        opt.header.raw_value = scheme.encode("utf-8")

    new_keys = set()
    for k, v in headers:
        opt = mutation.set_headers.add()
        opt.header.key = k.lower()
        opt.header.raw_value = v.encode("utf-8") if isinstance(v, str) else bytes(v)
        new_keys.add(k.lower())

    if original_headers is not None:
        # Envoy's routing breaks without x-forwarded-proto, keep this header
        to_remove = {
            k.lower()
            for k in original_headers
            if not k.startswith(":") and k.lower() != "x-forwarded-proto" and k.lower() not in new_keys
        }
        for rh in sorted(to_remove):
            mutation.remove_headers.append(rh)

    return mutation


class ExternalProcessorService(external_processor_pb2_grpc.ExternalProcessorServicer):
    """External Processor servicer handling bidirectional gRPC streams."""

    def __init__(
        self,
        config: ProxyConfig,
        upstream_ssl_context: ssl.SSLContext | None = None,
        session_registry: SessionRegistry | None = None,
    ) -> None:
        self.config = config
        self._session: aiohttp.ClientSession | None = None
        self._upstream_ssl_context = upstream_ssl_context
        self.session_registry = session_registry if session_registry is not None else SessionRegistry()
        self.target_url_parsed = urlparse(config.ext_proc_target)

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or initialize the aiohttp.ClientSession."""
        if self._session is None or self._session.closed:
            self._session = create_client_session(
                config=self.config,
                upstream_ssl_context=self._upstream_ssl_context,
            )
        return self._session

    async def close(self) -> None:
        """Close internal session if initialized."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _fetch_upstream_stream(
        self,
        session: aiohttp.ClientSession,
        method: str,
        target_url: str,
        headers: multidict.CIMultiDict,
        body_queue: asyncio.Queue[RequestBodyChunk] | None,
        resp_queue: asyncio.Queue[UpstreamResponseItem],
    ) -> None:
        """Execute upstream request and stream body/response via queues."""
        try:
            body_data = None
            if body_queue is not None:

                async def body_stream_generator() -> AsyncIterator[bytes]:
                    while True:
                        chunk_item = await body_queue.get()
                        if chunk_item.data:
                            yield chunk_item.data
                        if chunk_item.is_last:
                            break

                body_data = body_stream_generator()

            async with session.request(
                method=method,
                url=target_url,
                headers=headers,
                data=body_data,
                allow_redirects=False,
            ) as upstream_resp:
                # Filter hop-by-hop headers from upstream response
                filtered_resp_headers = filter_response_headers(upstream_resp.headers)
                resp_headers = list(filtered_resp_headers.items())

                is_empty_body = is_bodyless_response(method, upstream_resp.status)

                await resp_queue.put(
                    UpstreamHeaders(
                        status=upstream_resp.status,
                        headers=resp_headers,
                        is_empty_body=is_empty_body,
                    )
                )

                if not is_empty_body:
                    content_iter = upstream_resp.content.iter_any()
                    try:
                        prev_chunk = await anext(content_iter)
                    except StopAsyncIteration:
                        await resp_queue.put(UpstreamBodyChunk(data=b"", is_last=True))
                    else:
                        async for next_chunk in content_iter:
                            await resp_queue.put(UpstreamBodyChunk(data=prev_chunk, is_last=False))
                            prev_chunk = next_chunk
                        await resp_queue.put(UpstreamBodyChunk(data=prev_chunk, is_last=True))

        except (aiohttp.ClientError, TimeoutError, ssl.SSLError, OSError) as err:
            status_code, err_msg = classify_upstream_error(err)
            logger.warning("Upstream request error for %s: %s", target_url, err)
            await resp_queue.put(UpstreamError(status=status_code, message=err_msg))

    async def _stream_paired_request_body_to_envoy(
        self,
        ext_proc_session: ExtProcSession,
    ) -> AsyncIterator[external_processor_pb2.ProcessingResponse]:
        has_yielded_body = False
        while True:
            body_item = await ext_proc_session.request_to_envoy_queue.get()
            if isinstance(body_item, SessionAbort):
                break
            if isinstance(body_item, UpstreamRequestBodyChunk):
                if body_item.data or (body_item.is_last and not has_yielded_body):
                    streamed_body = external_processor_pb2.StreamedBodyResponse(
                        body=body_item.data,
                        end_of_stream=body_item.is_last,
                    )
                    body_mut = external_processor_pb2.BodyMutation(streamed_response=streamed_body)
                    body_common = external_processor_pb2.CommonResponse(
                        body_mutation=body_mut,
                        status=external_processor_pb2.CommonResponse.ResponseStatus.CONTINUE,
                    )
                    has_yielded_body = True
                    yield external_processor_pb2.ProcessingResponse(
                        request_body=external_processor_pb2.BodyResponse(response=body_common)
                    )
                if body_item.is_last:
                    break

    async def _read_target_response_from_envoy(
        self,
        request_iterator: AsyncIterator[external_processor_pb2.ProcessingRequest],
        ext_proc_session: ExtProcSession,
    ) -> list[str]:
        target_response_headers: list[str] = []
        async for req in request_iterator:
            if req.HasField("response_headers"):
                raw_resp_headers = multidict.CIMultiDict()
                for h in req.response_headers.headers.headers:
                    raw_resp_headers.add(
                        h.key, h.value or (h.raw_value.decode("utf-8", "ignore") if h.raw_value else "")
                    )
                target_response_headers = [h.key for h in req.response_headers.headers.headers]

                status_str = raw_resp_headers.get(":status", "200")
                try:
                    resp_status = int(status_str)
                except ValueError:
                    resp_status = 200

                filtered_hdrs = filter_response_headers(raw_resp_headers)
                is_empty = req.response_headers.end_of_stream

                if not await ext_proc_session.put_response(
                    EnvoyResponseHeaders(
                        status=resp_status,
                        headers=list(filtered_hdrs.items()),
                        is_empty_body=is_empty,
                    )
                ):
                    break

                if is_empty:
                    break

            elif req.HasField("response_body"):
                chunk_data = req.response_body.body
                is_last_chunk = req.response_body.end_of_stream
                if not await ext_proc_session.put_response(
                    EnvoyResponseBodyChunk(
                        data=chunk_data,
                        is_last=is_last_chunk,
                    )
                ):
                    break
                if is_last_chunk:
                    break
        return target_response_headers

    async def _stream_paired_final_response_to_envoy(
        self,
        resp_queue: asyncio.Queue[UpstreamResponseItem],
        target_response_headers: list[str],
        first_item: UpstreamResponseItem | None = None,
    ) -> AsyncIterator[external_processor_pb2.ProcessingResponse]:
        current_item: UpstreamResponseItem | None = first_item

        while True:
            item = current_item if current_item is not None else await resp_queue.get()
            current_item = None

            if isinstance(item, UpstreamError):
                err_bytes = item.message.encode("utf-8")
                hdr_mut = _build_header_mutation(
                    [],
                    status_code=item.status,
                    original_headers=target_response_headers,
                )
                streamed_body = external_processor_pb2.StreamedBodyResponse(
                    body=err_bytes,
                    end_of_stream=True,
                )
                body_mut = external_processor_pb2.BodyMutation(streamed_response=streamed_body)
                cr = external_processor_pb2.CommonResponse(
                    header_mutation=hdr_mut,
                    body_mutation=body_mut,
                )
                yield external_processor_pb2.ProcessingResponse(
                    response_headers=external_processor_pb2.HeadersResponse(response=cr)
                )
                break

            elif isinstance(item, UpstreamHeaders):
                hdr_mut = _build_header_mutation(
                    item.headers,
                    status_code=item.status,
                    original_headers=target_response_headers,
                )
                cr = external_processor_pb2.CommonResponse(header_mutation=hdr_mut)
                yield external_processor_pb2.ProcessingResponse(
                    response_headers=external_processor_pb2.HeadersResponse(response=cr)
                )
                if item.is_empty_body:
                    break

            elif isinstance(item, UpstreamBodyChunk):
                streamed_body = external_processor_pb2.StreamedBodyResponse(
                    body=item.data,
                    end_of_stream=item.is_last,
                )
                body_mut = external_processor_pb2.BodyMutation(streamed_response=streamed_body)
                cr = external_processor_pb2.CommonResponse(body_mutation=body_mut)
                yield external_processor_pb2.ProcessingResponse(
                    response_body=external_processor_pb2.BodyResponse(response=cr)
                )
                if item.is_last:
                    break

    async def _handle_paired_flow(
        self,
        proxy_first_item: UpstreamRequestHeaders,
        ext_proc_session: ExtProcSession,
        request_iterator: AsyncIterator[external_processor_pb2.ProcessingRequest],
        resp_queue: asyncio.Queue[UpstreamResponseItem],
        original_request_headers: Iterable[str] | None = None,
        upstream_first_item: UpstreamResponseItem | None = None,
    ) -> AsyncIterator[external_processor_pb2.ProcessingResponse]:
        """Handle PATH A: Paired mode when Upstream Service called HTTP Proxy."""
        header_mutation = _build_header_mutation(
            headers=proxy_first_item.headers,
            method=proxy_first_item.method,
            path=proxy_first_item.path,
            scheme=proxy_first_item.scheme,
            original_headers=original_request_headers,
        )
        common_resp = external_processor_pb2.CommonResponse(
            header_mutation=header_mutation,
            status=external_processor_pb2.CommonResponse.ResponseStatus.CONTINUE,
            clear_route_cache=True,
        )
        headers_resp = external_processor_pb2.HeadersResponse(response=common_resp)
        yield external_processor_pb2.ProcessingResponse(request_headers=headers_resp)

        if proxy_first_item.has_body:
            async for resp in self._stream_paired_request_body_to_envoy(ext_proc_session):
                yield resp

        # TODO: similarly to HTTP proxy, this prevents the handler from reading early
        # response, one that happens before full request body is sent to the target.
        target_response_headers = await self._read_target_response_from_envoy(request_iterator, ext_proc_session)

        async for resp in self._stream_paired_final_response_to_envoy(
            resp_queue,
            target_response_headers,
            first_item=upstream_first_item,
        ):
            yield resp

    async def _handle_unpaired_flow(
        self,
        first_item: UpstreamResponseItem,
        resp_queue: asyncio.Queue[UpstreamResponseItem],
    ) -> AsyncIterator[external_processor_pb2.ProcessingResponse]:
        """Handle PATH B: Unpaired / Direct Short-Circuit or Error."""
        current_item: UpstreamResponseItem | None = first_item

        while True:
            item = current_item if current_item is not None else await resp_queue.get()
            current_item = None

            if isinstance(item, UpstreamError):
                err_bytes = item.message.encode("utf-8")
                sir = external_processor_pb2.StreamedImmediateResponse()
                h_status = sir.headers_response.headers.headers.add()
                h_status.key = ":status"
                h_status.raw_value = str(item.status).encode("utf-8")
                h_ct = sir.headers_response.headers.headers.add()
                h_ct.key = "content-type"
                h_ct.raw_value = b"text/plain"
                sir.headers_response.end_of_stream = False
                yield external_processor_pb2.ProcessingResponse(streamed_immediate_response=sir)

                sir_body = external_processor_pb2.StreamedImmediateResponse()
                sir_body.body_response.body = err_bytes
                sir_body.body_response.end_of_stream = True
                yield external_processor_pb2.ProcessingResponse(streamed_immediate_response=sir_body)
                break

            elif isinstance(item, UpstreamHeaders):
                sir = external_processor_pb2.StreamedImmediateResponse()
                h_status = sir.headers_response.headers.headers.add()
                h_status.key = ":status"
                h_status.raw_value = str(item.status).encode("utf-8")

                for k, v in item.headers:
                    hv = sir.headers_response.headers.headers.add()
                    hv.key = k.lower()
                    hv.raw_value = v.encode("utf-8") if isinstance(v, str) else bytes(v)

                sir.headers_response.end_of_stream = item.is_empty_body
                yield external_processor_pb2.ProcessingResponse(streamed_immediate_response=sir)
                if item.is_empty_body:
                    break

            elif isinstance(item, UpstreamBodyChunk):
                sir = external_processor_pb2.StreamedImmediateResponse()
                sir.body_response.body = item.data
                sir.body_response.end_of_stream = item.is_last
                yield external_processor_pb2.ProcessingResponse(streamed_immediate_response=sir)
                if item.is_last:
                    break

    async def _forward_request_body_from_envoy(
        self,
        request_iterator: AsyncIterator[external_processor_pb2.ProcessingRequest],
        body_queue: asyncio.Queue[RequestBodyChunk],
        upstream_task: asyncio.Task | None = None,
    ) -> None:
        async for req in request_iterator:
            if upstream_task is not None and upstream_task.done():
                break
            if req.HasField("request_body"):
                chunk = req.request_body.body
                is_last = req.request_body.end_of_stream
                if upstream_task is not None:
                    put_task = asyncio.create_task(body_queue.put(RequestBodyChunk(data=chunk, is_last=is_last)))
                    try:
                        done, _ = await asyncio.wait([put_task, upstream_task], return_when=asyncio.FIRST_COMPLETED)
                        if upstream_task in done:
                            break
                    finally:
                        if not put_task.done():
                            put_task.cancel()
                            try:
                                await put_task
                            except asyncio.CancelledError:
                                pass
                else:
                    await body_queue.put(RequestBodyChunk(data=chunk, is_last=is_last))
                if is_last:
                    break

    async def _dispatch_flow(
        self,
        ext_proc_session: ExtProcSession,
        resp_queue: asyncio.Queue[UpstreamResponseItem],
        request_iterator: AsyncIterator[external_processor_pb2.ProcessingRequest],
        original_request_headers: list[str],
    ) -> AsyncIterator[external_processor_pb2.ProcessingResponse]:
        get_proxy_req = asyncio.create_task(ext_proc_session.request_to_envoy_queue.get())
        get_upstream_resp = asyncio.create_task(resp_queue.get())

        done, _ = await asyncio.wait(
            [get_proxy_req, get_upstream_resp],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if get_proxy_req in done:
            proxy_first_item = get_proxy_req.result()
            upstream_first_item: UpstreamResponseItem | None = None
            if get_upstream_resp in done:
                upstream_first_item = get_upstream_resp.result()
            else:
                get_upstream_resp.cancel()

            if isinstance(proxy_first_item, UpstreamRequestHeaders):
                async for resp in self._handle_paired_flow(
                    proxy_first_item=proxy_first_item,
                    ext_proc_session=ext_proc_session,
                    request_iterator=request_iterator,
                    resp_queue=resp_queue,
                    original_request_headers=original_request_headers,
                    upstream_first_item=upstream_first_item,
                ):
                    yield resp
        else:
            get_proxy_req.cancel()
            first_item = get_upstream_resp.result()

            async for resp in self._handle_unpaired_flow(
                first_item=first_item,
                resp_queue=resp_queue,
            ):
                yield resp

    async def Process(
        self,
        request_iterator: AsyncIterator[external_processor_pb2.ProcessingRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[external_processor_pb2.ProcessingResponse]:
        """Handle bidirectional ext_proc stream from Envoy."""
        session = await self.get_session()
        upstream_task: asyncio.Task | None = None
        ext_proc_session: ExtProcSession | None = None
        request_id: str | None = None

        try:
            # Read first request containing headers
            try:
                first_request = await anext(request_iterator)
            except StopAsyncIteration:
                return

            if first_request.observability_mode:
                # In observability mode, do not respond to messages
                async for _ in request_iterator:
                    pass
                return

            if not first_request.HasField("request_headers"):
                logger.warning("First request did not contain request_headers")
                return

            # Extract headers, method, path
            raw_headers = multidict.CIMultiDict()
            for h in first_request.request_headers.headers.headers:
                raw_headers.add(h.key, h.value or (h.raw_value.decode("utf-8", "ignore") if h.raw_value else ""))
            original_request_headers = [h.key for h in first_request.request_headers.headers.headers]

            method = raw_headers.get(":method", "GET").upper()
            path = raw_headers.get(":path", "/")
            if not path.startswith("/"):
                path = "/" + path

            # Construct target URL for Upstream Service
            target_base = self.config.ext_proc_target.rstrip("/")
            target_url = f"{target_base}{path}"

            # Create unique request ID and register ExtProcSession
            request_id = uuid.uuid4().hex
            ext_proc_session = ExtProcSession(request_id=request_id)
            self.session_registry.register(ext_proc_session)

            # Prepare outgoing headers for Upstream Service
            outgoing_headers = filter_request_headers(raw_headers)
            outgoing_headers[REQUEST_ID_HEADER] = request_id
            outgoing_headers["Host"] = self.target_url_parsed.netloc
            # Disable retries, we have no way to handle them.
            outgoing_headers["x-litellm-num-retries"] = "0"

            has_request_body = not first_request.request_headers.end_of_stream
            body_queue: asyncio.Queue[RequestBodyChunk] | None = (
                asyncio.Queue(maxsize=DEFAULT_QUEUE_MAXSIZE) if has_request_body else None
            )
            resp_queue: asyncio.Queue[UpstreamResponseItem] = asyncio.Queue(maxsize=DEFAULT_QUEUE_MAXSIZE)

            # Start upstream task to send initial request to Upstream Service
            upstream_task = asyncio.create_task(
                self._fetch_upstream_stream(
                    session=session,
                    method=method,
                    target_url=target_url,
                    headers=outgoing_headers,
                    body_queue=body_queue,
                    resp_queue=resp_queue,
                )
            )

            if has_request_body and body_queue is not None:
                await self._forward_request_body_from_envoy(request_iterator, body_queue, upstream_task=upstream_task)

            async for resp in self._dispatch_flow(
                ext_proc_session=ext_proc_session,
                resp_queue=resp_queue,
                request_iterator=request_iterator,
                original_request_headers=original_request_headers,
            ):
                yield resp

            await upstream_task

        except Exception:
            logger.exception("Error in ext_proc Process")
            raise
        finally:
            if ext_proc_session is not None and request_id is not None:
                self.session_registry.unregister(request_id)
                ext_proc_session.abort("ext_proc Process terminated")

            if upstream_task is not None and not upstream_task.done():
                upstream_task.cancel()
                try:
                    await upstream_task
                except asyncio.CancelledError:
                    pass


def create_ext_proc_server(
    config: ProxyConfig,
    server_credentials: grpc.ServerCredentials,
    upstream_ssl_context: ssl.SSLContext | None = None,
    session_registry: SessionRegistry | None = None,
) -> grpc.aio.Server:
    """Create and configure the TLS-secured gRPC ext_proc server."""
    server = grpc.aio.server()
    service = ExternalProcessorService(
        config=config,
        upstream_ssl_context=upstream_ssl_context,
        session_registry=session_registry,
    )
    external_processor_pb2_grpc.add_ExternalProcessorServicer_to_server(service, server)
    listen_addr = f"{config.ext_proc_host}:{config.ext_proc_port}"
    server.add_secure_port(listen_addr, server_credentials)
    server.ext_proc_service = service  # type: ignore[attr-defined]
    return server


async def run_grpc_server(
    config: ProxyConfig,
    server_credentials: grpc.ServerCredentials,
    session_registry: SessionRegistry,
) -> None:
    """Initialize and run the TLS-secured Envoy ext_proc gRPC server."""
    grpc_server = create_ext_proc_server(
        config=config,
        server_credentials=server_credentials,
        session_registry=session_registry,
    )
    await grpc_server.start()
    logger.info(
        "Envoy ext_proc gRPC server listening with TLS on %s:%d (target: %s)",
        config.ext_proc_host,
        config.ext_proc_port,
        config.ext_proc_target,
    )
    try:
        await grpc_server.wait_for_termination()
    finally:
        if hasattr(grpc_server, "ext_proc_service"):
            await grpc_server.ext_proc_service.close()
        await grpc_server.stop(grace=2.0)

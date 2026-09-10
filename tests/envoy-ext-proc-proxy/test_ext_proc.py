"""Integration tests for Envoy ext_proc gRPC server with TLS protection."""

import asyncio
import os
import ssl
import unittest
from typing import List, Optional

from aiohttp import web
import grpc

import envoy_ext_proc_proxy  # noqa: F401

# isort: split
from envoy.service.ext_proc.v3 import (
    external_processor_pb2,
    external_processor_pb2_grpc,
)
from envoy_ext_proc_proxy.cert_utils import (
    create_grpc_server_credentials,
    create_server_ssl_context,
    generate_self_signed_cert,
)
from envoy_ext_proc_proxy.config import ProxyConfig
from envoy_ext_proc_proxy.ext_proc_server import (
    _build_header_mutation,
    create_ext_proc_server,
)
from envoy_ext_proc_proxy.session_registry import (
    DEFAULT_QUEUE_MAXSIZE,
    EnvoyResponseBodyChunk,
    ExtProcSession,
    SessionAbort,
    UpstreamRequestBodyChunk,
)


class TestExtProcServer(unittest.IsolatedAsyncioTestCase):
    """Test suite for the Envoy ext_proc gRPC server."""

    @classmethod
    def setUpClass(cls):
        cls.server_cert, cls.server_key = generate_self_signed_cert(hostname="localhost")
        with open(cls.server_cert, "rb") as f:
            cls.cert_bytes = f.read()

    @classmethod
    def tearDownClass(cls):
        for f in (cls.server_cert, cls.server_key):
            if os.path.exists(f):
                os.remove(f)

    async def asyncSetUp(self):
        self.http_runners: List[web.AppRunner] = []
        self.grpc_servers: List[grpc.aio.Server] = []

    async def asyncTearDown(self):
        for server in reversed(self.grpc_servers):
            if hasattr(server, "ext_proc_service"):
                await server.ext_proc_service.close()
            await server.stop(grace=1.0)
        for runner in reversed(self.http_runners):
            await runner.cleanup()

    async def _start_mock_http_server(self, handler, ssl_context: ssl.SSLContext | None = None) -> int:
        """Start a mock HTTP/HTTPS backend server and return its port."""
        app = web.Application()
        app.router.add_route("*", "/{path_info:.*}", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        self.http_runners.append(runner)

        site = web.TCPSite(runner, host="127.0.0.1", port=0, ssl_context=ssl_context)
        await site.start()
        port = runner.addresses[0][1]
        return port

    async def _start_ext_proc_server(
        self,
        target_port: int,
        upstream_timeout: float = 5.0,
        target_scheme: str = "http",
        upstream_ssl_context: Optional[ssl.SSLContext] = None,
    ) -> int:
        """Start the ext_proc gRPC server with TLS and return its port."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            grpc_port = s.getsockname()[1]

        config = ProxyConfig(
            ext_proc_host="127.0.0.1",
            ext_proc_port=grpc_port,
            ext_proc_target=f"{target_scheme}://localhost:{target_port}",
            cert=self.server_cert,
            key=self.server_key,
            upstream_timeout=upstream_timeout,
        )

        creds = create_grpc_server_credentials(self.server_cert, self.server_key)
        grpc_server = create_ext_proc_server(
            config=config,
            server_credentials=creds,
            upstream_ssl_context=upstream_ssl_context,
        )
        await grpc_server.start()
        self.grpc_servers.append(grpc_server)
        return grpc_port

    def _get_client_credentials(self) -> grpc.ChannelCredentials:
        """Get channel credentials configured to trust the server certificate."""
        return grpc.ssl_channel_credentials(root_certificates=self.cert_bytes)

    async def test_ext_proc_get_request(self):
        """Test proxying a GET request without body via ext_proc over TLS."""
        received_request = {}

        async def mock_handler(request: web.Request):
            nonlocal received_request
            received_request = {
                "method": request.method,
                "path": request.path,
                "query": dict(request.query),
                "custom_header": request.headers.get("X-Custom-Client-Header"),
            }
            resp = web.Response(text="backend GET response ok")
            resp.headers["X-Backend-Header"] = "BackendVal"
            resp.headers["Connection"] = "close, X-Dynamic-Hop"
            resp.headers["X-Dynamic-Hop"] = "StripMe"
            return resp

        http_port = await self._start_mock_http_server(mock_handler)
        grpc_port = await self._start_ext_proc_server(http_port)

        creds = self._get_client_credentials()
        async with grpc.aio.secure_channel(f"localhost:{grpc_port}", creds) as channel:
            stub = external_processor_pb2_grpc.ExternalProcessorStub(channel)

            async def request_generator():
                req = external_processor_pb2.ProcessingRequest()
                h1 = req.request_headers.headers.headers.add()
                h1.key = ":method"
                h1.value = "GET"
                h2 = req.request_headers.headers.headers.add()
                h2.key = ":path"
                h2.value = "/api/v1/resource?filter=active"
                h3 = req.request_headers.headers.headers.add()
                h3.key = "x-custom-client-header"
                h3.value = "MyHeaderVal"
                req.request_headers.end_of_stream = True
                yield req

            responses = []
            async for resp in stub.Process(request_generator()):
                responses.append(resp)

            # Verify mock backend received request
            self.assertEqual(received_request["method"], "GET")
            self.assertEqual(received_request["path"], "/api/v1/resource")
            self.assertEqual(received_request["query"], {"filter": "active"})
            self.assertEqual(received_request["custom_header"], "MyHeaderVal")

            # Verify ext_proc responses
            self.assertGreaterEqual(len(responses), 2)
            first_resp = responses[0]
            self.assertTrue(first_resp.HasField("streamed_immediate_response"))
            sir = first_resp.streamed_immediate_response
            self.assertTrue(sir.HasField("headers_response"))

            # Check :status header and hop-by-hop filtered headers
            headers_dict = {
                h.key: (h.raw_value.decode("utf-8") if h.raw_value else h.value)
                for h in sir.headers_response.headers.headers
            }
            self.assertEqual(headers_dict.get(":status"), "200")
            self.assertEqual(headers_dict.get("x-backend-header"), "BackendVal")
            self.assertNotIn("connection", headers_dict)
            self.assertNotIn("x-dynamic-hop", headers_dict)

            # Collect body chunks
            body_chunks = [
                r.streamed_immediate_response.body_response.body
                for r in responses[1:]
                if r.streamed_immediate_response.HasField("body_response")
            ]
            full_body = b"".join(body_chunks).decode("utf-8")
            self.assertEqual(full_body, "backend GET response ok")

    async def test_ext_proc_streaming_request_body(self):
        """Test streaming request body chunk-by-chunk to backend over TLS."""
        received_body_chunks = []

        async def mock_handler(request: web.Request):
            async for chunk in request.content.iter_any():
                received_body_chunks.append(chunk)
            return web.Response(text="all chunks received successfully")

        http_port = await self._start_mock_http_server(mock_handler)
        grpc_port = await self._start_ext_proc_server(http_port)

        creds = self._get_client_credentials()
        async with grpc.aio.secure_channel(f"localhost:{grpc_port}", creds) as channel:
            stub = external_processor_pb2_grpc.ExternalProcessorStub(channel)

            input_queue = asyncio.Queue()

            async def request_generator():
                while True:
                    item = await input_queue.get()
                    if item is None:
                        break
                    yield item

            call = stub.Process(request_generator())

            # 1. Send request headers with end_of_stream = False
            req_headers = external_processor_pb2.ProcessingRequest()
            h1 = req_headers.request_headers.headers.headers.add()
            h1.key = ":method"
            h1.value = "POST"
            h2 = req_headers.request_headers.headers.headers.add()
            h2.key = ":path"
            h2.value = "/upload/stream"
            req_headers.request_headers.end_of_stream = False
            await input_queue.put(req_headers)

            # 2. Send chunk 1 (no intermediate confirmations needed in FULL_DUPLEX_STREAMED mode)
            chunk1 = external_processor_pb2.ProcessingRequest()
            chunk1.request_body.body = b"first chunk of data; "
            chunk1.request_body.end_of_stream = False
            await input_queue.put(chunk1)

            # 3. Send chunk 2 (final chunk)
            chunk2 = external_processor_pb2.ProcessingRequest()
            chunk2.request_body.body = b"second and final chunk."
            chunk2.request_body.end_of_stream = True
            await input_queue.put(chunk2)

            # Signal generator termination
            await input_queue.put(None)

            # 4. Receive streamed response from backend
            resp_headers = await call.read()
            self.assertTrue(resp_headers.HasField("streamed_immediate_response"))
            sir_hdrs = resp_headers.streamed_immediate_response.headers_response
            headers_dict = {
                h.key: (h.raw_value.decode("utf-8") if h.raw_value else h.value) for h in sir_hdrs.headers.headers
            }
            self.assertEqual(headers_dict.get(":status"), "200")

            # Receive body response
            body_chunks = []
            while True:
                resp_body = await call.read()
                if resp_body is grpc.aio.EOF:
                    break
                if resp_body.streamed_immediate_response.HasField("body_response"):
                    body_chunks.append(resp_body.streamed_immediate_response.body_response.body)
                    if resp_body.streamed_immediate_response.body_response.end_of_stream:
                        break

            full_received = b"".join(received_body_chunks)
            self.assertEqual(full_received, b"first chunk of data; second and final chunk.")
            self.assertEqual(b"".join(body_chunks), b"all chunks received successfully")

    async def test_ext_proc_large_response_streaming(self):
        """Test streaming large multi-megabyte response from backend over TLS."""
        payload = b"X" * (1024 * 1024)  # 1MB payload

        async def mock_handler(request: web.Request):
            resp = web.StreamResponse(status=200)
            await resp.prepare(request)
            for i in range(4):  # 4 * 1MB = 4MB total
                await resp.write(payload)
            await resp.write_eof()
            return resp

        http_port = await self._start_mock_http_server(mock_handler)
        grpc_port = await self._start_ext_proc_server(http_port)

        creds = self._get_client_credentials()
        async with grpc.aio.secure_channel(f"localhost:{grpc_port}", creds) as channel:
            stub = external_processor_pb2_grpc.ExternalProcessorStub(channel)

            async def request_generator():
                req = external_processor_pb2.ProcessingRequest()
                h1 = req.request_headers.headers.headers.add()
                h1.key = ":method"
                h1.value = "GET"
                h2 = req.request_headers.headers.headers.add()
                h2.key = ":path"
                h2.value = "/large-data"
                req.request_headers.end_of_stream = True
                yield req

            responses = []
            async for resp in stub.Process(request_generator()):
                responses.append(resp)

            self.assertGreater(len(responses), 2)
            first_resp = responses[0]
            headers_dict = {
                h.key: (h.raw_value.decode("utf-8") if h.raw_value else h.value)
                for h in first_resp.streamed_immediate_response.headers_response.headers.headers
            }
            self.assertEqual(headers_dict.get(":status"), "200")

            body_chunks = [
                r.streamed_immediate_response.body_response.body
                for r in responses[1:]
                if r.streamed_immediate_response.HasField("body_response")
            ]
            total_body = b"".join(body_chunks)
            self.assertEqual(len(total_body), 4 * len(payload))

    async def test_ext_proc_backend_connection_error(self):
        """Test backend connection error returns 502 Bad Gateway over TLS."""
        unused_port = 59888
        grpc_port = await self._start_ext_proc_server(unused_port)

        creds = self._get_client_credentials()
        async with grpc.aio.secure_channel(f"localhost:{grpc_port}", creds) as channel:
            stub = external_processor_pb2_grpc.ExternalProcessorStub(channel)

            async def request_generator():
                req = external_processor_pb2.ProcessingRequest()
                h1 = req.request_headers.headers.headers.add()
                h1.key = ":method"
                h1.value = "GET"
                h2 = req.request_headers.headers.headers.add()
                h2.key = ":path"
                h2.value = "/down"
                req.request_headers.end_of_stream = True
                yield req

            responses = []
            async for resp in stub.Process(request_generator()):
                responses.append(resp)

            self.assertGreaterEqual(len(responses), 1)
            first_resp = responses[0]
            headers_dict = {
                h.key: (h.raw_value.decode("utf-8") if h.raw_value else h.value)
                for h in first_resp.streamed_immediate_response.headers_response.headers.headers
            }
            self.assertEqual(headers_dict.get(":status"), "502")

    async def test_ext_proc_backend_timeout(self):
        """Test backend timeout returns 504 Gateway Timeout over TLS."""

        async def slow_handler(request: web.Request):
            await asyncio.sleep(2.0)
            return web.Response(text="slow response")

        http_port = await self._start_mock_http_server(slow_handler)
        grpc_port = await self._start_ext_proc_server(http_port, upstream_timeout=0.5)

        creds = self._get_client_credentials()
        async with grpc.aio.secure_channel(f"localhost:{grpc_port}", creds) as channel:
            stub = external_processor_pb2_grpc.ExternalProcessorStub(channel)

            async def request_generator():
                req = external_processor_pb2.ProcessingRequest()
                h1 = req.request_headers.headers.headers.add()
                h1.key = ":method"
                h1.value = "GET"
                h2 = req.request_headers.headers.headers.add()
                h2.key = ":path"
                h2.value = "/slow"
                req.request_headers.end_of_stream = True
                yield req

            responses = []
            async for resp in stub.Process(request_generator()):
                responses.append(resp)

            self.assertGreaterEqual(len(responses), 1)
            first_resp = responses[0]
            headers_dict = {
                h.key: (h.raw_value.decode("utf-8") if h.raw_value else h.value)
                for h in first_resp.streamed_immediate_response.headers_response.headers.headers
            }
            self.assertEqual(headers_dict.get(":status"), "504")

    async def test_ext_proc_observability_mode(self):
        """Test that observability mode produces no responses over TLS."""

        async def mock_handler(request: web.Request):
            return web.Response(text="ok")

        http_port = await self._start_mock_http_server(mock_handler)
        grpc_port = await self._start_ext_proc_server(http_port)

        creds = self._get_client_credentials()
        async with grpc.aio.secure_channel(f"localhost:{grpc_port}", creds) as channel:
            stub = external_processor_pb2_grpc.ExternalProcessorStub(channel)

            async def request_generator():
                req = external_processor_pb2.ProcessingRequest()
                req.observability_mode = True
                h1 = req.request_headers.headers.headers.add()
                h1.key = ":method"
                h1.value = "GET"
                h2 = req.request_headers.headers.headers.add()
                h2.key = ":path"
                h2.value = "/obs"
                req.request_headers.end_of_stream = True
                yield req

            responses = []
            async for resp in stub.Process(request_generator()):
                responses.append(resp)

            self.assertEqual(len(responses), 0)

    async def test_ext_proc_insecure_connection_rejected(self):
        """Test that unencrypted plaintext connections to the TLS gRPC server fail."""

        async def mock_handler(request: web.Request):
            return web.Response(text="ok")

        http_port = await self._start_mock_http_server(mock_handler)
        grpc_port = await self._start_ext_proc_server(http_port)

        async with grpc.aio.insecure_channel(f"127.0.0.1:{grpc_port}") as channel:
            stub = external_processor_pb2_grpc.ExternalProcessorStub(channel)

            async def request_generator():
                req = external_processor_pb2.ProcessingRequest()
                h1 = req.request_headers.headers.headers.add()
                h1.key = ":method"
                h1.value = "GET"
                req.request_headers.end_of_stream = True
                yield req

            with self.assertRaises(grpc.RpcError):
                async for _ in stub.Process(request_generator()):
                    pass

    async def test_ext_proc_gzipped_response_not_decompressed(self):
        """Test that gzipped response is forwarded without decompressing in memory."""
        import gzip

        raw_payload = b"Hello compressed world! " * 50
        compressed_payload = gzip.compress(raw_payload)

        async def mock_handler(request: web.Request):
            resp = web.Response(body=compressed_payload, status=200)
            resp.headers["Content-Encoding"] = "gzip"
            resp.headers["Content-Type"] = "text/plain"
            return resp

        http_port = await self._start_mock_http_server(mock_handler)
        grpc_port = await self._start_ext_proc_server(http_port)

        creds = self._get_client_credentials()
        async with grpc.aio.secure_channel(f"localhost:{grpc_port}", creds) as channel:
            stub = external_processor_pb2_grpc.ExternalProcessorStub(channel)

            async def request_generator():
                req = external_processor_pb2.ProcessingRequest()
                h1 = req.request_headers.headers.headers.add()
                h1.key = ":method"
                h1.value = "GET"
                h2 = req.request_headers.headers.headers.add()
                h2.key = ":path"
                h2.value = "/compressed"
                req.request_headers.end_of_stream = True
                yield req

            responses = []
            async for resp in stub.Process(request_generator()):
                responses.append(resp)

            sir_headers = responses[0].streamed_immediate_response.headers_response.headers.headers
            headers_dict = {h.key: (h.raw_value.decode("utf-8") if h.raw_value else h.value) for h in sir_headers}
            self.assertEqual(headers_dict.get("content-encoding"), "gzip")

            received_body = b"".join(
                r.streamed_immediate_response.body_response.body
                for r in responses[1:]
                if r.streamed_immediate_response.HasField("body_response")
            )
            self.assertEqual(received_body, compressed_payload)
            self.assertEqual(gzip.decompress(received_body), raw_payload)

    async def test_ext_proc_upstream_https_with_custom_ssl_context(self):
        """Test proxying to an HTTPS backend with custom upstream_ssl_context."""
        import ssl

        backend_cert, backend_key = generate_self_signed_cert(hostname="localhost")
        try:
            backend_server_ssl = create_server_ssl_context(cert_file=backend_cert, key_file=backend_key)
            # Create client SSL context for proxy to verify backend cert
            proxy_to_backend_ssl = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=backend_cert)

            async def mock_handler(request: web.Request):
                return web.Response(text="secure backend response", status=200)

            https_port = await self._start_mock_http_server(mock_handler, ssl_context=backend_server_ssl)
            grpc_port = await self._start_ext_proc_server(
                target_port=https_port,
                target_scheme="https",
                upstream_ssl_context=proxy_to_backend_ssl,
            )

            creds = self._get_client_credentials()
            async with grpc.aio.secure_channel(f"localhost:{grpc_port}", creds) as channel:
                stub = external_processor_pb2_grpc.ExternalProcessorStub(channel)

                async def request_generator():
                    req = external_processor_pb2.ProcessingRequest()
                    h1 = req.request_headers.headers.headers.add()
                    h1.key = ":method"
                    h1.value = "GET"
                    h2 = req.request_headers.headers.headers.add()
                    h2.key = ":path"
                    h2.value = "/secure-endpoint"
                    req.request_headers.end_of_stream = True
                    yield req

                responses = []
                async for resp in stub.Process(request_generator()):
                    responses.append(resp)

                headers_dict = {
                    h.key: (h.raw_value.decode("utf-8") if h.raw_value else h.value)
                    for h in responses[0].streamed_immediate_response.headers_response.headers.headers
                }
                self.assertEqual(headers_dict.get(":status"), "200")

                body = b"".join(
                    r.streamed_immediate_response.body_response.body
                    for r in responses[1:]
                    if r.streamed_immediate_response.HasField("body_response")
                ).decode("utf-8")
                self.assertEqual(body, "secure backend response")
        finally:
            for f in (backend_cert, backend_key):
                if os.path.exists(f):
                    os.remove(f)

    def test_build_header_mutation_remove_headers(self):
        """Test _build_header_mutation calculates remove_headers diffing accurately."""
        # 1. Without original_headers, remove_headers is empty
        m1 = _build_header_mutation([("x-foo", "bar")])
        self.assertEqual(len(m1.remove_headers), 0)

        # 2. Original headers contains headers omitted in new headers
        orig = [":method", ":path", "X-Keep", "X-Drop-1", "x-drop-2"]
        new_hdrs = [("x-keep", "new-val"), ("x-added", "val")]
        m2 = _build_header_mutation(
            new_hdrs,
            status_code=200,
            method="GET",
            path="/test",
            scheme="https",
            original_headers=orig,
        )
        # Should exclude :method, :path, and x-keep
        # Should include sorted lowercase of x-drop-1 and x-drop-2
        self.assertEqual(list(m2.remove_headers), ["x-drop-1", "x-drop-2"])

        # Check set_headers
        set_map = {
            h.header.key: (h.header.raw_value.decode("utf-8") if h.header.raw_value else h.header.value)
            for h in m2.set_headers
        }
        self.assertEqual(set_map.get(":status"), "200")
        self.assertEqual(set_map.get(":method"), "GET")
        self.assertEqual(set_map.get(":path"), "/test")
        self.assertEqual(set_map.get(":scheme"), "https")
        self.assertEqual(set_map.get("x-keep"), "new-val")
        self.assertEqual(set_map.get("x-added"), "val")

    async def test_queue_maxsize_and_backpressure(self):
        """Test that queues enforce maxsize limits, apply backpressure, and abort cleanly when full."""
        self.assertGreater(DEFAULT_QUEUE_MAXSIZE, 0)
        session = ExtProcSession(request_id="test-maxsize")
        self.assertEqual(session.request_to_envoy_queue.maxsize, DEFAULT_QUEUE_MAXSIZE)
        self.assertEqual(session.response_from_envoy_queue.maxsize, DEFAULT_QUEUE_MAXSIZE)

        # Fill queue to capacity
        for i in range(DEFAULT_QUEUE_MAXSIZE):
            session.request_to_envoy_queue.put_nowait(UpstreamRequestBodyChunk(data=b"chunk", is_last=False))
        self.assertTrue(session.request_to_envoy_queue.full())

        # An additional put should block due to backpressure
        blocked_put = asyncio.create_task(
            session.request_to_envoy_queue.put(UpstreamRequestBodyChunk(data=b"overflow", is_last=False))
        )
        await asyncio.sleep(0.01)
        self.assertFalse(blocked_put.done())

        # Reading one item relieves backpressure and unblocks the pending put
        item = session.request_to_envoy_queue.get_nowait()
        self.assertEqual(item.data, b"chunk")
        await asyncio.wait_for(blocked_put, timeout=1.0)
        self.assertTrue(session.request_to_envoy_queue.full())

        # Aborting on a completely full queue must not raise QueueFull
        session.abort(reason="test abort")
        self.assertTrue(session.is_aborted)

        # Confirm SessionAbort was enqueued
        items = []
        while not session.request_to_envoy_queue.empty():
            items.append(session.request_to_envoy_queue.get_nowait())
        self.assertTrue(any(isinstance(it, SessionAbort) for it in items))

    async def test_put_request_and_response_wake_on_abort(self):
        """Test that producers blocked on full queues wake up immediately when session is aborted."""
        session = ExtProcSession(request_id="test-wake-abort")
        for _ in range(DEFAULT_QUEUE_MAXSIZE):
            session.request_to_envoy_queue.put_nowait(UpstreamRequestBodyChunk(data=b"data", is_last=False))
            session.response_from_envoy_queue.put_nowait(EnvoyResponseBodyChunk(data=b"data", is_last=False))

        self.assertTrue(session.request_to_envoy_queue.full())
        self.assertTrue(session.response_from_envoy_queue.full())

        req_put_task = asyncio.create_task(session.put_request(UpstreamRequestBodyChunk(data=b"more", is_last=False)))
        resp_put_task = asyncio.create_task(session.put_response(EnvoyResponseBodyChunk(data=b"more", is_last=False)))
        await asyncio.sleep(0.01)
        self.assertFalse(req_put_task.done())
        self.assertFalse(resp_put_task.done())

        session.abort(reason="aborted by test")

        req_res = await asyncio.wait_for(req_put_task, timeout=1.0)
        resp_res = await asyncio.wait_for(resp_put_task, timeout=1.0)
        self.assertFalse(req_res)
        self.assertFalse(resp_res)

        self.assertFalse(await session.put_request(UpstreamRequestBodyChunk(data=b"after", is_last=False)))
        self.assertFalse(await session.put_response(EnvoyResponseBodyChunk(data=b"after", is_last=False)))

    async def test_put_request_cancellation_cancels_child_task_without_orphan_write(self):
        """Test that cancelling a put_request task cleanly cancels background put and prevents orphan write."""
        session = ExtProcSession(request_id="test-cancel-clean")
        for _ in range(DEFAULT_QUEUE_MAXSIZE):
            session.request_to_envoy_queue.put_nowait(UpstreamRequestBodyChunk(data=b"fill", is_last=False))
        self.assertTrue(session.request_to_envoy_queue.full())

        put_caller_task = asyncio.create_task(
            session.put_request(UpstreamRequestBodyChunk(data=b"orphan-payload", is_last=False))
        )
        await asyncio.sleep(0.01)
        self.assertFalse(put_caller_task.done())

        put_caller_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await put_caller_task

        item = session.request_to_envoy_queue.get_nowait()
        self.assertEqual(item.data, b"fill")
        self.assertFalse(session.request_to_envoy_queue.full())

        await asyncio.sleep(0.02)

        self.assertFalse(session.request_to_envoy_queue.full())
        remaining = []
        while not session.request_to_envoy_queue.empty():
            remaining.append(session.request_to_envoy_queue.get_nowait())
        self.assertFalse(
            any(it.data == b"orphan-payload" for it in remaining if isinstance(it, UpstreamRequestBodyChunk))
        )


if __name__ == "__main__":
    unittest.main()

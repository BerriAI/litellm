"""Executed by the PyO3 retained test with the actual built module in `native`."""

import asyncio
import contextvars
import gc
import inspect
import json
import threading
import unittest
import weakref
from copy import deepcopy
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import ModuleType
from unittest.mock import patch

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.mistral.ocr.transformation import MistralOCRConfig
from litellm.rust_bridge.ocr_retained import OCREncoded, OCRRetainedBoundary, OCRRoots

native: ModuleType = globals()["native"]
context = contextvars.ContextVar("retained-real-boundary", default="unset")
MODEL = "mistral-ocr-latest"
RESPONSE = {"pages": [{"index": 0, "markdown": "local OCR"}], "model": MODEL, "usage_info": {"pages_processed": 1}}


class Graph(dict):
    pass


class Header(str):
    pass


class PreCallAbort(BaseException):
    pass


class CopiedDocumentBoundary(OCRRetainedBoundary):
    def prepare(self) -> OCRRoots:
        self.document = deepcopy(self.document)
        return super().prepare()

    async def aprepare(self) -> OCRRoots:
        self.document = deepcopy(self.document)
        return await super().aprepare()


class ReboundBodyBoundary(OCRRetainedBoundary):
    def encode(self, roots: OCRRoots) -> OCREncoded:
        headers, url, _body, files = roots
        view = self.logging_obj.model_call_details["additional_args"]
        return super().encode((headers, url, view["complete_input_dict"], files))


class ReboundHeadersBoundary(OCRRetainedBoundary):
    def encode(self, roots: OCRRoots) -> OCREncoded:
        _headers, url, body, files = roots
        view = self.logging_obj.model_call_details["additional_args"]
        return super().encode((view["headers"], url, body, files))


class Callback(CustomLogger):
    def __init__(self, action):
        super().__init__()
        self.action = action
        self.failures = []
        self.calls = 0

    def log_pre_api_call(self, model, messages, kwargs):
        self.calls += 1
        try:
            return self.action(kwargs["additional_args"])
        except AssertionError as error:
            self.failures.append(str(error))
            raise


class Server:
    def __init__(self):
        self.requests = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                owner.requests.append((self.path, sorted((k.lower(), v) for k, v in self.headers.items()), body))
                if self.path == "/blocked/v1/ocr":
                    owner.started.set()
                    if not owner.release.wait(10):
                        owner.finished.set()
                        return
                failed = self.path == "/error/v1/ocr"
                payload = b'{"message":"controlled HTTP failure"}' if failed else json.dumps(RESPONSE).encode()
                self.send_response(429 if failed else 200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    if self.path == "/blocked/v1/ocr":
                        owner.finished.set()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"

    def close(self):
        self.release.set()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(5)
        assert not self.thread.is_alive()


def inputs(server, callbacks=(), *, document=None, optional=None, path="", client=None):
    return {
        "model": MODEL,
        "document": Graph(type="document_url", document_url="https://example.test/original.pdf")
        if document is None
        else document,
        "optional_params": {} if optional is None else optional,
        "logging_obj": Logging(
            model=MODEL,
            messages=[],
            stream=False,
            call_type="ocr",
            start_time=datetime.now(),
            litellm_call_id="retained-real",
            function_id="retained-real",
            dynamic_input_callbacks=list(callbacks),
            supports_correlation_logging=False,
        ),
        "api_key": "local-test-key",
        "api_base": server.url + path,
        "headers": {"X-Proof": "original"},
        "provider_config": MistralOCRConfig(),
        "litellm_params": {},
        "custom_llm_provider": "mistral",
        "timeout": 5.0,
        "client": client,
    }


def invoke(mode, kwargs, *, boundary_factory=OCRRetainedBoundary):
    handler = BaseLLMHTTPHandler()
    if mode == "python-sync":
        return handler.ocr(**kwargs)
    if mode == "python-async":
        return handler.async_ocr(**kwargs)
    if mode == "native-sync":
        return native.ocr_retained(boundary_factory(handler=handler, **kwargs))
    assert mode == "native-async"
    return native.aocr_retained(boundary_factory(handler=handler, **kwargs))


class RealBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.server = Server()
        self.addCleanup(self.server.close)
        self.sync_client = HTTPHandler(timeout=5.0)
        self.addCleanup(self.sync_client.close)

    def check_callbacks(self, callbacks):
        for callback in callbacks:
            self.assertEqual(callback.failures, [])
            self.assertEqual(callback.calls, 1)

    async def differential(self, mode, *, boundary_factory=OCRRetainedBoundary):
        context.set("caller")
        thread = threading.get_ident()
        task = asyncio.current_task()
        document = Graph(type="document_url", document_url="https://example.test/original.pdf")
        nested = Graph(values=[1])
        optional = {"unknown_python_json": {7: ("tuple", 2)}, "nested": nested, "nested_alias": nested["values"]}
        retained = {}
        events = []

        def phase(name, expected):
            self.assertEqual(threading.get_ident(), thread)
            self.assertIs(asyncio.current_task(), task)
            self.assertEqual(context.get(), expected)
            events.append(name)

        def mutate(view):
            phase("mutate", "caller")
            body, headers = view["complete_input_dict"], view["headers"]
            self.assertIs(body["document"], document, "caller document identity was not retained")
            self.assertIs(body["nested"], nested)
            self.assertIs(body["nested_alias"], nested["values"])
            self.assertIs(body["unknown_python_json"], optional["unknown_python_json"])
            retained.update(body=body, headers=headers, view=view)
            headers["X-Proof"] = "in-place"
            body["body_mutation"] = True
            nested["values"].append(2)
            view["headers"] = {"X-Proof": "must-not-send"}
            view["complete_input_dict"] = {"document": {"document_url": "must-not-send"}}
            context.set("mutated")
            child_callback = Callback(lambda _: events.append("reentry"))
            child = invoke("native-sync", inputs(self.server, [child_callback], path="/child", client=self.sync_client))
            self.assertEqual(child.pages[0].markdown, "local OCR")
            self.check_callbacks([child_callback])
            return {"headers": {"X-Proof": "ignored-return"}, "complete_input_dict": {"invalid": object()}}

        def mutate_then_raise(view):
            phase("raise", "mutated")
            self.assertIs(view, retained["view"])
            self.assertEqual(view["complete_input_dict"], {"document": {"document_url": "must-not-send"}})
            self.assertEqual(view["headers"], {"X-Proof": "must-not-send"})
            retained["body"]["before_error"] = True
            retained["headers"]["X-Before-Error"] = "yes"
            context.set("caught")
            raise RuntimeError("intentional non-blocking pre_call error")

        def closure_only(view):
            phase("later", "caught")
            self.assertIs(view, retained["view"])
            self.assertIsNot(view["complete_input_dict"], retained["body"])
            self.assertIsNot(view["headers"], retained["headers"])
            self.assertTrue(retained["body"]["before_error"])
            self.assertEqual(retained["headers"]["X-Before-Error"], "yes")
            self.assertIs(retained["body"]["nested_alias"], nested["values"])
            self.assertEqual(retained["body"]["nested_alias"], [1, 2])
            view["complete_input_dict"]["observed"] = True
            view["headers"]["X-View-Only"] = "not-on-wire"
            document["document_url"] = "https://example.test/closure.pdf"
            context.set("later")

        callbacks = [Callback(action) for action in (mutate, mutate_then_raise, closure_only)]
        async_client = AsyncHTTPHandler(timeout=5.0)
        try:
            kwargs = inputs(
                self.server,
                callbacks,
                document=document,
                optional=optional,
                client=async_client if mode.endswith("async") else self.sync_client,
            )
            logging_ref = weakref.ref(kwargs["logging_obj"])
            before = len(self.server.requests)
            pending = invoke(mode, kwargs, boundary_factory=boundary_factory)
            if mode.endswith("async"):
                self.assertTrue(inspect.iscoroutine(pending))
                self.assertEqual(events, [])
                self.assertEqual(len(self.server.requests), before)
                self.assertNotIn("additional_args", kwargs["logging_obj"].model_call_details)
                response = await pending
            else:
                response = pending
            self.check_callbacks(callbacks)
            self.assertEqual(events, ["mutate", "reentry", "raise", "later"])
            self.assertEqual(context.get(), "later")
            self.assertEqual(len(self.server.requests), before + 2)
            wire = self.server.requests[-1]
            self.assertEqual(wire[0], "/v1/ocr")
            expected = (
                b'{"model":"mistral-ocr-latest","document":{"type":"document_url",'
                b'"document_url":"https://example.test/closure.pdf"},"unknown_python_json":{"7":["tuple",2]},'
                b'"nested":{"values":[1,2]},"nested_alias":[1,2],"body_mutation":true,"before_error":true}'
            )
            self.assertEqual(wire[2], expected, "wire body must encode retained mutations after pre_call")
            self.assertIn(("x-proof", "in-place"), wire[1], "wire headers must use retained execution headers")
            self.assertIn(("x-before-error", "yes"), wire[1])
            self.assertIn(("authorization", "Bearer local-test-key"), wire[1])
            self.assertFalse(any(name == "x-view-only" for name, _ in wire[1]))
            self.assertEqual(
                retained["view"]["complete_input_dict"],
                {"document": {"document_url": "must-not-send"}, "observed": True},
            )
            self.assertEqual(retained["view"]["headers"], {"X-Proof": "must-not-send", "X-View-Only": "not-on-wire"})
            del kwargs, pending
            gc.collect()
            self.assertIsNone(logging_ref(), "logging owner survived the completed call")
            self.assertIs(retained["body"]["document"], document)
            retained["body"]["nested"]["values"].append(3)
            retained["headers"]["X-After-Return"] = "usable"
            self.assertEqual(optional["nested"]["values"], [1, 2, 3])
            self.assertIs(retained["body"]["nested_alias"], optional["nested"]["values"])
            self.assertEqual(retained["body"]["nested_alias"], [1, 2, 3])
            self.assertEqual(retained["headers"]["X-After-Return"], "usable")
            self.assertEqual(wire[2], expected)
            self.assertEqual(response.pages[0].markdown, "local OCR")
            return wire, response.model_dump()
        finally:
            await async_client.close()

    def test_differential_callbacks_wire(self):
        async def exercise():
            baseline = await self.differential("python-sync")
            for mode in ("python-async", "native-sync", "native-async"):
                with self.subTest(mode=mode):
                    self.assertEqual(await self.differential(mode), baseline)

        asyncio.run(exercise())

    def check_negative_control(self, boundary_factory, failure, expected_requests):
        async def exercise():
            baseline = await self.differential("python-sync")
            for mode in ("native-sync", "native-async"):
                with self.subTest(mode=mode):
                    symbol = "aocr_retained" if mode.endswith("async") else "ocr_retained"
                    self.assertTrue(inspect.isbuiltin(getattr(native, symbol)))
                    self.assertEqual(await self.differential(mode), baseline)
                    before = len(self.server.requests)
                    with self.assertRaisesRegex(AssertionError, failure):
                        await self.differential(mode, boundary_factory=boundary_factory)
                    self.assertEqual(len(self.server.requests), before + expected_requests)
                    self.assertEqual(self.server.requests[-1][0], "/v1/ocr")

        asyncio.run(exercise())

    def test_negative_control_copied_caller_document(self):
        self.check_negative_control(CopiedDocumentBoundary, "caller document identity was not retained", 1)

    def test_negative_control_rebound_logging_body(self):
        self.check_negative_control(ReboundBodyBoundary, "wire body must encode retained mutations after pre_call", 2)

    def test_negative_control_rebound_logging_headers(self):
        self.check_negative_control(ReboundHeadersBoundary, "wire headers must use retained execution headers", 2)

    def test_differential_callback_retained_mutation_after_post_received(self):
        async def suspended(mode):
            self.server.started.clear()
            self.server.release.clear()
            self.server.finished.clear()
            retained = {}

            def retain(view):
                retained.update(body=view["complete_input_dict"], headers=view["headers"], view=view)
                view["complete_input_dict"] = {"replacement": True}
                view["headers"] = {"X-Proof": "must-not-send"}

            callback = Callback(retain)
            client = AsyncHTTPHandler(timeout=5.0)
            kwargs = inputs(self.server, [callback], path="/blocked", client=client)
            logging = kwargs["logging_obj"]
            logging.log_raw_request_response = True
            before = len(self.server.requests)
            task = asyncio.create_task(invoke(mode, kwargs))
            try:
                self.assertTrue(await asyncio.to_thread(self.server.started.wait, 5))
                self.assertFalse(task.done())
                self.assertFalse(self.server.release.is_set())
                self.assertFalse(self.server.finished.is_set())
                self.check_callbacks([callback])
                self.assertEqual(len(self.server.requests), before + 1)
                path, headers, body = self.server.requests[-1]
                wire = (path, tuple(headers), body)
                expected = (
                    b'{"model":"mistral-ocr-latest","document":{"type":"document_url",'
                    b'"document_url":"https://example.test/original.pdf"}}'
                )
                self.assertEqual(path, "/blocked/v1/ocr")
                self.assertEqual(body, expected)
                self.assertIn(("x-proof", "original"), headers)
                self.assertNotIn(("x-proof", "must-not-send"), headers)
                logged_body = logging.model_call_details["raw_request_typed_dict"]["raw_request_body"]
                self.assertIs(logged_body, retained["body"])
                self.assertIs(logged_body["document"], kwargs["document"])
                self.assertIs(logging.model_call_details["additional_args"], retained["view"])
                self.assertIsNot(retained["view"]["complete_input_dict"], retained["body"])
                self.assertIsNot(retained["view"]["headers"], retained["headers"])

                retained["body"]["after_encoding"] = True
                retained["body"]["document"]["document_url"] = "https://example.test/while-blocked.pdf"
                retained["headers"]["X-Proof"] = "while-blocked"
                self.assertTrue(logged_body["after_encoding"])
                self.assertEqual(kwargs["document"]["document_url"], "https://example.test/while-blocked.pdf")
                self.assertEqual(logged_body["document"]["document_url"], "https://example.test/while-blocked.pdf")
                self.assertEqual(retained["headers"]["X-Proof"], "while-blocked")
                self.assertEqual(retained["view"]["complete_input_dict"], {"replacement": True})
                self.assertEqual(retained["view"]["headers"], {"X-Proof": "must-not-send"})
                self.assertFalse(task.done())
                self.assertFalse(self.server.finished.is_set())
                self.assertEqual((path, tuple(headers), body), wire)

                self.server.release.set()
                response = await asyncio.wait_for(task, 5)
                self.assertEqual(response.pages[0].markdown, "local OCR")
                self.assertIs(logging.model_call_details["raw_request_typed_dict"]["raw_request_body"], logged_body)
                self.assertTrue(logged_body["after_encoding"])
                self.assertEqual(len(self.server.requests), before + 1)
                received_path, received_headers, received_body = self.server.requests[-1]
                self.assertEqual((received_path, tuple(received_headers), received_body), wire)
                self.check_callbacks([callback])
                return wire, logged_body, retained["headers"], response.model_dump()
            finally:
                self.server.release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await client.close()
                if self.server.started.is_set():
                    self.assertTrue(await asyncio.to_thread(self.server.finished.wait, 5))

        async def exercise():
            baseline = await suspended("python-async")
            self.assertEqual(await suspended("native-async"), baseline)

        asyncio.run(exercise())

    def test_public_rust_dispatch_wire_fallback_and_escaping_base_exception(self):
        async def exercise():
            for asynchronous in (False, True):
                for outcome in ("success", "missing-symbol", "pre-call-abort", "disabled"):
                    with self.subTest(asynchronous=asynchronous, outcome=outcome):
                        symbol = "aocr_retained" if asynchronous else "ocr_retained"
                        self.assertTrue(inspect.isbuiltin(getattr(native, symbol)))
                        missing_symbol = ModuleType("native_without_" + symbol)
                        missing_symbol.__dict__.update(
                            (name, value) for name, value in vars(native).items() if name != symbol
                        )
                        escaped = PreCallAbort("public pre_call must escape unchanged")
                        before = len(self.server.requests)

                        def mutate(view, outcome=outcome, escaped=escaped):
                            view["headers"]["X-Proof"] = "public-in-place"
                            view["complete_input_dict"]["public_mutation"] = True
                            view["complete_input_dict"]["document"]["document_url"] = (
                                "https://example.test/public-mutated.pdf"
                            )
                            if outcome == "pre-call-abort":
                                raise escaped

                        callback = Callback(mutate)
                        kwargs = inputs(self.server, [callback])
                        with patch(
                            "litellm.rust_bridge.get_native_bridge",
                            return_value=missing_symbol if outcome == "missing-symbol" else native,
                        ) as loader:
                            public_kwargs = {
                                **{
                                    key: kwargs[key]
                                    for key in (
                                        "model",
                                        "document",
                                        "api_key",
                                        "api_base",
                                        "custom_llm_provider",
                                        "timeout",
                                    )
                                },
                                "extra_headers": kwargs["headers"],
                                "litellm_logging_obj": kwargs["logging_obj"],
                                "rust": outcome != "disabled",
                            }

                            async def call(asynchronous=asynchronous, public_kwargs=public_kwargs):
                                if asynchronous:
                                    return await litellm.aocr(**public_kwargs)
                                return litellm.ocr(**public_kwargs)

                            if outcome == "pre-call-abort":
                                with self.assertRaises(PreCallAbort) as caught:
                                    await call()
                                self.assertIs(caught.exception, escaped)
                            else:
                                response = await call()
                                self.assertEqual(response.pages[0].markdown, "local OCR")

                        if outcome == "disabled":
                            loader.assert_not_called()
                        else:
                            loader.assert_called_once_with()
                        self.check_callbacks([callback])
                        self.assertEqual(len(self.server.requests), before + int(outcome != "pre-call-abort"))
                        if outcome != "pre-call-abort":
                            path, headers, body = self.server.requests[-1]
                            self.assertEqual(path, "/v1/ocr")
                            self.assertIn(("x-proof", "public-in-place"), headers)
                            self.assertIn(("authorization", "Bearer local-test-key"), headers)
                            self.assertEqual(
                                json.loads(body),
                                {
                                    "model": MODEL,
                                    "document": {
                                        "type": "document_url",
                                        "document_url": "https://example.test/public-mutated.pdf",
                                    },
                                    "public_mutation": True,
                                },
                            )

        asyncio.run(exercise())

    def lifecycle_inputs(self, outcome, retained):
        refs = []

        def callback(view):
            body, headers = view["complete_input_dict"], view["headers"]
            body["sentinel"] = Graph(alive=True)
            headers["X-Sentinel"] = Header("alive")
            refs.extend((weakref.ref(body["sentinel"]), weakref.ref(headers["X-Sentinel"])))
            if outcome == "encoding":
                body["not_json"] = object()
            if retained is not None:
                retained.extend((body, headers))
            view["complete_input_dict"] = {}
            view["headers"] = {}
            if outcome == "pre-call-abort":
                raise PreCallAbort("lifecycle pre_call abort")

        logger = Callback(callback)
        kwargs = inputs(
            self.server,
            [logger],
            optional={"nested": Graph(alive=True)},
            path={"http": "/error", "cancel": "/blocked"}.get(outcome, ""),
            client=self.sync_client,
        )
        refs.extend(weakref.ref(kwargs[key]) for key in ("document", "logging_obj"))
        refs.append(weakref.ref(kwargs["optional_params"]["nested"]))
        return kwargs, logger, refs

    async def lifecycle(self, mode, outcome, retained=None):
        kwargs, logger, refs = self.lifecycle_inputs(outcome, retained)
        before = len(self.server.requests)
        async_client = AsyncHTTPHandler(timeout=5.0)
        if mode.endswith("async"):
            kwargs["client"] = async_client
        try:
            try:
                pending = invoke(mode, kwargs)
                response = await pending if mode.endswith("async") else pending
            except BaseLLMException as error:
                self.assertIn(outcome, ("encoding", "http"))
                self.assertEqual(error.status_code, 500 if outcome == "encoding" else 429)
                signature = (type(error), error.status_code, str(error))
            except PreCallAbort as error:
                self.assertEqual(outcome, "pre-call-abort")
                signature = (type(error), str(error))
            else:
                self.assertEqual(outcome, "success")
                self.assertEqual(response.pages[0].markdown, "local OCR")
                signature = response.model_dump()
            self.check_callbacks([logger])
            self.assertEqual(len(refs), 5)
            self.assertEqual(len(self.server.requests), before + (outcome not in ("encoding", "pre-call-abort")))
            return refs, signature
        finally:
            await async_client.close()

    def test_collection_after_success_and_failures(self):
        async def exercise():
            for outcome in ("success", "encoding", "http", "pre-call-abort"):
                baseline = None
                for mode in ("python-sync", "python-async", "native-sync", "native-async"):
                    with self.subTest(mode=mode, outcome=outcome):
                        refs, signature = await self.lifecycle(mode, outcome)
                        gc.collect()
                        self.assertTrue(all(ref() is None for ref in refs), f"request graph leaked: {mode=} {outcome=}")
                        if baseline is None:
                            baseline = signature
                        self.assertEqual(signature, baseline)

        asyncio.run(exercise())

    def test_callback_retained_graph_remains_usable_then_collects(self):
        async def exercise():
            for outcome in ("success", "encoding", "http", "pre-call-abort"):
                for mode in ("python-sync", "python-async", "native-sync", "native-async"):
                    with self.subTest(mode=mode, outcome=outcome):
                        retained = []
                        refs, _ = await self.lifecycle(mode, outcome, retained)
                        gc.collect()
                        self.assertIsNone(refs[1](), f"logging owner survived: {mode=} {outcome=}")
                        self.assertTrue(all(refs[index]() is not None for index in (0, 2, 3, 4)))
                        self.assertIs(retained[0]["document"], refs[0]())
                        self.assertIs(retained[0]["nested"], refs[2]())
                        self.assertIs(retained[0]["sentinel"], refs[3]())
                        self.assertIs(retained[1]["X-Sentinel"], refs[4]())
                        retained[0]["document"]["after_return"] = "usable"
                        retained[0]["nested"]["alive"] = "nested still usable"
                        retained[0]["sentinel"]["alive"] = "still usable"
                        retained[1]["X-After-Return"] = "usable"
                        self.assertEqual(refs[0]()["after_return"], "usable")
                        self.assertEqual(refs[2]()["alive"], "nested still usable")
                        self.assertEqual(refs[3]()["alive"], "still usable")
                        self.assertEqual(retained[1]["X-After-Return"], "usable")
                        retained.clear()
                        gc.collect()
                        self.assertTrue(all(ref() is None for ref in refs))

        asyncio.run(exercise())

    def test_collection_after_cancellation_during_blocked_transport(self):
        async def cancel():
            kwargs, logger, refs = self.lifecycle_inputs("cancel", None)
            client = AsyncHTTPHandler(timeout=5.0)
            kwargs["client"] = client
            task = asyncio.create_task(invoke("native-async", kwargs))
            try:
                self.assertTrue(await asyncio.to_thread(self.server.started.wait, 5))
                self.check_callbacks([logger])
                self.assertEqual(len(refs), 5)
                gc.collect()
                self.assertTrue(all(ref() is not None for ref in refs))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertFalse(self.server.release.is_set())
                return refs
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                await client.close()

        async def exercise():
            refs = await cancel()
            barrier = asyncio.get_running_loop().create_future()
            asyncio.get_running_loop().call_soon(barrier.set_result, None)
            await barrier
            gc.collect()
            self.assertTrue(all(ref() is None for ref in refs), "cancelled native call retained the request graph")
            self.server.release.set()
            self.assertTrue(await asyncio.to_thread(self.server.finished.wait, 5))
            self.assertEqual(len(self.server.requests), 1)

        asyncio.run(exercise())

"""Strict whole-argument OCR dispatch and opt-in native transport contracts."""

import asyncio
import atexit
import builtins
import contextvars
import gc
import importlib
import inspect
import json
import os
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import ModuleType
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge import configuration

ocr_main = importlib.import_module("litellm.ocr.main")
rust_bridge = importlib.import_module("litellm.rust_bridge.ocr")
rust_bridge_bindings = importlib.import_module("litellm.rust_bridge.bindings")
rust_bridge_loader = importlib.import_module("litellm.rust_bridge.loader")

MODEL = "mistral/mistral-ocr-latest"
RESPONSE_DATA = {
    "pages": [
        {
            "index": 0,
            "markdown": "proof",
            "images": [{"id": "image-0", "image_base64": "aW1hZ2U="}],
            "dimensions": {"dpi": 200, "height": 2200, "width": 1700},
        }
    ],
    "model": "mistral-ocr-latest",
    "document_annotation": {"title": "Test document"},
    "usage_info": {"pages_processed": 1, "doc_size_bytes": 1234},
    "object": "ocr",
}


@pytest.fixture(autouse=True)
def reset_rust_state(monkeypatch):
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    configuration.reset_rust_configuration()
    rust_bridge._OCR.reset()
    rust_bridge._AOCR.reset()
    rust_bridge_loader.reset_native_bridge_cache()
    try:
        yield
    finally:
        rust_bridge._OCR.reset()
        rust_bridge._AOCR.reset()
        configuration.reset_rust_configuration()
        rust_bridge_loader.reset_native_bridge_cache()


@pytest.fixture
def document():
    return {"type": "document_url", "document_url": "https://example.invalid/document.pdf"}


@pytest.fixture
def response():
    return OCRResponse.model_validate(RESPONSE_DATA)


@pytest.fixture
def injected_native(response):
    sync = Mock(return_value=response)
    asynchronous = AsyncMock(return_value=response)
    rust_bridge._OCR.override(sync)
    rust_bridge._AOCR.override(asynchronous)
    return sync, asynchronous


@pytest.fixture
def no_python_ocr(monkeypatch):
    prepare = Mock(side_effect=lambda **kwargs: pytest.fail("Python OCR preparation ran"))
    handler = Mock(side_effect=lambda **kwargs: pytest.fail("Python OCR transport ran"))
    mapping = Mock(side_effect=lambda **kwargs: pytest.fail("Python exception mapping ran"))
    monkeypatch.setattr(ocr_main, "_prepare_ocr_request", prepare)
    monkeypatch.setattr(ocr_main.base_llm_http_handler, "_prepare_ocr_request", prepare)
    monkeypatch.setattr(ocr_main.base_llm_http_handler, "_async_prepare_ocr_request", prepare)
    monkeypatch.setattr(ocr_main.base_llm_http_handler, "ocr", handler)
    monkeypatch.setattr(ocr_main.base_llm_http_handler, "async_ocr", handler)
    monkeypatch.setattr(litellm, "exception_type", mapping)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("enable_with", ["global", "environment"])
async def test_unwrapped_dispatch_preserves_every_argument(
    monkeypatch, no_python_ocr, injected_native, response, asynchronous, enable_with
):
    if enable_with == "global":
        litellm.rust(True)
    else:
        monkeypatch.setenv("LITELLM_RUST", "1")
    opaque = object()
    document = {"type": "file", "file": opaque, "mime_type": "application/pdf"}
    metadata = {"opaque": opaque, "nested": []}
    arguments = {
        "model": "unresolved-provider/opaque-model",
        "document": document,
        "api_key": opaque,
        "api_base": opaque,
        "timeout": httpx.Timeout(30, read=12.5),
        "custom_llm_provider": "unported-provider",
        "extra_headers": {"x-opaque": opaque},
        "metadata": metadata,
        "litellm_metadata": {"opaque": opaque},
        "litellm_logging_obj": opaque,
        "litellm_call_id": "whole-arguments",
        "pages": [0, 2],
        "include_image_base64": True,
        "document_annotation_format": {"schema": opaque},
        "req_format": "native",
        "client": opaque,
        "arbitrary_option": opaque,
        "optional_params": {"opaque": opaque},
        "kwargs": {"caller_owned": opaque},
        "aocr": opaque,
    }
    result = (
        await inspect.unwrap(ocr_main.aocr)(**arguments) if asynchronous else inspect.unwrap(ocr_main.ocr)(**arguments)
    )

    sync, async_native = injected_native
    selected, unused = (async_native, sync) if asynchronous else (sync, async_native)
    selected.assert_called_once()
    unused.assert_not_called()
    if asynchronous:
        async_native.assert_awaited_once()
    assert result is response
    assert selected.call_args.kwargs == {}
    (forwarded,) = selected.call_args.args
    assert isinstance(forwarded, dict)
    assert forwarded.keys() == arguments.keys()
    for name, value in arguments.items():
        assert forwarded[name] is value, name
    assert document == {"type": "file", "file": opaque, "mime_type": "application/pdf"}
    assert metadata == {"opaque": opaque, "nested": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_unwrapped_dispatch_keeps_unresolved_defaults(
    no_python_ocr, injected_native, document, response, asynchronous
):
    litellm.rust(True)
    result = (
        await inspect.unwrap(ocr_main.aocr)(MODEL, document)
        if asynchronous
        else inspect.unwrap(ocr_main.ocr)(MODEL, document)
    )
    selected = injected_native[int(asynchronous)]
    selected.assert_called_once_with(
        {
            "model": MODEL,
            "document": document,
            "api_key": None,
            "api_base": None,
            "timeout": None,
            "custom_llm_provider": None,
            "extra_headers": None,
        }
    )
    assert result is response


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_public_decorator_routes_full_kwargs(no_python_ocr, injected_native, document, response, asynchronous):
    litellm.rust(True)
    metadata = {"test_tag": "whole-arguments"}
    pages = [0, 2]
    arguments = {
        "model": MODEL,
        "document": document,
        "api_key": "sk-test",
        "api_base": "https://example.invalid",
        "timeout": 12.5,
        "extra_headers": {"x-trace-id": "trace-1"},
        "pages": pages,
        "include_image_base64": True,
        "metadata": metadata,
        "arbitrary_option": {"nested": ["preserved"]},
        "num_retries": 0,
    }
    result = await litellm.aocr(**arguments) if asynchronous else litellm.ocr(**arguments)

    selected = injected_native[int(asynchronous)]
    selected.assert_called_once()
    injected_native[not asynchronous].assert_not_called()
    (forwarded,) = selected.call_args.args
    assert result is response
    for name in ("model", "api_key", "api_base", "timeout", "extra_headers", "arbitrary_option", "num_retries"):
        assert forwarded[name] == arguments[name]
    assert forwarded["document"] is document
    assert forwarded["pages"] is pages
    assert forwarded["include_image_base64"] is True
    assert forwarded["metadata"]["test_tag"] == "whole-arguments"
    assert forwarded["custom_llm_provider"] is None
    assert "litellm_logging_obj" not in forwarded
    assert "litellm_call_id" not in forwarded
    assert "kwargs" not in forwarded


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_bridge_passes_same_dictionary_and_response(injected_native, document, response, asynchronous):
    arguments = {"model": MODEL, "document": document, "opaque": object()}
    result = await rust_bridge.aocr(arguments) if asynchronous else rust_bridge.ocr(arguments)
    selected = injected_native[int(asynchronous)]
    selected.assert_called_once_with(arguments)
    assert selected.call_args.args[0] is arguments
    assert result is response


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("public", [False, True], ids=["unwrapped", "decorated"])
@pytest.mark.parametrize("failure", ["missing", "unsupported", "runtime"])
async def test_native_failures_propagate_without_fallback(
    monkeypatch, no_python_ocr, injected_native, document, asynchronous, public, failure
):
    litellm.rust(True)
    error = (
        NotImplementedError("native OCR provider is not implemented")
        if failure == "unsupported"
        else RuntimeError("native OCR failed")
    )
    if failure == "missing":
        rust_bridge._OCR.reset()
        rust_bridge._AOCR.reset()
        monkeypatch.setattr(rust_bridge_bindings, "get_native_bridge", lambda: None)
    else:
        injected_native[int(asynchronous)].side_effect = error
    function = litellm.aocr if asynchronous else litellm.ocr
    route = function if public else inspect.unwrap(function)
    with pytest.raises(RuntimeError if failure == "missing" else type(error)) as caught:
        result = route(model=MODEL, document=document, api_key="sk-test", num_retries=0)
        if asynchronous:
            await result
    if failure == "missing":
        assert "OCR" in str(caught.value).upper()
    else:
        assert caught.value is error
        injected_native[int(asynchronous)].assert_called_once()
    injected_native[not asynchronous].assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_bridge_missing_binding_is_strict(document, asynchronous):
    rust_bridge._OCR.override(None)
    rust_bridge._AOCR.override(None)
    with pytest.raises(RuntimeError, match="(?i)ocr"):
        if asynchronous:
            await rust_bridge.aocr({"model": MODEL, "document": document})
        else:
            rust_bridge.ocr({"model": MODEL, "document": document})


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("setting", ["default", "disabled", "overrides-environment"])
async def test_rust_off_keeps_python_preparation_and_transport(
    monkeypatch, injected_native, response, asynchronous, setting
):
    if setting == "overrides-environment":
        monkeypatch.setenv("LITELLM_RUST", "1")
    if setting != "default":
        litellm.rust(False)
    prepare = Mock(wraps=ocr_main._prepare_ocr_request)
    handler = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    lookup = Mock(side_effect=lambda: pytest.fail("disabled Rust binding was consulted"))
    monkeypatch.setattr(ocr_main, "_prepare_ocr_request", prepare)
    monkeypatch.setattr(ocr_main.base_llm_http_handler, "ocr", handler)
    monkeypatch.setattr(rust_bridge, "load_rust_ocr", lookup)
    monkeypatch.setattr(rust_bridge, "load_rust_aocr", lookup)
    arguments = {
        "model": MODEL,
        "document": {"type": "file", "file": b"%PDF-1.4", "mime_type": "application/pdf"},
        "api_key": "sk-test",
        "timeout": 12.5,
        "extra_headers": {"x-trace-id": "python"},
        "pages": [0],
        "include_image_base64": True,
        "num_retries": 0,
    }
    result = await litellm.aocr(**arguments) if asynchronous else litellm.ocr(**arguments)

    assert result is response
    prepare.assert_called_once()
    handler.assert_called_once()
    if asynchronous:
        handler.assert_awaited_once()
    call = handler.call_args.kwargs
    assert call["model"] == "mistral-ocr-latest"
    assert call["custom_llm_provider"] == "mistral"
    assert call["document"] == {
        "type": "document_url",
        "document_url": "data:application/pdf;base64,JVBERi0xLjQ=",
    }
    assert call["optional_params"] == {"pages": [0], "include_image_base64": True}
    assert call["timeout"] == 12.5
    assert call["headers"] == arguments["extra_headers"]
    assert call["aocr"] is asynchronous
    lookup.assert_not_called()
    for native in injected_native:
        native.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_rust_off_preserves_python_exception_mapping(monkeypatch, injected_native, document, asynchronous):
    litellm.rust(False)
    original_error = ValueError("Python transport failed")
    mapped_error = RuntimeError("mapped Python error")
    mapping = Mock(return_value=mapped_error)
    handler = AsyncMock(side_effect=original_error) if asynchronous else Mock(side_effect=original_error)
    monkeypatch.setattr(litellm, "exception_type", mapping)
    monkeypatch.setattr(ocr_main.base_llm_http_handler, "ocr", handler)
    arguments = dict(model=MODEL, document=document, api_key="sk-test", litellm_logging_obj=Mock())
    with pytest.raises(RuntimeError) as caught:
        if asynchronous:
            await inspect.unwrap(ocr_main._legacy_aocr)(**arguments)
        else:
            inspect.unwrap(ocr_main._legacy_ocr)(**arguments)
    assert caught.value is mapped_error
    handler.assert_called_once()
    mapping.assert_called_once()
    assert mapping.call_args.kwargs["original_exception"] is original_error
    assert mapping.call_args.kwargs["model"] == "mistral-ocr-latest"
    assert mapping.call_args.kwargs["custom_llm_provider"] == "mistral"
    for native in injected_native:
        native.assert_not_called()


def test_global_toggle_preserves_injected_bindings(injected_native):
    for enabled in (True, False, True):
        litellm.rust(enabled)
        assert configuration.rust_enabled() is enabled
        assert rust_bridge.load_rust_ocr() is injected_native[0]
        assert rust_bridge.load_rust_aocr() is injected_native[1]


@pytest.mark.parametrize("name", ["ocr", "aocr"])
def test_binding_override_none_and_reset(monkeypatch, name):
    native = ModuleType("litellm.rust_bridge._native")
    implementation = Mock()
    setattr(native, name, implementation)
    monkeypatch.setattr(rust_bridge_bindings, "get_native_bridge", lambda: native)
    binding = rust_bridge._OCR if name == "ocr" else rust_bridge._AOCR
    assert binding.load() is implementation
    override = Mock()
    binding.override(override)
    assert binding.load() is override
    binding.override(None)
    assert binding.load() is None
    binding.reset()
    assert binding.load() is implementation
    setattr(native, name, object())
    assert binding.load() is None


def test_loader_caches_missing_extension_until_reset(monkeypatch):
    original_import = builtins.__import__
    attempts = Mock()

    def import_without_native(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "litellm.rust_bridge" and "_native" in fromlist:
            attempts()
            raise ImportError("extension not installed")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_native)
    assert rust_bridge_loader.get_native_bridge() is None
    assert rust_bridge_loader.native_bridge_available() is False
    attempts.assert_called_once()
    rust_bridge_loader.reset_native_bridge_cache()
    assert rust_bridge_loader.get_native_bridge() is None
    assert attempts.call_count == 2


@pytest.fixture
def native_ocr():
    native = rust_bridge_loader.get_native_bridge()
    try:
        available = native is not None and all(
            tuple(inspect.signature(getattr(native, name)).parameters) == ("arguments",) for name in ("ocr", "aocr")
        )
    except (AttributeError, TypeError, ValueError):
        available = False
    if not available:
        message = "native whole-argument OCR extension unavailable or stale; rebuild the extension"
        if os.environ.get("LITELLM_REQUIRE_NATIVE_OCR") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return native


class WireRecorder:
    def __init__(self):
        self.requests = []
        self.received = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.release.set()
        self.status = 200
        self.response_data = RESPONSE_DATA
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                recorder.requests.append(
                    {
                        "path": self.path,
                        "headers": {name.lower(): value for name, value in self.headers.items()},
                        "body": json.loads(body),
                    }
                )
                recorder.received.set()
                try:
                    if not recorder.release.wait(timeout=10):
                        return
                    payload = json.dumps(recorder.response_data).encode()
                    self.send_response(recorder.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    recorder.finished.set()

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def api_base(self):
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def stop(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def wire_recorder():
    recorder = WireRecorder()
    try:
        yield recorder
    finally:
        recorder.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("callback_raises", [False, True], ids=["ignored-return", "caught-error"])
async def test_native_mistral_wire_response_and_callback_identity(
    native_ocr, wire_recorder, monkeypatch, no_python_ocr, document, asynchronous, callback_raises
):
    litellm.rust(True)
    context = contextvars.ContextVar("ocr-callback-context", default="caller")
    caller = (threading.get_ident(), asyncio.current_task())
    events = []
    retained = []
    observations = []
    pages = [0]

    class Mutate(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            events.append(("mutate", context.get(), threading.get_ident(), asyncio.current_task()))
            context.set("callback")
            view = kwargs["additional_args"]
            body, headers = view["complete_input_dict"], view["headers"]
            retained.append((kwargs, body, headers))
            headers["X-Proof"] = "mutated"
            body["document"]["document_url"] = "https://example.invalid/mutated.pdf"
            document["document_name"] = "closure-mutation"
            pages.append(2)
            view["headers"] = {"X-Replacement": "logging-only"}
            view["complete_input_dict"] = {"model": "logging-only"}
            if callback_raises:
                raise RuntimeError("expected callback failure")
            return {"additional_args": {"headers": {"X-Return": "ignored"}}}

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            events.append(("observe", context.get(), threading.get_ident(), asyncio.current_task()))
            observations.append(
                (
                    kwargs is retained[0][0],
                    dict(kwargs["additional_args"]["headers"]),
                    dict(kwargs["additional_args"]["complete_input_dict"]),
                )
            )

    monkeypatch.setattr(litellm, "input_callback", [Mutate(), Observe()])
    arguments = {
        "model": MODEL,
        "document": document,
        "api_key": "sk-test",
        "api_base": wire_recorder.api_base,
        "timeout": httpx.Timeout(5, read=3),
        "extra_headers": {"X-Trace": "caller"},
        "pages": pages,
        "include_image_base64": True,
        "metadata": {"opaque": object()},
        "arbitrary_option": object(),
        "num_retries": 0,
    }
    response = await litellm.aocr(**arguments) if asynchronous else litellm.ocr(**arguments)

    assert isinstance(response, OCRResponse)
    assert response.object == "ocr"
    assert response.model == RESPONSE_DATA["model"]
    assert response.pages[0].index == 0
    assert response.pages[0].markdown == "proof"
    assert response.pages[0].dimensions.dpi == 200
    assert response.pages[0].images[0].image_base64 == "aW1hZ2U="
    assert response.document_annotation == {"title": "Test document"}
    assert response.usage_info.pages_processed == 1
    assert response.usage_info.doc_size_bytes == 1234
    assert response.get_provider_native_response() is None
    assert events == [("mutate", "caller", *caller), ("observe", "callback", *caller)]
    assert context.get() == "callback"
    assert observations == [(True, {"X-Replacement": "logging-only"}, {"model": "logging-only"})]
    assert len(retained) == 1
    assert retained[0][1]["document"] is document
    assert retained[0][1]["pages"] is pages
    assert len(wire_recorder.requests) == 1
    request = wire_recorder.requests[0]
    assert request["path"] == "/v1/ocr"
    assert request["headers"]["authorization"] == "Bearer sk-test"
    assert request["headers"]["x-trace"] == "caller"
    assert request["headers"]["x-proof"] == "mutated"
    assert "x-replacement" not in request["headers"]
    assert "x-return" not in request["headers"]
    assert request["body"] == {
        "model": "mistral-ocr-latest",
        "document": {
            "type": "document_url",
            "document_url": "https://example.invalid/mutated.pdf",
            "document_name": "closure-mutation",
        },
        "pages": [0, 2],
        "include_image_base64": True,
    }
    document["document_url"] = "https://example.invalid/after-send.pdf"
    assert request["body"]["document"]["document_url"] == "https://example.invalid/mutated.pdf"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("failure", [False, True], ids=["success", "failure"])
@pytest.mark.parametrize("callback_source", ["global", "per-call"])
async def test_public_native_callback_lifecycle(
    native_ocr, wire_recorder, monkeypatch, no_python_ocr, document, asynchronous, failure, callback_source
):
    from litellm import utils
    from litellm.litellm_core_utils import litellm_logging, logging_worker

    litellm.rust(True)
    wire_recorder.status = 429 if failure else 200
    monkeypatch.setattr(utils, "function_setup", Mock(side_effect=AssertionError("native OCR entered function_setup")))
    context = contextvars.ContextVar("ocr-lifecycle-context", default="caller")
    caller_thread, caller_task = threading.get_ident(), asyncio.current_task()
    shared = {"phase": "pre_call"}
    pre_calls = []
    terminal_calls = []
    finished = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    worker = logging_worker.LoggingWorker(timeout=5, concurrency=1)
    monkeypatch.setattr(utils, "executor", executor)
    monkeypatch.setattr(litellm_logging, "executor", executor)
    monkeypatch.setattr(logging_worker, "GLOBAL_LOGGING_WORKER", worker)

    class Lifecycle(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            pre_calls.append((kwargs, context.get(), threading.get_ident(), asyncio.current_task()))
            kwargs["ocr_lifecycle_state"] = shared
            context.set("pre_call")

        def record(self, event, kwargs, response_obj, start_time, end_time):
            terminal_calls.append(
                {
                    "event": event,
                    "kwargs": kwargs,
                    "response": response_obj,
                    "exception": kwargs.get("exception"),
                    "shared": kwargs.get("ocr_lifecycle_state"),
                    "phase": kwargs.get("ocr_lifecycle_state", {}).get("phase"),
                    "context": context.get(),
                    "thread": threading.get_ident(),
                    "task": asyncio.current_task() if threading.get_ident() == caller_thread else None,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )
            kwargs["ocr_lifecycle_state"]["phase"] = "terminal"
            context.set("terminal")
            finished.set()

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.record("sync_success", kwargs, response_obj, start_time, end_time)

        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            self.record("sync_failure", kwargs, response_obj, start_time, end_time)

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            await asyncio.sleep(0)
            self.record("async_success", kwargs, response_obj, start_time, end_time)

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            await asyncio.sleep(0)
            self.record("async_failure", kwargs, response_obj, start_time, end_time)

    callback = Lifecycle()
    monkeypatch.setattr(litellm, "callbacks", [callback] if callback_source == "global" else [])
    arguments = dict(
        model=MODEL,
        document=document,
        api_key="sk-test",
        api_base=wire_recorder.api_base,
        timeout=5,
        num_retries=0,
        **({"callbacks": [callback]} if callback_source == "per-call" else {}),
    )
    try:
        if failure:
            with pytest.raises(litellm.RateLimitError) as caught:
                if asynchronous:
                    await litellm.aocr(**arguments)
                else:
                    litellm.ocr(**arguments)
            assert caught.value.status_code == 429
            assert caught.value.model == "mistral-ocr-latest"
            assert caught.value.llm_provider == "mistral"
        else:
            response = await litellm.aocr(**arguments) if asynchronous else litellm.ocr(**arguments)
            assert isinstance(response, OCRResponse)
            assert response.pages[0].markdown == "proof"
        assert await asyncio.to_thread(finished.wait, 5), "terminal callback was not delivered"
    finally:
        try:
            await asyncio.wait_for(worker.flush(), timeout=5)
            await asyncio.wait_for(asyncio.wrap_future(executor.submit(lambda: None)), timeout=5)
        finally:
            try:
                await asyncio.wait_for(worker.stop(), timeout=5)
            finally:
                atexit.unregister(worker._flush_on_exit)
                executor.shutdown(wait=False, cancel_futures=True)

    assert len(wire_recorder.requests) == 1
    assert len(pre_calls) == 1
    details, pre_context, pre_thread, pre_task = pre_calls[0]
    assert (pre_context, pre_thread, pre_task) == ("caller", caller_thread, caller_task)
    assert details["additional_args"]["complete_input_dict"]["document"] is document
    expected_events = (
        {"sync_failure", "async_failure"}
        if failure and asynchronous
        else {f"{'async' if asynchronous else 'sync'}_{'failure' if failure else 'success'}"}
    )
    assert len(terminal_calls) == len(expected_events)
    assert {terminal["event"] for terminal in terminal_calls} == expected_events
    assert shared == {"phase": "terminal"}
    assert details["litellm_call_id"]
    for terminal in terminal_calls:
        assert terminal["kwargs"] is details
        assert terminal["shared"] is shared
        expected_phase = "terminal" if terminal["event"] == "async_failure" else "pre_call"
        assert terminal["phase"] == expected_phase
        assert terminal["context"] == expected_phase
        assert terminal["start_time"] <= terminal["end_time"]
        if failure:
            assert terminal["response"] is None
            assert terminal["exception"] is caught.value
            assert "RateLimitError" in details["traceback_exception"]
            assert (terminal["thread"], terminal["task"]) == (caller_thread, caller_task)
        else:
            assert terminal["response"] is response
            assert terminal["exception"] is None
            assert terminal["task"] is not caller_task
            assert (terminal["thread"] == caller_thread) is asynchronous
            if asynchronous:
                assert terminal["task"] is not None
    assert context.get() == ("terminal" if failure else "pre_call")


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("public", [False, True], ids=["unwrapped", "public"])
@pytest.mark.parametrize(
    "model,provider,options,message",
    [
        ("azure_ai/doc-intelligence/prebuilt-read", None, {}, "Document Intelligence OCR polling"),
        ("documentintelligence/prebuilt-layout", "azure_ai", {}, "Document Intelligence OCR polling"),
        ("azure_ai/mistral-ocr-latest", None, {}, "HTTP document URL to data URI conversion"),
        ("vertex_ai/mistral-ocr-latest", None, {}, "HTTP document URL to data URI conversion"),
        ("mistral-ocr-latest", "vertex_ai", {}, "HTTP document URL to data URI conversion"),
        ("azure_ai/mistral-ocr-latest", None, {"api_key": None}, "Azure OCR credential acquisition"),
        ("vertex_ai/mistral-ocr-latest", None, {"api_key": None}, "Vertex OCR credential acquisition"),
        ("vertex_ai/deepseek-ocr-maas", None, {"api_key": None}, "Vertex OCR credential acquisition"),
        ("azure_ai/cohere/parse-v5.0", None, {}, "Cohere OCR request transformation"),
        ("cohere/parse-v5.0", None, {}, "OCR provider"),
        ("vertex_ai/deepseek-ocr-maas", None, {"stream": True}, "OCR streaming response handling"),
        (MODEL, None, {"document": {"type": "file", "file": b"%PDF-1.4"}}, "file"),
        (MODEL, None, {"req_format": "native"}, "native"),
    ],
)
async def test_native_unsupported_requests_never_prepare_or_send(
    native_ocr,
    wire_recorder,
    monkeypatch,
    no_python_ocr,
    document,
    asynchronous,
    public,
    model,
    provider,
    options,
    message,
):
    monkeypatch.setenv("LITELLM_RUST", "1")
    for name in ("AZURE_AI_API_KEY", "VERTEX_AI_API_KEY", "VERTEXAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    logger = Mock()
    arguments = {
        "model": model,
        "document": document,
        "custom_llm_provider": provider,
        "api_key": "sk-test",
        "api_base": wire_recorder.api_base,
        "timeout": 3,
        **({} if public else {"litellm_logging_obj": logger}),
        "num_retries": 0,
        **options,
    }
    with pytest.raises(NotImplementedError, match=message):
        function = litellm.aocr if asynchronous else litellm.ocr
        result = (function if public else inspect.unwrap(function))(**arguments)
        if asynchronous:
            await result
    logger.assert_not_called()
    assert logger.mock_calls == []
    assert wire_recorder.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("auth", ["key", "header", "environment"])
@pytest.mark.parametrize("provider", ["azure_ai", "vertex_ai", "deepseek"])
async def test_public_native_cloud_wire_and_shallow_boundaries(
    native_ocr, wire_recorder, monkeypatch, no_python_ocr, asynchronous, auth, provider
):
    monkeypatch.setenv("LITELLM_RUST", "1")
    deepseek = provider == "deepseek"
    model = "vertex_ai/deepseek-ai/deepseek-ocr-maas" if deepseek else f"{provider}/mistral-ocr-latest"
    document = {"type": "image_url", "image_url": "data:image/png;base64,YWJj", "nested": []}
    shared_param = ["stop"] if deepseek else [0]
    param = "stop" if deepseek else "pages"
    captured = []
    opaque = object()

    class Mutate(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            view = kwargs["additional_args"]
            body = view["complete_input_dict"]
            captured.append(body)
            assert body[param] is shared_param
            shared_param.append("changed" if deepseek else 2)
            document["image_url"] = "https://example.invalid/original-only.png"
            if deepseek:
                assert "document" not in body
                body["messages"][0]["content"][0]["image_url"] = "data:image/png;base64,ZGVm"
            else:
                assert body["document"] is not document
                assert body["document"]["nested"] is document["nested"]
                document["nested"].append("shared")
                assert body["document"]["image_url"] == "data:image/png;base64,YWJj"
                body["document"]["image_url"] = "data:image/png;base64,ZGVm"
            view["headers"]["X-Callback"] = "native"
            view["complete_input_dict"] = {"replacement": True}

    monkeypatch.setattr(litellm, "input_callback", [Mutate()])
    if deepseek:
        wire_recorder.response_data = {
            "choices": [{"message": {"content": "proof"}}],
            "usage": {"pages_processed": 1},
        }
    if auth == "environment":
        monkeypatch.setenv("AZURE_AI_API_KEY" if provider == "azure_ai" else "VERTEX_AI_API_KEY", "native-token")
    arguments = dict(
        model=model,
        document=document,
        api_key="native-token" if auth == "key" else None,
        extra_headers={"aUtHoRiZaTiOn": "Bearer native-token"} if auth == "header" else {},
        api_base=wire_recorder.api_base,
        vertex_ai_project="test-project",
        vertex_ai_location="europe-west4",
        timeout=3,
        num_retries=0,
        metadata={"opaque": opaque},
        vertex_credentials=opaque,
        arbitrary_option=opaque,
        **{param: shared_param},
    )
    response = await litellm.aocr(**arguments) if asynchronous else litellm.ocr(**arguments)
    assert isinstance(response, OCRResponse)
    assert response.pages[0].markdown == "proof"
    assert response.usage_info.pages_processed == 1
    assert len(captured) == len(wire_recorder.requests) == 1
    sent = wire_recorder.requests[0]
    auth_header = "api-key" if provider == "azure_ai" and auth != "header" else "authorization"
    assert sent["headers"][auth_header] == ("native-token" if auth_header == "api-key" else "Bearer native-token")
    assert sent["headers"]["x-callback"] == "native"
    assert sent["body"][param] == shared_param
    assert "arbitrary_option" not in sent["body"] and "metadata" not in sent["body"]
    assert "vertex_credentials" not in sent["body"] and "vertex_ai_project" not in sent["body"]
    if deepseek:
        assert sent["path"] == "/v1/projects/test-project/locations/europe-west4/endpoints/openapi/chat/completions"
        assert sent["body"] == {
            "model": "deepseek-ai/deepseek-ocr-maas",
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": "data:image/png;base64,ZGVm"}]}
            ],
            "stop": ["stop", "changed"],
        }
    else:
        expected_path = (
            "/providers/mistral/azure/ocr"
            if provider == "azure_ai"
            else "/v1/projects/test-project/locations/europe-west4/publishers/mistralai/models/mistral-ocr-latest:rawPredict"
        )
        assert sent["path"] == expected_path
        assert sent["body"] == {
            "model": "mistral-ocr-latest",
            "document": {"type": "image_url", "image_url": "data:image/png;base64,ZGVm", "nested": ["shared"]},
            "pages": [0, 2],
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_public_native_azure_supplied_entra_token(
    native_ocr, wire_recorder, monkeypatch, no_python_ocr, asynchronous
):
    monkeypatch.setenv("LITELLM_RUST", "1")
    monkeypatch.delenv("AZURE_AI_API_KEY", raising=False)
    arguments = dict(
        model="azure_ai/mistral-ocr-latest",
        document={"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
        azure_ad_token="entra-token",
        api_base=wire_recorder.api_base,
        timeout=3,
        num_retries=0,
    )
    response = await litellm.aocr(**arguments) if asynchronous else litellm.ocr(**arguments)
    assert response.pages[0].markdown == "proof"
    assert wire_recorder.requests[0]["headers"]["authorization"] == "Bearer entra-token"


class PreCallAbort(BaseException):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_callback_escape_never_sends_or_replays(
    native_ocr, wire_recorder, monkeypatch, no_python_ocr, document, asynchronous
):
    litellm.rust(True)
    error = PreCallAbort("stop before POST")
    calls = []

    class Abort(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            calls.append(kwargs["additional_args"]["complete_input_dict"]["document"])
            raise error

    monkeypatch.setattr(litellm, "input_callback", [Abort()])
    arguments = dict(model=MODEL, document=document, api_key="sk-test", api_base=wire_recorder.api_base, num_retries=0)
    with pytest.raises(PreCallAbort) as caught:
        if asynchronous:
            await litellm.aocr(**arguments)
        else:
            litellm.ocr(**arguments)
    assert caught.value is error
    assert len(calls) == 1 and calls[0] is document
    assert wire_recorder.requests == []


def test_native_sync_callback_reentry_without_event_loop(
    native_ocr, wire_recorder, monkeypatch, no_python_ocr, document
):
    litellm.rust(True)
    context = contextvars.ContextVar("ocr-reentry", default="caller")
    events = []
    arguments = dict(model=MODEL, document=document, api_key="sk-test", api_base=wire_recorder.api_base, num_retries=0)

    class Reenter(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            with pytest.raises(RuntimeError, match="no running event loop"):
                asyncio.get_running_loop()
            events.append((context.get(), threading.get_ident()))
            if len(events) == 1:
                context.set("nested")
                result = litellm.ocr(**arguments)
                events.append((result.pages[0].markdown, threading.get_ident()))

    monkeypatch.setattr(litellm, "input_callback", [Reenter()])
    response = litellm.ocr(**arguments)
    assert response.pages[0].markdown == "proof"
    assert events == [(value, threading.get_ident()) for value in ("caller", "nested", "proof")]
    assert context.get() == "nested"
    assert len(wire_recorder.requests) == 2


class Opaque:
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_native_retains_opaque_arguments_until_terminal_cleanup(native_ocr, wire_recorder, document, outcome):
    wire_recorder.release.clear()
    wire_recorder.status = 429 if outcome == "error" else 200
    opaque = Opaque()
    reference = weakref.ref(opaque)
    logger = Mock()
    logger.async_success_handler = AsyncMock()
    logger.async_failure_handler = AsyncMock()
    logger._should_run_sync_callbacks_for_async_calls.return_value = False
    arguments = dict(
        model=MODEL,
        document=document,
        api_key="sk-test",
        api_base=wire_recorder.api_base,
        timeout=5,
        litellm_logging_obj=logger,
        opaque=opaque,
    )
    task = asyncio.create_task(native_ocr.aocr(arguments))
    del arguments, opaque
    try:
        assert await asyncio.to_thread(wire_recorder.received.wait, 5), "POST never reached server"
        assert not task.done()
        logger.update_from_kwargs.assert_called_once()
        logger.pre_call.assert_called_once()
        logger.reset_mock()
        gc.collect()
        assert reference() is not None
        if outcome == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif outcome == "error":
            wire_recorder.release.set()
            with pytest.raises(litellm.RateLimitError) as caught:
                await task
            assert caught.value.status_code == 429
            del caught
        else:
            wire_recorder.release.set()
            result = await task
            assert result.pages[0].markdown == "proof"
    finally:
        wire_recorder.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert await asyncio.to_thread(wire_recorder.finished.wait, 5)
    del task
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    if outcome == "success":
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), 5)
    logger.reset_mock()
    await asyncio.sleep(0)
    gc.collect()
    assert reference() is None, "native execution retained opaque kwargs after cleanup"
    assert len(wire_recorder.requests) == 1

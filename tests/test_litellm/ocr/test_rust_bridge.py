"""Public Python OCR routing and private native OCR proof contracts."""

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
async def test_bridge_preserves_every_argument(no_python_ocr, injected_native, response, asynchronous):
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
    result = await rust_bridge.aocr(arguments) if asynchronous else rust_bridge.ocr(arguments)

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
async def test_public_wrapper_passes_defaults_to_legacy(monkeypatch, injected_native, document, response, asynchronous):
    litellm.rust(True)
    legacy = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    monkeypatch.setattr(ocr_main, "_legacy_aocr" if asynchronous else "_legacy_ocr", legacy)
    result = await litellm.aocr(MODEL, document) if asynchronous else litellm.ocr(MODEL, document)
    legacy.assert_called_once_with(MODEL, document, None, None, None, None, None)
    if asynchronous:
        legacy.assert_awaited_once()
    assert result is response
    for native in injected_native:
        native.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_public_wrapper_passes_full_kwargs_to_legacy(
    monkeypatch, injected_native, document, response, asynchronous
):
    litellm.rust(True)
    legacy = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    monkeypatch.setattr(ocr_main, "_legacy_aocr" if asynchronous else "_legacy_ocr", legacy)
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

    legacy.assert_called_once_with(
        MODEL,
        document,
        "sk-test",
        "https://example.invalid",
        12.5,
        None,
        arguments["extra_headers"],
        pages=pages,
        include_image_base64=True,
        metadata=metadata,
        arbitrary_option=arguments["arbitrary_option"],
        num_retries=0,
    )
    if asynchronous:
        legacy.assert_awaited_once()
    assert result is response
    assert legacy.call_args.args[1] is document
    assert legacy.call_args.kwargs["pages"] is pages
    assert legacy.call_args.kwargs["metadata"] is metadata
    for native in injected_native:
        native.assert_not_called()


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
@pytest.mark.parametrize("failure", ["missing", "unsupported", "runtime"])
async def test_bridge_failures_propagate_without_fallback(
    monkeypatch, no_python_ocr, injected_native, document, asynchronous, failure
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
    arguments = dict(model=MODEL, document=document, api_key="sk-test", num_retries=0)
    with pytest.raises(RuntimeError if failure == "missing" else type(error)) as caught:
        await rust_bridge.aocr(arguments) if asynchronous else rust_bridge.ocr(arguments)
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
    arguments = {"model": MODEL, "document": document}
    with pytest.raises(RuntimeError, match=r"(?i)ocr"):
        await rust_bridge.aocr(arguments) if asynchronous else rust_bridge.ocr(arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["ocr", "aocr", "ocr-async"])
@pytest.mark.parametrize("setting", ["default", "disabled", "overrides-environment", "global", "environment"])
async def test_public_ocr_always_keeps_python_preparation_and_transport(
    monkeypatch, injected_native, response, route, setting
):
    asynchronous = route != "ocr"
    if setting in ("overrides-environment", "environment"):
        monkeypatch.setenv("LITELLM_RUST", "1")
    if setting in ("disabled", "overrides-environment"):
        litellm.rust(False)
    if setting == "global":
        litellm.rust(True)
    prepare = Mock(wraps=ocr_main._prepare_ocr_request)
    handler = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    lookup = Mock(side_effect=lambda: pytest.fail("public OCR consulted a native binding"))
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
        **({"aocr": True} if route == "ocr-async" else {}),
    }
    function = litellm.aocr if route == "aocr" else litellm.ocr
    result = await function(**arguments) if asynchronous else function(**arguments)

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
@pytest.mark.parametrize("enabled", [False, True])
async def test_legacy_preserves_python_exception_mapping(monkeypatch, injected_native, document, asynchronous, enabled):
    litellm.rust(enabled)
    original_error = ValueError("Python transport failed")
    mapped_error = RuntimeError("mapped Python error")
    mapping = Mock(return_value=mapped_error)
    handler = AsyncMock(side_effect=original_error) if asynchronous else Mock(side_effect=original_error)
    monkeypatch.setattr(litellm, "exception_type", mapping)
    monkeypatch.setattr(ocr_main.base_llm_http_handler, "ocr", handler)
    arguments = dict(model=MODEL, document=document, api_key="sk-test", litellm_logging_obj=Mock())
    with pytest.raises(RuntimeError) as caught:
        await inspect.unwrap(ocr_main._legacy_aocr)(**arguments) if asynchronous else inspect.unwrap(
            ocr_main._legacy_ocr
        )(**arguments)
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
def native_ocr(monkeypatch, reset_rust_state):
    """PRIVATE, test-only route selection; public OCR stays Python until full lifecycle parity."""
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
    python_ocr, python_aocr = litellm.ocr, litellm.aocr
    signature = inspect.signature(python_ocr)

    def native_arguments(args, kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return {**bound.arguments.pop("kwargs"), **bound.arguments}

    def private_ocr(*args, **kwargs):
        if not configuration.rust_enabled():
            return python_ocr(*args, **kwargs)
        arguments = native_arguments(args, kwargs)
        return rust_bridge.aocr(arguments) if arguments.get("aocr") is True else rust_bridge.ocr(arguments)

    async def private_aocr(*args, **kwargs):
        if not configuration.rust_enabled():
            return await python_aocr(*args, **kwargs)
        return await rust_bridge.aocr(native_arguments(args, kwargs))

    monkeypatch.setattr(litellm, "ocr", private_ocr)
    monkeypatch.setattr(litellm, "aocr", private_aocr)
    return native


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["ocr", "aocr", "ocr-async"])
async def test_private_fixture_selects_native_only_when_enabled(
    request, monkeypatch, injected_native, document, response, route
):
    native = ModuleType("litellm.rust_bridge._native")
    native.ocr = rust_bridge.ocr
    native.aocr = rust_bridge.aocr
    monkeypatch.setattr(rust_bridge_loader, "get_native_bridge", lambda: native)
    python_ocr, python_aocr = litellm.ocr, litellm.aocr
    asynchronous = route != "ocr"
    legacy = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    monkeypatch.setattr(ocr_main, "_legacy_aocr" if route == "aocr" else "_legacy_ocr", legacy)
    assert request.getfixturevalue("native_ocr") is native
    assert ocr_main.ocr is python_ocr and ocr_main.aocr is python_aocr
    arguments = {"metadata": {"opaque": object()}, **({"aocr": True} if route == "ocr-async" else {})}
    for enabled in (True, False, True):
        litellm.rust(enabled)
        function = litellm.aocr if route == "aocr" else litellm.ocr
        result = function(MODEL, document, **arguments)
        assert (await result if asynchronous else result) is response
    legacy.assert_called_once_with(MODEL, document, None, None, None, None, None, **arguments)
    selected, unused = injected_native[int(asynchronous)], injected_native[not asynchronous]
    assert selected.call_count == 2
    unused.assert_not_called()
    assert selected.call_args.args[0] == {
        "model": MODEL,
        "document": document,
        "api_key": None,
        "api_base": None,
        "timeout": None,
        "custom_llm_provider": None,
        "extra_headers": None,
        **arguments,
    }
    assert selected.call_args.args[0]["document"] is document
    assert selected.call_args.args[0]["metadata"] is arguments["metadata"]


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
@pytest.mark.parametrize("callback_source", ["global", "per-call", "terminal-list"])
async def test_private_native_callback_lifecycle(
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
            kwargs.setdefault("ocr_lifecycle_state", shared)
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
        **(
            {"failure_callback" if failure else "success_callback": [callback]}
            if callback_source == "terminal-list"
            else {}
        ),
    )
    try:
        if failure:
            with pytest.raises(litellm.RateLimitError) as caught:  # noqa: PT012  # parametrized sync and async calls require distinct statements
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
    assert len(pre_calls) == (0 if callback_source == "terminal-list" else 1)
    details = terminal_calls[0]["kwargs"] if callback_source == "terminal-list" else pre_calls[0][0]
    if callback_source != "terminal-list":
        _, pre_context, pre_thread, pre_task = pre_calls[0]
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
        expected_context = (
            "caller" if callback_source == "terminal-list" and expected_phase == "pre_call" else expected_phase
        )
        assert terminal["context"] == expected_context
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
    expected_context = "terminal" if failure else "caller" if callback_source == "terminal-list" else "pre_call"
    assert context.get() == expected_context


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True], ids=["success", "failure"])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync-request", "async-request"])
@pytest.mark.parametrize("callback_kind", ["sync", "async", "async-object"])
@pytest.mark.parametrize("callback_source", ["terminal-list", "global-list", "callbacks", "per-call", "overlap"])
async def test_private_native_callable_terminal_callback(
    native_ocr,
    wire_recorder,
    monkeypatch,
    no_python_ocr,
    document,
    failure,
    asynchronous,
    callback_kind,
    callback_source,
):
    from litellm import utils
    from litellm.litellm_core_utils import litellm_logging, logging_worker

    litellm.rust(True)
    wire_recorder.status = 429 if failure else 200
    calls = []
    finished = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    worker = logging_worker.LoggingWorker(timeout=5, concurrency=1)
    monkeypatch.setattr(utils, "executor", executor)
    monkeypatch.setattr(litellm_logging, "executor", executor)
    monkeypatch.setattr(logging_worker, "GLOBAL_LOGGING_WORKER", worker)
    monkeypatch.setattr(litellm_logging, "customLogger", None)
    monkeypatch.setattr(utils, "callback_list", [])
    monkeypatch.setattr(utils, "function_setup", Mock(side_effect=AssertionError("native OCR entered function_setup")))

    def record(kwargs, *terminal):
        if terminal:
            assert kwargs["log_event_type"] == "post_api_call"
            calls.append((kwargs, *terminal))
            finished.set()

    async def async_callback(kwargs, *terminal):
        await asyncio.sleep(0)
        record(kwargs, *terminal)

    class AsyncCallable:
        async def __call__(self, kwargs, *terminal):
            await async_callback(kwargs, *terminal)

    callback = record if callback_kind == "sync" else async_callback if callback_kind == "async" else AsyncCallable()
    callback_list = [callback, callback]
    terminal_name = "failure_callback" if failure else "success_callback"
    if callback_source in ("global-list", "overlap"):
        monkeypatch.setattr(litellm, terminal_name, [callback])
    if callback_source == "overlap":
        monkeypatch.setattr(litellm, f"_async_{terminal_name}", [callback])
    if callback_source == "callbacks":
        monkeypatch.setattr(litellm, "callbacks", [callback])

    arguments = {
        "model": MODEL,
        "document": document,
        "api_key": "sk-test",
        "api_base": wire_recorder.api_base,
        "timeout": 5,
        "num_retries": 0,
        **({terminal_name: callback_list} if callback_source in ("terminal-list", "overlap") else {}),
        **({"callbacks": callback_list} if callback_source in ("per-call", "overlap") else {}),
    }
    try:
        if failure:
            with pytest.raises(litellm.RateLimitError) as caught:  # noqa: PT012  # parametrized sync and async calls require distinct statements
                if asynchronous:
                    await litellm.aocr(**arguments)
                else:
                    litellm.ocr(**arguments)
            expected_response = None
        else:
            expected_response = await litellm.aocr(**arguments) if asynchronous else litellm.ocr(**arguments)

        if asynchronous or callback_kind == "sync":
            assert await asyncio.to_thread(finished.wait, 5), "callable callback was not delivered"
    finally:
        try:
            await asyncio.wait_for(worker.flush(), 5)
            await asyncio.wait_for(asyncio.wrap_future(executor.submit(lambda: None)), 5)
        finally:
            await asyncio.wait_for(worker.stop(), 5)
            atexit.unregister(worker._flush_on_exit)
            executor.shutdown(wait=True, cancel_futures=True)

    assert callback_list == [callback, callback]
    assert callback not in utils.callback_list
    if callback_source in ("terminal-list", "overlap"):
        assert arguments[terminal_name] is callback_list
    if callback_source in ("per-call", "overlap"):
        assert arguments["callbacks"] is callback_list
        assert callback not in litellm.callbacks
        assert callback not in litellm.input_callback
    if not asynchronous and callback_kind != "sync":
        assert calls == []
        return
    assert len(calls) == 1
    details, callback_response, start_time, end_time = calls[0]
    assert details["model"] == "mistral-ocr-latest"
    assert details["litellm_call_id"]
    assert callback_response is expected_response
    assert start_time <= end_time
    if failure:
        assert details["exception"] is caught.value


@pytest.mark.asyncio
async def test_native_named_callback_initialization_preserves_aliases(monkeypatch, document):
    from datetime import datetime

    from litellm.litellm_core_utils import litellm_logging

    events = []

    class NamedLogger(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            events.append("input")

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            events.append("success")

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            events.append("failure")

    callback = NamedLogger()
    monkeypatch.setattr(litellm_logging, "_init_custom_logger_compatible_class", Mock(return_value=callback))
    monkeypatch.setattr(litellm, "input_callback", [callback])
    monkeypatch.setattr(litellm, "_async_success_callback", [callback])
    monkeypatch.setattr(litellm, "_async_failure_callback", [callback])
    callbacks = ["lago", callback, "lago"]
    metadata = {"opaque": object()}
    arguments = {"model": MODEL, "document": document, "callbacks": callbacks, "metadata": metadata}
    logger = rust_bridge.initialize_logging(arguments, True)
    assert arguments["document"] is document
    assert arguments["metadata"] is metadata
    assert arguments["callbacks"] is callbacks
    assert callbacks == ["lago", callback, "lago"]
    assert rust_bridge.initialize_logging(arguments, True) is logger
    logger.pre_call(input="document", api_key="sk-test")
    result = OCRResponse.model_validate(RESPONSE_DATA)
    await logger.async_success_handler(result, datetime.now(), datetime.now())
    await logger.async_failure_handler(ValueError("test"), "test", datetime.now(), datetime.now())
    assert events == ["input", "success", "failure"]


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
        (
            "azure_ai/mistral-ocr-latest",
            None,
            {"api_key": None, "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"}},
            "OCR credential acquisition",
        ),
        (
            "vertex_ai/mistral-ocr-latest",
            None,
            {"api_key": None, "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"}},
            "OCR credential acquisition",
        ),
        (
            "vertex_ai/deepseek-ocr-maas",
            None,
            {"api_key": None, "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"}},
            "OCR credential acquisition",
        ),
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
    with pytest.raises(NotImplementedError, match=message):  # noqa: PT012  # parametrized native and public entry points differ
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
async def test_private_native_cloud_wire_and_shallow_boundaries(
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
async def test_private_native_azure_supplied_entra_token(
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
    with pytest.raises(PreCallAbort) as caught:  # noqa: PT012  # parametrized sync and async calls require distinct statements
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


@pytest.fixture
async def isolated_ocr_logging_worker(monkeypatch):
    from litellm.litellm_core_utils import logging_worker

    worker = logging_worker.LoggingWorker(timeout=5, concurrency=1)
    monkeypatch.setattr(logging_worker, "GLOBAL_LOGGING_WORKER", worker)
    try:
        yield worker
    finally:
        try:
            await asyncio.wait_for(worker.flush(), 5)
        finally:
            try:
                await asyncio.wait_for(worker.stop(), 5)
            finally:
                atexit.unregister(worker._flush_on_exit)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("outcome", ["mutation", "replacement", "failure"])
async def test_public_deployment_callback_parity(
    native_ocr, wire_recorder, monkeypatch, document, enabled, outcome, isolated_ocr_logging_worker
):
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging

    events, captured, responses, snapshots, terminals = [], [], [], [], []
    metadata = {"deployment_state": {"phase": "caller"}}
    state = metadata["deployment_state"]
    replacement = OCRResponse.model_validate({**RESPONSE_DATA, "document_annotation": {"reviewed": True}})
    decoy_logger = object()

    class DeploymentLogger(CustomLogger):
        def __init__(self, index):
            super().__init__()
            self.index = index

        async def async_pre_call_deployment_hook(self, kwargs, call_type):
            assert kwargs["metadata"] is metadata
            if self.index == 0:
                assert kwargs["litellm_logging_obj"] is logger
                captured.append(kwargs["litellm_logging_obj"])
                state["phase"] = "first"
                return {**kwargs, "pages": [2], "litellm_logging_obj": decoy_logger}
            assert kwargs["litellm_logging_obj"] is decoy_logger
            assert state["phase"] == "first" and kwargs["pages"] == [2]
            kwargs["pages"].append(3)
            state["phase"] = "second"
            events.append(("deployment_pre", call_type.value))

        def log_pre_api_call(self, model, messages, kwargs):
            assert kwargs is captured[0].model_call_details
            assert kwargs["litellm_params"]["metadata"]["deployment_state"] is state
            assert state["phase"] == "second"
            events.append(("pre_api", kwargs["additional_args"]["complete_input_dict"].get("pages")))

        async def async_post_call_success_deployment_hook(self, request_data, response, call_type):
            assert request_data["metadata"] is metadata
            assert request_data["litellm_logging_obj"] is captured[0]
            responses.append(response)
            if self.index == 0:
                response.document_annotation = {"mutated": True}
                state["phase"] = "success"
                return replacement if outcome == "replacement" else None
            assert state["phase"] == "success"
            assert response is (replacement if outcome == "replacement" else responses[0])
            response.document_annotation["second"] = True
            events.append(("deployment_success", call_type.value))

        async def async_post_call_failure_deployment_hook(self, request_data, exception, call_type, **kwargs):
            assert request_data["metadata"] is metadata
            assert request_data["litellm_logging_obj"] is captured[0]
            snapshots.append(exception)
            if self.index == 0:
                assert exception.status_code == 429
                state["phase"] = "error"
                captured[0].model_call_details["deployment_error_state"] = state
                exception.status_code = 418
                raise RuntimeError("observer failure must not replace the provider error")
            assert exception is snapshots[0] and exception.status_code == 418
            assert captured[0].model_call_details["deployment_error_state"] is state
            assert state["phase"] == "error"
            events.append(("deployment_failure", call_type.value))

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            terminals.append((kwargs, response_obj))

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            terminals.append((kwargs, kwargs["exception"]))

    callbacks = [DeploymentLogger(0), DeploymentLogger(1)]
    monkeypatch.setattr(litellm, "callbacks", callbacks)
    logger = Logging(
        model=MODEL,
        messages=[],
        stream=False,
        call_type="aocr",
        start_time=datetime.now(),
        litellm_call_id=f"deployment-{enabled}-{outcome}",
        function_id="",
        dynamic_input_callbacks=callbacks,
        dynamic_async_success_callbacks=callbacks,
        dynamic_async_failure_callbacks=callbacks,
    )
    wire_recorder.status = 429 if outcome == "failure" else 200
    litellm.rust(enabled)
    arguments = dict(
        model=MODEL,
        document=document,
        api_key="sk-test",
        api_base=wire_recorder.api_base,
        timeout=5,
        num_retries=0,
        metadata=metadata,
        litellm_logging_obj=logger,
    )
    if outcome == "failure":
        with pytest.raises(litellm.RateLimitError) as caught:
            await litellm.aocr(**arguments)
        result = caught.value
        assert result.status_code == 429
        assert len(snapshots) == 2 and all(snapshot is not result for snapshot in snapshots)
    else:
        result = await litellm.aocr(**arguments)
        assert len(responses) == 2 and result is responses[1]
        assert (result is responses[0]) is (outcome == "mutation")
        assert result.document_annotation == {
            "reviewed" if outcome == "replacement" else "mutated": True,
            "second": True,
        }
    await asyncio.sleep(0)
    await asyncio.wait_for(isolated_ocr_logging_worker.flush(), 5)
    assert len(captured) == 1 and captured[0] is logger
    assert len(terminals) == 2
    for details, terminal in terminals:
        assert details is captured[0].model_call_details and terminal is result
        assert details["litellm_params"]["metadata"]["deployment_state"] is state
        if outcome == "failure":
            assert details["deployment_error_state"] is state
    terminal_event = "deployment_failure" if outcome == "failure" else "deployment_success"
    assert events == [("deployment_pre", "aocr"), ("pre_api", [2, 3]), ("pre_api", [2, 3]), (terminal_event, "aocr")]
    assert len(wire_recorder.requests) == 1 and wire_recorder.requests[0]["body"]["pages"] == [2, 3]


@pytest.mark.asyncio
async def test_public_deferred_success_callback_parity(
    native_ocr, wire_recorder, monkeypatch, document, isolated_ocr_logging_worker
):
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging

    worker = isolated_ocr_logging_worker
    observations = []

    class DeferredLogger(CustomLogger):
        def __init__(self):
            self.calls = []
            self.delivered = asyncio.Event()

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.calls.append(response_obj.document_annotation)
            self.delivered.set()

    try:
        for enabled in (False, True):
            callback = DeferredLogger()
            logger = Logging(
                model=MODEL,
                messages=[],
                stream=False,
                call_type="aocr",
                start_time=datetime.now(),
                litellm_call_id=f"deferred-{enabled}",
                function_id="",
                dynamic_async_success_callbacks=[callback],
            )
            logger._defer_async_logging = True
            litellm.rust(enabled)
            result = await litellm.aocr(
                model=MODEL,
                document=document,
                api_key="sk-test",
                api_base=wire_recorder.api_base,
                timeout=5,
                num_retries=0,
                litellm_logging_obj=logger,
            )
            enqueue = getattr(logger, "_enqueue_deferred_logging", None)
            assert callable(enqueue)
            assert not callback.calls
            result.document_annotation = {"reviewed": True}
            enqueue()
            logger._enqueue_deferred_logging = None
            await asyncio.wait_for(callback.delivered.wait(), 5)
            await asyncio.wait_for(worker.flush(), 5)
            observations.append((callable(enqueue), tuple(callback.calls)))
    finally:
        await asyncio.wait_for(worker.flush(), 5)
    assert len(wire_recorder.requests) == 2
    assert observations[0] == (True, ({"reviewed": True},))
    assert len(observations[1][1]) == 1
    assert observations[1] == observations[0]


def test_public_cold_callable_input_callback_parity(native_ocr, wire_recorder, monkeypatch, document):
    from litellm import utils
    from litellm.litellm_core_utils import litellm_logging

    observations = []
    for enabled in (False, True):
        calls = []

        def callback(kwargs, calls=calls):
            calls.append(kwargs["log_event_type"])

        monkeypatch.setattr(litellm, "input_callback", [callback])
        monkeypatch.setattr(litellm_logging, "customLogger", None)
        monkeypatch.setattr(utils, "callback_list", [])
        litellm.rust(enabled)
        result = litellm.ocr(
            model=MODEL,
            document=document,
            api_key="sk-test",
            api_base=wire_recorder.api_base,
            timeout=5,
            num_retries=0,
        )
        assert result.pages[0].markdown == "proof"
        observations.append(tuple(calls))
    assert len(wire_recorder.requests) == 2
    assert observations[0] == ("pre_api_call",)
    assert observations[1] == observations[0]


class Opaque:
    pass


@pytest.mark.asyncio
async def test_native_deferred_callback_retains_shared_arguments(
    native_ocr, wire_recorder, document, isolated_ocr_logging_worker
):
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging

    class Arguments(dict):
        pass

    entered, release = asyncio.Event(), asyncio.Event()
    observed = []

    class DeferredLogger(CustomLogger):
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            entered.set()
            await release.wait()
            shared = arguments_ref()
            observed.append(
                (
                    shared is not None and shared["opaque"] is opaque_ref(),
                    shared is not None and shared["unknown_option"]["nested"] is opaque_ref(),
                    shared is not None and shared["document"] is document,
                    response_obj is result,
                )
            )

    logger = Logging(
        model=MODEL,
        messages=[],
        stream=False,
        call_type="aocr",
        start_time=datetime.now(),
        litellm_call_id="retained-deferred",
        function_id="",
        dynamic_async_success_callbacks=[DeferredLogger()],
    )
    logger._defer_async_logging = True
    opaque = Opaque()
    arguments = Arguments(
        model=MODEL,
        document=document,
        api_key="sk-test",
        api_base=wire_recorder.api_base,
        timeout=5,
        num_retries=0,
        litellm_logging_obj=logger,
        opaque=opaque,
        unknown_option={"nested": opaque},
    )
    arguments_ref, opaque_ref = weakref.ref(arguments), weakref.ref(opaque)
    try:
        result = await native_ocr.aocr(arguments)
        assert result.pages[0].markdown == "proof" and not entered.is_set()
        logger._enqueue_deferred_logging()
        logger._enqueue_deferred_logging = None
        del arguments, opaque
        await asyncio.wait_for(entered.wait(), 5)
        gc.collect()
        assert arguments_ref() is not None and opaque_ref() is not None
        assert arguments_ref()["unknown_option"]["nested"] is opaque_ref()
        assert not observed
        shared = arguments_ref()
    finally:
        release.set()
        await asyncio.wait_for(isolated_ocr_logging_worker.flush(), 5)
    assert observed == [(True, True, True, True)]
    assert shared["opaque"] is shared["unknown_option"]["nested"] is opaque_ref()
    assert shared["document"] is document and shared["litellm_logging_obj"] is logger
    del shared
    await asyncio.sleep(0)
    gc.collect()
    assert arguments_ref() is None and opaque_ref() is None
    assert len(wire_recorder.requests) == 1
    assert "opaque" not in wire_recorder.requests[0]["body"]
    assert "unknown_option" not in wire_recorder.requests[0]["body"]


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_correlation_context_restored_in_calling_task(
    native_ocr, wire_recorder, monkeypatch, document, enabled, outcome, isolated_ocr_logging_worker
):
    from litellm._logging import session_id_var, trace_id_var

    during, restored = [], []

    class CorrelationLogger(CustomLogger):
        async def async_pre_call_deployment_hook(self, kwargs, call_type):
            during.append((trace_id_var.get(), session_id_var.get(), asyncio.current_task()))

    monkeypatch.setattr(litellm, "request_correlation_in_logs", True)
    monkeypatch.setattr(litellm, "callbacks", [CorrelationLogger()])
    litellm.rust(enabled)
    wire_recorder.status = 429 if outcome == "failure" else 200
    wire_recorder.release.clear()

    async def call():
        trace_token = trace_id_var.set("outer-trace")
        session_token = session_id_var.set("outer-session")
        try:
            return await litellm.aocr(
                model=MODEL,
                document=document,
                api_key="sk-test",
                api_base=wire_recorder.api_base,
                timeout=5,
                num_retries=0,
                litellm_trace_id="request-trace",
                litellm_session_id="request-session",
            )
        finally:
            restored.append((trace_id_var.get(), session_id_var.get(), asyncio.current_task()))
            trace_id_var.reset(trace_token)
            session_id_var.reset(session_token)

    task = asyncio.create_task(call())
    try:
        assert await asyncio.to_thread(wire_recorder.received.wait, 5), "POST never reached server"
        assert during == [("request-trace", "request-session", task)]
        if outcome == "cancel":
            task.cancel()
        else:
            wire_recorder.release.set()
        if outcome == "success":
            assert (await asyncio.wait_for(task, 5)).pages[0].markdown == "proof"
        else:
            with pytest.raises(asyncio.CancelledError if outcome == "cancel" else litellm.RateLimitError):
                await asyncio.wait_for(task, 5)
        assert restored == [("outer-trace", "outer-session", task)]
    finally:
        wire_recorder.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert await asyncio.to_thread(wire_recorder.finished.wait, 5)
        await asyncio.wait_for(isolated_ocr_logging_worker.flush(), 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_native_retains_opaque_arguments_until_terminal_cleanup(
    native_ocr, wire_recorder, document, outcome, isolated_ocr_logging_worker
):
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
    if outcome == "success":
        await asyncio.wait_for(isolated_ocr_logging_worker.flush(), 5)
    logger.reset_mock()
    await asyncio.sleep(0)
    gc.collect()
    assert reference() is None, "native execution retained opaque kwargs after cleanup"
    assert len(wire_recorder.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True], ids=["python", "native"])
async def test_public_cancellation_does_not_emit_failure_callbacks(
    native_ocr, wire_recorder, monkeypatch, document, enabled
):
    calls = []

    class CancellationLogger(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            calls.append("pre_call")

        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            calls.append("sync_failure")

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            calls.append("async_failure")

    monkeypatch.setattr(litellm, "callbacks", [CancellationLogger()])
    litellm.rust(enabled)
    wire_recorder.release.clear()
    task = asyncio.create_task(
        litellm.aocr(
            model=MODEL,
            document=document,
            api_key="sk-test",
            api_base=wire_recorder.api_base,
            timeout=5,
            num_retries=0,
        )
    )
    try:
        assert await asyncio.to_thread(wire_recorder.received.wait, 5), "POST never reached server"
        assert calls == ["pre_call"]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        wire_recorder.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert await asyncio.to_thread(wire_recorder.finished.wait, 5)
    await asyncio.sleep(0)
    assert calls == ["pre_call"]
    assert len(wire_recorder.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True], ids=["python", "native"])
async def test_public_concurrent_callback_isolation_and_cleanup(request, monkeypatch, enabled):
    from contextlib import ExitStack

    from litellm import utils
    from litellm.litellm_core_utils import litellm_logging, logging_worker

    if enabled:
        request.getfixturevalue("native_ocr")
        request.getfixturevalue("no_python_ocr")
    litellm.rust(enabled)
    context = contextvars.ContextVar("ocr-stress-context", default="parent")
    pre_calls, terminals = [], []
    entered, release = {}, {}
    worker = logging_worker.LoggingWorker(timeout=5, concurrency=2)
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(logging_worker, "GLOBAL_LOGGING_WORKER", worker)
    monkeypatch.setattr(utils, "executor", executor)
    monkeypatch.setattr(litellm_logging, "executor", executor)

    class ConcurrentLogger(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            tag = kwargs["litellm_call_id"]
            state = {"owner": tag, "phase": "pre"}
            pre_calls.append((tag, kwargs, state, context.get()))
            kwargs["ocr_stress_state"] = state
            kwargs["additional_args"]["headers"]["X-Callback"] = tag
            context.set(f"{tag}:pre")

        def record(self, event, kwargs, response_obj):
            tag = kwargs["litellm_call_id"]
            state = kwargs.get("ocr_stress_state", {})
            terminals.append(
                (tag, event, kwargs, state, dict(state), context.get(), response_obj, kwargs.get("exception"))
            )
            state["phase"] = event
            context.set(f"{tag}:{event}")

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.record("sync_success", kwargs, response_obj)

        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            self.record("sync_failure", kwargs, response_obj)

        async def terminal(self, event, kwargs, response_obj):
            tag = kwargs["litellm_call_id"]
            entered[tag].set()
            await asyncio.wait_for(release[tag].wait(), 5)
            self.record(event, kwargs, response_obj)

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            await self.terminal("async_success", kwargs, response_obj)

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            await self.terminal("async_failure", kwargs, response_obj)

    monkeypatch.setattr(litellm, "callbacks", [ConcurrentLogger()])

    async def call(tag, recorder):
        context.set(tag)
        return await litellm.aocr(
            model=MODEL,
            document={"type": "document_url", "document_url": f"https://example.invalid/{tag}.pdf"},
            api_key="sk-test",
            api_base=recorder.api_base,
            litellm_call_id=tag,
            timeout=5,
            num_retries=0,
        )

    async def batch(prefix, outcomes):
        with ExitStack() as stack:
            recorders = {f"{prefix}-{outcome}": WireRecorder() for outcome in outcomes}
            for tag, recorder in recorders.items():
                stack.callback(recorder.stop)
                recorder.release.clear()
                recorder.status = 429 if tag.endswith("error") else 200
                entered[tag], release[tag] = asyncio.Event(), asyncio.Event()
            tasks = {tag: asyncio.create_task(call(tag, recorder)) for tag, recorder in recorders.items()}
            try:
                assert all(await asyncio.gather(*(asyncio.to_thread(r.received.wait, 5) for r in recorders.values())))
                assert all(not task.done() for task in tasks.values())
                for tag, task in tasks.items():
                    if tag.endswith("cancel"):
                        task.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await asyncio.wait_for(task, 5)
                    else:
                        recorders[tag].release.set()
                active = tuple(tag for tag in tasks if not tag.endswith("cancel"))
                await asyncio.wait_for(asyncio.gather(*(entered[tag].wait() for tag in active)), 5)
                for tag in reversed(active):
                    release[tag].set()
                results = await asyncio.wait_for(asyncio.gather(*tasks.values(), return_exceptions=True), 5)
                await asyncio.wait_for(worker.flush(), 5)
                await asyncio.wait_for(asyncio.wrap_future(executor.submit(lambda: None)), 5)
                for (tag, recorder), result in zip(recorders.items(), results):
                    matching_pre = [entry for entry in pre_calls if entry[0] == tag]
                    assert len(matching_pre) == len(recorder.requests) == 1
                    _, details, state, pre_context = matching_pre[0]
                    assert pre_context == tag
                    sent = recorder.requests[0]
                    assert sent["path"] == "/v1/ocr"
                    assert sent["headers"]["x-callback"] == tag
                    assert sent["body"]["document"]["document_url"] == f"https://example.invalid/{tag}.pdf"
                    expected = (
                        []
                        if tag.endswith("cancel")
                        else ["sync_failure", "async_failure"]
                        if tag.endswith("error")
                        else ["async_success"]
                    )
                    matching_terminal = [entry for entry in terminals if entry[0] == tag]
                    assert [entry[1] for entry in matching_terminal] == expected
                    if tag.endswith("cancel"):
                        assert isinstance(result, asyncio.CancelledError)
                    elif tag.endswith("error"):
                        assert isinstance(result, litellm.RateLimitError) and result.status_code == 429
                    else:
                        assert isinstance(result, OCRResponse) and result.pages[0].markdown == "proof"
                    for _, event, kwargs, shared, snapshot, terminal_context, response, error in matching_terminal:
                        phase = "sync_failure" if event == "async_failure" else "pre"
                        assert kwargs is details and shared is state
                        assert snapshot == {"owner": tag, "phase": phase}
                        assert terminal_context == f"{tag}:{phase}"
                        assert response is (None if tag.endswith("error") else result)
                        assert error is (result if tag.endswith("error") else None)
                assert context.get() == "parent"
            finally:
                for tag, recorder in recorders.items():
                    recorder.release.set()
                    release[tag].set()
                    if not tasks[tag].done():
                        tasks[tag].cancel()
                await asyncio.wait_for(asyncio.gather(*tasks.values(), return_exceptions=True), 5)
                assert all(await asyncio.gather(*(asyncio.to_thread(r.finished.wait, 5) for r in recorders.values())))

    try:
        for index in range(2):
            await batch(str(index), ("success", "error", "cancel"))
        await batch("recovery", ("success",))
        assert len(pre_calls) == 7 and len(terminals) == 7
        assert len({id(entry[1]) for entry in pre_calls}) == len({id(entry[2]) for entry in pre_calls}) == 7
    finally:
        try:
            await asyncio.wait_for(worker.flush(), 5)
        finally:
            try:
                await asyncio.wait_for(worker.stop(), 5)
            finally:
                atexit.unregister(worker._flush_on_exit)
                await asyncio.wait_for(asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True), 5)

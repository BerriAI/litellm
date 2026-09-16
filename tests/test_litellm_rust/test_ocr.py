import asyncio
import base64
import gc
import json
import os
import threading
import weakref
from collections.abc import Coroutine, Generator
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Final, Protocol

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.azure_ai.ocr.cohere_parse_transformation import AzureAICohereParseConfig
from litellm.llms.azure_ai.ocr.document_intelligence.transformation import AzureDocumentIntelligenceOCRConfig
from litellm.llms.azure_ai.ocr.transformation import AzureAIOCRConfig
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.llms.cohere.ocr.transformation import CohereParseConfig
from litellm.llms.mistral.ocr.transformation import MistralOCRConfig
from litellm.llms.reducto.ocr.transformation import ReductoParseLegacyConfig, ReductoParseV3Config
from litellm.llms.vertex_ai.ocr.deepseek_transformation import VertexAIDeepSeekOCRConfig
from litellm.llms.vertex_ai.ocr.transformation import VertexAIOCRConfig
from tests.test_litellm_rust.support.recording_server import ResponseSpec, recording_service

pytestmark = pytest.mark.requires_rust_extension


class _PythonOcrRequest(Protocol):
    @property
    def data(self) -> object: ...


class _PythonOcrConfig(Protocol):
    def map_ocr_params(
        self, non_default_params: dict[str, object], optional_params: dict[str, object], model: str
    ) -> dict[str, object]: ...

    def transform_ocr_request(
        self, model: str, document: dict[str, str], optional_params: dict[str, object], headers: dict[str, str]
    ) -> _PythonOcrRequest: ...

    def transform_ocr_response(self, model: str, raw_response: httpx.Response, logging_obj: Logging) -> OCRResponse: ...


@pytest.mark.parametrize(
    "provider,config,document,options,payload",
    [
        pytest.param(
            provider,
            config,
            {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
            {"pages": None, "include_image_base64": False},
            {
                "pages": [{"index": "2", "markdown": "text", "dimensions": {"width": 1.0}, "extension": False}],
                "usage_info": {"pages_processed": True, "credits": "1.5", "custom": 0},
                "extra": "ignored",
            },
            id=provider,
        )
        for provider, config in (
            ("mistral", MistralOCRConfig()),
            ("azure_ai", AzureAIOCRConfig()),
            ("vertex_ai", VertexAIOCRConfig()),
        )
    ]
    + [
        pytest.param(
            "azure_ai",
            AzureAICohereParseConfig(),
            {"type": "image_url", "image_url": "data:image/png;base64,YWJj"},
            {"output_format": "blocks"},
            {"pages": [{"blocks": [{"type": "future", "data": None}]}]},
            id="azure-cohere",
        ),
        pytest.param(
            "azure_ai",
            AzureDocumentIntelligenceOCRConfig(),
            {"type": "document_url", "document_url": "https://example.com/a.pdf"},
            {"pages": [0, 2], "features": ["languages"]},
            {
                "status": "succeeded",
                "analyzeResult": {
                    "pages": [{"pageNumber": 2, "width": 1.5, "height": 2, "lines": [{"content": "text"}]}],
                    "content": "text",
                    "tables": [{"extension": None}],
                    "keyValuePairs": [],
                },
            },
            id="azure-document-intelligence",
        ),
        pytest.param(
            "cohere",
            CohereParseConfig(),
            {"type": "image_url", "image_url": "https://example.com/a.png", "ignored": "field"},
            {"output_format": None},
            {
                "pages": [
                    {
                        "markdown": {
                            "images": [{"bounding_box": {"x": 1}, "category": "future"}, {"image_base64": "encoded"}]
                        },
                        "blocks": [{"type": "text", "text": "Total Due: $4.00"}, {"type": "future", "payload": None}],
                    }
                ],
                "meta": {"billed_units": {"pages": 0}},
            },
            id="cohere",
        ),
        pytest.param(
            "reducto",
            ReductoParseV3Config(),
            {"type": "document_url", "document_url": "reducto://ready.pdf"},
            {"formatting": None, "settings": {"future": True}},
            {
                "result": {
                    "chunks": [
                        {
                            "blocks": [
                                {"content": "text", "bbox": {"page": "2"}, "extension": None},
                                {"content": "ignored", "bbox": {"page": "bad"}},
                            ]
                        }
                    ]
                },
                "usage": {"num_pages": "1", "credits": True},
            },
            id="reducto-v3",
        ),
        pytest.param(
            "reducto",
            ReductoParseLegacyConfig(),
            {"type": "document_url", "document_url": "reducto://ready.pdf"},
            {"enhance": None},
            {"result": None, "chunks": [{"content": "ignored"}]},
            id="reducto-legacy",
        ),
    ]
    + [
        pytest.param(
            "vertex_ai",
            VertexAIDeepSeekOCRConfig(),
            {"type": "document_url", "document_url": "gs://bucket/a.pdf"},
            {"temperature": None},
            {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 1}},
            id=f"deepseek-{name}",
        )
        for name, content in (
            ("markdown", "text"),
            ("empty-pages", {"pages": []}),
            (
                "structured",
                {
                    "pages": [{"index": "2", "markdown": "text", "ignored": True}],
                    "document_annotation": {"language": "en"},
                    "ignored": True,
                },
            ),
            ("explicit-null-usage", {"model": "returned", "usage_info": None}),
            ("object-text", {"z": ["한글", "😀"], "a": "last"}),
        )
    ],
)
def test_native_provider_transforms_match_python_shapes(
    provider: str,
    config: _PythonOcrConfig,
    document: dict[str, str],
    options: dict[str, object],
    payload: dict[str, object],
) -> None:
    from litellm.rust_bridge import _native

    model: Final = (
        "deepseek-ocr"
        if isinstance(config, VertexAIDeepSeekOCRConfig)
        else "cohere-parse"
        if isinstance(config, AzureAICohereParseConfig)
        else "doc-intelligence/prebuilt-read"
        if isinstance(config, AzureDocumentIntelligenceOCRConfig)
        else "parse-legacy"
        if provider == "reducto" and "enhance" in options
        else "model"
    )
    mapped: Final = config.map_ocr_params(
        options, options if isinstance(config, VertexAIDeepSeekOCRConfig) else {}, model
    )
    expected_request: Final = config.transform_ocr_request(model, document, mapped, {}).data
    logging: Final = Logging(
        model=model,
        messages=[],
        stream=False,
        call_type="ocr",
        start_time=datetime.now(timezone.utc),
        litellm_call_id="ocr-shape-parity",
        function_id="ocr-shape-parity",
    )
    expected_response: Final = config.transform_ocr_response(
        model, httpx.Response(200, json=payload), logging
    ).model_dump()
    with recording_service() as server:
        server.enqueue(ResponseSpec(body=payload))
        response: Final = _native.ocr(
            model=model,
            document={**document},
            api_key="test-key",
            api_base=server.base_url,
            custom_llm_provider=provider,
            extra_headers=None,
            timeout=3,
            **{**options, **({"vertex_project": "project"} if provider == "vertex_ai" else {})},
        )
        assert response.model_dump() == expected_response
        assert server.requests[0].body == expected_request


@pytest.fixture
def ocr_server() -> Generator[tuple[ThreadingHTTPServer, list[dict[str, object]]]]:
    requests: Final[list[dict[str, object]]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requests.append(
                {
                    "headers": {name.lower(): value for name, value in self.headers.items()},
                    "body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                }
            )
            if self.headers.get("x-test-stall") == "true":
                self.connection.settimeout(2)
                try:
                    self.rfile.read(1)
                except TimeoutError:
                    pass
                return
            if self.headers.get("User-Agent", "").startswith("python-httpx"):
                self.send_response(418)
                self.end_headers()
                return
            status = int(self.headers.get("x-test-status", "200"))
            if status != 200:
                body = b'{"error":"provider unavailable"}'
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            response: Final = json.dumps(
                {
                    "pages": [{"index": 0, "markdown": "native OCR response", "images": [], "dimensions": None}],
                    "model": "mistral-ocr-latest",
                    "usage_info": {"pages_processed": 1, "doc_size_bytes": 3},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread: Final = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield server, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_native_ocr_with_compiled_rust_extension(
    ocr_server: tuple[ThreadingHTTPServer, list[dict[str, object]]],
) -> None:
    from litellm.rust_bridge import _native

    server, requests = ocr_server
    address: Final = server.server_address
    host: Final = str(address[0])
    port: Final = int(address[1])

    response: Final = _native.ocr(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
        api_key="test-key",
        api_base=f"http://{host}:{port}",
        custom_llm_provider="mistral",
        extra_headers=None,
        timeout=None,
    )

    assert response is not None
    assert response.pages[0].markdown == "native OCR response"
    assert len(requests) == 1
    assert not requests[0]["headers"].get("user-agent", "").startswith("python-httpx")
    assert requests[0]["body"] == {
        "model": "mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
    }


@pytest.mark.parametrize(
    "file_input,mime_type,expected_type,expected_field,expected_uri",
    [
        (b"abc", "application/pdf", "document_url", "document_url", "data:application/pdf;base64,YWJj"),
        (BytesIO(b"abc"), "image/png", "image_url", "image_url", "data:image/png;base64,YWJj"),
    ],
)
def test_native_lifecycle_core_encodes_python_file_input(
    ocr_server,
    file_input,
    mime_type,
    expected_type,
    expected_field,
    expected_uri,
):
    server, requests = ocr_server
    litellm.rust(True)
    response = litellm.ocr(
        model="mistral/mistral-ocr-latest",
        document={"type": "file", "file": file_input, "mime_type": mime_type},
        api_key="test-key",
        api_base=f"http://127.0.0.1:{server.server_port}",
        opaque_extension={"nested": [None, False, 0]},
    )
    assert response.pages[0].markdown == "native OCR response"
    assert requests[0]["body"]["document"] == {
        "type": expected_type,
        expected_field: expected_uri,
    }
    assert requests[0]["body"]["opaque_extension"] == {"nested": [None, False, 0]}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("source", ["path", "pathlike", "reader", "text", "bytes"])
@pytest.mark.asyncio
async def test_native_file_sources_preserve_sdk_behavior(
    ocr_server: tuple[ThreadingHTTPServer, list[dict[str, object]]],
    tmp_path: Path,
    asynchronous: bool,
    source: str,
) -> None:
    server, requests = ocr_server
    path: Final = tmp_path / "scan.png"
    content: Final = b"document bytes"
    path.write_bytes(content)

    class CustomPath:
        def __fspath__(self) -> str:
            return str(path)

        def __str__(self) -> str:
            return "/not/the/document.pdf"

    class Reader:
        name = "scan.png"

        def __init__(self) -> None:
            self.stream = BytesIO(b"prefix" + content)
            self.stream.seek(6)
            self.calls = 0

        def read(self) -> bytes | str:
            assert threading.get_ident() == caller_thread
            if asynchronous:
                assert asyncio.current_task() is caller_task
            self.calls += 1
            value: Final = self.stream.read()
            return value.decode() if source == "text" else value

    caller_thread: Final = threading.get_ident()
    caller_task: Final = asyncio.current_task()
    reader: Final = Reader()
    file_input: Final = {
        "path": path,
        "pathlike": CustomPath(),
        "reader": reader,
        "text": reader,
        "bytes": content,
    }[source]
    document: Final = {"type": "file", "file": file_input, **({"mime_type": "image/png"} if source == "bytes" else {})}
    litellm.rust(True)
    kwargs: Final = {
        "model": "mistral/mistral-ocr-latest",
        "document": document,
        "api_key": "test-key",
        "api_base": f"http://127.0.0.1:{server.server_port}",
    }
    response: Final = await litellm.aocr(**kwargs) if asynchronous else litellm.ocr(**kwargs)
    assert isinstance(response, OCRResponse)
    assert response.pages[0].markdown == "native OCR response"
    assert len(requests) == 1
    body: Final = requests[0]["body"]
    assert isinstance(body, dict)
    assert body["document"] == {
        "type": "image_url",
        "image_url": "data:image/png;base64," + base64.b64encode(content).decode(),
    }
    assert reader.calls == (1 if source in {"reader", "text"} else 0)
    assert not reader.stream.closed


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_native_file_failures_preserve_identity_and_do_not_send(
    ocr_server: tuple[ThreadingHTTPServer, list[dict[str, object]]],
    tmp_path: Path,
    asynchronous: bool,
) -> None:
    from litellm.rust_bridge import _native

    server, requests = ocr_server
    failure: Final = KeyError("reader failed")

    class Reader:
        def __init__(self) -> None:
            self.calls = 0

        def read(self) -> bytes:
            self.calls += 1
            raise failure

    reader: Final = Reader()
    path: Final = tmp_path / "missing.pdf"

    async def invoke(source: Reader | Path) -> None:
        if asynchronous:
            await _native.aocr(
                model="mistral/mistral-ocr-latest",
                document={"type": "file", "file": source},
                api_key="test-key",
                api_base=f"http://127.0.0.1:{server.server_port}",
            )
        else:
            _native.ocr(
                model="mistral/mistral-ocr-latest",
                document={"type": "file", "file": source},
                api_key="test-key",
                api_base=f"http://127.0.0.1:{server.server_port}",
            )

    for source, error in ((reader, KeyError), (path, FileNotFoundError)):
        with pytest.raises(error) as caught:
            await invoke(source)
        if source is reader:
            assert caught.value is failure
        else:
            assert isinstance(caught.value, FileNotFoundError)
            assert getattr(caught.value, "filename") == str(path)
    assert reader.calls == 1
    assert requests == []


def test_unstarted_native_file_call_does_not_read_and_releases_reader() -> None:
    from litellm.rust_bridge import _native

    class Reader:
        pending: Coroutine[object, object, OCRResponse] | None = None

        def read(self) -> bytes:
            raise AssertionError("unstarted call consumed its document")

    def create_call() -> tuple[Coroutine[object, object, OCRResponse], weakref.ReferenceType[Reader]]:
        reader: Final = Reader()
        pending: Final = _native.aocr(model="mistral/mistral-ocr-latest", document={"type": "file", "file": reader})
        reader.pending = pending
        return pending, weakref.ref(reader)

    pending, reference = create_call()
    pending.close()
    del pending
    gc.collect()
    assert reference() is None


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires Unix named pipes")
@pytest.mark.asyncio
async def test_native_path_cancellation_waits_for_read_completion(
    ocr_server: tuple[ThreadingHTTPServer, list[dict[str, object]]], tmp_path: Path
) -> None:
    from litellm.rust_bridge import _native

    server, requests = ocr_server
    path: Final = tmp_path / "document.pdf"
    os.mkfifo(path)
    entered: Final = threading.Event()
    release: Final = threading.Event()

    def supply_document() -> None:
        with path.open("wb") as stream:
            stream.write(b"document")
            stream.flush()
            entered.set()
            release.wait(5)

    writer: Final = threading.Thread(target=supply_document, daemon=True)
    writer.start()
    task: Final = asyncio.create_task(
        _native.aocr(
            model="mistral/mistral-ocr-latest",
            document={"type": "file", "file": path},
            api_key="test-key",
            api_base=f"http://127.0.0.1:{server.server_port}",
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done()
    finally:
        release.set()
        await asyncio.to_thread(writer.join, 3)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert requests == []


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("model", ["mistral/mistral-ocr-latest", "azure_ai/doc-intelligence/prebuilt-read"])
@pytest.mark.asyncio
async def test_native_public_ocr_matches_python(model, asynchronous):
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from typing import Final
    from urllib.parse import parse_qsl, urlsplit

    from litellm.rust_bridge import _native

    assert callable(_native.ocr)
    calls: Final = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body: Final = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            target: Final = urlsplit(self.path)
            calls.append(
                (
                    target.path,
                    parse_qsl(target.query),
                    self.headers.get("Authorization"),
                    self.headers.get("Ocp-Apim-Subscription-Key"),
                    body,
                )
            )
            payload: Final = (
                {"status": "succeeded", "analyzeResult": {"pages": []}}
                if "doc-intelligence" in model
                else {"pages": [{"index": 0, "markdown": "hello"}]}
            )
            encoded: Final = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            pass

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread: Final = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        litellm.rust(True)
        arguments: Final = {
            "model": model,
            "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
            "api_key": "test-key",
            "api_base": f"http://127.0.0.1:{server.server_port}",
            "pages": [0, 2],
            "timeout": 3.0,
        }
        response: Final = await litellm.aocr(**arguments) if asynchronous else litellm.ocr(**arguments)
        response_data: Final = response.model_dump()
        assert len(calls) == 1
        assert response_data["object"] == "ocr"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_native_ocr_failures_do_not_retry_on_python(ocr_server, asynchronous):
    server, requests = ocr_server
    arguments = {
        "model": "mistral-ocr-latest",
        "custom_llm_provider": "mistral",
        "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
        "api_key": "test-key",
        "api_base": f"http://127.0.0.1:{server.server_port}",
        "extra_headers": {"x-test-status": "503"},
        "num_retries": 0,
    }
    litellm.rust(True)
    with pytest.raises(litellm.ServiceUnavailableError) as caught:
        await litellm.aocr(**arguments) if asynchronous else litellm.ocr(**arguments)
    assert caught.value.status_code == 503
    assert len(requests) == 1
    assert not requests[0]["headers"].get("user-agent", "").startswith("python-httpx")


@pytest.mark.parametrize(
    "custom_provider,error_type,message",
    [
        ("mistral", litellm.BadRequestError, "document"),
        ("not-a-provider", litellm.APIConnectionError, "invalid provider"),
    ],
)
def test_native_ocr_rejects_invalid_input_before_network(ocr_server, custom_provider, error_type, message):
    from litellm.rust_bridge import _native

    server, requests = ocr_server
    with pytest.raises(error_type, match=message):
        _native.ocr(
            model="mistral-ocr-latest",
            custom_llm_provider=custom_provider,
            document={"type": "document_url"},
            api_key="test-key",
            api_base=f"http://127.0.0.1:{server.server_port}",
        )
    assert requests == []


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_native_ocr_enforces_request_deadline_without_fallback(ocr_server, asynchronous):
    import asyncio
    import time

    server, requests = ocr_server
    litellm.rust(True)
    arguments = {
        "model": "mistral/mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
        "api_key": "test-key",
        "api_base": f"http://127.0.0.1:{server.server_port}",
        "extra_headers": {"x-test-stall": "true"},
        "timeout": 0.1,
        "num_retries": 0,
    }
    started = time.monotonic()
    with pytest.raises(litellm.Timeout):
        await asyncio.wait_for(
            litellm.aocr(**arguments) if asynchronous else asyncio.to_thread(litellm.ocr, **arguments),
            timeout=3,
        )
    assert 0.09 <= time.monotonic() - started < 3
    assert len(requests) == 1
    assert not requests[0]["headers"].get("user-agent", "").startswith("python-httpx")

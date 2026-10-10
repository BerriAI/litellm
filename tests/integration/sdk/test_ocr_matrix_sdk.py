from __future__ import annotations

import asyncio
import base64
import io
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from email.parser import BytesParser
from email.policy import HTTP
from pathlib import Path
from typing import Final, Literal, Protocol, cast

import pytest
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.base_llm.ocr.transformation import OCRResponse

pytestmark = pytest.mark.asyncio

Document = Mapping[str, object]
AuthMode = Literal["explicit", "env"]
CallStyle = Literal["sync", "async"]

PAGE_TEXT: Final = "scripted ocr page"
PDF_BYTES: Final = b"%PDF-1.4\n%ocr matrix document\n%%EOF\n"
PNG_BYTES: Final = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
DOCUMENTS: Final = {
    "/documents/scan.pdf": (PDF_BYTES, "application/pdf"),
    "/documents/scan.png": (PNG_BYTES, "image/png"),
}
DI_ANALYZE_PATH: Final = "/documentintelligence/documentModels/prebuilt-layout"
OCR_CALL_TYPES: Final = frozenset({"ocr", "aocr"})
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_STRING_KEYED: Final = TypeAdapter(dict[str, object])
_FLOAT: Final = TypeAdapter(float)


class _OCRClient(Protocol):
    def ocr(self, *, model: str, document: Document, **kwargs: object) -> OCRResponse: ...

    async def aocr(self, *, model: str, document: Document, **kwargs: object) -> OCRResponse: ...


_OCR_CLIENT: Final = cast(_OCRClient, litellm)


def _json_reply(body: Mapping[str, object]) -> Reply:
    return Reply(body=json.dumps(body).encode())


def _respond(request: Request) -> Reply:
    path: Final = request.target.split("?", 1)[0]
    if request.method == "GET" and path in DOCUMENTS:
        content, content_type = DOCUMENTS[path]
        return Reply(body=content, content_type=content_type)
    if path == "/upload":
        return _json_reply({"file_id": "reducto://scripted-upload"})
    if path == "/parse":
        return _json_reply(
            {
                "job_id": "scripted-job",
                "usage": {"num_pages": 1, "credits": 1},
                "result": {"chunks": [{"content": PAGE_TEXT, "blocks": [{"content": PAGE_TEXT, "bbox": {"page": 1}}]}]},
            }
        )
    if path == f"{DI_ANALYZE_PATH}:analyze":
        operation: Final = f"http://{request.headers['host']}{DI_ANALYZE_PATH}/analyzeResults/scripted?api-version=1"
        return Reply(status=202, body=b"", headers={"operation-location": operation})
    if path.startswith(f"{DI_ANALYZE_PATH}/analyzeResults/"):
        return _json_reply(
            {
                "status": "succeeded",
                "analyzeResult": {
                    "content": PAGE_TEXT,
                    "pages": [{"pageNumber": 1, "lines": [{"content": PAGE_TEXT}]}],
                },
            }
        )
    if path == "/v2/parse":
        return _json_reply(
            {"pages": [{"index": 0, "markdown": {"content": PAGE_TEXT}}], "meta": {"billed_units": {"pages": 1}}}
        )
    model: Final = _JSON.validate_json(request.body)["model"]
    return _json_reply(
        {"pages": [{"index": 0, "markdown": PAGE_TEXT}], "model": model, "usage_info": {"pages_processed": 1}}
    )


@dataclass(frozen=True, slots=True)
class LoggedCall:
    payload: Mapping[str, object]
    response: object


class RecordingLogger(CustomLogger):
    def __init__(self) -> None:
        self.calls: tuple[LoggedCall, ...] = ()

    def _record(self, kwargs: Mapping[str, object], response_obj: object) -> None:
        payload: Final = _STRING_KEYED.validate_python(kwargs.get("standard_logging_object"))
        if payload.get("call_type") in OCR_CALL_TYPES:
            self.calls = (*self.calls, LoggedCall(payload, response_obj))

    def log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self._record(kwargs, response_obj)

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self._record(kwargs, response_obj)

    async def wait_for_call(self, timeout: float = 10.0) -> LoggedCall:
        deadline: Final = asyncio.get_running_loop().time() + timeout
        while not self.calls:
            assert asyncio.get_running_loop().time() < deadline, "success callback never fired"
            await asyncio.sleep(0.05)
        assert len(self.calls) == 1, self.calls
        return self.calls[0]


@pytest.fixture
def logger(monkeypatch: pytest.MonkeyPatch) -> RecordingLogger:
    recorder: Final = RecordingLogger()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    return recorder


class _NamedReader(io.BytesIO):
    def __init__(self, path: Path) -> None:
        super().__init__(path.read_bytes())
        self.name: Final = path.name


def _data_uri(content: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(content).decode()}"


def _written(tmp_path: Path, name: str, content: bytes) -> Path:
    path: Final = tmp_path / name
    path.write_bytes(content)
    return path


@dataclass(frozen=True, slots=True)
class Input:
    id: str
    build: Callable[[str, Path], Document]
    content: bytes
    url_path: str | None = None


PDF_BY_URL: Final = Input(
    "pdf_url",
    lambda base, _: {"type": "document_url", "document_url": f"{base}/documents/scan.pdf"},
    PDF_BYTES,
    "/documents/scan.pdf",
)
PNG_BY_URL: Final = Input(
    "image_url",
    lambda base, _: {"type": "image_url", "image_url": f"{base}/documents/scan.png"},
    PNG_BYTES,
    "/documents/scan.png",
)
PDF_DATA_URI: Final = Input(
    "pdf_data_uri",
    lambda *_: {"type": "document_url", "document_url": _data_uri(PDF_BYTES, "application/pdf")},
    PDF_BYTES,
)
PNG_DATA_URI: Final = Input(
    "image_data_uri", lambda *_: {"type": "image_url", "image_url": _data_uri(PNG_BYTES, "image/png")}, PNG_BYTES
)
PDF_AS_PATH: Final = Input(
    "pdf_path", lambda _, tmp: {"type": "file", "file": _written(tmp, "scan.pdf", PDF_BYTES)}, PDF_BYTES
)
PDF_AS_BYTES: Final = Input(
    "pdf_bytes", lambda *_: {"type": "file", "file": PDF_BYTES, "mime_type": "application/pdf"}, PDF_BYTES
)
PNG_AS_BYTES: Final = Input(
    "image_bytes", lambda *_: {"type": "file", "file": PNG_BYTES, "mime_type": "image/png"}, PNG_BYTES
)
PNG_AS_FILE_OBJECT: Final = Input(
    "image_file_object",
    lambda _, tmp: {"type": "file", "file": _NamedReader(_written(tmp, "scan.png", PNG_BYTES))},
    PNG_BYTES,
)


@dataclass(frozen=True, slots=True)
class Provider:
    id: str
    model: str
    key_env: str
    auth_header: str
    base_env: str | None = None
    downloads_urls: bool = False

    def credential_header(self, key: str) -> str:
        return key if self.auth_header == "ocp-apim-subscription-key" else f"Bearer {key}"


MISTRAL: Final = Provider("mistral", "mistral/mistral-ocr-latest", "MISTRAL_API_KEY", "authorization")
AZURE_AI_MISTRAL: Final = Provider(
    "azure_ai_mistral",
    "azure_ai/mistral-document-ai-2512",
    "AZURE_AI_API_KEY",
    "authorization",
    base_env="AZURE_AI_API_BASE",
    downloads_urls=True,
)
AZURE_DOC_INTELLIGENCE: Final = Provider(
    "azure_doc_intelligence",
    "azure_ai/doc-intelligence/prebuilt-layout",
    "AZURE_DOCUMENT_INTELLIGENCE_API_KEY",
    "ocp-apim-subscription-key",
    base_env="AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
)
COHERE: Final = Provider("cohere", "cohere/parse-v5.0", "COHERE_API_KEY", "authorization")
REDUCTO_V3: Final = Provider("reducto_v3", "reducto/parse-v3", "REDUCTO_API_KEY", "authorization")
REDUCTO_LEGACY: Final = Provider("reducto_legacy", "reducto/parse-legacy", "REDUCTO_API_KEY", "authorization")


@dataclass(frozen=True, slots=True)
class Case:
    provider: Provider
    auth: AuthMode
    document: Input
    call: CallStyle

    @property
    def id(self) -> str:
        return f"{self.provider.id}-{self.auth}-{self.document.id}-{self.call}"

    @property
    def key(self) -> str:
        return f"{self.provider.id}-{self.auth}-key"

    def bind_credentials(self, monkeypatch: pytest.MonkeyPatch, base_url: str) -> Mapping[str, str]:
        for env_var in (self.provider.key_env, self.provider.base_env):
            if env_var is not None:
                monkeypatch.delenv(env_var, raising=False)
        if self.auth == "explicit":
            return {"api_key": self.key, "api_base": base_url}
        monkeypatch.setenv(self.provider.key_env, self.key)
        if self.provider.base_env is None:
            return {"api_base": base_url}
        monkeypatch.setenv(self.provider.base_env, base_url)
        return {}

    async def run(self, document: Document, credentials: Mapping[str, str]) -> OCRResponse:
        response: Final = (
            await _OCR_CLIENT.aocr(model=self.provider.model, document=document, **credentials)
            if self.call == "async"
            else _OCR_CLIENT.ocr(model=self.provider.model, document=document, **credentials)
        )
        assert isinstance(response, OCRResponse)
        return response


CASES: Final = (
    Case(MISTRAL, "explicit", PDF_BY_URL, "sync"),
    Case(MISTRAL, "env", PNG_BY_URL, "async"),
    Case(MISTRAL, "explicit", PDF_AS_PATH, "sync"),
    Case(MISTRAL, "explicit", PNG_AS_BYTES, "async"),
    Case(MISTRAL, "explicit", PNG_AS_FILE_OBJECT, "sync"),
    Case(AZURE_AI_MISTRAL, "explicit", PDF_BY_URL, "sync"),
    Case(AZURE_AI_MISTRAL, "env", PNG_BY_URL, "async"),
    Case(AZURE_DOC_INTELLIGENCE, "explicit", PDF_BY_URL, "sync"),
    Case(AZURE_DOC_INTELLIGENCE, "env", PNG_AS_BYTES, "async"),
    Case(COHERE, "explicit", PNG_BY_URL, "sync"),
    Case(COHERE, "env", PNG_DATA_URI, "async"),
    Case(REDUCTO_V3, "explicit", PDF_AS_PATH, "sync"),
    Case(REDUCTO_V3, "env", PNG_AS_BYTES, "async"),
    Case(REDUCTO_V3, "explicit", PDF_DATA_URI, "async"),
    Case(REDUCTO_LEGACY, "explicit", PDF_AS_BYTES, "sync"),
)


def _decoded(value: str) -> bytes | str:
    if not value.startswith("data:"):
        return value
    return base64.b64decode(value.split(",", 1)[1])


def _uploaded_file(request: Request) -> bytes:
    message: Final = BytesParser(policy=HTTP).parsebytes(
        f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    )
    parts: Final = tuple(
        part for part in message.iter_parts() if part.get_param("name", header="content-disposition") == "file"
    )
    assert len(parts) == 1, request.body[:200]
    payload: Final = parts[0].get_payload(decode=True)
    assert isinstance(payload, bytes)
    return payload


def _sent_document(request: Request) -> bytes | str:
    if "multipart/form-data" in request.headers.get("content-type", ""):
        return _uploaded_file(request)
    body: Final = _JSON.validate_json(request.body)
    if "urlSource" in body:
        return str(body["urlSource"])
    if "base64Source" in body:
        return base64.b64decode(str(body["base64Source"]))
    document: Final = _STRING_KEYED.validate_python(body["document"])
    return _decoded(str(document.get("document_url") or document["image_url"]))


def _assert_sent(case: Case, requests: tuple[Request, ...], document: Document) -> None:
    provider_calls: Final = tuple(request for request in requests if request.target.split("?", 1)[0] not in DOCUMENTS)
    assert provider_calls, requests
    assert {request.headers.get(case.provider.auth_header) for request in provider_calls} == {
        case.provider.credential_header(case.key)
    }
    downloads: Final = tuple(request for request in requests if request.target.split("?", 1)[0] in DOCUMENTS)
    url: Final = document.get("document_url") or document.get("image_url")
    passes_url_through: Final = case.document.url_path is not None and not case.provider.downloads_urls
    assert _sent_document(provider_calls[0]) == (url if passes_url_through else case.document.content)
    assert [request.target for request in downloads] == (
        [case.document.url_path] if case.document.url_path is not None and case.provider.downloads_urls else []
    )


def _assert_ocr_response(response: OCRResponse, model: str) -> None:
    assert response.object == "ocr"
    assert response.model == model.split("/", 1)[1]
    assert [page.index for page in response.pages] == list(range(len(response.pages)))
    assert PAGE_TEXT in " ".join(page.markdown for page in response.pages)
    assert response.usage_info is not None
    assert response.usage_info.pages_processed == len(response.pages)


def _assert_logged(logged: LoggedCall, response: OCRResponse, model: str, call: CallStyle) -> None:
    assert logged.response is response
    assert logged.payload["status"] == "success"
    assert logged.payload["call_type"] == ("aocr" if call == "async" else "ocr")
    assert logged.payload["custom_llm_provider"] == model.split("/", 1)[0]
    assert logged.payload["model"] == response.model
    assert _FLOAT.validate_python(logged.payload["response_cost"]) > 0


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
async def test_ocr_sends_the_bound_credential_and_document_then_logs_a_priced_call(
    case: Case,
    monkeypatch: pytest.MonkeyPatch,
    logger: RecordingLogger,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    with wire_server(_respond) as wire:
        monkeypatch.setattr(litellm, "user_url_allowed_hosts", [wire.url.removeprefix("http://")])
        credentials: Final = case.bind_credentials(monkeypatch, wire.url)
        document: Final = case.document.build(wire.url, tmp_path)
        response: Final = await case.run(document, credentials)
        logged: Final = await logger.wait_for_call()
        requests: Final = wire.drain()

    _assert_ocr_response(response, case.provider.model)
    _assert_sent(case, requests, document)
    _assert_logged(logged, response, case.provider.model, case.call)

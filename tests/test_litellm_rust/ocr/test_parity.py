import re
import socket
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.prompt_templates import image_handling
from litellm.llms.azure_ai.ocr.document_intelligence import transformation as document_intelligence
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from tests.test_litellm_rust.support.parity import Divergence, Masked, Outcome, assert_parity, run_on
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import OCR_DOCUMENT

pytestmark = pytest.mark.requires_rust_extension

IMAGE_DOCUMENT: Final = {"type": "image_url", "image_url": "data:image/png;base64,YWJj"}
REDUCTO_DOCUMENT: Final = {"type": "document_url", "document_url": "reducto://ready.pdf"}
VERTEX: Final = {"vertex_project": "test-project", "vertex_location": "us-central1"}


@dataclass(frozen=True, slots=True)
class Provider:
    model: str
    document: Mapping[str, object]
    arguments: Mapping[str, object]


PROVIDERS: Final = {
    "mistral": Provider("mistral/mistral-ocr-latest", OCR_DOCUMENT, {}),
    "azure-mistral": Provider("azure_ai/mistral-ocr-latest", OCR_DOCUMENT, {}),
    "azure-document-intelligence": Provider("azure_ai/doc-intelligence/prebuilt-layout", OCR_DOCUMENT, {}),
    "azure-cohere": Provider("azure_ai/Cohere-parse-v5.0", IMAGE_DOCUMENT, {}),
    "cohere": Provider("cohere/parse-v5.0", IMAGE_DOCUMENT, {}),
    "vertex-mistral": Provider("vertex_ai/mistral-ocr-latest", OCR_DOCUMENT, VERTEX),
    "vertex-deepseek": Provider("vertex_ai/deepseek-ocr", OCR_DOCUMENT, VERTEX),
    "reducto-v3": Provider("reducto/parse-v3", REDUCTO_DOCUMENT, {}),
    "reducto-legacy": Provider("reducto/parse-legacy", REDUCTO_DOCUMENT, {}),
}

STATUSES: Final = (400, 401, 403, 404, 408, 409, 413, 422, 429, 498, 500, 502, 503, 504, 529)

BODIES: Final = {
    "json-object": {"message": "rejected"},
    "json-string": "rejected",
    "empty-object": {},
    "rate-limit-text": {"message": "rate limit reached for requests"},
    "context-window-text": {"message": "This model's maximum context length is 10 tokens"},
    "content-policy-text": {"message": "content_policy_violation"},
    "invalid-request-text": {"error": {"type": "invalid_request_error", "message": "bad field"}},
    "model-not-found-text": {"error": {"type": "invalid_request_error", "code": "model_not_found"}},
    "timeout-text": {"message": "Request timed out"},
    "vertex-quota-text": {"message": "429 Quota exceeded for aiplatform"},
    "vertex-api-key-text": {"message": "API key not valid. Please pass a valid API key."},
    "cohere-api-token-text": {"message": "invalid api token"},
    "cohere-internal-text": {"message": "Internal Server Error"},
}

REPRESENTATIVE_STATUSES: Final = (400, 429, 500)

NO_RETRIES: Final = MappingProxyType({"num_retries": 0})
CALL_OPTIONS: Final = {
    "no-retries": NO_RETRIES,
    "retries-and-timeout": MappingProxyType({"num_retries": 0, "timeout": 7.5}),
    "defaults": MappingProxyType({}),
}

VERTEX_UPSTREAM_CONTENT: Final = tuple(
    Divergence(field, "Rust keeps the upstream body and headers that _map_vertex_exception replaces with a stub")
    for field in ("response_text", "response_headers")
)


def upstream_divergences(
    asynchronous: bool, provider: Provider, options: Mapping[str, object], python: Outcome
) -> tuple[Divergence, ...]:
    vertex: Final = provider.model.startswith("vertex_ai/") and python.response_text == ""
    return VERTEX_UPSTREAM_CONTENT if vertex else ()


def arguments(server: RecordingServer, provider: Provider, **extra: object) -> dict[str, object]:
    return {
        "model": provider.model,
        "document": dict(provider.document),
        "api_key": "test-key",
        "api_base": server.base_url,
        **provider.arguments,
        **extra,
    }


async def call(
    server: RecordingServer, asynchronous: bool, provider: Provider, options: Mapping[str, object]
) -> OCRResponse:
    if asynchronous:
        return await litellm.aocr(**arguments(server, provider, **options))
    return litellm.ocr(**arguments(server, provider, **options))


async def compare_upstream(
    server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    provider: Provider,
    response: ResponseSpec,
    asynchronous: bool,
    options: Mapping[str, object] = NO_RETRIES,
    masks: tuple[Masked, ...] = (),
) -> None:
    server.expected_requests = None
    server.enqueue(response)
    python: Final = await run_on(False, monkeypatch, lambda: call(server, asynchronous, provider, options))
    server.enqueue(response)
    rust: Final = await run_on(True, monkeypatch, lambda: call(server, asynchronous, provider, options))
    assert_parity(python, rust, upstream_divergences(asynchronous, provider, options, python), masks)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("status", STATUSES)
@pytest.mark.parametrize("provider", PROVIDERS.values(), ids=PROVIDERS.keys())
async def test_upstream_status_parity(
    recording_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    provider: Provider,
    status: int,
    asynchronous: bool,
) -> None:
    await compare_upstream(
        recording_server,
        monkeypatch,
        provider,
        ResponseSpec(body=BODIES["json-object"], status=status, headers={"retry-after": "7"}),
        asynchronous,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("status", REPRESENTATIVE_STATUSES)
@pytest.mark.parametrize("body", BODIES.values(), ids=BODIES.keys())
@pytest.mark.parametrize("provider", PROVIDERS.values(), ids=PROVIDERS.keys())
async def test_upstream_body_parity(
    recording_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    provider: Provider,
    body: object,
    status: int,
    asynchronous: bool,
) -> None:
    await compare_upstream(
        recording_server, monkeypatch, provider, ResponseSpec(body=body, status=status), asynchronous
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("options", CALL_OPTIONS.values(), ids=CALL_OPTIONS.keys())
@pytest.mark.parametrize("status", (401, 500))
@pytest.mark.parametrize("provider", PROVIDERS.values(), ids=PROVIDERS.keys())
async def test_call_option_parity(
    recording_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    provider: Provider,
    status: int,
    options: Mapping[str, object],
    asynchronous: bool,
) -> None:
    await compare_upstream(
        recording_server,
        monkeypatch,
        provider,
        ResponseSpec(body=BODIES["json-object"], status=status),
        asynchronous,
        options,
    )


def closed_port_base_url() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        host, port = probe.getsockname()
    return f"http://{host}:{port}"


@dataclass(frozen=True, slots=True)
class Transport:
    response: ResponseSpec | None
    options: Mapping[str, object]


TRANSPORTS: Final = {
    "read-timeout": Transport(ResponseSpec(body={}, delay=1.5), {**NO_RETRIES, "timeout": 0.5}),
    "connection-refused": Transport(None, NO_RETRIES),
    "not-json": Transport(ResponseSpec(body=None, raw=b"<html>bad gateway</html>"), NO_RETRIES),
    "missing-fields": Transport(ResponseSpec(body={"unexpected": True}), NO_RETRIES),
    "oversized": Transport(ResponseSpec(body={"padding": "x" * 64}), {**NO_RETRIES, "max_response_bytes": 16}),
}


def replaced(pattern: re.Pattern[str], placeholder: str) -> Callable[[object], object]:
    return lambda value: pattern.sub(placeholder, value) if isinstance(value, str) else value


ELAPSED: Final = tuple(
    Masked(
        field,
        "both handlers report how long the timed-out call ran",
        replaced(re.compile(r"time taken=[0-9.]+ seconds"), "time taken=<elapsed> seconds"),
    )
    for field in ("text", "message")
)

REFUSED_CONNECTION: Final = re.compile(r"\[Errno \d+\] Connection refused|error sending request")


def refused_connection_divergences(provider: Provider) -> tuple[Divergence, ...]:
    reason: Final = "httpx and reqwest word a refused connection differently"
    text: Final = tuple(
        Divergence(field, reason, replaced(REFUSED_CONNECTION, "<connection refused>")) for field in ("text", "message")
    )
    vertex: Final = (Divergence("response_text", reason),) if provider.model.startswith("vertex_ai/") else ()
    return text + vertex


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("transport", TRANSPORTS.values(), ids=TRANSPORTS.keys())
@pytest.mark.parametrize("provider", PROVIDERS.values(), ids=PROVIDERS.keys())
async def test_transport_parity(
    recording_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    provider: Provider,
    transport: Transport,
    asynchronous: bool,
) -> None:
    if transport.response is not None:
        await compare_upstream(
            recording_server, monkeypatch, provider, transport.response, asynchronous, transport.options, ELAPSED
        )
        return
    recording_server.expected_requests = 0
    unreachable: Final = {**transport.options, "api_base": closed_port_base_url()}
    python: Final = await run_on(
        False, monkeypatch, lambda: call(recording_server, asynchronous, provider, unreachable)
    )
    rust: Final = await run_on(True, monkeypatch, lambda: call(recording_server, asynchronous, provider, unreachable))
    assert_parity(
        python,
        rust,
        upstream_divergences(asynchronous, provider, unreachable, python) + refused_connection_divergences(provider),
    )


@dataclass(frozen=True, slots=True)
class Validation:
    document: Callable[[Path], Mapping[str, object]] | None = None
    options: Mapping[str, object] = NO_RETRIES
    api_key: str | None = "test-key"
    environment: Callable[[Path], Mapping[str, str]] = lambda _: MappingProxyType({})


def file_document(file: object, **extra: object) -> Mapping[str, object]:
    return {"type": "file", "file": file, **extra}


def unreadable_file(directory: Path) -> Mapping[str, object]:
    path: Final = directory / "unreadable.pdf"
    if not path.exists():
        path.write_bytes(b"%PDF-1.4")
        path.chmod(0)
    return file_document(path)


class RaisingReader:
    def read(self) -> bytes:
        raise ValueError("reader exploded")


def raising_token_provider() -> str:
    raise ValueError("token unavailable")


def url_document(url: str) -> Callable[[Path], Mapping[str, object]]:
    return lambda _: {"type": "document_url", "document_url": url}


GENERIC_VALIDATIONS: Final = {
    "bad-req-format": Validation(options={**NO_RETRIES, "req_format": "it's-bogus"}),
    "empty-file": Validation(lambda _: file_document(b"")),
    "missing-file": Validation(lambda directory: file_document(directory / "missing.pdf")),
    "unreadable-file": Validation(unreadable_file),
    "missing-document-url": Validation(url_document("")),
    "invalid-mime-type": Validation(lambda _: file_document(b"abc", mime_type="not a mime type")),
    "invalid-data-uri": Validation(url_document("data:application/pdf;base64")),
}

SPECIFIC_VALIDATIONS: Final = {
    "cohere-pdf": ("cohere", Validation(url_document(str(OCR_DOCUMENT["document_url"])))),
    "azure-cohere-pdf": ("azure-cohere", Validation(url_document(str(OCR_DOCUMENT["document_url"])))),
    "reducto-plain-url": ("reducto-v3", Validation(url_document("https://example.com/scan.pdf"))),
    "reducto-legacy-plain-url": ("reducto-legacy", Validation(url_document("https://example.com/scan.pdf"))),
    "invalid-pages": ("azure-document-intelligence", Validation(options={**NO_RETRIES, "pages": [-1]})),
    "missing-azure-ai-credentials": ("azure-mistral", Validation(api_key=None)),
    "missing-document-intelligence-credentials": ("azure-document-intelligence", Validation(api_key=None)),
    "missing-reducto-key": ("reducto-v3", Validation(api_key=None)),
    "raising-file-reader": ("mistral", Validation(lambda _: file_document(RaisingReader()))),
    "raising-token-provider": (
        "azure-mistral",
        Validation(options={**NO_RETRIES, "azure_ad_token_provider": raising_token_provider}, api_key=None),
    ),
    "vertex-api-key-env-without-google-credentials": (
        "vertex-mistral",
        Validation(
            api_key=None,
            environment=lambda directory: {
                "VERTEX_AI_API_KEY": "env-key",
                "GOOGLE_APPLICATION_CREDENTIALS": str(directory / "missing-credentials.json"),
            },
        ),
    ),
}

VALIDATION_ROWS: Final = {
    **{
        f"{name}-{provider}": (PROVIDERS[provider], validation)
        for name, validation in GENERIC_VALIDATIONS.items()
        for provider in PROVIDERS
    },
    **{name: (PROVIDERS[provider], validation) for name, (provider, validation) in SPECIFIC_VALIDATIONS.items()},
    "invalid-provider": (Provider("not_a_provider/ocr-model", OCR_DOCUMENT, {}), Validation()),
    "model-of-a-provider-without-ocr": (Provider("gpt-4o", OCR_DOCUMENT, {}), Validation()),
    "model-without-provider-prefix": (Provider("mistral-ocr-latest", OCR_DOCUMENT, {}), Validation()),
}

CREDENTIAL_VARIABLES: Final = (
    "AZURE_DOCUMENT_INTELLIGENCE_API_KEY",
    "REDUCTO_API_KEY",
    "MISTRAL_API_KEY",
    "VERTEX_AI_API_KEY",
    "VERTEXAI_API_KEY",
    "VERTEXAI_CREDENTIALS",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("row", VALIDATION_ROWS.values(), ids=VALIDATION_ROWS.keys())
async def test_validation_parity(
    recording_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    isolated_azure_auth: None,
    tmp_path: Path,
    row: tuple[Provider, Validation],
    asynchronous: bool,
) -> None:
    provider, validation = row
    for variable in CREDENTIAL_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    for variable, value in validation.environment(tmp_path).items():
        monkeypatch.setenv(variable, value)
    recording_server.expected_requests = None

    def invoke() -> Awaitable[OCRResponse]:
        document: Final = provider.document if validation.document is None else validation.document(tmp_path)
        options: Final = {**validation.options, "document": dict(document), "api_key": validation.api_key}
        return call(recording_server, asynchronous, provider, options)

    python: Final = await run_on(False, monkeypatch, invoke)
    python_requests: Final = len(recording_server.requests)
    rust: Final = await run_on(True, monkeypatch, invoke)
    assert len(recording_server.requests) - python_requests == python_requests
    assert_parity(python, rust, upstream_divergences(asynchronous, provider, validation.options, python))


@dataclass(frozen=True, slots=True)
class SentRequest:
    method: str
    path: str
    headers: Mapping[str, str]
    body: object


TRANSPORT_HEADERS: Final = frozenset(
    {"host", "content-length", "accept", "accept-encoding", "connection", "user-agent"}
)


def sent_since(server: RecordingServer, start: int) -> tuple[SentRequest, ...]:
    return tuple(
        SentRequest(
            method=request.method,
            path=request.path,
            headers=MappingProxyType(
                {name: value for name, value in request.headers.items() if name not in TRANSPORT_HEADERS}
            ),
            body=request.body if request.body is not None else request.raw_body,
        )
        for request in server.requests[start:]
    )


async def sent_on(
    backend: bool, server: RecordingServer, monkeypatch: pytest.MonkeyPatch, call: Callable[[], Awaitable[OCRResponse]]
) -> tuple[SentRequest, ...]:
    start: Final = len(server.requests)
    await run_on(backend, monkeypatch, call)
    return sent_since(server, start)


REQUEST_ROWS: Final = {
    **{name: (provider, MappingProxyType({})) for name, provider in PROVIDERS.items()},
    "extra-body-mistral": (PROVIDERS["mistral"], MappingProxyType({"extra_body": {"foo": 1}})),
    "sampling-params-vertex-deepseek": (
        PROVIDERS["vertex-deepseek"],
        MappingProxyType({"temperature": 0, "max_tokens": 5, "stream": True}),
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("row", REQUEST_ROWS.values(), ids=REQUEST_ROWS.keys())
async def test_request_parity(
    recording_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    row: tuple[Provider, Mapping[str, object]],
    asynchronous: bool,
) -> None:
    provider, options = row
    recording_server.expected_requests = None
    python: Final = await sent_on(
        False, recording_server, monkeypatch, lambda: call(recording_server, asynchronous, provider, options)
    )
    rust: Final = await sent_on(
        True, recording_server, monkeypatch, lambda: call(recording_server, asynchronous, provider, options)
    )
    assert python
    assert rust == python


@pytest.mark.asyncio
async def test_request_user_agent_parity(recording_server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> None:
    recording_server.expected_requests = 2
    for backend in (False, True):
        await run_on(backend, monkeypatch, lambda: call(recording_server, True, PROVIDERS["mistral"], NO_RETRIES))
    python, rust = (request.headers.get("user-agent") for request in recording_server.requests)
    assert rust == python


DOCUMENT_INTELLIGENCE_RESULT: Final = {
    "status": "succeeded",
    "analyzeResult": {
        "content": "scanned",
        "pages": [{"pageNumber": 1, "width": 2, "height": 3, "unit": "inch", "lines": [{"content": "scanned"}]}],
    },
}

DOCUMENT_INTELLIGENCE_SETTINGS: Final = {
    "defaults": MappingProxyType({}),
    "api-version": MappingProxyType({"AZURE_DOCUMENT_INTELLIGENCE_API_VERSION": "2023-07-31"}),
    "dpi": MappingProxyType({"AZURE_DOCUMENT_INTELLIGENCE_DEFAULT_DPI": 300}),
}


def analyze(server: RecordingServer) -> Awaitable[OCRResponse]:
    return call(server, True, PROVIDERS["azure-document-intelligence"], NO_RETRIES)


async def analyzed_on(
    backend: bool, server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> tuple[tuple[str, ...], Outcome]:
    start: Final = len(server.requests)
    server.enqueue(ResponseSpec(body=DOCUMENT_INTELLIGENCE_RESULT))
    outcome: Final = await run_on(backend, monkeypatch, lambda: analyze(server))
    return tuple(request.path for request in server.requests[start:]), outcome


@pytest.mark.asyncio
@pytest.mark.parametrize("settings", DOCUMENT_INTELLIGENCE_SETTINGS.values(), ids=DOCUMENT_INTELLIGENCE_SETTINGS.keys())
async def test_document_intelligence_settings_parity(
    recording_server: RecordingServer, monkeypatch: pytest.MonkeyPatch, settings: Mapping[str, object]
) -> None:
    for name, value in settings.items():
        monkeypatch.setenv(name, str(value))
        monkeypatch.setattr(document_intelligence, name, value)
    recording_server.expected_requests = 2
    python_paths, python = await analyzed_on(False, recording_server, monkeypatch)
    rust_paths, rust = await analyzed_on(True, recording_server, monkeypatch)
    assert python.kind == "ok"
    assert rust_paths == python_paths
    assert_parity(python, rust)


async def failed_analysis_on(backend: bool, server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> Outcome:
    operation: Final = f"{server.base_url}/documentintelligence/documentModels/prebuilt-layout/analyzeResults/1"
    failed: Final = {"status": "failed", "error": {"code": "InvalidContent", "message": "The file is corrupted"}}
    server.enqueue(ResponseSpec(body=None, status=202, headers={"Operation-Location": operation}))
    server.enqueue(ResponseSpec(body=failed, headers={"retry-after": "1"}))
    return await run_on(backend, monkeypatch, lambda: analyze(server))


@pytest.mark.asyncio
async def test_document_intelligence_failed_operation_parity(
    recording_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording_server.expected_requests = 4
    python: Final = await failed_analysis_on(False, recording_server, monkeypatch)
    rust: Final = await failed_analysis_on(True, recording_server, monkeypatch)
    assert python.message is not None and "The file is corrupted" in python.message
    assert_parity(python, rust)


REMOTE_DOCUMENT_POLICIES: Final = {
    "default-policy": (False, 50.0),
    "allowlisted-host": (True, 50.0),
    "allowlisted-host-downloads-disabled": (True, 0.0),
}


async def fetched_on(backend: bool, server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    start: Final = len(server.requests)
    document: Final = {"type": "document_url", "document_url": f"{server.base_url}/scan.pdf"}
    await run_on(
        backend,
        monkeypatch,
        lambda: call(server, True, PROVIDERS["azure-mistral"], {**NO_RETRIES, "document": document}),
    )
    return tuple(request.path for request in server.requests[start:] if request.method == "GET")


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", REMOTE_DOCUMENT_POLICIES.values(), ids=REMOTE_DOCUMENT_POLICIES.keys())
async def test_remote_document_fetch_parity(
    recording_server: RecordingServer, monkeypatch: pytest.MonkeyPatch, policy: tuple[bool, float]
) -> None:
    allowlisted, download_limit_mb = policy
    host: Final = str(recording_server.server.server_address[0])
    monkeypatch.setattr(litellm, "user_url_allowed_hosts", [host] if allowlisted else [])
    monkeypatch.setenv("MAX_IMAGE_URL_DOWNLOAD_SIZE_MB", str(download_limit_mb))
    monkeypatch.setattr(image_handling, "MAX_IMAGE_URL_DOWNLOAD_SIZE_MB", download_limit_mb)
    recording_server.expected_requests = None
    recording_server.default_response = ResponseSpec(body=None, raw=b"%PDF-1.4 scanned")
    python: Final = await fetched_on(False, recording_server, monkeypatch)
    rust: Final = await fetched_on(True, recording_server, monkeypatch)
    assert rust == python

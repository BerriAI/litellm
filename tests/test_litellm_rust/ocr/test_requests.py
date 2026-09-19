import json
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import (
    OCR_DOCUMENT,
    OCR_RESPONSE,
    call_native,
    call_native_aocr,
    call_native_ocr,
)

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_ocr_contract_upstream_status(
    ocr_server: RecordingServer,
    ocr_backend: bool,
    asynchronous: bool,
) -> None:
    upstream: Final = ResponseSpec(body={"detail": "invalid provider option"}, status=422)
    ocr_server.enqueue(upstream)
    arguments: Final = {
        "model": "vertex_ai/mistral-ocr-latest",
        "vertex_project": "test-project",
        "vertex_location": "us-central1",
        "num_retries": 0,
    }
    with pytest.raises(litellm.BadRequestError) as caught:
        await call_native(ocr_server, asynchronous, **arguments)
    assert caught.value.status_code == upstream.status
    assert caught.value.response.status_code == upstream.status


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("preserved", ["body", "headers"])
async def test_ocr_contract_provider_error_details(
    ocr_server: RecordingServer,
    ocr_backend: bool,
    asynchronous: bool,
    preserved: str,
) -> None:
    payload: Final = {"message": "rate limited"}
    headers: Final = {"Retry-After": "17", "X-Request-ID": "ocr-request-123", "X-Future-Header": "retained"}
    ocr_server.enqueue(ResponseSpec(body=payload, status=429, headers=headers))
    with pytest.raises(litellm.RateLimitError) as caught:
        await call_native(ocr_server, asynchronous, num_retries=0)
    response: Final = caught.value.response
    assert isinstance(response, httpx.Response)
    if preserved == "body":
        assert response.content == json.dumps(payload).encode()
    else:
        for name, value in headers.items():
            assert response.headers.get(name.lower()) == value
            assert response.headers.get(name.upper()) == value


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_ocr_contract_invalid_response_format(
    ocr_server: RecordingServer,
    ocr_backend: bool,
    asynchronous: bool,
) -> None:
    ocr_server.expected_requests = 0
    with pytest.raises(litellm.UnsupportedParamsError) as caught:
        await call_native(ocr_server, asynchronous, req_format="bogus", num_retries=0)
    assert caught.value.status_code == 400
    for value in ("req_format", "bogus", "native", "litellm"):
        assert value in str(caught.value)
    assert ocr_server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "document,field",
    [
        ([], "document"),
        ({"document_url": "https://example.com/a.pdf"}, "type"),
        ({"type": "text"}, "type"),
    ],
)
async def test_ocr_contract_malformed_document_is_actionable(
    ocr_server: RecordingServer,
    ocr_backend: bool,
    asynchronous: bool,
    document: JsonValue,
    field: str,
) -> None:
    ocr_server.expected_requests = None
    with pytest.raises(litellm.BadRequestError) as caught:
        await call_native(ocr_server, asynchronous, document=document, num_retries=0)
    assert caught.value.status_code == 400
    assert field.lower() in str(caught.value).lower()
    assert "NoneType: None" not in str(caught.value)
    assert "indices must be" not in str(caught.value)
    assert ocr_server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("option,value,field", [("pages", [-1], "pages"), ("features", [1], "features")])
async def test_ocr_contract_azure_invalid_options_are_bad_requests(
    ocr_server: RecordingServer,
    ocr_backend: bool,
    asynchronous: bool,
    option: str,
    value: JsonValue,
    field: str,
) -> None:
    ocr_server.expected_requests = 0
    arguments: Final = {"model": "azure_ai/doc-intelligence/prebuilt-read", option: value, "num_retries": 0}
    with pytest.raises(litellm.BadRequestError) as caught:
        await call_native(ocr_server, asynchronous, **arguments)
    assert caught.value.status_code == 400
    assert field in str(caught.value)
    assert ocr_server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("model", ["mistral/mistral-ocr-latest", "azure_ai/mistral-ocr-latest", "reducto/parse-v3"])
async def test_ocr_contract_native_format_supported(
    ocr_server: RecordingServer,
    ocr_backend: bool,
    asynchronous: bool,
    model: str,
) -> None:
    ocr_server.expected_requests = None
    payload: Final = (
        {"result": {"chunks": [{"content": "native OCR response"}]}, "usage": {"num_pages": 1}}
        if model.startswith("reducto/")
        else OCR_RESPONSE
    )
    ocr_server.default_response = ResponseSpec(body=payload)
    arguments: Final = {
        "model": model,
        "req_format": "native",
        "num_retries": 0,
        "document": {"type": "document_url", "document_url": "reducto://ready.pdf"}
        if model.startswith("reducto/")
        else OCR_DOCUMENT,
    }
    response: Final = (
        await call_native_aocr(ocr_server, **arguments) if asynchronous else call_native_ocr(ocr_server, **arguments)
    )
    assert response.pages[0].markdown == "native OCR response"
    assert response.get_provider_native_response() == payload
    assert len(ocr_server.requests) == 1
    if ocr_backend:
        assert_native_request(ocr_server)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_ocr_contract_unknown_reducto_model_reaches_provider(
    ocr_server: RecordingServer,
    ocr_backend: bool,
    asynchronous: bool,
) -> None:
    ocr_server.default_response = ResponseSpec(body={"result": {"chunks": [{"content": "future model response"}]}})
    arguments: Final = {
        "model": "reducto/future-parse-model",
        "document": {"type": "document_url", "document_url": "reducto://ready.pdf"},
        "num_retries": 0,
    }
    response: Final = (
        await call_native_aocr(ocr_server, **arguments) if asynchronous else call_native_ocr(ocr_server, **arguments)
    )
    assert response.model == "future-parse-model"
    assert response.pages[0].markdown == "future model response"
    assert len(ocr_server.requests) == 1
    assert ocr_server.requests[0].path == "/parse"
    assert ocr_server.requests[0].body == {"input": "reducto://ready.pdf"}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_azure_ocr_uses_token_provider_result_as_bearer_token(
    ocr_server: RecordingServer, isolated_azure_auth: None, asynchronous: bool
) -> None:
    calls: Final = []

    def token_provider() -> str:
        calls.append("token")
        return "callback-token"

    arguments: Final = {
        "model": "azure_ai/mistral-ocr-latest",
        "api_key": None,
        "azure_ad_token_provider": token_provider,
    }
    response: Final = (
        await call_native_aocr(ocr_server, **arguments) if asynchronous else call_native_ocr(ocr_server, **arguments)
    )

    assert calls == ["token"]
    assert response.pages[0].markdown == "native OCR response"
    assert_native_request(ocr_server)
    assert ocr_server.requests[0].headers["authorization"] == "Bearer callback-token"


@pytest.fixture
def ocr_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=OCR_RESPONSE)
    return recording_server


def assert_native_request(server: RecordingServer) -> None:
    assert len(server.requests) == 1
    assert not server.requests[0].headers.get("user-agent", "").startswith("python-httpx")


def test_native_ocr_maps_provider_400_with_public_provider_details(ocr_server: RecordingServer) -> None:
    ocr_server.enqueue(ResponseSpec(body={"message": "invalid OCR request"}, status=400))

    with pytest.raises(litellm.BadRequestError) as caught:
        call_native_ocr(ocr_server)

    assert caught.value.status_code == 400
    assert caught.value.model == "mistral-ocr-latest"
    assert caught.value.llm_provider == "mistral"
    assert "invalid OCR request" in str(caught.value)


class TokenAbort(BaseException):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("failure", ["ordinary", "abort"], ids=["value-error", "base-exception"])
async def test_native_azure_ocr_token_provider_failure_prevents_pre_call_callback_and_request(
    ocr_server: RecordingServer,
    isolated_azure_auth: None,
    asynchronous: bool,
    failure: str,
) -> None:
    ocr_server.expected_requests = 0
    calls: Final = []
    recorder: Final = RecordingLogger()
    original: Final = {"ordinary": ValueError("token unavailable"), "abort": TokenAbort("abort")}

    def token_provider() -> object:
        calls.append("token")
        raise original[failure]

    arguments: Final = {
        "model": "azure_ai/mistral-ocr-latest",
        "api_key": None,
        "azure_ad_token_provider": token_provider,
        "callbacks": [recorder],
    }
    expected: Final = TokenAbort if failure == "abort" else litellm.APIConnectionError
    with pytest.raises(expected) as caught:
        await call_native_aocr(ocr_server, **arguments) if asynchronous else call_native_ocr(ocr_server, **arguments)
    assert calls == ["token"]
    assert ocr_server.requests == []
    assert "log_pre_api_call" not in recorder.names
    if failure == "ordinary":
        assert "Failed to get Azure AD token: token unavailable" in str(caught.value)
        assert isinstance(caught.value.__context__, RuntimeError)
        assert caught.value.__context__.__cause__ is original[failure]
    else:
        assert caught.value is original[failure]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "override, expected_key",
    [
        ({}, "credential-key"),
        ({"api_key": "explicit-key"}, "explicit-key"),
        ({"api_key": None}, "environment-key"),
    ],
    ids=["inherit", "explicit", "explicit-none"],
)
async def test_native_ocr_inherits_named_credentials_without_overwriting_arguments(
    ocr_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    override: dict[str, object],
    expected_key: str,
) -> None:
    from litellm.models.credentials import CredentialItem

    pages: Final = [0]
    opaque: Final = object()
    monkeypatch.setenv("MISTRAL_API_KEY", "environment-key")
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(credential_name="other", credential_info={}, credential_values={"api_key": "wrong-key"}),
            CredentialItem(
                credential_name="ocr-test",
                credential_info={},
                credential_values={
                    "api_key": "credential-key",
                    "api_base": ocr_server.base_url,
                    "pages": pages,
                    "opaque": opaque,
                },
            ),
            CredentialItem(credential_name="ocr-test", credential_info={}, credential_values={"api_key": "later-key"}),
        ],
    )

    class Observer(RecordingLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            super().log_pre_api_call(model, messages, kwargs)
            pages.append(2)

    arguments: Final = {
        "model": "mistral/mistral-ocr-latest",
        "document": OCR_DOCUMENT,
        "litellm_credential_name": "ocr-test",
        "callbacks": [Observer()],
        **override,
    }
    response: Final = await litellm.aocr(**arguments) if asynchronous else litellm.ocr(**arguments)
    assert response.pages[0].markdown == "native OCR response"
    assert ocr_server.requests[0].headers["authorization"] == f"Bearer {expected_key}"
    assert ocr_server.requests[0].body["pages"] == [0, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_native_file_preparation_preserves_reader_exception(
    ocr_server: RecordingServer, asynchronous: bool
) -> None:
    ocr_server.expected_requests = 0
    failure: Final = RuntimeError("reader failed")

    class Reader:
        def read(self) -> bytes:
            raise failure

    document: Final = {"type": "file", "file": Reader()}
    with pytest.raises(litellm.APIConnectionError, match="reader failed") as caught:
        await call_native_aocr(ocr_server, document=document) if asynchronous else call_native_ocr(
            ocr_server, document=document
        )
    assert caught.value.__context__ is failure


COHERE_IMAGE: Final = {"type": "image_url", "image_url": "data:image/png;base64,YWJj"}
FILE_SIZE_LIMIT: Final = 50 * 1024 * 1024


class IntReader:
    def read(self) -> int:
        return 1


def oversized_file(tmp_path: Path) -> Path:
    path: Final = tmp_path / "large.pdf"
    with path.open("wb") as stream:
        stream.truncate(FILE_SIZE_LIMIT + 1)
    return path


def empty_token() -> str:
    return ""


def unused_token() -> str:
    raise AssertionError("the token provider must not run")


@dataclass(frozen=True, slots=True)
class PublicFailure:
    arguments: Callable[[Path], dict[str, object]]
    error: type[Exception]
    match: str
    provider_requests: int = 0
    response: ResponseSpec | None = None
    cause: type[BaseException] | None = None


PUBLIC_FAILURES: Final = {
    "unknown-req-format": PublicFailure(
        lambda _: {"req_format": "raw"}, litellm.BadRequestError, "Invalid `req_format`"
    ),
    "empty-file": PublicFailure(
        lambda _: {"document": {"type": "file", "file": BytesIO(b"")}}, litellm.BadRequestError, "File is empty"
    ),
    "oversized-file": PublicFailure(
        lambda tmp_path: {"document": {"type": "file", "file": oversized_file(tmp_path)}},
        litellm.BadRequestError,
        "exceeds the size limit",
    ),
    "missing-file": PublicFailure(
        lambda tmp_path: {"document": {"type": "file", "file": tmp_path / "missing.pdf"}},
        litellm.APIConnectionError,
        "File not found",
        cause=FileNotFoundError,
    ),
    "reader-returns-non-bytes": PublicFailure(
        lambda _: {"document": {"type": "file", "file": IntReader()}},
        litellm.APIConnectionError,
        "bytes or str",
        cause=TypeError,
    ),
    "cohere-non-image": PublicFailure(
        lambda _: {"model": "cohere/parse-v5.0"}, litellm.BadRequestError, "only accepts `image_url`"
    ),
    "cohere-unknown-format": PublicFailure(
        lambda _: {"model": "cohere/parse-v5.0", "document": COHERE_IMAGE, "output_format": "html"},
        litellm.BadRequestError,
        "output_format",
    ),
    "azure-missing-api-base": PublicFailure(
        lambda _: {
            "model": "azure_ai/mistral-ocr-latest",
            "api_key": None,
            "api_base": None,
            "azure_ad_token_provider": unused_token,
        },
        litellm.APIConnectionError,
        "Missing Azure AI API Base",
    ),
    "azure-empty-token": PublicFailure(
        lambda _: {
            "model": "azure_ai/mistral-ocr-latest",
            "api_key": None,
            "azure_ad_token": "static-token",
            "azure_ad_token_provider": empty_token,
        },
        litellm.APIConnectionError,
        "Missing Azure AI credentials",
    ),
    "upstream-500": PublicFailure(
        lambda _: {},
        litellm.InternalServerError,
        "provider unavailable",
        provider_requests=1,
        response=ResponseSpec(body={"message": "provider unavailable"}, status=500),
    ),
    "invalid-provider-response": PublicFailure(
        lambda _: {},
        litellm.APIConnectionError,
        "pages",
        provider_requests=1,
        response=ResponseSpec(body={"pages": "invalid"}),
    ),
    "response-over-limit": PublicFailure(
        lambda _: {"max_response_bytes": len(json.dumps(OCR_RESPONSE).encode()) - 1},
        litellm.APIConnectionError,
        "OCR response exceeds the size limit",
        provider_requests=1,
    ),
    "timeout": PublicFailure(
        lambda _: {"timeout": 0.01},
        litellm.Timeout,
        "",
        provider_requests=1,
        response=ResponseSpec(body=OCR_RESPONSE, delay=0.2),
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("failure", PUBLIC_FAILURES.values(), ids=PUBLIC_FAILURES.keys())
async def test_native_failures_raise_the_public_exception_class(
    ocr_server: RecordingServer,
    isolated_azure_auth: None,
    tmp_path: Path,
    asynchronous: bool,
    failure: PublicFailure,
) -> None:
    ocr_server.expected_requests = failure.provider_requests
    if failure.response is not None:
        ocr_server.enqueue(failure.response)

    with pytest.raises(failure.error, match=failure.match) as caught:
        await call_native(ocr_server, asynchronous, **failure.arguments(tmp_path))

    assert len(ocr_server.requests) == failure.provider_requests
    if failure.cause is not None:
        assert isinstance(caught.value.__context__, failure.cause)

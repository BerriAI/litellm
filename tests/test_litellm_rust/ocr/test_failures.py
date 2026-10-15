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
from tests.test_litellm_rust.ocr.data import (
    OCR_RESPONSE,
    call,
    call_aocr,
    call_ocr,
)
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_ocr_contract_upstream_status(
    server: RecordingServer,
    ocr_backend: bool,
    asynchronous: bool,
) -> None:
    upstream: Final = ResponseSpec(body={"detail": "invalid provider option"}, status=422)
    server.enqueue(upstream)
    arguments: Final = {
        "model": "vertex_ai/mistral-ocr-latest",
        "vertex_project": "test-project",
        "vertex_location": "us-central1",
        "num_retries": 0,
    }
    with pytest.raises(litellm.BadRequestError) as caught:
        await call(server, asynchronous, **arguments)
    assert caught.value.status_code == upstream.status
    assert caught.value.response.status_code == upstream.status


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("preserved", ["body", "headers"])
async def test_ocr_contract_provider_error_details(
    server: RecordingServer,
    ocr_backend: bool,
    asynchronous: bool,
    preserved: str,
) -> None:
    payload: Final = {"message": "rate limited"}
    headers: Final = {"Retry-After": "17", "X-Request-ID": "ocr-request-123", "X-Future-Header": "retained"}
    server.enqueue(ResponseSpec(body=payload, status=429, headers=headers))
    with pytest.raises(litellm.RateLimitError) as caught:
        await call(server, asynchronous, num_retries=0)
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
    server: RecordingServer,
    ocr_backend: bool,
    asynchronous: bool,
) -> None:
    server.expected_requests = 0
    with pytest.raises(litellm.UnsupportedParamsError) as caught:
        await call(server, asynchronous, req_format="bogus", num_retries=0)
    assert caught.value.status_code == 400
    for value in ("req_format", "bogus", "native", "litellm"):
        assert value in str(caught.value)
    assert server.requests == []


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
    server: RecordingServer,
    ocr_backend: bool,
    asynchronous: bool,
    document: JsonValue,
    field: str,
) -> None:
    server.expected_requests = None
    with pytest.raises(litellm.BadRequestError) as caught:
        await call(server, asynchronous, document=document, num_retries=0)
    assert caught.value.status_code == 400
    assert field.lower() in str(caught.value).lower()
    assert "NoneType: None" not in str(caught.value)
    assert "indices must be" not in str(caught.value)
    assert server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("option,value,field", [("pages", [-1], "pages"), ("features", [1], "features")])
async def test_ocr_contract_azure_invalid_options_are_bad_requests(
    server: RecordingServer,
    ocr_backend: bool,
    asynchronous: bool,
    option: str,
    value: JsonValue,
    field: str,
) -> None:
    server.expected_requests = 0
    arguments: Final = {"model": "azure_ai/doc-intelligence/prebuilt-read", option: value, "num_retries": 0}
    with pytest.raises(litellm.BadRequestError) as caught:
        await call(server, asynchronous, **arguments)
    assert caught.value.status_code == 400
    assert field in str(caught.value)
    assert server.requests == []


def test_native_ocr_maps_provider_400_with_public_provider_details(server: RecordingServer) -> None:
    server.enqueue(ResponseSpec(body={"message": "invalid OCR request"}, status=400))

    with pytest.raises(litellm.BadRequestError) as caught:
        call_ocr(server)

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
    server: RecordingServer,
    isolated_azure_auth: None,
    asynchronous: bool,
    failure: str,
) -> None:
    server.expected_requests = 0
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
        await call_aocr(server, **arguments) if asynchronous else call_ocr(server, **arguments)
    assert calls == ["token"]
    assert server.requests == []
    assert "log_pre_api_call" not in recorder.names
    if failure == "ordinary":
        assert "Failed to get Azure AD token: token unavailable" in str(caught.value)
        assert isinstance(caught.value.__context__, RuntimeError)
        assert caught.value.__context__.__cause__ is original[failure]
    else:
        assert caught.value is original[failure]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_native_file_preparation_preserves_reader_exception(server: RecordingServer, asynchronous: bool) -> None:
    server.expected_requests = 0
    failure: Final = RuntimeError("reader failed")

    class Reader:
        def read(self) -> bytes:
            raise failure

    document: Final = {"type": "file", "file": Reader()}
    with pytest.raises(litellm.APIConnectionError, match="reader failed") as caught:
        await call_aocr(server, document=document) if asynchronous else call_ocr(server, document=document)
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
    server: RecordingServer,
    isolated_azure_auth: None,
    tmp_path: Path,
    asynchronous: bool,
    failure: PublicFailure,
) -> None:
    server.expected_requests = failure.provider_requests
    if failure.response is not None:
        server.enqueue(failure.response)

    with pytest.raises(failure.error, match=failure.match) as caught:
        await call(server, asynchronous, **failure.arguments(tmp_path))

    assert len(server.requests) == failure.provider_requests
    if failure.cause is not None:
        assert isinstance(caught.value.__context__, failure.cause)

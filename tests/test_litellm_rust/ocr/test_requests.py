import json
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Final, NoReturn

import httpx
import pytest
from pydantic import JsonValue

import litellm
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


@pytest.fixture(params=[False, True], ids=["python", "rust"])
def ocr_backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> bool:
    enabled: Final = bool(request.param)
    monkeypatch.setenv("LITELLM_RUST", "1" if enabled else "0")
    return enabled


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


def test_native_ocr_encodes_python_file_input_and_drops_unknown_arguments(ocr_server: RecordingServer) -> None:
    response: Final = call_native_ocr(
        ocr_server,
        document={"type": "file", "file": BytesIO(b"abc"), "mime_type": "image/png"},
        opaque_extension=object(),
    )

    assert response.pages[0].markdown == "native OCR response"
    assert_native_request(ocr_server)
    assert ocr_server.requests[0].body == {
        "model": "mistral-ocr-latest",
        "document": {"type": "image_url", "image_url": "data:image/png;base64,YWJj"},
    }


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


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "name,value",
    [
        ("ssl_verify", object()),
        ("ssl_certificate", 1),
        ("ssl_certificate", ""),
        ("vertex_project", 1),
        ("vertex_location", ["region"]),
        ("user_url_allowed_hosts", ["example.test", 1]),
    ],
)
async def test_native_settings_fail_before_provider_io(
    ocr_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    name: str,
    value: object,
) -> None:
    ocr_server.expected_requests = 0
    monkeypatch.setattr(litellm, name, value)
    with pytest.raises(ValueError, match=r"http_settings|provider_defaults|url_policy"):
        await call_native(ocr_server, asynchronous, num_retries=0)
    assert ocr_server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_ssl_context_is_terminal_configuration(ocr_server: RecordingServer, asynchronous: bool) -> None:
    import ssl

    ocr_server.expected_requests = 0
    context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with pytest.raises(ValueError, match=r"request\.ssl_verify.*SSLContext"):
        await call_native(ocr_server, asynchronous, ssl_verify=context, num_retries=0)
    assert ocr_server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_settings_preserve_protocol_failures(
    ocr_server: RecordingServer, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    ocr_server.expected_requests = 0
    failure: Final = LookupError("settings truth test failed")
    cause: Final = RuntimeError("settings cause")

    class RaisesBool:
        def __bool__(self) -> bool:
            raise failure from cause

    monkeypatch.setattr(litellm, "force_ipv4", RaisesBool())
    with pytest.raises(LookupError) as caught:
        await call_native(ocr_server, asynchronous, num_retries=0)
    assert caught.value is failure
    assert caught.value.__cause__ is cause
    assert caught.value.__traceback__ is not None
    assert ocr_server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_settings_observe_mutation_between_calls(
    ocr_server: RecordingServer, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    monkeypatch.setattr(litellm, "force_ipv4", "yes")
    monkeypatch.setattr(litellm, "http2", 1)
    monkeypatch.setattr(litellm, "vertex_project", [])
    monkeypatch.setattr(litellm, "vertex_location", 0)
    monkeypatch.setattr(litellm, "user_url_allowed_hosts", "EXAMPLE.TEST.")
    response: Final = await call_native(ocr_server, asynchronous, num_retries=0)
    assert response.pages[0].markdown == "native OCR response"
    assert_native_request(ocr_server)
    monkeypatch.setattr(litellm, "ssl_certificate", 1)
    with pytest.raises(ValueError, match=r"http_settings\.ssl_certificate"):
        await call_native(ocr_server, asynchronous, num_retries=0)
    assert len(ocr_server.requests) == 1


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("failure", ["invalid", "live", "schema"])
def test_native_projection_errors_never_select_python(
    ocr_server: RecordingServer, monkeypatch: pytest.MonkeyPatch, required: bool, failure: str
) -> None:
    import dataclasses
    import ssl

    from litellm.rust_bridge import runtime, settings
    from litellm.rust_bridge.catalog import Route, RouteContext, RouteRule
    from litellm.rust_bridge.configuration import Rollout
    from litellm.rust_bridge.ocr.entrypoints import NATIVE_OCR, LiteLLMOcrRequest

    ocr_server.expected_requests = 0
    snapshot: Final = dataclasses.replace(settings.http_settings(), user_agent=1)
    if failure == "schema":
        monkeypatch.setattr(settings, "http_settings", lambda: snapshot)
    else:
        monkeypatch.setattr(
            litellm, "ssl_verify", ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT) if failure == "live" else object()
        )
    request: Final = LiteLLMOcrRequest(
        model="mistral/mistral-ocr-latest",
        document=OCR_DOCUMENT,
        api_key="test-key",
        api_base=ocr_server.base_url,
        timeout=None,
        custom_llm_provider="mistral",
        extra_headers=None,
        kwargs={},
    )

    def python_fallback() -> NoReturn:
        pytest.fail("projection failures must not select Python")

    with pytest.raises(RuntimeError if failure == "schema" else ValueError, match="http_settings"):
        runtime.run(
            RouteContext(Route.OCR, provider="mistral"),
            binding=NATIVE_OCR,
            native=lambda native: native(request, (), {}),
            python=python_fallback,
            rules=(RouteRule(Route.OCR, Rollout.RUST_REQUIRED if required else Rollout.RUST_OPT_OUT),),
        )
    assert ocr_server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("present", [False, True], ids=["missing", "invalid-pem"])
async def test_native_client_certificate_is_validated_before_io(
    ocr_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    asynchronous: bool,
    present: bool,
) -> None:
    ocr_server.expected_requests = 0
    certificate: Final = tmp_path / "client.pem"
    if present:
        certificate.write_text("invalid certificate")
    monkeypatch.setattr(litellm, "ssl_certificate", str(certificate))
    with pytest.raises(ValueError, match=r"http_settings\.ssl_certificate.*PEM") as caught:
        await call_native(ocr_server, asynchronous, num_retries=0)
    assert str(certificate) not in str(caught.value)
    assert ocr_server.requests == []

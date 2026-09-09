from typing import Final

import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from tests.test_litellm_rust.support.requests import (
    OCR_DOCUMENT,
    OCR_MODEL,
    OCR_RESPONSE,
    call_native_ocr,
    call_aocr,
    call_ocr as call_public_ocr,
)
from litellm.rust_bridge.provenance import has_rust_response_marker
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_azure_ocr_calls_python_token_provider(
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
        await call_aocr(ocr_server, **arguments) if asynchronous else call_public_ocr(ocr_server, **arguments)
    )

    assert calls == ["token"]
    assert has_rust_response_marker(response)
    assert response.pages[0].markdown == "native OCR response"
    assert_native_request(ocr_server)
    assert ocr_server.requests[0].headers["authorization"] == "Bearer callback-token"


@pytest.fixture
def ocr_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=OCR_RESPONSE)
    return recording_server


def call_ocr(server: RecordingServer, **kwargs: object) -> OCRResponse:
    return call_native_ocr(server, **kwargs)


def assert_native_request(server: RecordingServer) -> None:
    assert len(server.requests) == 1
    assert not server.requests[0].headers.get("user-agent", "").startswith("python-httpx")


def test_ocr_sends_expected_provider_request(ocr_server: RecordingServer) -> None:
    response: Final = call_ocr(ocr_server)

    assert response.pages[0].markdown == "native OCR response"
    assert_native_request(ocr_server)
    assert ocr_server.requests[0].path == "/v1/ocr"
    assert ocr_server.requests[0].body == {"model": "mistral-ocr-latest", "document": OCR_DOCUMENT}


def test_ocr_rejects_unsupported_file_document_before_callbacks(ocr_server: RecordingServer) -> None:
    ocr_server.expected_requests = 0
    recorder: Final = RecordingLogger()

    with pytest.raises(NotImplementedError, match="OCR file document preparation"):
        call_native_ocr(
            ocr_server,
            document={"type": "file", "file": b"%PDF-1.4", "mime_type": "application/pdf"},
            callbacks=[recorder],
        )

    assert ocr_server.requests == []
    assert recorder.events == ()


def test_ocr_sends_optional_parameters(ocr_server: RecordingServer) -> None:
    call_ocr(ocr_server, pages=[0, 2], include_image_base64=True)

    assert ocr_server.requests[0].body["pages"] == [0, 2]
    assert ocr_server.requests[0].body["include_image_base64"] is True


def test_ocr_sends_custom_headers(ocr_server: RecordingServer) -> None:
    call_ocr(ocr_server, extra_headers={"x-trace-id": "trace-1"})

    assert ocr_server.requests[0].headers["authorization"] == "Bearer test-key"
    assert ocr_server.requests[0].headers["x-trace-id"] == "trace-1"


def test_ocr_resolves_provider_credentials(ocr_server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "environment-key")

    call_native_ocr(ocr_server, api_key=None)

    assert ocr_server.requests[0].headers["authorization"] == "Bearer environment-key"


def test_ocr_explicit_credentials_override_defaults(
    ocr_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "environment-key")

    call_ocr(ocr_server)

    assert ocr_server.requests[0].headers["authorization"] == "Bearer test-key"


def test_ocr_resolves_provider_endpoint(ocr_server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_AI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_AI_API_BASE", ocr_server.base_url)

    call_native_ocr(ocr_server, model="azure_ai/pixtral-12b-2409", api_key=None, api_base=None)

    assert_native_request(ocr_server)
    assert ocr_server.requests[0].path == "/providers/mistral/azure/ocr"
    assert ocr_server.requests[0].headers["authorization"] == "Bearer azure-key"


def test_ocr_resolves_vertex_project_and_location(ocr_server: RecordingServer) -> None:
    call_native_ocr(
        ocr_server,
        model="vertex_ai/mistral-ocr-2505",
        api_key="vertex-token",
        vertex_project="project-1",
        vertex_location="us-central1",
    )

    assert_native_request(ocr_server)
    assert ocr_server.requests[0].path == (
        "/v1/projects/project-1/locations/us-central1/publishers/mistralai/models/mistral-ocr-2505:rawPredict"
    )


def test_ocr_returns_normalized_response(ocr_server: RecordingServer) -> None:
    response: Final = call_ocr(ocr_server)

    assert isinstance(response, OCRResponse)
    assert response.model == "mistral-ocr-latest"
    assert response.usage_info.pages_processed == 1


def test_ocr_provider_error_preserves_status_and_context(ocr_server: RecordingServer) -> None:
    ocr_server.enqueue(ResponseSpec(body={"message": "invalid OCR request"}, status=400))

    with pytest.raises(litellm.BadRequestError) as caught:
        call_ocr(ocr_server)

    assert caught.value.status_code == 400
    assert caught.value.model == "mistral-ocr-latest"
    assert caught.value.llm_provider == "mistral"
    assert "invalid OCR request" not in str(caught.value)


def test_ocr_honors_request_timeout(ocr_server: RecordingServer) -> None:
    ocr_server.enqueue(ResponseSpec(body=OCR_RESPONSE, delay=0.2))

    with pytest.raises(RuntimeError, match="OCR transport failed"):
        call_ocr(ocr_server, timeout=0.01)

    assert len(ocr_server.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize(
    "credentials, expected_token, expected_calls",
    [
        ({"api_key": "resource-key"}, "resource-key", 0),
        ({"azure_ad_token": "static-token"}, "callback-1", 1),
        ({"extra_headers": {"Authorization": "Bearer override"}}, "override", 1),
    ],
)
async def test_azure_ocr_token_provider_precedence(
    ocr_server: RecordingServer,
    isolated_azure_auth: None,
    asynchronous: bool,
    backend: str,
    credentials: dict[str, object],
    expected_token: str,
    expected_calls: int,
) -> None:
    litellm.rust(backend == "rust")
    calls: Final = []

    def token_provider() -> str:
        calls.append("token")
        return f"callback-{len(calls)}"

    arguments: Final = {
        "model": "azure_ai/mistral-ocr-latest",
        "api_key": None,
        "azure_ad_token_provider": token_provider,
        **credentials,
    }
    response: Final = (
        await call_aocr(ocr_server, **arguments) if asynchronous else call_public_ocr(ocr_server, **arguments)
    )
    assert has_rust_response_marker(response) == (backend == "rust")
    assert len(calls) == expected_calls
    assert len(ocr_server.requests) == 1
    assert ocr_server.requests[0].headers["authorization"] == f"Bearer {expected_token}"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_azure_ocr_does_not_cache_caller_tokens(
    ocr_server: RecordingServer,
    isolated_azure_auth: None,
    asynchronous: bool,
) -> None:
    calls: Final = []
    ocr_server.expected_requests = 2

    def token_provider() -> str:
        calls.append("token")
        return f"callback-{len(calls)}"

    for _ in range(2):
        arguments: Final = {
            "model": "azure_ai/mistral-ocr-latest",
            "api_key": None,
            "azure_ad_token_provider": token_provider,
        }
        response: Final = (
            await call_aocr(ocr_server, **arguments) if asynchronous else call_public_ocr(ocr_server, **arguments)
        )
        assert has_rust_response_marker(response)
    assert len(calls) == 2
    assert [request.headers["authorization"] for request in ocr_server.requests] == [
        "Bearer callback-1",
        "Bearer callback-2",
    ]


class TokenAbort(BaseException):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("failure", ["non_string", "type_error", "ordinary", "abort"])
async def test_azure_ocr_token_failure_stops_execution(
    ocr_server: RecordingServer,
    isolated_azure_auth: None,
    asynchronous: bool,
    backend: str,
    failure: str,
) -> None:
    litellm.rust(backend == "rust")
    ocr_server.expected_requests = 0
    calls: Final = []
    recorder: Final = RecordingLogger()
    original: Final = {
        "type_error": TypeError("token type"),
        "ordinary": ValueError("token unavailable"),
        "abort": TokenAbort("abort"),
    }

    def token_provider() -> object:
        calls.append("token")
        if failure == "non_string":
            return 123
        raise original[failure]

    arguments: Final = {
        "model": "azure_ai/mistral-ocr-latest",
        "api_key": None,
        "azure_ad_token_provider": token_provider,
        "callbacks": [recorder],
    }
    expected: Final = TokenAbort if failure == "abort" else litellm.APIConnectionError
    with pytest.raises(expected) as caught:
        await call_aocr(ocr_server, **arguments) if asynchronous else call_public_ocr(ocr_server, **arguments)
    assert calls == ["token"]
    assert ocr_server.requests == []
    assert "log_pre_api_call" not in recorder.names
    if failure == "ordinary":
        assert "Failed to get Azure AD token: token unavailable" in str(caught.value)
        assert isinstance(caught.value.__context__, RuntimeError)
        assert caught.value.__context__.__cause__ is original[failure]
    elif failure == "abort":
        assert caught.value is original[failure]
    elif failure == "type_error":
        assert caught.value.__context__ is original[failure]
    else:
        assert isinstance(caught.value.__context__, TypeError)


@pytest.mark.parametrize(
    "configuration",
    [
        {"azure_ad_token": "oidc/assertion", "client_id": "client", "tenant_id": "tenant"},
        {"model": "azure_ai/doc-intelligence/prebuilt-read"},
        {"document": {"type": "file", "file": b"pdf"}},
    ],
)
def test_azure_unsupported_native_auth_does_not_invoke_provider(
    ocr_server: RecordingServer,
    isolated_azure_auth: None,
    configuration: dict[str, object],
) -> None:
    ocr_server.expected_requests = 0
    calls: Final = []
    recorder: Final = RecordingLogger()

    def provider() -> str:
        calls.append("token")
        return "unused"

    arguments: Final = {
        "model": "azure_ai/mistral-ocr-latest",
        "api_key": None,
        "azure_ad_token_provider": provider,
        "callbacks": [recorder],
        **configuration,
    }
    with pytest.raises(NotImplementedError):
        call_native_ocr(ocr_server, **arguments)
    assert calls == []
    assert recorder.events == ()
    assert ocr_server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["python", "rust"])
async def test_azure_missing_endpoint_prevents_token_callback(
    ocr_server: RecordingServer,
    isolated_azure_auth: None,
    backend: str,
) -> None:
    litellm.rust(backend == "rust")
    ocr_server.expected_requests = 0
    calls: Final = []

    def provider() -> str:
        calls.append("token")
        return "unused"

    with pytest.raises(litellm.APIConnectionError, match="Missing Azure AI API Base"):
        await call_aocr(
            ocr_server,
            model="azure_ai/mistral-ocr-latest",
            api_key=None,
            api_base=None,
            azure_ad_token_provider=provider,
        )
    assert calls == []
    assert ocr_server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["python", "rust"])
async def test_azure_empty_callback_token_does_not_restore_static_token(
    ocr_server: RecordingServer,
    isolated_azure_auth: None,
    backend: str,
) -> None:
    litellm.rust(backend == "rust")
    ocr_server.expected_requests = 0

    def provider() -> str:
        return ""

    with pytest.raises(litellm.APIConnectionError, match="Missing Azure AI credentials"):
        await call_aocr(
            ocr_server,
            model="azure_ai/mistral-ocr-latest",
            api_key=None,
            azure_ad_token="static-token",
            azure_ad_token_provider=provider,
        )
    assert ocr_server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["python", "rust"])
async def test_azure_falsey_callable_leaves_static_token_unchanged(
    ocr_server: RecordingServer,
    isolated_azure_auth: None,
    backend: str,
) -> None:
    litellm.rust(backend == "rust")
    calls: Final = []

    class Provider:
        def __bool__(self) -> bool:
            return False

        def __call__(self) -> str:
            calls.append("token")
            return "unused"

    response: Final = await call_aocr(
        ocr_server,
        model="azure_ai/mistral-ocr-latest",
        api_key=None,
        azure_ad_token="static-token",
        azure_ad_token_provider=Provider(),
    )
    assert has_rust_response_marker(response) == (backend == "rust")
    assert calls == []
    assert ocr_server.requests[0].headers["authorization"] == "Bearer static-token"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["python", "rust"])
async def test_azure_token_callback_does_not_await_coroutine_result(
    ocr_server: RecordingServer,
    isolated_azure_auth: None,
    backend: str,
) -> None:
    litellm.rust(backend == "rust")
    ocr_server.expected_requests = 0
    calls: Final = []

    async def acquire() -> str:
        calls.append("awaited")
        return "unused"

    coroutine: Final = acquire()

    def provider() -> object:
        return coroutine

    try:
        with pytest.raises(litellm.APIConnectionError, match="Azure AD token must be a string"):
            await call_aocr(
                ocr_server, model="azure_ai/mistral-ocr-latest", api_key=None, azure_ad_token_provider=provider
            )
    finally:
        coroutine.close()
    assert calls == []
    assert ocr_server.requests == []

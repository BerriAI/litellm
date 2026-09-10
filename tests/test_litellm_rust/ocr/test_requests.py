from typing import Final

import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests.test_litellm_rust.support.requests import (
    OCR_DOCUMENT,
    OCR_RESPONSE,
    call_native_aocr,
    call_native_ocr,
)
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension


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


def test_native_ocr_sends_model_and_document_to_mistral_ocr_path(ocr_server: RecordingServer) -> None:
    response: Final = call_native_ocr(ocr_server)

    assert response.pages[0].markdown == "native OCR response"
    assert_native_request(ocr_server)
    assert ocr_server.requests[0].path == "/v1/ocr"
    assert ocr_server.requests[0].body == {"model": "mistral-ocr-latest", "document": OCR_DOCUMENT}


def test_native_ocr_prepares_file_document_like_python(ocr_server: RecordingServer) -> None:
    response: Final = call_native_ocr(
        ocr_server,
        document={"type": "file", "file": b"%PDF-1.4", "mime_type": "application/pdf"},
    )

    assert response.pages[0].markdown == "native OCR response"
    assert_native_request(ocr_server)
    assert ocr_server.requests[0].body == {
        "model": "mistral-ocr-latest",
        "document": {
            "type": "document_url",
            "document_url": "data:application/pdf;base64,JVBERi0xLjQ=",
        },
    }


def test_native_ocr_sends_pages_and_image_options(ocr_server: RecordingServer) -> None:
    call_native_ocr(ocr_server, pages=[0, 2], include_image_base64=True)

    assert ocr_server.requests[0].body["pages"] == [0, 2]
    assert ocr_server.requests[0].body["include_image_base64"] is True


def test_native_ocr_merges_custom_headers_with_authorization(ocr_server: RecordingServer) -> None:
    call_native_ocr(ocr_server, extra_headers={"x-trace-id": "trace-1"})

    assert ocr_server.requests[0].headers["authorization"] == "Bearer test-key"
    assert ocr_server.requests[0].headers["x-trace-id"] == "trace-1"


def test_native_mistral_ocr_uses_environment_api_key_when_argument_is_missing(
    ocr_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "environment-key")

    call_native_ocr(ocr_server, api_key=None)

    assert ocr_server.requests[0].headers["authorization"] == "Bearer environment-key"


def test_native_mistral_ocr_prefers_explicit_api_key_over_environment(
    ocr_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "environment-key")

    call_native_ocr(ocr_server)

    assert ocr_server.requests[0].headers["authorization"] == "Bearer test-key"


def test_native_azure_ocr_uses_environment_endpoint_and_api_key(
    ocr_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AZURE_AI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_AI_API_BASE", ocr_server.base_url)

    call_native_ocr(ocr_server, model="azure_ai/pixtral-12b-2409", api_key=None, api_base=None)

    assert_native_request(ocr_server)
    assert ocr_server.requests[0].path == "/providers/mistral/azure/ocr"
    assert ocr_server.requests[0].headers["authorization"] == "Bearer azure-key"


def test_native_vertex_ocr_builds_path_from_project_and_location(ocr_server: RecordingServer) -> None:
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


def test_native_ocr_normalizes_provider_response_model_and_usage(ocr_server: RecordingServer) -> None:
    response: Final = call_native_ocr(ocr_server)

    assert isinstance(response, OCRResponse)
    assert response.model == "mistral-ocr-latest"
    assert response.usage_info.pages_processed == 1


def test_native_ocr_maps_provider_400_without_exposing_response_body(ocr_server: RecordingServer) -> None:
    ocr_server.enqueue(ResponseSpec(body={"message": "invalid OCR request"}, status=400))

    with pytest.raises(litellm.BadRequestError) as caught:
        call_native_ocr(ocr_server)

    assert caught.value.status_code == 400
    assert caught.value.model == "mistral-ocr-latest"
    assert caught.value.llm_provider == "mistral"
    assert "invalid OCR request" not in str(caught.value)


def test_native_ocr_raises_transport_error_when_request_exceeds_timeout(ocr_server: RecordingServer) -> None:
    ocr_server.enqueue(ResponseSpec(body=OCR_RESPONSE, delay=0.2))

    with pytest.raises(RuntimeError, match="OCR transport failed"):
        call_native_ocr(ocr_server, timeout=0.01)

    assert len(ocr_server.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "credentials, expected_token, expected_calls",
    [
        ({"api_key": "resource-key"}, "resource-key", 0),
        ({"azure_ad_token": "static-token"}, "callback-1", 1),
        ({"extra_headers": {"Authorization": "Bearer override"}}, "override", 1),
    ],
    ids=["api-key-skips-provider", "provider-overrides-static-token", "header-overrides-provider"],
)
async def test_native_azure_ocr_applies_python_credential_precedence(
    ocr_server: RecordingServer,
    isolated_azure_auth: None,
    asynchronous: bool,
    credentials: dict[str, object],
    expected_token: str,
    expected_calls: int,
) -> None:
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
        await call_native_aocr(ocr_server, **arguments) if asynchronous else call_native_ocr(ocr_server, **arguments)
    )
    assert response.pages[0].markdown == "native OCR response"
    assert len(calls) == expected_calls
    assert len(ocr_server.requests) == 1
    assert ocr_server.requests[0].headers["authorization"] == f"Bearer {expected_token}"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_azure_ocr_calls_token_provider_for_each_request(
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
            await call_native_aocr(ocr_server, **arguments)
            if asynchronous
            else call_native_ocr(ocr_server, **arguments)
        )
        assert response.pages[0].markdown == "native OCR response"
    assert len(calls) == 2
    assert [request.headers["authorization"] for request in ocr_server.requests] == [
        "Bearer callback-1",
        "Bearer callback-2",
    ]


class TokenAbort(BaseException):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "failure",
    ["non_string", "type_error", "ordinary", "abort"],
    ids=["non-string-result", "type-error", "value-error", "base-exception"],
)
async def test_native_azure_ocr_token_provider_failure_prevents_pre_call_callback_and_request(
    ocr_server: RecordingServer,
    isolated_azure_auth: None,
    asynchronous: bool,
    failure: str,
) -> None:
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
        await call_native_aocr(ocr_server, **arguments) if asynchronous else call_native_ocr(ocr_server, **arguments)
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
    ],
    ids=["oidc-assertion", "document-intelligence-model"],
)
def test_native_azure_ocr_rejects_unsupported_configuration_before_token_or_callbacks(
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
async def test_native_azure_ocr_validates_endpoint_before_calling_token_provider(
    ocr_server: RecordingServer,
    isolated_azure_auth: None,
) -> None:
    ocr_server.expected_requests = 0
    calls: Final = []

    def provider() -> str:
        calls.append("token")
        return "unused"

    with pytest.raises(litellm.APIConnectionError, match="Missing Azure AI API Base"):
        await call_native_aocr(
            ocr_server,
            model="azure_ai/mistral-ocr-latest",
            api_key=None,
            api_base=None,
            azure_ad_token_provider=provider,
        )
    assert calls == []
    assert ocr_server.requests == []


@pytest.mark.asyncio
async def test_native_azure_ocr_does_not_fall_back_to_static_token_after_empty_provider_result(
    ocr_server: RecordingServer,
    isolated_azure_auth: None,
) -> None:
    ocr_server.expected_requests = 0

    def provider() -> str:
        return ""

    with pytest.raises(litellm.APIConnectionError, match="Missing Azure AI credentials"):
        await call_native_aocr(
            ocr_server,
            model="azure_ai/mistral-ocr-latest",
            api_key=None,
            azure_ad_token="static-token",
            azure_ad_token_provider=provider,
        )
    assert ocr_server.requests == []


@pytest.mark.asyncio
async def test_native_azure_ocr_ignores_falsey_token_provider_and_uses_static_token(
    ocr_server: RecordingServer,
    isolated_azure_auth: None,
) -> None:
    calls: Final = []

    class Provider:
        def __bool__(self) -> bool:
            return False

        def __call__(self) -> str:
            calls.append("token")
            return "unused"

    response: Final = await call_native_aocr(
        ocr_server,
        model="azure_ai/mistral-ocr-latest",
        api_key=None,
        azure_ad_token="static-token",
        azure_ad_token_provider=Provider(),
    )
    assert response.pages[0].markdown == "native OCR response"
    assert calls == []
    assert ocr_server.requests[0].headers["authorization"] == "Bearer static-token"


@pytest.mark.asyncio
async def test_native_azure_ocr_rejects_coroutine_returned_by_sync_token_provider(
    ocr_server: RecordingServer,
    isolated_azure_auth: None,
) -> None:
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
            await call_native_aocr(
                ocr_server, model="azure_ai/mistral-ocr-latest", api_key=None, azure_ad_token_provider=provider
            )
    finally:
        coroutine.close()
    assert calls == []
    assert ocr_server.requests == []

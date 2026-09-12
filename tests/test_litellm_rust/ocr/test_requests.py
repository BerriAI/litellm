from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import (
    OCR_DOCUMENT,
    OCR_RESPONSE,
    call_native_aocr,
    call_native_ocr,
)

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


def test_native_ocr_reads_sdk_path_input(ocr_server: RecordingServer, tmp_path: Path) -> None:
    document_path: Final = tmp_path / "document.pdf"
    document_path.write_bytes(b"%PDF-1.4")

    response: Final = call_native_ocr(
        ocr_server,
        document={"type": "file", "file": document_path},
    )

    assert response.pages[0].markdown == "native OCR response"
    assert ocr_server.requests[0].body["document"] == {
        "type": "document_url",
        "document_url": "data:application/pdf;base64,JVBERi0xLjQ=",
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


def test_native_ocr_maps_provider_400_with_public_provider_details(ocr_server: RecordingServer) -> None:
    ocr_server.enqueue(ResponseSpec(body={"message": "invalid OCR request"}, status=400))

    with pytest.raises(litellm.BadRequestError) as caught:
        call_native_ocr(ocr_server)

    assert caught.value.status_code == 400
    assert caught.value.model == "mistral-ocr-latest"
    assert caught.value.llm_provider == "mistral"
    assert "invalid OCR request" in str(caught.value)


def test_native_ocr_rejects_unknown_response_format_before_provider_request(ocr_server: RecordingServer) -> None:
    ocr_server.expected_requests = 0

    with pytest.raises(litellm.BadRequestError, match="Invalid `req_format`"):
        call_native_ocr(ocr_server, req_format="raw")

    assert ocr_server.requests == []


def test_ocr_raises_public_timeout_when_request_exceeds_timeout(ocr_server: RecordingServer) -> None:
    litellm.rust(True)
    ocr_server.enqueue(ResponseSpec(body=OCR_RESPONSE, delay=0.2))

    with pytest.raises(litellm.Timeout):
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
    [{"azure_ad_token": "oidc/assertion", "client_id": "client", "tenant_id": "tenant"}],
    ids=["invalid-oidc-assertion"],
)
def test_public_azure_ocr_maps_invalid_oidc_configuration_before_token_or_request(
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
    with pytest.raises(litellm.APIConnectionError):
        call_native_ocr(ocr_server, **arguments)
    assert calls == []
    assert "log_pre_api_call" not in recorder.names
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


@pytest.mark.parametrize("source", ["sdk", "proxy"])
@pytest.mark.parametrize(
    "filename,mime", [("scan.PNG", "image/png"), ("document.pdf", "application/pdf"), ("note.txt", "text/plain")]
)
def test_ocr_file_helpers_use_native_document_preparation(source: str, filename: str, mime: str) -> None:
    from io import BytesIO

    from litellm.ocr.input import convert_file_document_to_url_document, get_mime_type
    from litellm.proxy.ocr_endpoints.endpoints import _build_document_from_upload

    file: Final = BytesIO(b"abc")
    file.name = filename
    document: Final = (
        convert_file_document_to_url_document({"type": "file", "file": file})
        if source == "sdk"
        else _build_document_from_upload(b"abc", filename, "application/octet-stream; charset=utf-8")
    )
    field: Final = "image_url" if mime.startswith("image/") else "document_url"
    assert get_mime_type(filename) == mime
    assert document == {"type": field, field: f"data:{mime};base64,YWJj"}


@pytest.mark.parametrize("attribute", ["read", "name"])
def test_native_file_preparation_preserves_property_errors(attribute: str) -> None:
    from litellm.ocr.input import convert_file_document_to_url_document

    failure: Final = LookupError("file property failed")

    class File:
        def __getattribute__(self, name: str):
            if name == attribute:
                raise failure
            return super().__getattribute__(name)

        def read(self):
            return b"abc"

    with pytest.raises(LookupError) as caught:
        convert_file_document_to_url_document({"type": "file", "file": File()})
    assert caught.value is failure


@pytest.mark.parametrize("kind", ["bytes", "path", "reader"])
def test_native_file_preparation_rejects_oversized_input(kind: str, tmp_path: Path) -> None:
    from litellm.ocr.input import FileDocument, convert_file_document_to_url_document, get_max_file_bytes

    limit: Final = get_max_file_bytes()
    path: Final = tmp_path / "large.pdf"
    with path.open("wb") as stream:
        stream.truncate(limit + 1)

    class Reader:
        def read(self) -> bytes:
            return b"a" * (limit + 1)

    document: Final[FileDocument] = {
        "type": "file",
        "file": path if kind == "path" else Reader() if kind == "reader" else b"a" * (limit + 1),
    }
    with pytest.raises(ValueError, match="exceeds the size limit"):
        convert_file_document_to_url_document(document)


@pytest.mark.parametrize("kind", ["str", "path", "reader"])
def test_native_upload_binding_rejects_filesystem_inputs(kind: str, tmp_path: Path) -> None:
    from io import BytesIO
    from typing import cast  # noqa: TID251  # deliberately invalid inputs exercise the native runtime boundary

    from litellm.ocr.input import convert_upload_to_url_document

    path: Final = tmp_path / "secret.pdf"
    path.write_bytes(b"server secret")
    source: Final = str(path) if kind == "str" else path if kind == "path" else BytesIO(b"abc")
    with pytest.raises(TypeError):
        convert_upload_to_url_document(cast(bytes, source), "document.pdf", None)


@pytest.mark.parametrize("extra_bytes", [0, 1])
def test_native_upload_enforces_file_size_limit(extra_bytes: int) -> None:
    import base64

    from litellm.ocr.input import convert_upload_to_url_document, get_max_file_bytes

    content: Final = b"a" * (get_max_file_bytes() + extra_bytes)
    if extra_bytes:
        with pytest.raises(ValueError, match="exceeds the size limit"):
            convert_upload_to_url_document(content, "scan.pdf", None)
        return
    document: Final = convert_upload_to_url_document(content, "scan.pdf", None)
    assert document["type"] == "document_url"
    assert base64.b64decode(document["document_url"].split(",", 1)[1]) == content


def test_native_file_preparation_preserves_reader_exception() -> None:
    from litellm.ocr.input import convert_file_document_to_url_document

    failure: Final = RuntimeError("reader failed")

    class Reader:
        def read(self) -> bytes:
            raise failure

    with pytest.raises(RuntimeError) as caught:
        convert_file_document_to_url_document({"type": "file", "file": Reader()})
    assert caught.value is failure

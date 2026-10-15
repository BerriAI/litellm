from pathlib import Path
from typing import Final

import pytest

import litellm
from tests.test_litellm_rust.ocr.data import (
    OCR_DOCUMENT,
    call,
)
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests.test_litellm_rust.support.recording_server import RecordingServer
from tests.test_litellm_rust.support.routing import assert_served_natively

pytestmark = pytest.mark.requires_rust_extension


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
    server: RecordingServer,
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
                    "api_base": server.base_url,
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
    assert server.requests[0].headers["authorization"] == f"Bearer {expected_key}"
    assert server.requests[0].body["pages"] == [0, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "name,value",
    [
        ("ssl_verify", object()),
        ("ssl_certificate", 1),
        ("ssl_certificate", ""),
        ("user_url_allowed_hosts", ["example.test", 1]),
    ],
)
async def test_native_settings_fail_before_provider_io(
    server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    name: str,
    value: object,
) -> None:
    server.expected_requests = 0
    monkeypatch.setattr(litellm, name, value)
    with pytest.raises(ValueError, match=r"http_settings|url_policy"):
        await call(server, asynchronous, num_retries=0)
    assert server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "name,value",
    [
        ("vertex_project", 1),
        ("vertex_location", ["region"]),
    ],
)
async def test_native_vertex_globals_fail_before_provider_io_only_for_vertex_calls(
    server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    name: str,
    value: object,
) -> None:
    server.expected_requests = 1
    monkeypatch.setattr(litellm, name, value)
    await call(server, asynchronous, num_retries=0)
    with pytest.raises(ValueError, match=r"provider_defaults"):
        await call(server, asynchronous, model="vertex_ai/mistral-ocr-latest", num_retries=0)
    assert len(server.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_ssl_context_is_terminal_configuration(server: RecordingServer, asynchronous: bool) -> None:
    import ssl

    server.expected_requests = 0
    context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with pytest.raises(ValueError, match=r"request\.ssl_verify.*SSLContext"):
        await call(server, asynchronous, ssl_verify=context, num_retries=0)
    assert server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_settings_preserve_protocol_failures(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    server.expected_requests = 0
    failure: Final = LookupError("settings truth test failed")
    cause: Final = RuntimeError("settings cause")

    class RaisesBool:
        def __bool__(self) -> bool:
            raise failure from cause

    monkeypatch.setattr(litellm, "force_ipv4", RaisesBool())
    with pytest.raises(LookupError) as caught:
        await call(server, asynchronous, num_retries=0)
    assert caught.value is failure
    assert caught.value.__cause__ is cause
    assert caught.value.__traceback__ is not None
    assert server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_settings_observe_mutation_between_calls(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    monkeypatch.setattr(litellm, "force_ipv4", "yes")
    monkeypatch.setattr(litellm, "http2", 1)
    monkeypatch.setattr(litellm, "vertex_project", [])
    monkeypatch.setattr(litellm, "vertex_location", 0)
    monkeypatch.setattr(litellm, "user_url_allowed_hosts", "EXAMPLE.TEST.")
    response: Final = await call(server, asynchronous, num_retries=0)
    assert response.pages[0].markdown == "native OCR response"
    assert_served_natively(server)
    monkeypatch.setattr(litellm, "ssl_certificate", 1)
    with pytest.raises(ValueError, match=r"http_settings\.ssl_certificate"):
        await call(server, asynchronous, num_retries=0)
    assert len(server.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("present", [False, True], ids=["missing", "invalid-pem"])
async def test_native_client_certificate_is_validated_before_io(
    server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    asynchronous: bool,
    present: bool,
) -> None:
    server.expected_requests = 0
    certificate: Final = tmp_path / "client.pem"
    if present:
        certificate.write_text("invalid certificate")
    monkeypatch.setattr(litellm, "ssl_certificate", str(certificate))
    with pytest.raises(ValueError, match=r"http_settings\.ssl_certificate.*PEM") as caught:
        await call(server, asynchronous, num_retries=0)
    assert str(certificate) not in str(caught.value)
    assert server.requests == []

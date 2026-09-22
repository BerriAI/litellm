from __future__ import annotations

import asyncio
import traceback
from typing import Final
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.integrations.custom_secret_manager import CustomSecretManager
from litellm.llms.custom_httpx import llm_http_handler
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.rust_bridge import bindings, configuration
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem
from tests.test_litellm_rust.support.recording_server import ResponseSpec, recording_service


class _VaultSecrets(CustomSecretManager):
    def __init__(self, failure: BaseException | None = None) -> None:
        super().__init__(secret_manager_name="rust_bridge_ocr_test")
        self.failure = failure

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: dict[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> str | None:
        return "vault-key" if secret_name == "MISTRAL_API_KEY" else None

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: dict[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> str | None:
        if secret_name == "MISTRAL_API_KEY" and self.failure is not None:
            raise self.failure
        return "vault-key" if secret_name == "MISTRAL_API_KEY" else None


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Mock:
    response: Final = {
        "pages": [{"index": 0, "markdown": "parsed document", "images": []}],
        "model": "mistral-ocr-latest",
        "usage_info": {"pages_processed": 1},
    }
    request_handler: Final = Mock(return_value=httpx.Response(200, json=response))
    client: Final = httpx.Client(transport=httpx.MockTransport(request_handler))
    monkeypatch.setattr(llm_http_handler, "_get_httpx_client", lambda: HTTPHandler(client=client))
    monkeypatch.setattr(litellm, "secret_manager_client", _VaultSecrets())
    monkeypatch.setattr(litellm, "_key_management_system", KeyManagementSystem.CUSTOM)
    monkeypatch.setattr(
        litellm,
        "_key_management_settings",
        KeyManagementSettings(access_mode="read_only", hosted_keys=["MISTRAL_API_KEY"]),
    )
    yield request_handler
    client.close()


def _call(api_base: str | None = None) -> object:
    return litellm.ocr(
        model="mistral/mistral-ocr-latest",
        document={"type": "document_url", "document_url": "https://example.com/document.pdf"},
        api_base=api_base,
    )


async def _call_with_mode(asynchronous: bool, api_base: str) -> object:
    if not asynchronous:
        return _call(api_base)
    return await litellm.aocr(
        model="mistral/mistral-ocr-latest",
        document={"type": "document_url", "document_url": "https://example.com/document.pdf"},
        api_base=api_base,
    )


def test_bridge_disabled_uses_custom_secret_manager(provider: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "0")
    configuration.reset_rust_configuration()

    result: Final = _call()

    assert result.pages[0].markdown == "parsed document"
    assert provider.call_args.args[0].headers["Authorization"] == "Bearer vault-key"


@pytest.mark.parametrize("restrict_hosted_keys", (False, True))
def test_bridge_enabled_with_native_extension_uses_custom_secret_manager(
    provider: Mock, monkeypatch: pytest.MonkeyPatch, restrict_hosted_keys: bool
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    monkeypatch.setenv("LITELLM_RUST", "1")
    monkeypatch.setattr(
        litellm,
        "_key_management_settings",
        KeyManagementSettings(access_mode="read_only", hosted_keys=["MISTRAL_API_KEY"] if restrict_hosted_keys else None),
    )
    configuration.reset_rust_configuration()

    response: Final = {
        "pages": [{"index": 0, "markdown": "parsed document", "images": []}],
        "model": "mistral-ocr-latest",
        "usage_info": {"pages_processed": 1},
    }
    with recording_service() as server:
        server.default_response = ResponseSpec(body=response)
        result: Final = _call(server.base_url)

    assert result.pages[0].markdown == "parsed document"
    assert server.requests[0].headers["authorization"] == "Bearer vault-key"
    assert provider.call_count == 0


def test_bridge_enabled_uses_manually_assigned_secret_manager(provider: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    monkeypatch.setenv("LITELLM_RUST", "1")
    monkeypatch.setattr(litellm, "_key_management_system", None)
    configuration.reset_rust_configuration()

    response: Final = {
        "pages": [{"index": 0, "markdown": "parsed document", "images": []}],
        "model": "mistral-ocr-latest",
        "usage_info": {"pages_processed": 1},
    }
    with recording_service() as server:
        server.default_response = ResponseSpec(body=response)
        result: Final = _call(server.base_url)

    assert result.pages[0].markdown == "parsed document"
    assert server.requests[0].headers["authorization"] == "Bearer vault-key"
    assert provider.call_count == 0


def test_missing_native_module_uses_python_path(provider: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "1")
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: None)
    configuration.reset_rust_configuration()

    result: Final = _call()

    assert result.pages[0].markdown == "parsed document"
    assert provider.call_args.args[0].headers["Authorization"] == "Bearer vault-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("environment_key", (None, "environment-key"))
@pytest.mark.parametrize("failure_type", (ValueError, asyncio.CancelledError))
async def test_native_secret_callback_preserves_exception_without_provider_io(
    provider: Mock,
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    environment_key: str | None,
    failure_type: type[BaseException],
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    failure: Final = failure_type("secret manager failed")
    cause: Final = RuntimeError("original cause")
    context: Final = RuntimeError("original context")
    failure.__cause__ = cause
    failure.__context__ = context
    monkeypatch.setattr(litellm, "secret_manager_client", _VaultSecrets(failure))
    monkeypatch.setenv("LITELLM_RUST", "1")
    if environment_key is None:
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    else:
        monkeypatch.setenv("MISTRAL_API_KEY", environment_key)
    configuration.reset_rust_configuration()

    with recording_service() as server:
        server.expected_requests = 0
        with pytest.raises(failure_type) as raised:
            await _call_with_mode(asynchronous, server.base_url)

    assert raised.value is failure
    assert raised.value.__cause__ is cause
    assert raised.value.__context__ is context
    assert "sync_read_secret" in tuple(frame.name for frame in traceback.extract_tb(raised.value.__traceback__))
    assert server.requests == []
    assert provider.call_count == 0

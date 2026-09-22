from __future__ import annotations

from typing import Final

import httpx
import pytest

import litellm
from litellm.integrations.custom_secret_manager import CustomSecretManager
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge import configuration
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem
from tests.test_litellm_rust.support.recording_server import ResponseSpec, recording_service


class _VaultSecrets(CustomSecretManager):
    def __init__(self) -> None:
        super().__init__(secret_manager_name="rust_bridge_ocr_test")

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
        return "vault-key" if secret_name == "MISTRAL_API_KEY" else None


async def _call(asynchronous: bool, api_base: str) -> OCRResponse:
    if asynchronous:
        return await litellm.aocr(
            model="mistral/mistral-ocr-latest",
            document={"type": "document_url", "document_url": "https://example.com/document.pdf"},
            api_base=api_base,
        )
    return litellm.ocr(
        model="mistral/mistral-ocr-latest",
        document={"type": "document_url", "document_url": "https://example.com/document.pdf"},
        api_base=api_base,
    )


_RESPONSE: Final = {
    "pages": [{"index": 0, "markdown": "parsed document", "images": []}],
    "model": "mistral-ocr-latest",
    "usage_info": {"pages_processed": 1},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("rust_enabled", ("0", "1"))
@pytest.mark.parametrize("access_mode", ("read_only", "read_and_write"))
@pytest.mark.parametrize("system", (None, KeyManagementSystem.CUSTOM))
async def test_readable_secret_managers_keep_python_ocr_fallback(
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    rust_enabled: str,
    access_mode: str,
    system: KeyManagementSystem | None,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    monkeypatch.setenv("LITELLM_RUST", rust_enabled)
    monkeypatch.setenv("MISTRAL_API_KEY", "environment-key")
    monkeypatch.setattr(litellm, "secret_manager_client", _VaultSecrets())
    monkeypatch.setattr(litellm, "_key_management_system", system)
    monkeypatch.setattr(
        litellm,
        "_key_management_settings",
        KeyManagementSettings(access_mode=access_mode, hosted_keys=["MISTRAL_API_KEY"]),
    )
    configuration.reset_rust_configuration()

    with recording_service() as server:
        server.default_response = ResponseSpec(body=_RESPONSE)
        result: Final = await _call(asynchronous, server.base_url)

    assert result.pages[0].markdown == "parsed document"
    assert len(server.requests) == 1
    expected_key: Final = "vault-key" if system is KeyManagementSystem.CUSTOM else "environment-key"
    assert server.requests[0].headers["authorization"] == f"Bearer {expected_key}"
    assert "x-litellm-rust" not in result._hidden_params.get("additional_headers", {})


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_no_secret_client_leaves_dormant_binding_settings_unread(
    monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    monkeypatch.setenv("LITELLM_RUST", "1")
    monkeypatch.setenv("MISTRAL_API_KEY", "environment-key")
    monkeypatch.setattr(litellm, "secret_manager_client", None)
    monkeypatch.setattr(litellm, "_key_management_settings", object())
    configuration.reset_rust_configuration()

    with recording_service() as server:
        server.default_response = ResponseSpec(body=_RESPONSE)
        result: Final = await _call(asynchronous, server.base_url)

    assert result.pages[0].markdown == "parsed document"
    assert len(server.requests) == 1
    assert server.requests[0].headers["authorization"] == "Bearer environment-key"
    assert result._hidden_params["additional_headers"]["x-litellm-rust"] == "true"

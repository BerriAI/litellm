from typing import Final

import pytest

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.dulus.chat.transformation import DulusConfig

DEFAULT_API_BASE: Final = "https://control.dulus.ai/v1"


def test_resolves_environment_and_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DULUS_API_KEY", "dulus_sk_fake-environment-key")
    monkeypatch.delenv("DULUS_API_BASE", raising=False)
    assert get_llm_provider(model="dulus/dulus-b-27b") == (
        "dulus-b-27b",
        "dulus",
        "dulus_sk_fake-environment-key",
        DEFAULT_API_BASE,
    )
    assert get_llm_provider(
        model="dulus/dulus-b-27b", api_key="dulus_sk_fake-explicit-key", api_base="https://gateway.example/v1"
    ) == ("dulus-b-27b", "dulus", "dulus_sk_fake-explicit-key", "https://gateway.example/v1")

    monkeypatch.setenv("DULUS_API_BASE", "https://env-gateway.example/v1")
    assert get_llm_provider(model="dulus/dulus-b-27b")[3] == "https://env-gateway.example/v1"


def test_endpoint_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DULUS_API_KEY", "dulus_sk_fake-environment-key")
    model, provider, key, base = get_llm_provider(
        model="dulus-b-27b", api_base="https://control.dulus.ai/v1"
    )
    assert (provider, key, base) == ("dulus", "dulus_sk_fake-environment-key", "https://control.dulus.ai/v1")


def test_config_provider_info_and_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DULUS_API_KEY", raising=False)
    monkeypatch.delenv("DULUS_API_BASE", raising=False)
    config = DulusConfig()
    assert config.custom_llm_provider == "dulus"
    assert config.get_openai_compatible_provider_info(None, None) == (DEFAULT_API_BASE, None)
    params = config.get_supported_openai_params(model="dulus-b-27b")
    assert "tools" in params and "stream" in params

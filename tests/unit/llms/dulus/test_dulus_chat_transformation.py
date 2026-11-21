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
    monkeypatch.setenv("DULUS_API_KEY", "dulus_sk_fake-environment-key")
    monkeypatch.delenv("DULUS_API_BASE", raising=False)
    config = DulusConfig()
    assert config.custom_llm_provider == "dulus"
    assert config.get_openai_compatible_provider_info(None, None) == (
        DEFAULT_API_BASE,
        "dulus_sk_fake-environment-key",
    )
    params = config.get_supported_openai_params(model="dulus-b-27b")
    assert "tools" in params and "stream" in params


def test_missing_key_fails_closed_no_openai_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without DULUS_API_KEY we must fail closed, never inherit OPENAI_API_KEY."""
    import litellm

    from litellm.exceptions import AuthenticationError

    monkeypatch.delenv("DULUS_API_KEY", raising=False)
    monkeypatch.delenv("DULUS_API_BASE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-openai-key")
    monkeypatch.setattr(litellm, "openai_key", "sk-fake-openai-key", raising=False)
    config = DulusConfig()
    with pytest.raises(AuthenticationError):
        config.get_openai_compatible_provider_info(None, None)
    with pytest.raises(AuthenticationError):
        config.validate_environment(
            headers={},
            model="dulus-b-27b",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )
    # endpoint inference must not leak the OpenAI key either
    _model, provider, key, base = get_llm_provider(
        model="dulus-b-27b", api_base="https://control.dulus.ai/v1"
    )
    assert provider == "dulus"
    assert key != "sk-fake-openai-key"


def test_cost_map_has_pricing_for_budgets() -> None:
    """Paid Dulus models must consume proxy budgets; free-tier models stay 0."""
    import json
    import pathlib

    cost_map = json.loads(
        pathlib.Path("model_prices_and_context_window.json").read_text(encoding="utf-8")
    )
    dulus_models = sorted(k for k in cost_map if k.startswith("dulus/"))
    assert len(dulus_models) == 19
    for model in dulus_models:
        entry = cost_map[model]
        assert "input_cost_per_token" in entry, model
        assert "output_cost_per_token" in entry, model
        if "free" in model:
            assert entry["input_cost_per_token"] == 0, model
            assert entry["output_cost_per_token"] == 0, model
        else:
            assert entry["input_cost_per_token"] > 0, model
            assert entry["output_cost_per_token"] > 0, model


def test_environment_key_never_goes_over_plain_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """DULUS_API_KEY from the environment is only attached to an HTTPS api_base."""
    from litellm.exceptions import AuthenticationError

    monkeypatch.setenv("DULUS_API_KEY", "dulus_sk_fake-environment-key")
    monkeypatch.delenv("DULUS_API_BASE", raising=False)
    _model, provider, key, _base = get_llm_provider(model="dulus-b-27b", api_base="http://control.dulus.ai/v1")
    assert provider == "dulus"
    assert key is None

    config = DulusConfig()
    with pytest.raises(AuthenticationError):
        config.get_openai_compatible_provider_info("http://control.dulus.ai/v1", None)
    monkeypatch.setenv("DULUS_API_BASE", "http://localhost:8080/v1")
    with pytest.raises(AuthenticationError):
        config.get_openai_compatible_provider_info(None, None)

    # an explicit key for a local HTTP gateway is the caller's choice and still works
    assert config.get_openai_compatible_provider_info("http://localhost:8080/v1", "dulus_sk_fake-explicit-key") == (
        "http://localhost:8080/v1",
        "dulus_sk_fake-explicit-key",
    )

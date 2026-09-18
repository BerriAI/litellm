from typing import Final

import pytest

import litellm
from litellm import CustomLLM
from litellm.litellm_core_utils.get_llm_provider_logic import (
    get_llm_provider,
    is_registered_custom_provider,
    resolve_model_and_provider_for_metadata,
)

CUSTOM_PROVIDER: Final = "test-onprem-llm"


@pytest.fixture
def registered_custom_provider(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": CUSTOM_PROVIDER, "custom_handler": CustomLLM()}])
    monkeypatch.setattr(litellm, "provider_list", list(litellm.provider_list))
    monkeypatch.setattr(litellm, "_custom_providers", list(litellm._custom_providers))
    return CUSTOM_PROVIDER


def test_get_llm_provider_resolves_custom_provider_map_prefix_before_first_completion(
    registered_custom_provider: str,
) -> None:
    assert registered_custom_provider not in litellm.provider_list

    model, provider, dynamic_api_key, api_base = get_llm_provider(model=f"{registered_custom_provider}/my-model")

    assert (model, provider, dynamic_api_key, api_base) == ("my-model", registered_custom_provider, None, None)


def test_get_llm_provider_strips_prefix_when_custom_provider_passed_explicitly(
    registered_custom_provider: str,
) -> None:
    model, provider, _, api_base = get_llm_provider(
        model="my-model",
        custom_llm_provider=registered_custom_provider,
        api_base="http://onprem.internal:8080",
    )

    assert (model, provider, api_base) == ("my-model", registered_custom_provider, "http://onprem.internal:8080")


def test_get_llm_provider_still_rejects_unregistered_prefix(registered_custom_provider: str) -> None:
    with pytest.raises(litellm.BadRequestError, match="LLM Provider NOT provided"):
        get_llm_provider(model="not-registered-llm/my-model")


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [(CUSTOM_PROVIDER, True), ("not-registered-llm", False), (None, False), ("", False)],
)
def test_is_registered_custom_provider(registered_custom_provider: str, candidate: str | None, expected: bool) -> None:
    assert is_registered_custom_provider(candidate) is expected


@pytest.fixture
def chatgpt_login_tripwire(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.llms.chatgpt.authenticator import Authenticator

    def _tripwire(self: Authenticator) -> str:
        raise AssertionError("metadata lookup ran the ChatGPT OAuth device flow")

    monkeypatch.setenv("CHATGPT_TOKEN_DIR", "/nonexistent/chatgpt-token-dir")
    monkeypatch.setattr(Authenticator, "get_access_token", _tripwire)


@pytest.mark.parametrize(
    ("model", "custom_llm_provider", "expected"),
    [
        ("chatgpt/gpt-5.4", None, ("gpt-5.4", "chatgpt")),
        ("gpt-5.4", "chatgpt", ("gpt-5.4", "chatgpt")),
        ("github_copilot/gpt-4o", None, ("gpt-4o", "github_copilot")),
        ("openai/gpt-5.4", None, ("gpt-5.4", "openai")),
    ],
)
def test_resolve_model_and_provider_for_metadata_never_logs_in(
    chatgpt_login_tripwire: None, model: str, custom_llm_provider: str | None, expected: tuple[str, str]
) -> None:
    assert resolve_model_and_provider_for_metadata(model, custom_llm_provider) == expected


def test_metadata_callers_never_log_in_for_a_declared_chatgpt_deployment(
    chatgpt_login_tripwire: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.litellm_core_utils.llm_response_utils.get_api_base import get_api_base
    from litellm.proxy.spend_tracking.savings import _resolve_model
    from litellm.router import Router
    from litellm.types.router import LiteLLM_Params
    from litellm.utils import supports_native_streaming

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    assert supports_native_streaming("chatgpt/gpt-5.4", None) is True
    assert get_api_base("chatgpt/gpt-5.4", {"model": "chatgpt/gpt-5.4"}) is None
    identity = _resolve_model("chatgpt/gpt-5.4", None)
    assert identity is not None and identity.provider == "chatgpt"
    assert Router._model_group_llm_provider(LiteLLM_Params(model="chatgpt/gpt-5.4")) == ("gpt-5.4", "chatgpt")

from typing import Final

import pytest

import litellm
from litellm import CustomLLM
from litellm.litellm_core_utils.get_llm_provider_logic import (
    get_llm_provider,
    is_registered_custom_provider,
)

CUSTOM_PROVIDER: Final = "test-onprem-llm"


@pytest.mark.parametrize("model", ("groq/compound", "groq/compound-mini"))
def test_get_llm_provider_keeps_groq_prefix_for_compound_models(model: str) -> None:
    """
    https://github.com/BerriAI/litellm/issues/32467

    Groq's own model ids for these two models include a "groq/" segment
    (confirmed via Groq's /v1/models endpoint), unlike every other Groq
    model where "groq/" is purely litellm's routing prefix. Stripping it
    here sent Groq a model id it doesn't recognize.
    """
    resolved_model, provider, _, _ = get_llm_provider(model=model)

    assert resolved_model == model
    assert provider == "groq"


def test_get_llm_provider_strips_groq_prefix_for_regular_models() -> None:
    resolved_model, provider, _, _ = get_llm_provider(model="groq/llama-3.3-70b-versatile")

    assert resolved_model == "llama-3.3-70b-versatile"
    assert provider == "groq"


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

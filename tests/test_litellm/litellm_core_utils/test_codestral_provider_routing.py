import pytest

from litellm.litellm_core_utils.get_llm_provider_logic import (
    _get_codestral_provider_from_api_base,
    get_llm_provider,
)


def test_codestral_chat_endpoint_routes_to_codestral_provider():
    _, provider, _, _ = get_llm_provider(
        model="codestral-latest",
        api_base="https://codestral.mistral.ai/v1/chat/completions",
    )
    assert provider == "codestral"


def test_codestral_fim_endpoint_routes_to_text_completion_provider():
    _, provider, _, _ = get_llm_provider(
        model="codestral-latest",
        api_base="https://codestral.mistral.ai/v1/fim/completions",
    )
    assert provider == "text-completion-codestral"


@pytest.mark.parametrize(
    "api_base",
    [
        "https://codestral.mistral.ai",
        "https://codestral.mistral.ai/v1",
        "https://codestral.mistral.ai/v1/chat/completions/",
        "https://CODESTRAL.MISTRAL.AI/v1",
    ],
)
def test_codestral_chat_provider_routing_with_host_variants(api_base):
    _, provider, _, _ = get_llm_provider(
        model="codestral-latest",
        api_base=api_base,
    )
    assert provider == "codestral"


def test_codestral_helper_ignores_similar_domain():
    assert (
        _get_codestral_provider_from_api_base(
            "https://codestral.mistral.ai.evil.com/v1/fim/completions"
        )
        is None
    )

from typing import Final

import httpx
import pytest

import litellm
from litellm import CustomLLM
from litellm.litellm_core_utils import get_llm_provider_logic
from litellm.litellm_core_utils.get_llm_provider_logic import (
    get_llm_provider,
    inferred_provider,
    is_registered_custom_provider,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler

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


def test_get_llm_provider_leaves_fal_ai_api_base_unset_for_global_fallback() -> None:
    _, provider, _, api_base = get_llm_provider(model="fal_ai/fal-ai/flux/schnell")
    assert provider == "fal_ai"
    assert api_base is None

    _, _, _, explicit = get_llm_provider(model="fal_ai/fal-ai/flux/schnell", api_base="http://edge.local/fal")
    assert explicit == "http://edge.local/fal"


def test_image_generation_fal_ai_egresses_to_global_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "api_base", "http://gateway.local/fal")
    monkeypatch.setenv("FAL_AI_API_KEY", "test")
    seen: Final[list[httpx.URL]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json={"images": [{"url": "https://fal.media/a.png"}]})

    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handler)))
    litellm.image_generation(model="fal_ai/fal-ai/flux/schnell", prompt="a red kite", client=client)
    assert str(seen[0]).startswith("http://gateway.local/fal")


def test_inferred_provider_adopts_a_declared_authenticating_provider_without_resolving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _oauth_tripwire(model: str, *args: object, **kwargs: object) -> None:
        raise AssertionError(f"get_llm_provider would run the OAuth device flow for {model}")

    monkeypatch.setattr(get_llm_provider_logic, "get_llm_provider", _oauth_tripwire)

    assert inferred_provider("github_copilot/gpt-5.5") == "github_copilot"


def test_inferred_provider_matches_the_resolver_for_a_bare_model_name() -> None:
    assert inferred_provider("claude-sonnet-4-6") == get_llm_provider(model="claude-sonnet-4-6")[1]
    assert inferred_provider("gpt-5.5-pro") == get_llm_provider(model="gpt-5.5-pro")[1]


@pytest.mark.parametrize("model", ["some-unknown-model-xyz", "", None], ids=["unknown", "empty", "missing"])
def test_inferred_provider_is_none_when_nothing_resolves(model: str | None) -> None:
    assert inferred_provider(model) is None

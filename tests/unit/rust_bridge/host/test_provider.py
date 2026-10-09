from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils import get_llm_provider_logic
from litellm.rust_bridge.host.provider import resolve_provider

pytestmark = pytest.mark.usefixtures("local_model_cost_map")


def test_bare_model_known_to_the_catalog_resolves_to_its_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "anthropic_models", {"claude-test-bare"})

    assert resolve_provider("claude-test-bare", None) == ("claude-test-bare", "anthropic")


def test_prefix_and_explicit_provider_give_the_same_target() -> None:
    assert resolve_provider("anthropic/claude-test-prefixed", None) == ("claude-test-prefixed", "anthropic")
    assert resolve_provider("claude-test-prefixed", "anthropic") == ("claude-test-prefixed", "anthropic")


def test_unclaimed_model_is_none_rather_than_an_exception() -> None:
    assert resolve_provider("no-provider-claims-this-model", None) is None


def test_declared_authenticating_provider_never_reaches_the_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(
        *args: object, **kwargs: object
    ) -> tuple[str, str, str | None, str | None]:  # kwargs-ok: shadows the resolver's signature
        pytest.fail("get_llm_provider would start the device flow for this provider")

    monkeypatch.setattr(get_llm_provider_logic, "get_llm_provider", forbidden)
    declared: Final = next(iter(get_llm_provider_logic.PROVIDERS_THAT_AUTHENTICATE_ON_PROVIDER_INFO))

    assert resolve_provider(f"{declared}/some-model", None) == ("some-model", declared)
    assert resolve_provider("some-model", declared) == ("some-model", declared)

from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.model_latest_aliases import (
    expand_model_latest_alias,
    get_model_latest_aliases,
    is_model_latest_alias,
    resolve_model_latest_alias,
)


@pytest.mark.parametrize("model", ("", "kimi-latest", "openai/unknown", "gemini/gemini-pro-latest"))
def test_non_alias_expansion_keeps_the_requested_name(model: str) -> None:
    assert expand_model_latest_alias(model) == (model,)


def test_alias_expansion_keeps_requested_and_target_names() -> None:
    for provider, aliases in get_model_latest_aliases().items():
        for alias, target in aliases.items():
            assert expand_model_latest_alias(f"{provider}/{alias}") == (f"{provider}/{alias}", f"{provider}/{target}")


def test_bare_alias_expands_to_its_cost_map_owner() -> None:
    alias, target = next(iter(get_model_latest_aliases()["openai"].items()))
    assert expand_model_latest_alias(alias) == (alias, target)


def test_latest_aliases_price_and_route_like_their_targets() -> None:
    aliases: Final = get_model_latest_aliases()
    assert aliases
    for provider, provider_aliases in aliases.items():
        for alias, target in provider_aliases.items():
            prefixed: Final = f"{provider}/{alias}"
            alias_key: Final = prefixed if prefixed in litellm.model_cost else alias
            target_key: Final = f"{provider}/{target}" if f"{provider}/{target}" in litellm.model_cost else target
            entry: Final = litellm.model_cost[alias_key]
            assert (entry["litellm_provider"], entry["mode"]) == (provider, "chat")
            assert entry == litellm.model_cost[target_key]
            alias_info: Final = {**litellm.get_model_info(prefixed), "key": target}
            assert alias_info == {**litellm.get_model_info(f"{provider}/{target}"), "key": target}
            assert litellm.get_llm_provider(prefixed)[:2] == (target, provider)
            assert litellm.get_llm_provider(alias, custom_llm_provider=provider)[:2] == (target, provider)


def test_alias_resolution_is_scoped_to_the_owning_provider() -> None:
    alias, target = next(iter(get_model_latest_aliases()["openai"].items()))
    assert resolve_model_latest_alias(alias, "openai") == target
    assert resolve_model_latest_alias(alias, "anthropic") == alias
    assert resolve_model_latest_alias(target, "openai") == target


@pytest.mark.parametrize(
    ("model", "provider", "expected"),
    (
        ("openai/gpt-luna-latest", None, True),
        ("gpt-luna-latest", "openai", True),
        ("gpt-luna-latest", None, True),
        ("gpt-luna-latest", "anthropic", False),
        ("moonshot/kimi-latest", None, True),
        ("kimi-latest", None, False),
        ("gemini/gemini-pro-latest", None, False),
        ("", None, False),
        ("openai/unknown-model", None, False),
    ),
)
def test_latest_alias_detection_follows_the_cost_map_owner(model: str, provider: str | None, expected: bool) -> None:
    assert is_model_latest_alias(model, provider) is expected

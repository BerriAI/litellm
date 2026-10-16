import json
from pathlib import Path
from typing import Final

import pytest

from litellm.proxy.common_utils.token_count_settings import (
    TokenCountSettings,
    count_source_fields,
    resolve_token_count_settings,
)
from litellm.types.utils import TokenCountResponse

_DEFAULT: Final = {"token_count_framing": 97, "token_count_fallback_model": "anthropic.claude-opus-5-5"}
_COST_MAP: Final = Path(__file__).parents[4] / "model_prices_and_context_window.json"


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ({}, TokenCountSettings(framing=97, fallback_model="anthropic.claude-opus-5-5")),
        (
            {"token_count_framing": 5, "token_count_fallback_model": "custom-model"},
            TokenCountSettings(framing=5, fallback_model="custom-model"),
        ),
        ({"token_count_framing": 0}, TokenCountSettings(framing=0, fallback_model="anthropic.claude-opus-5-5")),
        (
            {"token_count_framing": None, "token_count_fallback_model": None},
            TokenCountSettings(framing=97, fallback_model="anthropic.claude-opus-5-5"),
        ),
        (
            {"token_count_framing": True, "token_count_fallback_model": 5},
            TokenCountSettings(framing=97, fallback_model="anthropic.claude-opus-5-5"),
        ),
        ({"token_count_framing": 78.0}, TokenCountSettings(framing=78, fallback_model="anthropic.claude-opus-5-5")),
        ({"token_count_framing": 1.5}, TokenCountSettings(framing=97, fallback_model="anthropic.claude-opus-5-5")),
        ({"token_count_framing": -1}, TokenCountSettings(framing=97, fallback_model="anthropic.claude-opus-5-5")),
        ({"token_count_framing": "78"}, TokenCountSettings(framing=97, fallback_model="anthropic.claude-opus-5-5")),
        ({"token_count_fallback_model": ""}, TokenCountSettings(framing=97, fallback_model=None)),
        ({"token_count_fallback_model": "  "}, TokenCountSettings(framing=97, fallback_model=None)),
    ],
)
def test_config_overrides_the_cost_map_default_key_by_key(
    configured: dict[str, object], expected: TokenCountSettings
) -> None:
    assert resolve_token_count_settings(configured=configured, default=_DEFAULT) == expected


def test_no_setting_anywhere_means_no_fallback_and_no_framing() -> None:
    assert resolve_token_count_settings(configured={}, default={}) == TokenCountSettings(
        framing=None, fallback_model=None
    )


def _counted(source: str | None, model_used: str = "us.anthropic.claude-opus-5-5") -> TokenCountResponse:
    return TokenCountResponse(
        total_tokens=7,
        request_model="sonnet",
        model_used=model_used,
        tokenizer_type="bedrock_api",
        count_source=source,
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (None, {}),
        ("provider", {"litellm_count_source": "provider"}),
        ("local_estimate", {"litellm_count_source": "local_estimate"}),
        (
            "fallback_model",
            {"litellm_count_source": "fallback_model", "litellm_count_fallback_model": "us.anthropic.claude-opus-5-5"},
        ),
    ],
)
def test_count_source_fields_name_the_fallback_model_only_for_a_fallback_count(
    source: str | None, expected: dict[str, str]
) -> None:
    assert count_source_fields(_counted(source)) == expected


def test_every_shipped_fallback_model_is_a_mapped_model_of_the_same_provider() -> None:
    cost_map: Final = json.loads(_COST_MAP.read_text())
    fallbacks: Final = {
        name: entry["token_count_fallback_model"]
        for name, entry in cost_map.items()
        if isinstance(entry, dict) and "token_count_fallback_model" in entry
    }
    assert fallbacks
    assert {
        name: (cost_map.get(fallback, {}).get("litellm_provider"), cost_map[name]["litellm_provider"])
        for name, fallback in fallbacks.items()
        if cost_map.get(fallback, {}).get("litellm_provider") != cost_map[name]["litellm_provider"]
        or fallback == name
    } == {}

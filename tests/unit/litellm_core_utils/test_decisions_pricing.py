from collections.abc import Mapping
from typing import Final

from litellm.litellm_core_utils.decisions_pricing import (
    decisions_pricing_key,
    decisions_pricing_model,
    is_decisions_pricing_key,
)


def test_decisions_pricing_key_uses_provider_decisions_model_order() -> None:
    pricing_key: Final = decisions_pricing_key("openai", "gpt-6-luna")

    assert pricing_key == "/".join(("openai", "decisions", "gpt-6-luna"))


def test_is_decisions_pricing_key_matches_only_the_provider_prefix() -> None:
    decision_key: Final = decisions_pricing_key("openai", "gpt-6-luna")
    plain_key: Final = "openai/gpt-6-luna"
    other_provider_key: Final = decisions_pricing_key("perplexity", "gpt-6-luna")

    assert is_decisions_pricing_key(decision_key, "openai")
    assert not is_decisions_pricing_key(plain_key, "openai")
    assert not is_decisions_pricing_key(other_provider_key, "openai")


def test_decisions_pricing_model_uses_variant_or_falls_back_to_plain_model() -> None:
    pricing_key: Final = decisions_pricing_key("openai", "gpt-6-luna")
    variant_cost_map: Final[Mapping[str, object]] = {pricing_key: {}}
    plain_cost_map: Final[Mapping[str, object]] = {"openai/gpt-6-luna": {}}

    assert decisions_pricing_model("openai", "gpt-6-luna", variant_cost_map) == pricing_key
    assert decisions_pricing_model("openai", "gpt-6-luna", plain_cost_map) == "openai/gpt-6-luna"

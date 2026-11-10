import pytest

from litellm.litellm_core_utils.cost_discount import (
    parse_cost_discount_key,
    resolve_cost_discount,
)


def test_parse_cost_discount_key_bare_provider():
    parsed = parse_cost_discount_key("vertex_ai")
    assert parsed.provider == "vertex_ai"
    assert parsed.model_pattern is None


def test_parse_cost_discount_key_splits_on_first_slash():
    parsed = parse_cost_discount_key("vertex_ai/claude-*")
    assert parsed.provider == "vertex_ai"
    assert parsed.model_pattern == "claude-*"


def test_parse_cost_discount_key_pattern_containing_slash_stays_in_pattern():
    parsed = parse_cost_discount_key("vertex_ai/a/b")
    assert parsed.provider == "vertex_ai"
    assert parsed.model_pattern == "a/b"


def test_parse_cost_discount_key_empty_pattern_after_slash():
    parsed = parse_cost_discount_key("vertex_ai/")
    assert parsed.provider == "vertex_ai"
    assert parsed.model_pattern == ""


def test_resolve_cost_discount_exact_key_beats_glob_and_bare():
    config = {"vertex_ai/claude-sonnet-4-5": 0.3, "vertex_ai/claude-*": 0.2, "vertex_ai": 0.05}
    assert resolve_cost_discount(config, "vertex_ai", "claude-sonnet-4-5") == 0.3


def test_resolve_cost_discount_glob_beats_bare_provider():
    config = {"vertex_ai/claude-*": 0.2, "vertex_ai": 0.05}
    assert resolve_cost_discount(config, "vertex_ai", "claude-sonnet-4-5") == 0.2


def test_resolve_cost_discount_longest_literal_glob_wins():
    config = {"vertex_ai/claude-*": 0.2, "vertex_ai/claude-sonnet-*": 0.25, "vertex_ai": 0.05}
    assert resolve_cost_discount(config, "vertex_ai", "claude-sonnet-4-5") == 0.25


def test_resolve_cost_discount_strips_provider_prefix_from_model():
    config = {"vertex_ai/claude-*": 0.2}
    assert resolve_cost_discount(config, "vertex_ai", "vertex_ai/claude-sonnet-4-5") == 0.2


@pytest.mark.parametrize("model", ["openai/openai/gpt-4", "openai/openai/openai/gpt-4"])
def test_resolve_cost_discount_collapses_repeated_provider_prefix(model: str):
    config = {"openai": 0.05, "openai/gpt-*": 0.20}
    assert resolve_cost_discount(config, "openai", model) == 0.20


def test_resolve_cost_discount_prefixed_pattern_beats_stripped_glob():
    config = {"openrouter": 0.05, "openrouter/openrouter/aurora-*": 0.30, "openrouter/aurora-*": 0.20}
    assert resolve_cost_discount(config, "openrouter", "openrouter/openrouter/aurora-alpha") == 0.30


def test_resolve_cost_discount_fully_stripped_name_still_matches():
    config = {"openrouter": 0.05, "openrouter/aurora-*": 0.20}
    assert resolve_cost_discount(config, "openrouter", "openrouter/openrouter/aurora-alpha") == 0.20


def test_resolve_cost_discount_exact_prefixed_name_beats_stripped_exact():
    config = {"openrouter/openrouter/auto": 0.40, "openrouter/auto": 0.10}
    assert resolve_cost_discount(config, "openrouter", "openrouter/openrouter/auto") == 0.40
    assert resolve_cost_discount(config, "openrouter", "openrouter/auto") == 0.10


def test_resolve_cost_discount_longest_literal_across_candidate_names():
    config = {"openai/*": 0.10, "openai/gpt-*": 0.20}
    assert resolve_cost_discount(config, "openai", "openai/openai/gpt-4") == 0.20


def test_resolve_cost_discount_glob_skips_prefixed_candidate_names():
    config = {"openai/o*": 0.30, "openai/gpt-*": 0.20}
    assert resolve_cost_discount(config, "openai", "openai/openai/gpt-4") == 0.20


def test_resolve_cost_discount_prefixless_glob_only_sees_fully_stripped_name():
    config = {"openai": 0.05, "openai/o*": 0.30}
    assert resolve_cost_discount(config, "openai", "openai/openai/gpt-4") == 0.05


def test_resolve_cost_discount_slash_glob_sees_prefixed_candidate_names():
    config = {"openrouter": 0.05, "openrouter/*/auto": 0.25}
    assert resolve_cost_discount(config, "openrouter", "openrouter/openrouter/auto") == 0.25


def test_resolve_cost_discount_non_matching_model_falls_back_to_bare():
    config = {"vertex_ai/claude-*": 0.2, "vertex_ai": 0.05}
    assert resolve_cost_discount(config, "vertex_ai", "gemini-3-pro-preview") == 0.05


def test_resolve_cost_discount_non_matching_model_no_bare_returns_none():
    config = {"vertex_ai/claude-*": 0.2}
    assert resolve_cost_discount(config, "vertex_ai", "gemini-3-pro-preview") is None


def test_resolve_cost_discount_other_provider_untouched():
    config = {"vertex_ai/claude-*": 0.2, "vertex_ai": 0.05}
    assert resolve_cost_discount(config, "openai", "claude-sonnet-4-5") is None


def test_resolve_cost_discount_none_provider_returns_none():
    config = {"vertex_ai": 0.05}
    assert resolve_cost_discount(config, None, "claude-sonnet-4-5") is None


def test_resolve_cost_discount_empty_provider_returns_none():
    config = {"vertex_ai": 0.05}
    assert resolve_cost_discount(config, "", "claude-sonnet-4-5") is None


def test_resolve_cost_discount_none_model_only_bare_matches():
    config = {"vertex_ai/claude-*": 0.2, "vertex_ai": 0.05}
    assert resolve_cost_discount(config, "vertex_ai", None) == 0.05
    assert resolve_cost_discount({"vertex_ai/claude-*": 0.2}, "vertex_ai", None) is None


def test_resolve_cost_discount_char_class_counts_as_wildcard():
    config = {"vertex_ai/claude-sonnet-*": 0.3, "vertex_ai/claude-[abcdefghijklmnopqrstuvwxyz]*": 0.1}
    assert resolve_cost_discount(config, "vertex_ai", "claude-sonnet-4-5") == 0.3


def test_resolve_cost_discount_unclosed_bracket_matches_literally():
    config = {"vertex_ai/weird[": 0.2}
    assert resolve_cost_discount(config, "vertex_ai", "weird[") == 0.2
    assert resolve_cost_discount(config, "vertex_ai", "weirdx") is None


def test_resolve_cost_discount_unclosed_negated_class_counts_as_literal():
    config = {"vertex_ai/gemini-?*": 0.2, "vertex_ai/gemini-[!]*": 0.3}
    assert resolve_cost_discount(config, "vertex_ai", "gemini-[!]") == 0.3


def test_resolve_cost_discount_char_class_with_leading_bracket():
    config = {"vertex_ai/a[]x]*": 0.1, "vertex_ai/a]*": 0.2}
    assert resolve_cost_discount(config, "vertex_ai", "a]q") == 0.2
    assert resolve_cost_discount(config, "vertex_ai", "axq") == 0.1


def test_resolve_cost_discount_glob_crosses_slash_in_model():
    config = {"bedrock/*anthropic.claude-*": 0.15, "bedrock": 0.05}
    assert resolve_cost_discount(config, "bedrock", "bedrock/us-east-1/anthropic.claude-v2:1") == 0.15


def test_resolve_cost_discount_exact_pattern_key_does_not_count_as_exact():
    config = {"vertex_ai/claude-*": 0.2, "vertex_ai": 0.05}
    assert resolve_cost_discount(config, "vertex_ai", "vertex_ai") == 0.05


def test_resolve_cost_discount_region_stripped_name_matches_pattern():
    config = {"bedrock": 0.05, "bedrock/claude-*": 0.2}
    assert resolve_cost_discount(config, "bedrock", "bedrock/us-gov-west-1/claude-x", "us-gov-west-1") == 0.2


def test_resolve_cost_discount_region_stripped_exact_beats_glob():
    config = {"bedrock/claude-x": 0.3, "bedrock/claude-*": 0.2}
    assert resolve_cost_discount(config, "bedrock", "bedrock/us-gov-west-1/claude-x", "us-gov-west-1") == 0.3


def test_resolve_cost_discount_region_qualified_pattern_wins_longest_literal():
    config = {"bedrock": 0.05, "bedrock/us-gov-west-1/*": 0.4, "bedrock/claude-*": 0.2}
    assert resolve_cost_discount(config, "bedrock", "bedrock/us-gov-west-1/claude-x", "us-gov-west-1") == 0.4


def test_resolve_cost_discount_prefixless_glob_cannot_see_region_prefix():
    config = {"bedrock": 0.05, "bedrock/us*": 0.9}
    assert resolve_cost_discount(config, "bedrock", "bedrock/us-gov-west-1/claude-x", "us-gov-west-1") == 0.05


def test_resolve_cost_discount_no_region_name_keeps_qualified_name_hidden():
    config = {"bedrock": 0.05, "bedrock/claude-*": 0.2}
    assert resolve_cost_discount(config, "bedrock", "bedrock/us-gov-west-1/claude-x") == 0.05

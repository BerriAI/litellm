from litellm.litellm_core_utils.cost_discount import parse_cost_discount_key


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

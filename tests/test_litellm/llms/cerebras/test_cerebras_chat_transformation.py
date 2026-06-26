from litellm.llms.cerebras.chat import CerebrasConfig


def test_tools_and_response_format_drops_response_format():
    """Cerebras rejects requests that contain both tools and response_format.
    Regression test for: tools + response_format causes 400 from Cerebras API."""
    config = CerebrasConfig()
    non_default_params = {
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "Answer"}},
    }
    result = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params={},
        model="llama-4-scout-17b-16e-instruct",
        drop_params=False,
    )
    assert "tools" in result
    assert "response_format" not in result


def test_response_format_without_tools_passes_through():
    """response_format alone is valid and must not be dropped."""
    config = CerebrasConfig()
    non_default_params = {
        "response_format": {"type": "json_object"},
    }
    result = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params={},
        model="llama-4-scout-17b-16e-instruct",
        drop_params=False,
    )
    assert result.get("response_format") == {"type": "json_object"}


def test_tools_without_response_format_passes_through():
    """tools alone must not be affected."""
    config = CerebrasConfig()
    tools = [{"type": "function", "function": {"name": "search"}}]
    non_default_params = {"tools": tools}
    result = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params={},
        model="llama-4-scout-17b-16e-instruct",
        drop_params=False,
    )
    assert result.get("tools") == tools
    assert "response_format" not in result

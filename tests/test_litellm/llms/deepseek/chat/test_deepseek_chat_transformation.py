from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig


def _function_tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object"}},
    }


def test_drop_unsupported_tools_keeps_function_tools_only():
    optional_params = {
        "tools": [
            _function_tool("shell"),
            {"type": "namespace", "name": "container.exec"},
            _function_tool("apply_patch"),
        ],
        "tool_choice": "auto",
    }

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert [tool["function"]["name"] for tool in result["tools"]] == [
        "shell",
        "apply_patch",
    ]
    assert all(tool["type"] == "function" for tool in result["tools"])
    assert result["tool_choice"] == "auto"


def test_drop_unsupported_tools_drops_dangling_tool_choice_when_none_survive():
    optional_params = {
        "tools": [{"type": "namespace", "name": "container.exec"}],
        "tool_choice": "required",
        "parallel_tool_calls": True,
        "temperature": 0.2,
    }

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert "tools" not in result
    assert "tool_choice" not in result
    assert "parallel_tool_calls" not in result
    assert result["temperature"] == 0.2


def test_drop_unsupported_tools_is_noop_for_function_only():
    optional_params = {
        "tools": [_function_tool("shell")],
        "tool_choice": "auto",
    }

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert result is optional_params


def test_drop_unsupported_tools_is_noop_without_tools():
    optional_params = {"temperature": 0.7}

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert result is optional_params


def test_transform_request_strips_unsupported_tools_from_body():
    config = DeepSeekChatConfig()
    body = config.transform_request(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "tools": [
                _function_tool("shell"),
                {"type": "namespace", "name": "container.exec"},
            ],
            "tool_choice": "auto",
        },
        litellm_params={},
        headers={},
    )

    assert [tool["type"] for tool in body["tools"]] == ["function"]
    assert body["tools"][0]["function"]["name"] == "shell"


async def test_async_transform_request_strips_unsupported_tools_from_body():
    config = DeepSeekChatConfig()
    body = await config.async_transform_request(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "tools": [
                _function_tool("shell"),
                {"type": "namespace", "name": "container.exec"},
            ],
            "tool_choice": "auto",
        },
        litellm_params={},
        headers={},
    )

    assert [tool["type"] for tool in body["tools"]] == ["function"]
    assert body["tools"][0]["function"]["name"] == "shell"


def test_thinking_mode_active_bool_thinking_returns_false_without_crashing():
    config = DeepSeekChatConfig()
    assert config._thinking_mode_active(model="deepseek-reasoner", optional_params={"thinking": True}) is False


def _map(model: str, **non_default_params) -> dict:
    """Map params the way litellm does for a real request, from an empty optional_params."""
    return DeepSeekChatConfig().map_openai_params(
        non_default_params=non_default_params,
        optional_params={},
        model=model,
        drop_params=False,
    )


def test_v4_rejects_tool_choice_required_so_it_is_downgraded_to_auto():
    """deepseek-v4 runs in thinking mode by default, which 400s on tool_choice='required'."""
    result = _map("deepseek/deepseek-v4-pro", tool_choice="required")

    assert result["tool_choice"] == "auto"


def test_v4_leaves_tool_choice_any_alone():
    """
    DeepSeek rejects "any" in every mode, thinking or not, with a
    deserialization error listing the values it accepts. That is a different bug
    from the thinking-mode restriction, so rewriting it here would hide an
    invalid value rather than work around a provider limit.
    """
    result = _map("deepseek-v4-flash", tool_choice="any")

    assert result["tool_choice"] == "any"


def test_v4_named_function_tool_choice_is_downgraded_to_auto():
    result = _map("deepseek/deepseek-v4-pro", tool_choice={"type": "function", "function": {"name": "get_weather"}})

    assert result["tool_choice"] == "auto"


def test_v4_keeps_tool_choice_auto():
    result = _map("deepseek/deepseek-v4-pro", tool_choice="auto")

    assert result["tool_choice"] == "auto"


def test_v4_keeps_tool_choice_none():
    """'none' forbids tool calls and is accepted in thinking mode - downgrading it would allow them."""
    result = _map("deepseek/deepseek-v4-pro", tool_choice="none")

    assert result["tool_choice"] == "none"


def test_non_reasoning_model_keeps_tool_choice_required():
    result = _map("deepseek/deepseek-chat", tool_choice="required")

    assert result["tool_choice"] == "required"


def test_opt_in_reasoning_model_keeps_tool_choice_required_while_thinking_is_off():
    """
    deepseek-v3.2 carries supports_reasoning=true but only enters thinking mode
    on request, so it still accepts 'required'. Keying the downgrade off the
    capability flag instead of the model family would break this call.
    """
    result = _map("deepseek/deepseek-v3.2", tool_choice="required")

    assert result["tool_choice"] == "required"


def test_explicit_thinking_downgrades_tool_choice_on_any_model():
    result = _map("deepseek-reasoner", thinking={"type": "enabled"}, tool_choice="required")

    assert result["tool_choice"] == "auto"


def test_v4_keeps_tool_choice_required_when_thinking_is_explicitly_disabled():
    """Disabling thinking is the caller's escape hatch for keeping a forced tool_choice."""
    result = _map("deepseek/deepseek-v4-pro", thinking={"type": "disabled"}, tool_choice="required")

    assert result["tool_choice"] == "required"


def test_v4_keeps_tool_choice_required_when_reasoning_effort_is_none():
    result = _map("deepseek/deepseek-v4-pro", reasoning_effort="none", tool_choice="required")

    assert result["tool_choice"] == "required"

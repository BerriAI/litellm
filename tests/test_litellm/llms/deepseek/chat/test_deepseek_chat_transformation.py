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


_JSON_SCHEMA_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def test_downgrade_json_schema_to_json_object():
    config = DeepSeekChatConfig()
    messages, optional_params = config._downgrade_json_schema_to_json_object(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Is the sky blue?"}],
        optional_params={"response_format": dict(_JSON_SCHEMA_RESPONSE_FORMAT)},
    )

    assert optional_params["response_format"] == {"type": "json_object"}
    injected = messages[-1]
    assert injected["role"] == "user"
    # DeepSeek json_object mode requires the word "json" in the prompt
    assert "json" in injected["content"].lower()
    assert "answer" in injected["content"]


def test_downgrade_json_schema_without_schema_body_still_downgrades():
    """DeepSeek rejects the json_schema type itself, so a json_schema response_format
    with no nested schema must still be downgraded to json_object (nothing to inject,
    but the format must not reach the API unchanged and 400)."""
    config = DeepSeekChatConfig()
    original_messages = [{"role": "user", "content": "Reply in json."}]
    for response_format in (
        {"type": "json_schema"},
        {"type": "json_schema", "json_schema": {}},
    ):
        messages, optional_params = config._downgrade_json_schema_to_json_object(
            model="deepseek-chat",
            messages=list(original_messages),
            optional_params={"response_format": response_format},
        )
        assert optional_params["response_format"] == {"type": "json_object"}
        assert messages == original_messages


def test_downgrade_reads_response_schema_key():
    """Some callers pass the schema under a top-level `response_schema` key
    (mirrors GroqChatConfig's extraction); it is downgraded and injected too."""
    config = DeepSeekChatConfig()
    messages, optional_params = config._downgrade_json_schema_to_json_object(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Is the sky blue?"}],
        optional_params={
            "response_format": {
                "type": "json_schema",
                "response_schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            }
        },
    )

    assert optional_params["response_format"] == {"type": "json_object"}
    assert "answer" in messages[-1]["content"]


def test_downgrade_is_noop_for_json_object():
    config = DeepSeekChatConfig()
    original_messages = [{"role": "user", "content": "Is the sky blue?"}]
    messages, optional_params = config._downgrade_json_schema_to_json_object(
        model="deepseek-chat",
        messages=original_messages,
        optional_params={"response_format": {"type": "json_object"}},
    )

    assert optional_params["response_format"] == {"type": "json_object"}
    assert messages == original_messages


def test_transform_request_downgrades_json_schema_response_format():
    config = DeepSeekChatConfig()
    body = config.transform_request(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Is the sky blue?"}],
        optional_params={"response_format": dict(_JSON_SCHEMA_RESPONSE_FORMAT)},
        litellm_params={},
        headers={},
    )

    assert body["response_format"] == {"type": "json_object"}
    assert any("answer" in m.get("content", "") for m in body["messages"])


async def test_async_transform_request_downgrades_json_schema_response_format():
    config = DeepSeekChatConfig()
    body = await config.async_transform_request(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Is the sky blue?"}],
        optional_params={"response_format": dict(_JSON_SCHEMA_RESPONSE_FORMAT)},
        litellm_params={},
        headers={},
    )

    assert body["response_format"] == {"type": "json_object"}
    assert any("answer" in m.get("content", "") for m in body["messages"])

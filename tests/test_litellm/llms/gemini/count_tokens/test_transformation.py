from litellm.llms.gemini.count_tokens.transformation import build_count_tokens_payload


def test_build_count_tokens_payload_translates_anthropic_request():
    payload = build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hello world"}],
        system="You are a helpful assistant",
        tools=[
            {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
    )

    assert payload.contents
    assert payload.contents[0]["parts"][0].get("text") == "hello world"
    assert payload.system_instruction is not None
    assert payload.system_instruction["parts"][0].get("text") == "You are a helpful assistant"
    assert payload.tools is not None
    function_declarations = payload.tools[0]["function_declarations"]
    assert function_declarations[0]["name"] == "get_weather"
    assert function_declarations[0].get("parameters", {}).get("required") == ["city"]


def test_build_count_tokens_payload_without_system_or_tools():
    payload = build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        system=None,
        tools=None,
    )

    assert payload.contents
    assert payload.system_instruction is None
    assert payload.tools is None


def test_build_count_tokens_payload_passes_openai_tools_through():
    payload = build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        system=None,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
    )

    assert payload.tools is not None
    assert payload.tools[0]["function_declarations"][0]["name"] == "get_weather"

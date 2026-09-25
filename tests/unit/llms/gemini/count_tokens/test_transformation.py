import json

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


def test_build_count_tokens_payload_keeps_openai_tool_calls_and_results():
    payload = build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "what's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {"role": "tool", "content": "sunny", "tool_call_id": "call_1"},
        ],
        system=None,
        tools=None,
    )

    assert payload.system_instruction is not None
    assert payload.system_instruction["parts"][0].get("text") == "be helpful"
    assert payload.contents[0]["parts"][0].get("text") == "what's the weather?"
    function_call = payload.contents[1]["parts"][0].get("function_call")
    assert function_call == {"name": "get_weather", "args": {"city": "Paris"}}
    function_response = payload.contents[2]["parts"][0].get("function_response")
    assert function_response["name"] == "get_weather"
    assert function_response["response"] == {"content": "sunny"}


def test_build_count_tokens_payload_maps_anthropic_web_search_tool():
    payload = build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        system=None,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
    )

    assert payload.tools == ({"googleSearch": {}},)


def test_build_count_tokens_payload_maps_openai_web_search_tool():
    payload = build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        system=None,
        tools=[{"type": "web_search_preview"}],
    )

    assert payload.tools == ({"googleSearch": {}},)


def test_build_count_tokens_payload_routes_openai_tool_types_to_openai_path():
    payload = build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=[
            {"role": "user", "content": "check it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {"role": "tool", "content": "sunny", "tool_call_id": "call_1"},
        ],
        system=None,
        tools=[
            {"type": "web_search_preview"},
            {"type": "computer_use", "display_width": 1024, "display_height": 768},
        ],
    )

    function_call = payload.contents[1]["parts"][0].get("function_call")
    assert function_call == {"name": "get_weather", "args": {"city": "Paris"}}
    function_response = payload.contents[2]["parts"][0].get("function_response")
    assert function_response["name"] == "get_weather"


def test_build_count_tokens_payload_wraps_responses_api_tool():
    payload = build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        system=None,
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
    )

    assert payload.tools is not None
    function_declaration = payload.tools[0]["function_declarations"][0]
    assert function_declaration["name"] == "get_weather"
    assert function_declaration["parameters"] == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
    }


def test_build_count_tokens_payload_drops_search_tool_when_mixed_with_functions():
    payload = build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        system=None,
        tools=[
            {"type": "web_search_20250305", "name": "web_search"},
            {"name": "get_weather", "input_schema": {"type": "object"}},
        ],
    )

    assert payload.tools == ({"function_declarations": [{"name": "get_weather", "parameters": {"type": "object"}}]},)


def test_build_count_tokens_payload_merges_thought_signature_into_one_part():
    payload = build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "let me reason about this",
                        "signature": "sig123",
                    },
                    {"type": "text", "text": "answer"},
                ],
            },
        ],
        system=None,
        tools=None,
    )

    assert payload.contents[1]["parts"] == (
        {"thought": True, "text": "let me reason about this", "thoughtSignature": "sig123"},
        {"text": "answer"},
    )


def test_build_count_tokens_payload_counts_server_side_blocks_as_text():
    payload = build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=[
            {
                "role": "assistant",
                "content": [{"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "q"}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "s1",
                        "content": [{"type": "web_search_result", "url": "u", "title": "t"}],
                    }
                ],
            },
        ],
        system=None,
        tools=None,
    )

    assert payload.contents[0]["parts"][0].get("text") == json.dumps(
        {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "q"}},
        ensure_ascii=False,
    )
    result_text = payload.contents[1]["parts"][0].get("text")
    assert isinstance(result_text, str)
    assert "web_search_tool_result" in result_text
    assert "tool_use_id" in result_text


def test_build_count_tokens_payload_maps_anthropic_hosted_tools_to_native_gemini_tools():
    payload = build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        system=None,
        tools=[{"type": "code_execution_20250522", "name": "code_execution"}],
    )

    assert payload.tools == ({"codeExecution": {}},)


def test_build_count_tokens_payload_maps_web_fetch_tool_to_url_context():
    payload = build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        system=None,
        tools=[{"type": "web_fetch_20250910", "name": "web_fetch"}],
    )

    assert payload.tools == ({"urlContext": {}},)


def test_build_count_tokens_payload_drops_url_context_when_mixed_with_functions():
    payload = build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        system=None,
        tools=[
            {"type": "web_fetch_20250910", "name": "web_fetch"},
            {"name": "get_weather", "input_schema": {"type": "object"}},
        ],
    )

    assert payload.tools == ({"function_declarations": [{"name": "get_weather", "parameters": {"type": "object"}}]},)


def test_build_count_tokens_payload_folds_system_into_contents_for_models_without_system_support():
    payload = build_count_tokens_payload(
        model="gemini-1.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        system="be nice",
        tools=None,
    )

    assert payload.system_instruction is None
    assert [part.get("text") for part in payload.contents[0]["parts"]] == ["be nice", "hi"]


def test_build_count_tokens_payload_returns_invalid_for_malformed_anthropic_input():
    from litellm.llms.gemini.count_tokens.transformation import InvalidCountTokensRequest

    payload = build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        system={"not": "a valid system prompt"},
        tools=[{"name": "get_weather", "input_schema": {"type": "object"}}],
    )

    assert isinstance(payload, InvalidCountTokensRequest)
    assert payload.message


def test_normalize_count_tokens_tools_handles_each_tool_shape():
    from litellm.llms.gemini.count_tokens.transformation import normalize_count_tokens_tools

    assert normalize_count_tokens_tools(None) is None
    assert normalize_count_tokens_tools([{"function_declarations": [{"name": "g"}]}]) == (
        {"function_declarations": [{"name": "g"}]},
    )
    assert normalize_count_tokens_tools([{"googleSearch": {}}]) == ({"googleSearch": {}},)
    assert normalize_count_tokens_tools(
        [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    ) == ({"function_declarations": [{"name": "f", "parameters": {"type": "object"}}]},)
    assert normalize_count_tokens_tools([{"name": "f", "input_schema": {"type": "object"}}]) == (
        {"function_declarations": [{"name": "f", "parameters": {"type": "object"}}]},
    )
    assert normalize_count_tokens_tools([{"googleSearch": {}}, {"type": "function", "function": {"name": "f"}}]) == (
        {"function_declarations": [{"name": "f"}]},
    )

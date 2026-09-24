from litellm.llms.gemini.count_tokens.transformation import build_count_tokens_payload

MODEL = "gemini-2.5-flash"


def test_anthropic_tool_turns_become_gemini_function_call_and_response_parts():
    payload = build_count_tokens_payload(
        model=MODEL,
        messages=[
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny"}]},
        ],
        system=None,
        tools=None,
    )

    assert payload.contents == [
        {"role": "user", "parts": [{"text": "What is the weather in Paris?"}]},
        {"role": "model", "parts": [{"function_call": {"name": "get_weather", "args": {"city": "Paris"}}}]},
        {
            "role": "user",
            "parts": [{"function_response": {"name": "get_weather", "response": {"content": "Sunny"}}}],
        },
    ], payload
    assert payload.system_instruction is None
    assert payload.tools is None


def test_system_prompt_is_lifted_out_of_contents_into_system_instruction():
    payload = build_count_tokens_payload(
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
        system=[{"type": "text", "text": "Be terse"}, {"type": "text", "text": "Answer in French"}],
        tools=None,
    )

    assert payload.contents == [{"role": "user", "parts": [{"text": "hi"}]}], payload
    assert payload.system_instruction is not None
    system_text = "".join(part["text"] for part in payload.system_instruction["parts"])
    assert "Be terse" in system_text and "Answer in French" in system_text, payload.system_instruction


def test_string_system_prompt_becomes_system_instruction():
    payload = build_count_tokens_payload(
        model=MODEL, messages=[{"role": "user", "content": "hi"}], system="You are terse", tools=None
    )

    assert payload.system_instruction == {"parts": [{"text": "You are terse"}]}
    assert payload.contents == [{"role": "user", "parts": [{"text": "hi"}]}], payload


def test_anthropic_tools_become_gemini_function_declarations():
    payload = build_count_tokens_payload(
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
        system=None,
        tools=[
            {
                "name": "get_weather",
                "description": "Weather lookup",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            }
        ],
    )

    assert payload.tools == [
        {
            "function_declarations": [
                {
                    "name": "get_weather",
                    "description": "Weather lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ]
        }
    ], payload.tools


def test_unrecognised_system_value_is_dropped_not_sent():
    payload = build_count_tokens_payload(
        model=MODEL, messages=[{"role": "user", "content": "hi"}], system={"unexpected": "shape"}, tools=None
    )

    assert payload.system_instruction is None
    assert payload.contents == [{"role": "user", "parts": [{"text": "hi"}]}], payload

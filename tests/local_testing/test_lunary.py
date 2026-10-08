import io


import litellm

litellm.failure_callback = ["lunary"]
litellm.success_callback = ["lunary"]
litellm.set_verbose = True








def test_lunary_with_tools():
    import litellm

    messages = [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco, Tokyo, and Paris?",
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    response = litellm.completion(
        model="gpt-6-luna",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )

    response_message = response.choices[0].message
    assert response.choices[0].message.tool_calls
    assert all(call.function.name == "get_current_weather" for call in response.choices[0].message.tool_calls)
    print("\nLLM Response:\n", response.choices[0].message)

import os
import sys
import traceback
from dotenv import load_dotenv

load_dotenv()
import pytest

from litellm import completion, acompletion, responses
from litellm.exceptions import APIConnectionError

@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_chat_completion_snowflake(sync_mode):
    try:
        messages = [
            {
                "role": "user",
                "content": "Write me a poem about the blue sky",
            },
        ]

        if sync_mode:
            response = completion(
                model="snowflake/mistral-7b",
                messages=messages,
                api_base = "https://exampleopenaiendpoint-production.up.railway.app/v1/chat/completions"
            )
            print(response)
            assert response is not None
        else:
            response = await acompletion(
                model="snowflake/mistral-7b",
                messages=messages,
                api_base = "https://exampleopenaiendpoint-production.up.railway.app/v1/chat/completions"
            )
            print(response)
            assert response is not None
    except APIConnectionError as e:
        # Skip test if Snowflake API is unavailable (502 error)
        if "Application failed to respond" in str(e) or "502" in str(e):
            pytest.skip(f"Snowflake API unavailable: {e}")
        else:
            raise  # Re-raise if it's a different APIConnectionError
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_chat_completion_snowflake_stream(sync_mode):
    try:
        set_verbose = True
        messages = [
            {
                "role": "user",
                "content": "Write me a poem about the blue sky",
            },
        ]
        
        if sync_mode is False:
            response = await acompletion(
                model="snowflake/mistral-7b",
                messages=messages,
                max_tokens=100,
                stream=True,
                api_base = "https://exampleopenaiendpoint-production.up.railway.app/v1/chat/completions"
            )
            
            async for chunk in response:
                print(chunk)
        else:
            response = completion(
                model="snowflake/mistral-7b",
                messages=messages,
                max_tokens=100,
                stream=True,
                api_base = "https://exampleopenaiendpoint-production.up.railway.app/v1/chat/completions"
            )

            for chunk in response:
                print(chunk)
    except APIConnectionError as e:
        # Skip test if Snowflake API is unavailable (502 error)
        if "Application failed to respond" in str(e) or "502" in str(e):
            pytest.skip(f"Snowflake API unavailable: {e}")
        else:
            raise  # Re-raise if it's a different APIConnectionError
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="Requires Snowflake credentials - run manually when needed")
def test_snowflake_tool_calling_responses_api():
    """
    Test Snowflake tool calling with Responses API.
    Requires SNOWFLAKE_JWT and SNOWFLAKE_ACCOUNT_ID environment variables.
    """
    import litellm

    # Skip if credentials not available
    if not os.getenv("SNOWFLAKE_JWT") or not os.getenv("SNOWFLAKE_ACCOUNT_ID"):
        pytest.skip("Snowflake credentials not available")

    litellm.drop_params = False  # We now support tools!

    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        }
    ]

    try:
        # Test with tool_choice to force tool use
        response = responses(
            model="snowflake/claude-3-5-sonnet",
            input="What's the weather in Paris?",
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            max_output_tokens=200,
        )

        assert response is not None
        assert hasattr(response, "output")
        assert len(response.output) > 0

        # Verify tool call was made
        tool_call_found = False
        for item in response.output:
            if hasattr(item, "type") and item.type == "function_call":
                tool_call_found = True
                assert item.name == "get_weather"
                assert hasattr(item, "arguments")
                print(f"✅ Tool call detected: {item.name}({item.arguments})")
                break

        assert tool_call_found, "Expected tool call but none was found"

    except APIConnectionError as e:
        if "JWT token is invalid" in str(e):
            pytest.skip("Invalid Snowflake JWT token")
        elif "Application failed to respond" in str(e) or "502" in str(e):
            pytest.skip(f"Snowflake API unavailable: {e}")
        else:
            raise

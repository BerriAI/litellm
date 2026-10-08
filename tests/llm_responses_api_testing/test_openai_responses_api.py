import json

import pytest
from base_responses_api import BaseResponsesAPITest

import litellm


class TestOpenAIResponsesAPITest(BaseResponsesAPITest):
    test_responses_api_with_tool_calls = None

    def get_base_completion_call_args(self):
        return {
            "model": "openai/gpt-5.5",
        }

    def get_advanced_model_for_shell_tool(self):
        return "openai/gpt-5.2"


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_openai_compact_responses_api(sync_mode):
    """
    Test the compact_responses API for OpenAI.

    This test verifies that the compact_responses endpoint works correctly
    for compressing conversation history.
    """
    litellm.turn_on_debug()
    litellm.set_verbose = True

    input_messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
        {"role": "user", "content": "What is the weather like today?"},
    ]

    try:
        if sync_mode:
            response = litellm.compact_responses(
                model="openai/gpt-5.5",
                input=input_messages,
                instructions="Be helpful and concise",
            )
        else:
            response = await litellm.acompact_responses(
                model="openai/gpt-5.5",
                input=input_messages,
                instructions="Be helpful and concise",
            )
    except litellm.InternalServerError:
        pytest.skip("Skipping test due to InternalServerError")
    except litellm.BadRequestError as e:
        # compact_responses may not be available for all models/accounts
        pytest.skip(f"Skipping test due to BadRequestError: {e}")

    print("compact_responses response=", json.dumps(response, indent=4, default=str))

    # Validate response structure
    assert response is not None
    assert "id" in response, "Response should have an 'id' field"
    assert "output" in response, "Response should have an 'output' field"
    assert isinstance(response["output"], list), "Output should be a list"

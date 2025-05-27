from base_llm_unit_tests import BaseLLMChatTest
import pytest


# Test implementations
@pytest.mark.skip(reason="Deepseek API is hanging")
class TestDeepSeekChatCompletion(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {
            "model": "deepseek/deepseek-reasoner",
        }

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass


@pytest.mark.parametrize("stream", [True, False])
def test_deepseek_mock_completion(stream):
    """
    Deepseek API is hanging. Mock the call, to a fake endpoint, so we can confirm our integration is working.
    """
    import litellm
    from litellm import completion

    litellm._turn_on_debug()

    response = completion(
        model="deepseek/deepseek-reasoner",
        messages=[{"role": "user", "content": "Hello, world!"}],
        api_base="https://exampleopenaiendpoint-production.up.railway.app/v1/chat/completions",
        stream=stream,
    )
    print(f"response: {response}")
    if stream:
        for chunk in response:
            print(chunk)
    else:
        assert response is not None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_deepseek_provider_async_completion(stream):
    """
    Test that Deepseek provider requests are formatted correctly with the proper parameters
    """
    import litellm
    import json
    from unittest.mock import patch, AsyncMock, MagicMock
    from litellm import acompletion

    litellm._turn_on_debug()

    # Set up the test parameters
    api_key = "fake_api_key"
    model = "deepseek/deepseek-reasoner"
    messages = [{"role": "user", "content": "Hello, world!"}]

    # Mock AsyncHTTPHandler.post method for async test
    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.AsyncHTTPHandler.post"
    ) as mock_post:
        mock_response_data = litellm.ModelResponse(
            choices=[
                litellm.Choices(
                    message=litellm.Message(content="Hello!"),
                    index=0,
                    finish_reason="stop",
                )
            ]
        ).model_dump()
        # Create a proper mock response
        mock_response = MagicMock()  # Use MagicMock instead of AsyncMock
        mock_response.status_code = 200
        mock_response.text = json.dumps(mock_response_data)
        mock_response.headers = {"Content-Type": "application/json"}

        # Make json() return a value directly, not a coroutine
        mock_response.json.return_value = mock_response_data

        # Set the return value for the post method
        mock_post.return_value = mock_response

        await acompletion(
            custom_llm_provider="deepseek",
            api_key=api_key,
            model=model,
            messages=messages,
            stream=stream,
        )

    # Verify the request was made with the correct parameters
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    print("request call=", json.dumps(call_args.kwargs, indent=4, default=str))

    # Check request body
    request_body = json.loads(call_args.kwargs["data"])
    assert call_args.kwargs["url"] == "https://api.deepseek.com/beta/chat/completions"
    assert (
        request_body["model"] == "deepseek-reasoner"
    )  # Model name should be stripped of provider prefix
    assert request_body["messages"] == messages
    assert request_body["stream"] == stream

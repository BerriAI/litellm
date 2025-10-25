import json
import pytest
import litellm
from unittest.mock import patch, MagicMock


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_gatewayz_provider_async_completion(stream):
    """
    Test that Gatewayz provider requests are formatted correctly with the proper parameters
    """
    from litellm import acompletion

    litellm._turn_on_debug()

    # Set up the test parameters
    api_key = "fake_api_key"
    model = "gatewayz/test-model"
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
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(mock_response_data)
        mock_response.headers = {"Content-Type": "application/json"}

        # Make json() return a value directly, not a coroutine
        mock_response.json.return_value = mock_response_data

        # Set the return value for the post method
        mock_post.return_value = mock_response

        await acompletion(
            custom_llm_provider="gatewayz",
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
    assert call_args.kwargs["url"] == "https://api.gatewayz.com/v1/chat/completions"
    assert (
        request_body["model"] == "test-model"
    )  # Model name should be stripped of provider prefix
    assert request_body["messages"] == messages
    assert request_body["stream"] == stream


@pytest.mark.parametrize("stream", [False, True])
def test_gatewayz_provider_completion(stream):
    """
    Test that Gatewayz provider requests are formatted correctly with the proper parameters (sync version)
    """
    from litellm import completion

    litellm._turn_on_debug()

    # Set up the test parameters
    api_key = "fake_api_key"
    model = "gatewayz/test-model"
    messages = [{"role": "user", "content": "Hello, world!"}]

    # Mock HTTPHandler.post method for sync test
    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post"
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
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(mock_response_data)
        mock_response.headers = {"Content-Type": "application/json"}

        # Make json() return a value directly
        mock_response.json.return_value = mock_response_data

        # Set the return value for the post method
        mock_post.return_value = mock_response

        completion(
            custom_llm_provider="gatewayz",
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
    assert call_args.kwargs["url"] == "https://api.gatewayz.com/v1/chat/completions"
    assert (
        request_body["model"] == "test-model"
    )  # Model name should be stripped of provider prefix
    assert request_body["messages"] == messages
    assert request_body["stream"] == stream


def test_gatewayz_custom_api_base():
    """
    Test that Gatewayz provider respects custom API base URL
    """
    from litellm import completion

    litellm._turn_on_debug()

    # Set up the test parameters
    api_key = "fake_api_key"
    custom_api_base = "https://custom.gatewayz.com"
    model = "gatewayz/test-model"
    messages = [{"role": "user", "content": "Hello, world!"}]

    # Mock HTTPHandler.post method for sync test
    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post"
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
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(mock_response_data)
        mock_response.headers = {"Content-Type": "application/json"}

        # Make json() return a value directly
        mock_response.json.return_value = mock_response_data

        # Set the return value for the post method
        mock_post.return_value = mock_response

        completion(
            custom_llm_provider="gatewayz",
            api_key=api_key,
            api_base=custom_api_base,
            model=model,
            messages=messages,
        )

    # Verify the request was made with the correct parameters
    mock_post.assert_called_once()
    call_args = mock_post.call_args

    # Check that custom API base was used
    assert (
        call_args.kwargs["url"]
        == f"{custom_api_base}/v1/chat/completions"
    )

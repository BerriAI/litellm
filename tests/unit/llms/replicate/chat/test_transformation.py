import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import litellm
import pytest
from pydantic import ValidationError

from litellm.llms.replicate.chat.handler import async_completion
from litellm.llms.replicate.chat.transformation import ReplicateConfig
from litellm.types.utils import ModelResponse


def _transform(raw_response: httpx.Response) -> ModelResponse:
    return ReplicateConfig().transform_response(
        model="acme/echo-model",
        raw_response=raw_response,
        model_response=ModelResponse(),
        logging_obj=Mock(),
        request_data={"input": {"prompt": "Hello"}},
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )


@pytest.mark.parametrize(
    ("output", "content"),
    [
        (["Hello", ", ", "world"], "Hello, world"),
        ("Hello", "Hello"),
        ([], " "),
        ("", " "),
        ([""], " "),
    ],
)
def test_transform_response_joins_the_prediction_output_into_the_message_content(output: object, content: str):
    response = _transform(httpx.Response(200, json={"status": "succeeded", "output": output}))

    assert response.choices[0].message.content == content
    assert response.model == "replicate/acme/echo-model"
    assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens


def test_transform_response_uses_a_blank_message_when_the_prediction_has_no_output():
    response = _transform(httpx.Response(200, json={"status": "succeeded"}))

    assert response.choices[0].message.content == " "


@pytest.mark.parametrize("output", [None, 7, True, ["secret-output", 7], ["secret-output", None], [["secret-output"]]])
def test_transform_response_rejects_an_output_that_is_not_made_of_strings_without_echoing_it(output: object):
    with pytest.raises(ValidationError) as exc_info:
        _transform(httpx.Response(200, json={"status": "succeeded", "output": output}))

    assert "secret-output" not in str(exc_info.value)


@pytest.mark.asyncio
@patch("litellm.llms.replicate.chat.handler.asyncio.sleep", new_callable=AsyncMock)
@patch("litellm.llms.replicate.chat.handler.get_async_httpx_client")
async def test_async_completion_handles_starting_status(mock_get_client, mock_sleep):
    """Test that async completion polls correctly when status is 'starting'"""
    mock_client = AsyncMock()
    mock_get_client.return_value = mock_client
    post_response = Mock()
    post_response.json.return_value = {
        "id": "test-prediction-id",
        "urls": {
            "get": "https://api.replicate.com/v1/predictions/test-id",
            "cancel": "https://api.replicate.com/v1/predictions/test-id/cancel",
        },
    }
    mock_client.post = AsyncMock(return_value=post_response)
    get_response_starting = Mock()
    get_response_starting.status_code = 200
    get_response_starting.json.return_value = {"id": "test-prediction-id", "status": "starting", "output": None}
    get_response_processing = Mock()
    get_response_processing.status_code = 200
    get_response_processing.json.return_value = {"id": "test-prediction-id", "status": "processing", "output": None}
    get_response_succeeded = Mock()
    get_response_succeeded.status_code = 200
    get_response_succeeded.json.return_value = {
        "id": "test-prediction-id",
        "status": "succeeded",
        "output": ["Hello", " from", " DeepSeek!"],
    }
    get_response_succeeded.text = json.dumps(get_response_succeeded.json.return_value)
    get_response_succeeded.headers = {}
    mock_client.get = AsyncMock(side_effect=[get_response_starting, get_response_processing, get_response_succeeded])
    model_response = litellm.ModelResponse()
    model_response.choices = [litellm.Choices()]
    model_response.choices[0].message = litellm.Message(content="")
    mock_logging = Mock()
    mock_logging.post_call = Mock()
    result = await async_completion(
        model_response=model_response,
        model="deepseek-ai/deepseek-v3",
        messages=[{"role": "user", "content": "Hi"}],
        encoding=None,
        optional_params={},
        litellm_params={},
        version_id="deepseek-ai/deepseek-v3",
        input_data={"input": {"prompt": "test"}},
        api_key="test-key",
        api_base="https://api.replicate.com",
        logging_obj=mock_logging,
        print_verbose=print,
        headers={"Authorization": "Token test-key"},
    )
    assert result is not None
    assert result.choices[0].message.content == "Hello from DeepSeek!"
    assert mock_client.get.call_count == 3


class TestReplicateOutputFormats:
    @pytest.mark.usefixtures("fake_provider_credentials")
    def test_transform_response_list_output(self):
        """Test standard list output format"""
        from litellm.llms.replicate.chat.transformation import ReplicateConfig

        config = ReplicateConfig()

        # Mock response with list output
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "succeeded",
            "output": ["Hello", " ", "world"],
        }
        mock_response.text = json.dumps(mock_response.json.return_value)
        mock_response.headers = {}

        model_response = litellm.ModelResponse()
        model_response.choices = [litellm.Choices()]
        model_response.choices[0].message = litellm.Message(content="")

        mock_logging = Mock()
        mock_logging.post_call = Mock()

        result = config.transform_response(
            model="meta/llama-2-70b-chat",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
            request_data={"input": {"prompt": "test"}},
            messages=[{"role": "user", "content": "Hi"}],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key="test-key",
        )

        assert result.choices[0].message.content == "Hello world"


def test_transform_response_string_output():
    """Test string output format (as used by some DeepSeek models)"""
    from litellm.llms.replicate.chat.transformation import ReplicateConfig

    config = ReplicateConfig()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "succeeded", "output": "Hello from DeepSeek"}
    mock_response.text = json.dumps(mock_response.json.return_value)
    mock_response.headers = {}
    model_response = litellm.ModelResponse()
    model_response.choices = [litellm.Choices()]
    model_response.choices[0].message = litellm.Message(content="")
    mock_logging = Mock()
    mock_logging.post_call = Mock()
    result = config.transform_response(
        model="deepseek-ai/deepseek-v3",
        raw_response=mock_response,
        model_response=model_response,
        logging_obj=mock_logging,
        request_data={"input": {"prompt": "test"}},
        messages=[{"role": "user", "content": "Hi"}],
        optional_params={},
        litellm_params={},
        encoding=None,
        api_key="test-key",
    )
    assert result.choices[0].message.content == "Hello from DeepSeek"

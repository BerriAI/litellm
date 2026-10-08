import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from pydantic import ValidationError

import litellm
from litellm.llms.replicate.chat.handler import async_completion
from litellm.llms.replicate.chat.transformation import ReplicateConfig
from litellm.llms.replicate.common_utils import ReplicateError
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


def test_transform_response_reads_a_succeeded_prediction_that_carries_unmodelled_fields() -> None:
    response = _transform(
        httpx.Response(
            200,
            json={
                "id": "p1",
                "status": "succeeded",
                "output": ["Hello", " there"],
                "metrics": {"predict_time": 1.5, "stages": [1, {"name": None}]},
                "urls": {"get": "https://api.replicate.com/v1/predictions/p1"},
                "error": None,
            },
        )
    )

    assert response.choices[0].message.content == "Hello there"
    assert response.model == "replicate/acme/echo-model"


@pytest.mark.parametrize(
    ("body", "reported"),
    [
        pytest.param(
            {"status": "failed", "error": "boom", "output": None},
            "{'status': 'failed', 'error': 'boom', 'output': None}",
            id="failed",
        ),
        pytest.param(
            {"status": "processing", "output": ["partial"]},
            "{'status': 'processing', 'output': ['partial']}",
            id="still-processing",
        ),
        pytest.param(
            {"detail": {"b": [1, 2.5, True], "a": "z"}},
            "{'detail': {'b': [1, 2.5, True], 'a': 'z'}}",
            id="no-status",
        ),
        pytest.param({}, "{}", id="empty-object"),
    ],
)
def test_transform_response_reports_an_unsuccessful_prediction_body_as_received(
    body: dict[str, object], reported: str
) -> None:
    with pytest.raises(ReplicateError) as exc_info:
        _transform(httpx.Response(500, json=body, headers={"retry-after": "3"}))

    assert exc_info.value.status_code == 422
    assert exc_info.value.message == f"LiteLLM Error - prediction not succeeded - {reported}"
    assert exc_info.value.headers["retry-after"] == "3"


@pytest.mark.parametrize(
    ("body", "echo"),
    [
        pytest.param(["secret-output"], "secret-output", id="list"),
        pytest.param("secret-output", "secret-output", id="string"),
        pytest.param(4815162342, "4815162342", id="number"),
        pytest.param(None, "NoneType", id="null"),
    ],
)
def test_transform_response_rejects_a_body_that_is_not_an_object_without_echoing_it(body: object, echo: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        _transform(httpx.Response(500, content=json.dumps(body).encode()))

    assert echo not in str(exc_info.value)


@pytest.mark.parametrize(
    ("supplied", "stored"),
    [
        pytest.param({}, {}, id="nothing-supplied"),
        pytest.param(
            {"system_prompt": "", "max_new_tokens": 0, "temperature": 1, "debug": False},
            {"system_prompt": "", "max_new_tokens": 0, "temperature": 1, "debug": False},
            id="falsy-values-are-kept",
        ),
        pytest.param({"top_k": 40, "seed": None, "stop_sequences": None}, {"top_k": 40}, id="none-is-skipped"),
    ],
)
def test_init_stores_each_supplied_value_except_none_as_class_level_config(
    supplied: dict[str, object], stored: dict[str, object]
) -> None:
    class ScopedReplicateConfig(ReplicateConfig):
        pass

    ScopedReplicateConfig(**supplied)

    assert ScopedReplicateConfig.get_config() == stored


def test_init_called_again_with_none_keeps_the_value_stored_earlier() -> None:
    class ScopedReplicateConfig(ReplicateConfig):
        pass

    ScopedReplicateConfig(top_k=40)
    ScopedReplicateConfig(top_k=None, seed=5)

    assert ScopedReplicateConfig.get_config() == {"top_k": 40, "seed": 5}


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

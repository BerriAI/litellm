import json
import traceback
from unittest.mock import MagicMock, patch

import litellm
import pytest

from litellm.llms.triton.embedding.transformation import TritonEmbeddingConfig


def test_split_embedding_by_shape_passes():
    try:
        data = [{"shape": [2, 3], "data": [1, 2, 3, 4, 5, 6]}]
        split_output_data = TritonEmbeddingConfig.split_embedding_by_shape(data[0]["data"], data[0]["shape"])
        assert split_output_data == [[1, 2, 3], [4, 5, 6]]
    except Exception as e:
        pytest.fail(f"An exception occured: {e}")


def test_split_embedding_by_shape_fails_with_shape_value_error():
    data = [{"shape": [2], "data": [1, 2, 3, 4, 5, 6]}]
    with pytest.raises(ValueError, match="Shape must be of length"):
        TritonEmbeddingConfig.split_embedding_by_shape(data[0]["data"], data[0]["shape"])


def test_triton_embedding_response_sets_usage_with_token_counter():
    config = TritonEmbeddingConfig()
    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = {
        "model_name": "gte-base-en-v1",
        "outputs": [{"name": "embedding", "shape": [1, 2], "data": [0.1, 0.2]}],
    }
    model_response = litellm.EmbeddingResponse()
    request_data = {
        "inputs": [{"name": "input_text", "shape": [1], "datatype": "BYTES", "data": ["hello from triton"]}]
    }
    with patch("litellm.llms.triton.embedding.transformation.token_counter", return_value=7):
        transformed = config.transform_embedding_response(
            model="triton/gte-base-en-v1",
            raw_response=mock_http_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data=request_data,
        )
    assert transformed.usage is not None
    assert transformed.usage.prompt_tokens == 7
    assert transformed.usage.completion_tokens == 0
    assert transformed.usage.total_tokens == 7


def test_triton_embedding_response_sets_usage_with_word_count_fallback():
    config = TritonEmbeddingConfig()
    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = {
        "model_name": "gte-base-en-v1",
        "outputs": [{"name": "embedding", "shape": [1, 2], "data": [0.1, 0.2]}],
    }
    model_response = litellm.EmbeddingResponse()
    request_data = {
        "inputs": [{"name": "input_text", "shape": [1], "datatype": "BYTES", "data": ["hello from triton"]}]
    }
    with patch("litellm.llms.triton.embedding.transformation.token_counter", side_effect=Exception("tokenizer error")):
        transformed = config.transform_embedding_response(
            model="triton/gte-base-en-v1",
            raw_response=mock_http_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data=request_data,
        )
    assert transformed.usage is not None
    assert transformed.usage.prompt_tokens == 3
    assert transformed.usage.completion_tokens == 0
    assert transformed.usage.total_tokens == 3


def test_triton_embedding_batch_usage_sums_per_input_token_counts():
    """Batch inputs must not be joined before token counting (avoids extra newline tokens)."""
    config = TritonEmbeddingConfig()
    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = {
        "model_name": "gte-base-en-v1",
        "outputs": [{"name": "embedding", "shape": [2, 2], "data": [0.1, 0.2, 0.3, 0.4]}],
    }
    model_response = litellm.EmbeddingResponse()
    request_data = {
        "inputs": [{"name": "input_text", "shape": [2], "datatype": "BYTES", "data": ["first input", "second input"]}]
    }
    with patch("litellm.llms.triton.embedding.transformation.token_counter", side_effect=[5, 7]):
        transformed = config.transform_embedding_response(
            model="triton/gte-base-en-v1",
            raw_response=mock_http_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data=request_data,
        )
    assert transformed.usage is not None
    assert transformed.usage.prompt_tokens == 12
    assert transformed.usage.total_tokens == 12


def test_completion_triton_infer_api():
    litellm.set_verbose = True
    try:
        mock_response = MagicMock()

        def return_val():
            return {
                "model_name": "basketgpt",
                "model_version": "2",
                "outputs": [
                    {
                        "name": "text_output",
                        "datatype": "BYTES",
                        "shape": [1],
                        "data": [
                            "0004900005024 0004900006774 0004900005024 0004900005027 0004900005026 0004900005025 0004900005027 0004900005024 0004900006774 0004900005027"
                        ],
                    },
                    {
                        "name": "debug_probs",
                        "datatype": "FP32",
                        "shape": [0],
                        "data": [],
                    },
                    {
                        "name": "debug_tokens",
                        "datatype": "BYTES",
                        "shape": [0],
                        "data": [],
                    },
                ],
            }

        mock_response.json = return_val
        mock_response.status_code = 200

        with patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            return_value=mock_response,
        ) as mock_post:
            response = litellm.completion(
                model="triton/llama-3-8b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": "0004900005025 0004900005026 0004900005027",
                    }
                ],
                api_base="http://localhost:8000/infer",
            )

            print("litellm response", response.model_dump_json(indent=4))

            # Verify the call was made
            mock_post.assert_called_once()

            # Get the arguments passed to the post request
            call_kwargs = mock_post.call_args.kwargs

            # Verify URL
            assert call_kwargs["url"] == "http://localhost:8000/infer"

            # Parse the request data from the JSON string
            request_data = json.loads(call_kwargs["data"])

            # Verify request matches expected Triton format
            assert request_data["inputs"][0]["name"] == "text_input"
            assert request_data["inputs"][0]["shape"] == [1]
            assert request_data["inputs"][0]["datatype"] == "BYTES"
            assert request_data["inputs"][0]["data"] == [
                "0004900005025 0004900005026 0004900005027"
            ]

            assert request_data["inputs"][1]["shape"] == [1]
            assert request_data["inputs"][1]["datatype"] == "INT32"
            assert request_data["inputs"][1]["data"] == [20]

            # Verify response format matches expected completion format
            assert (
                response.choices[0].message.content
                == "0004900005024 0004900006774 0004900005024 0004900005027 0004900005026 0004900005025 0004900005027 0004900005024 0004900006774 0004900005027"
            )
            assert response.choices[0].finish_reason == "stop"
            assert response.choices[0].index == 0
            assert response.object == "chat.completion"

    except Exception as e:
        print("exception", e)
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")

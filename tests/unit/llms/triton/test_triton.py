import json
import traceback
from unittest.mock import MagicMock, patch

import httpx
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


LLAMA_3_CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.respx(assert_all_called=False)
def test_completion_triton_generate_api(stream, monkeypatch, respx_mock):
    monkeypatch.setattr(litellm, "known_tokenizer_config", dict(litellm.known_tokenizer_config))
    respx_mock.get(host="huggingface.co").respond(404, text="Entry not found")
    try:
        mock_response = MagicMock()
        if stream:

            def mock_iter_lines():
                mock_output = "".join(
                    [
                        'data: {"model_name":"ensemble","model_version":"1","sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"'
                        + t
                        + '"}\n\n'
                        for t in ["I", " am", " an", " AI", " assistant"]
                    ]
                )
                for out in mock_output.split("\n"):
                    yield out

            mock_response.iter_lines = mock_iter_lines
        else:

            def return_val():
                return {
                    "text_output": "I am an AI assistant",
                }

            mock_response.json = return_val
        mock_response.status_code = 200

        with patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            return_value=mock_response,
        ) as mock_post:
            response = litellm.completion(
                model="triton/llama-3-8b-instruct",
                messages=[{"role": "user", "content": "who are u?"}],
                max_tokens=10,
                timeout=5,
                api_base="http://localhost:8000/generate",
                stream=stream,
            )

            mock_post.assert_called_once()

            call_kwargs = mock_post.call_args.kwargs

            if stream:
                assert call_kwargs["url"] == "http://localhost:8000/generate_stream"
            else:
                assert call_kwargs["url"] == "http://localhost:8000/generate"

            request_data = json.loads(call_kwargs["data"])

            assert request_data["text_input"] == "who are u?"
            assert request_data["parameters"]["max_tokens"] == 10

            if stream:
                tokens = ["I", " am", " an", " AI", " assistant", None]
                idx = 0
                for chunk in response:
                    assert chunk.choices[0].delta.content == tokens[idx]
                    idx += 1
                assert idx == len(tokens)
            else:
                assert response.choices[0].message.content == "I am an AI assistant"

    except Exception as e:
        print("exception", e)
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.respx(assert_all_called=False)
def test_triton_generate_raw_request(monkeypatch, respx_mock):
    monkeypatch.setattr(litellm, "known_tokenizer_config", dict(litellm.known_tokenizer_config))
    respx_mock.get("https://huggingface.co/llama-3-8b-instruct/raw/main/tokenizer_config.json").respond(
        404, text="Entry not found"
    )
    respx_mock.get("https://huggingface.co/llama-3-8b-instruct/raw/main/chat_template.jinja").respond(
        200, text=LLAMA_3_CHAT_TEMPLATE
    )
    from litellm.utils import return_raw_request
    from litellm.types.utils import CallTypes

    try:
        kwargs = {
            "model": "triton/llama-3-8b-instruct",
            "messages": [{"role": "user", "content": "who are u?"}],
            "api_base": "http://localhost:8000/generate",
        }
        raw_request = return_raw_request(endpoint=CallTypes.completion, kwargs=kwargs)
        assert raw_request is not None
        assert "bad_words" not in json.dumps(raw_request["raw_request_body"])
        assert "stop_words" not in json.dumps(raw_request["raw_request_body"])
        assert raw_request["raw_request_body"]["text_input"] == (
            "<|start_header_id|>user<|end_header_id|>\n\nwho are u?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

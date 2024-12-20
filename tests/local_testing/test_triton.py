import pytest
from litellm.llms.triton.completion.handler import TritonChatCompletion


def test_split_embedding_by_shape_passes():
    try:
        triton = TritonChatCompletion()
        data = [
            {
                "shape": [2, 3],
                "data": [1, 2, 3, 4, 5, 6],
            }
        ]
        split_output_data = triton.split_embedding_by_shape(
            data[0]["data"], data[0]["shape"]
        )
        assert split_output_data == [[1, 2, 3], [4, 5, 6]]
    except Exception as e:
        pytest.fail(f"An exception occured: {e}")


def test_split_embedding_by_shape_fails_with_shape_value_error():
    triton = TritonChatCompletion()
    data = [
        {
            "shape": [2],
            "data": [1, 2, 3, 4, 5, 6],
        }
    ]
    with pytest.raises(ValueError):
        triton.split_embedding_by_shape(data[0]["data"], data[0]["shape"])


def test_completion_triton():
    from litellm import completion
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    from unittest.mock import patch, MagicMock, AsyncMock

    client = HTTPHandler()
    with patch.object(client, "post") as mock_post:
        try:
            response = completion(
                model="triton/llama-3-8b-instruct",
                messages=[{"role": "user", "content": "who are u?"}],
                max_tokens=10,
                timeout=5,
                client=client,
                api_base="http://localhost:8000/generate",
            )
            print(response)
        except Exception as e:
            print(e)

        mock_post.assert_called_once()

        print(mock_post.call_args.kwargs)

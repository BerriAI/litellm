import pytest
from litellm.llms.triton import TritonChatCompletion


def test_split_embedding_by_shape_passes():
    try:
        triton = TritonChatCompletion()
        data = [
            {
                "shape": [2, 3],
                "data": [1, 2, 3, 4, 5, 6],
            }
        ]
        split_output_data = triton.split_embedding_by_shape(data[0]["data"], data[0]["shape"])
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

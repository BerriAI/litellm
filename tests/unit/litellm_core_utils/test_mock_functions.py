from typing import Final

import pytest

import litellm
from litellm.exceptions import BadRequestError


@pytest.mark.parametrize(
    ("embedding_input", "expected_count"),
    [
        ("hello", 1),
        (["first", "second", "third"], 3),
        ([101, 102, 103], 1),
        ([[101, 102], [103, 104]], 2),
    ],
)
def test_mock_embedding_returns_one_embedding_per_input(embedding_input: object, expected_count: int) -> None:
    vector: Final = [0.1, 0.2]

    response: Final = litellm.embedding(model="openai/test-model", input=embedding_input, mock_response=vector)

    assert [row["index"] for row in response.data] == list(range(expected_count))
    assert [row["embedding"] for row in response.data] == [vector] * expected_count


@pytest.mark.parametrize("embedding_input", [[], ["text"] * 2049])
def test_mock_embedding_rejects_input_count_a_provider_would_reject(embedding_input: list[str]) -> None:
    with pytest.raises(BadRequestError, match="between 1 and 2048 items"):
        litellm.embedding(model="openai/test-model", input=embedding_input, mock_response=[0.1, 0.2])

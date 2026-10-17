import datetime
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.ovhcloud.embedding.transformation import OVHCloudEmbeddingConfig
from litellm.types.utils import EmbeddingResponse

model = "ovhcloud/BGE-M3"


def mock_embedding_response(*args, **kwargs):
    class MockResponse:
        def __init__(self):
            self.data = [{"embedding": [0.1, 0.2, 0.3]}]
            self.usage = litellm.Usage()
            self.model = kwargs.get("model", model)
            self.object = "embedding"

        def __getitem__(self, key):
            return getattr(self, key)

    return MockResponse()


def test_ovhcloud_embeddings():
    with patch("litellm.embedding", side_effect=mock_embedding_response) as mock_embed:
        response = litellm.embedding(
            model,
            input=["good morning from litellm"],
        )

        mock_embed.assert_called_once_with(
            model,
            input=["good morning from litellm"],
        )

        assert isinstance(response.data, list)
        assert "embedding" in response.data[0]
        assert isinstance(response.data[0]["embedding"], list)
        assert response.model == model
        assert response.object == "embedding"


EMBEDDING_ROW = {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}


def _transform(payload: object) -> EmbeddingResponse:
    logging_obj = Logging(
        model="ovhcloud/BGE-M3",
        messages=[],
        stream=False,
        call_type="embedding",
        start_time=datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        litellm_call_id="call-id",
        function_id="function-id",
    )
    return OVHCloudEmbeddingConfig().transform_embedding_response(
        model="BGE-M3",
        raw_response=httpx.Response(200, json=payload),
        model_response=EmbeddingResponse(),
        logging_obj=logging_obj,
        api_key="sk-test",
        request_data={},
        optional_params={},
        litellm_params={},
    )


def _token_counts(response: EmbeddingResponse) -> tuple[int, int, int]:
    assert response.usage is not None
    return (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens)


def test_ovhcloud_embedding_response_carries_provider_fields_and_usage():
    response = _transform(
        {
            "object": "list",
            "model": "BGE-M3",
            "data": [EMBEDDING_ROW],
            "usage": {"prompt_tokens": 8, "total_tokens": 9},
        }
    )

    assert response.model == "BGE-M3"
    assert response.object == "list"
    assert response.data == [EMBEDDING_ROW]
    assert _token_counts(response) == (8, 0, 9)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"data": [EMBEDDING_ROW]}, (0, 0, 0)),
        ({"data": [EMBEDDING_ROW], "usage": {}}, (0, 0, 0)),
        ({"data": [EMBEDDING_ROW], "usage": {"prompt_tokens": 4}}, (4, 0, 0)),
        ({"data": [EMBEDDING_ROW], "usage": {"total_tokens": 6}}, (0, 0, 6)),
        ({"data": [EMBEDDING_ROW], "usage": {"prompt_tokens": None, "total_tokens": None}}, (0, 0, 0)),
        ({"data": [EMBEDDING_ROW], "usage": {"prompt_tokens": "", "total_tokens": []}}, (0, 0, 0)),
        ({"data": [EMBEDDING_ROW], "usage": {"prompt_tokens": "12", "total_tokens": 12.0}}, (12, 0, 12)),
        (
            {"data": [EMBEDDING_ROW], "usage": {"prompt_tokens": 3, "total_tokens": 5, "completion_tokens": 2}},
            (3, 0, 5),
        ),
    ],
)
def test_ovhcloud_embedding_usage_tolerates_missing_null_and_loosely_typed_counts(payload, expected):
    response = _transform(payload)

    assert response.data == [EMBEDDING_ROW]
    assert _token_counts(response) == expected


@pytest.mark.parametrize("usage", [None, [], "tokens", 3])
def test_ovhcloud_embedding_usage_that_is_not_an_object_is_rejected(usage):
    with pytest.raises(ValidationError):
        _transform({"data": [EMBEDDING_ROW], "usage": usage})


@pytest.mark.parametrize("usage", [{"prompt_tokens": "many"}, {"total_tokens": 1.5}, {"prompt_tokens": [1]}])
def test_ovhcloud_embedding_token_count_that_is_not_an_integer_is_rejected(usage):
    with pytest.raises(ValidationError):
        _transform({"data": [EMBEDDING_ROW], "usage": usage})

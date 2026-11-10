import datetime

import httpx
import pytest
from pydantic import ValidationError

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.cometapi.embed.transformation import CometAPIEmbeddingConfig
from litellm.types.utils import EmbeddingResponse

EMBEDDING_ROW = {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}


def _transform(payload: object) -> EmbeddingResponse:
    logging_obj = Logging(
        model="cometapi/text-embedding-3-small",
        messages=[],
        stream=False,
        call_type="embedding",
        start_time=datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        litellm_call_id="call-id",
        function_id="function-id",
    )
    return CometAPIEmbeddingConfig().transform_embedding_response(
        model="text-embedding-3-small",
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


def test_cometapi_embedding_response_carries_provider_fields_and_usage():
    response = _transform(
        {
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [EMBEDDING_ROW],
            "usage": {"prompt_tokens": 8, "total_tokens": 9},
        }
    )

    assert response.model == "text-embedding-3-small"
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
def test_cometapi_embedding_usage_tolerates_missing_null_and_loosely_typed_counts(payload, expected):
    response = _transform(payload)

    assert response.data == [EMBEDDING_ROW]
    assert _token_counts(response) == expected


@pytest.mark.parametrize("usage", [None, [], "tokens", 3])
def test_cometapi_embedding_usage_that_is_not_an_object_is_rejected(usage):
    with pytest.raises(ValidationError):
        _transform({"data": [EMBEDDING_ROW], "usage": usage})


@pytest.mark.parametrize("usage", [{"prompt_tokens": "many"}, {"total_tokens": 1.5}, {"prompt_tokens": [1]}])
def test_cometapi_embedding_token_count_that_is_not_an_integer_is_rejected(usage):
    with pytest.raises(ValidationError):
        _transform({"data": [EMBEDDING_ROW], "usage": usage})

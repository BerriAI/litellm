import datetime

import httpx
import pytest
from pydantic import ValidationError

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.cohere.embed.transformation import CohereEmbeddingConfig
from litellm.types.utils import EmbeddingResponse

FLOAT_VECTORS = [[0.1, 0.2], [0.3, 0.4]]
INT8_VECTORS = [[1, -2], [3, 4]]


def _transform(payload: object) -> EmbeddingResponse:
    logging_obj = Logging(
        model="embed-english-v3.0",
        messages=[],
        stream=False,
        call_type="embedding",
        start_time=datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        litellm_call_id="call-id",
        function_id="function-id",
    )
    logging_obj.model_call_details["input"] = ["hello", "world"]
    return CohereEmbeddingConfig().transform_embedding_response(
        model="embed-english-v3.0",
        raw_response=httpx.Response(200, json=payload),
        model_response=EmbeddingResponse(),
        logging_obj=logging_obj,
        api_key="sk-test",
        request_data={"model": "embed-english-v3.0", "texts": ["hello", "world"]},
        optional_params={},
        litellm_params={},
    )


def _token_counts(response: EmbeddingResponse) -> tuple[int, int, int]:
    assert response.usage is not None
    return (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens)


def test_cohere_embedding_response_lists_every_embedding_type_in_provider_order():
    response = _transform(
        {
            "id": "embed-id",
            "embeddings": {"float": FLOAT_VECTORS, "int8": INT8_VECTORS},
            "meta": {"billed_units": {"input_tokens": 5}},
        }
    )

    assert response.object == "list"
    assert response.model == "embed-english-v3.0"
    assert response.data == [
        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
        {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
        {"object": "embedding", "index": 0, "embedding": [1, -2]},
        {"object": "embedding", "index": 1, "embedding": [3, 4]},
    ]
    assert _token_counts(response) == (5, 0, 5)


@pytest.mark.parametrize(
    ("meta", "expected_counts", "expected_details"),
    [
        ({"billed_units": {"input_tokens": 5}}, (5, 0, 5), (5, None)),
        ({"billed_units": {"images": 3}}, (3, 0, 3), (None, 3)),
        ({"billed_units": {"input_tokens": 5, "images": 3}}, (8, 0, 8), (5, 3)),
        ({"billed_units": {"input_tokens": 0}}, (0, 0, 0), (0, None)),
        (
            {"api_version": {"version": "2"}, "billed_units": {"input_tokens": 2, "output_tokens": 9}},
            (2, 0, 2),
            (2, None),
        ),
    ],
)
def test_cohere_embedding_usage_comes_from_billed_units(meta, expected_counts, expected_details):
    response = _transform({"embeddings": {"float": FLOAT_VECTORS}, "meta": meta})

    assert response.usage is not None
    details = response.usage.prompt_tokens_details
    assert details is not None
    assert _token_counts(response) == expected_counts
    assert (details.text_tokens, details.image_tokens) == expected_details


@pytest.mark.parametrize(
    "payload",
    [
        {"embeddings": {"float": FLOAT_VECTORS}},
        {"embeddings": {"float": FLOAT_VECTORS}, "meta": {}},
        {"embeddings": {"float": FLOAT_VECTORS}, "meta": {"billed_units": {}}},
        {"embeddings": {"float": FLOAT_VECTORS}, "meta": {"billed_units": {"input_tokens": None, "images": None}}},
    ],
)
def test_cohere_embedding_usage_counts_input_tokens_when_billed_units_are_absent(payload):
    response = _transform(payload)

    assert response.usage is not None
    assert response.usage.prompt_tokens_details is None
    assert _token_counts(response) == (2, 0, 2)


@pytest.mark.parametrize(
    ("embeddings", "expected_vectors"),
    [
        ({}, []),
        ({"float": []}, []),
        ({"float": [], "int8": INT8_VECTORS}, INT8_VECTORS),
        ({"base64": ["AAAA", "BBBB"]}, ["AAAA", "BBBB"]),
        ({"float": [[0.1], "x", None, 3]}, [[0.1], "x", None, 3]),
    ],
)
def test_cohere_embedding_rows_pass_through_whatever_each_type_lists(embeddings, expected_vectors):
    response = _transform({"embeddings": embeddings, "meta": {"billed_units": {"input_tokens": 1}}})

    assert [row["embedding"] for row in response.data] == expected_vectors


@pytest.mark.parametrize("embeddings", [FLOAT_VECTORS, [], None, "vectors", 3])
def test_cohere_embeddings_that_are_not_keyed_by_type_are_rejected(embeddings):
    with pytest.raises(ValidationError):
        _transform({"embeddings": embeddings, "meta": {"billed_units": {"input_tokens": 1}}})


@pytest.mark.parametrize("vectors", [None, 3, False, 1.5])
def test_cohere_embedding_type_without_a_list_of_vectors_is_rejected(vectors):
    with pytest.raises(ValidationError):
        _transform({"embeddings": {"float": vectors}, "meta": {"billed_units": {"input_tokens": 1}}})


@pytest.mark.parametrize("meta", [None, [], "meta", 3])
def test_cohere_embedding_meta_that_is_not_an_object_is_rejected(meta):
    with pytest.raises(ValidationError):
        _transform({"embeddings": {"float": FLOAT_VECTORS}, "meta": meta})


@pytest.mark.parametrize("payload", [[1], "text", 3, True])
def test_cohere_embedding_body_that_is_not_an_object_is_rejected(payload):
    with pytest.raises(ValidationError):
        _transform(payload)


def test_cohere_embedding_body_without_embeddings_is_rejected():
    with pytest.raises(KeyError):
        _transform({"meta": {"billed_units": {"input_tokens": 1}}})

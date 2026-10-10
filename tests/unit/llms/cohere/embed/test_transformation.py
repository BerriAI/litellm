import datetime
from typing import Final

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.cohere.embed.transformation import CohereEmbeddingConfig
from litellm.types.utils import EmbeddingResponse


def _cohere_embedding_logging_obj() -> Logging:
    return Logging(
        model="embed-v4.0",
        messages=["first", "second"],
        stream=False,
        call_type="embedding",
        start_time=datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        litellm_call_id="cohere-embedding-test-call",
        function_id="cohere-embedding-test-function",
    )


def test_cohere_embed_v4_request_maps_input_options() -> None:
    config: Final = CohereEmbeddingConfig()

    request: Final = config.transform_embedding_request(
        model="embed-v4.0",
        input=["first", "second"],
        optional_params={
            "input_type": "search_document",
            "embedding_types": ["float"],
            "output_dimension": 256,
        },
        headers={},
    )

    assert request == {
        "model": "embed-v4.0",
        "texts": ["first", "second"],
        "input_type": "search_document",
        "embedding_types": ["float"],
        "output_dimension": 256,
    }


def test_cohere_embed_v4_request_routes_base64_images() -> None:
    config: Final = CohereEmbeddingConfig()

    request: Final = config.transform_embedding_request(
        model="embed-v4.0",
        input=["data:image/png;base64,aGVsbG8="],
        optional_params={"input_type": "search_document"},
        headers={},
    )

    assert request == {
        "model": "embed-v4.0",
        "images": ["data:image/png;base64,aGVsbG8="],
        "input_type": "search_document",
    }


def test_cohere_embed_v4_response_parses_multiple_embeddings() -> None:
    config: Final = CohereEmbeddingConfig()
    model_response: Final = EmbeddingResponse()
    raw_response: Final = httpx.Response(
        200,
        json={
            "embeddings": {
                "float": [[0.1, 0.2], [0.3, 0.4]],
                "int8": [[1, 2], [3, 4]],
            },
            "meta": {"billed_units": {"input_tokens": 6}},
        },
    )

    response: Final = config.transform_embedding_response(
        model="embed-v4.0",
        raw_response=raw_response,
        model_response=model_response,
        logging_obj=_cohere_embedding_logging_obj(),
        api_key="test-api-key",
        request_data={"texts": ["first", "second"]},
        optional_params={"embedding_types": ["float", "int8"]},
        litellm_params={},
    )

    assert response.object == "list"
    assert response.model == "embed-v4.0"
    assert [item["embedding"] for item in response.data] == [
        [0.1, 0.2],
        [0.3, 0.4],
        [1, 2],
        [3, 4],
    ]
    assert [item["index"] for item in response.data] == [0, 1, 0, 1]
    assert response.usage.prompt_tokens == 6
    assert response.usage.total_tokens == 6

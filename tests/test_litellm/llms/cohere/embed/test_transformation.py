from typing import Final
from unittest.mock import MagicMock, patch

import httpx

from litellm.llms.cohere.embed.transformation import CohereEmbeddingConfig
from litellm.types.llms.cohere import CohereEmbeddingInput
from litellm.types.utils import EmbeddingResponse


def test_transform_embedding_request_preserves_mixed_inputs() -> None:
    config: Final = CohereEmbeddingConfig()
    inputs: Final[list[CohereEmbeddingInput]] = [
        {
            "content": [
                {"type": "text", "text": "a red shoe"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ]
        }
    ]

    request: Final = config.transform_embedding_request(
        model="embed-v4.0",
        input=inputs,
        optional_params={
            "input_type": "search_document",
            "output_dimension": 1536,
            "embedding_types": ["float"],
        },
        headers={},
    )

    assert request == {
        "model": "embed-v4.0",
        "inputs": inputs,
        "input_type": "search_document",
        "output_dimension": 1536,
        "embedding_types": ["float"],
    }
    assert "texts" not in request
    assert "images" not in request


def test_transform_embedding_response_uses_multimodal_billing_metadata() -> None:
    config: Final = CohereEmbeddingConfig()
    inputs: Final[list[CohereEmbeddingInput]] = [
        {
            "content": [
                {"type": "text", "text": "a red shoe"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ]
        }
    ]
    response: Final = httpx.Response(
        200,
        json={
            "embeddings": {"float": [[0.1, 0.2, 0.3]]},
            "meta": {"billed_units": {"input_tokens": 3, "images": 1}},
        },
    )
    logging_obj: Final = MagicMock()
    logging_obj.model_call_details = {"input": inputs}
    encoding: Final = MagicMock()

    with patch("litellm.encoding", encoding):
        result: Final = config.transform_embedding_response(
            model="embed-v4.0",
            raw_response=response,
            model_response=EmbeddingResponse(),
            logging_obj=logging_obj,
            api_key="test-api-key",
            request_data={
                "model": "embed-v4.0",
                "inputs": inputs,
                "input_type": "search_document",
            },
            optional_params={},
            litellm_params={},
        )

    assert result.data == [
        {
            "object": "embedding",
            "index": 0,
            "embedding": [0.1, 0.2, 0.3],
        }
    ]
    assert result.usage.prompt_tokens == 4
    assert result.usage.prompt_tokens_details.text_tokens == 3
    assert result.usage.prompt_tokens_details.image_tokens == 1
    encoding.encode.assert_not_called()


def test_multimodal_usage_fallback_counts_text_content() -> None:
    config: Final = CohereEmbeddingConfig()
    inputs: Final[list[CohereEmbeddingInput]] = [
        {
            "content": [
                {"type": "text", "text": "a red shoe"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ]
        }
    ]
    encoding: Final = MagicMock()
    encoding.encode.return_value = [1, 2, 3]

    usage: Final = config._calculate_usage(inputs, encoding, {})

    assert usage.prompt_tokens == 3
    assert usage.total_tokens == 3
    assert usage.prompt_tokens_details is None
    encoding.encode.assert_called_once_with("a red shoe")

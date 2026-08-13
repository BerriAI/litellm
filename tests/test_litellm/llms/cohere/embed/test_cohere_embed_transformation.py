from unittest.mock import MagicMock

import httpx

from litellm.llms.cohere.embed.transformation import CohereEmbeddingConfig
from litellm.types.utils import EmbeddingResponse


class _FakeEncoding:
    """Minimal tiktoken-like stub: token count == character count.

    Mirrors tiktoken's real behavior of raising on non-str input, so this
    stub reproduces the TypeError a dict input would trigger in production.
    """

    def encode(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError(f"expected str, got {type(text)}")
        return list(text)


def test_transform_request_preserves_multimodal_inputs():
    inputs = [
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

    result = CohereEmbeddingConfig().transform_embedding_request(
        model="embed-v4.0",
        input=inputs,
        optional_params={"input_type": "search_document", "output_dimension": 1536},
        headers={},
    )

    assert result == {
        "model": "embed-v4.0",
        "inputs": inputs,
        "input_type": "search_document",
        "output_dimension": 1536,
    }


def test_transform_request_preserves_text_inputs():
    result = CohereEmbeddingConfig().transform_embedding_request(
        model="embed-v4.0",
        input=["a red shoe"],
        optional_params={"input_type": "search_document"},
        headers={},
    )

    assert result == {
        "model": "embed-v4.0",
        "texts": ["a red shoe"],
        "input_type": "search_document",
    }


def test_transform_request_preserves_image_inputs():
    image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"

    result = CohereEmbeddingConfig().transform_embedding_request(
        model="embed-v4.0",
        input=[image],
        optional_params={},
        headers={},
    )

    assert result == {
        "model": "embed-v4.0",
        "images": [image],
        "input_type": "image",
    }


def test_transform_response_handles_multimodal_inputs():
    """Multimodal (dict) inputs must not crash token counting in the response path."""
    inputs = [
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

    raw_response = httpx.Response(
        status_code=200,
        json={
            "embeddings": {"float": [[0.1, 0.2, 0.3]]},
            "meta": {},
        },
        request=httpx.Request("POST", "https://api.cohere.ai/v2/embed"),
    )

    result = CohereEmbeddingConfig()._transform_response(
        response=raw_response,
        api_key=None,
        logging_obj=MagicMock(),
        data={},
        model_response=EmbeddingResponse(),
        model="embed-v4.0",
        encoding=_FakeEncoding(),
        input=inputs,
    )

    assert result.data == [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}]
    assert result.usage is not None
    assert result.usage.prompt_tokens == len("a red shoe")

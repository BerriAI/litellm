from datetime import datetime, timezone
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.cohere.embed.transformation import CohereEmbeddingConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import CallTypes, EmbeddingResponse


def _logging_obj(inputs: list[str]) -> Logging:
    logging_obj: Final = Logging(
        model="embed-english-v3.0",
        messages=inputs,
        stream=False,
        call_type=CallTypes.embedding,
        start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        litellm_call_id="test-call",
        function_id="test-function",
    )
    logging_obj.model_call_details = {"input": inputs}
    return logging_obj


def test_cohere_text_embedding_request_maps_openai_params():
    config: Final = CohereEmbeddingConfig()
    request: Final = config.transform_embedding_request(
        model="embed-english-v3.0",
        input=["hello", "world"],
        optional_params={"output_dimension": 256, "embedding_types": ["float"]},
        headers={},
    )

    assert request == {
        "model": "embed-english-v3.0",
        "texts": ["hello", "world"],
        "input_type": litellm.COHERE_DEFAULT_EMBEDDING_INPUT_TYPE,
        "output_dimension": 256,
        "embedding_types": ["float"],
    }


def test_cohere_image_embedding_request_uses_images():
    config: Final = CohereEmbeddingConfig()
    request: Final = config.transform_embedding_request(
        model="embed-english-v3.0",
        input=["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"],
        optional_params={},
        headers={},
    )

    assert request == {
        "model": "embed-english-v3.0",
        "images": ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"],
        "input_type": "image",
    }


def test_cohere_embedding_response_preserves_vectors_and_provider_usage():
    config: Final = CohereEmbeddingConfig()
    inputs: Final = ["hello", "world"]
    logging_obj: Final = _logging_obj(inputs)
    response: Final = config.transform_embedding_response(
        model="embed-english-v3.0",
        raw_response=httpx.Response(
            200,
            json={
                "embeddings": {"float": [[0.1, 0.2], [0.3, 0.4]]},
                "meta": {"billed_units": {"input_tokens": 5}},
            },
        ),
        model_response=EmbeddingResponse(),
        logging_obj=logging_obj,
        api_key="test-key",
        request_data={"texts": inputs},
        optional_params={},
        litellm_params={},
    )

    assert response.model == "embed-english-v3.0"
    assert response.data == [
        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
        {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
    ]
    assert (response.usage.prompt_tokens, response.usage.total_tokens) == (5, 5)


def test_cohere_embedding_response_counts_tokens_when_provider_usage_is_missing():
    config: Final = CohereEmbeddingConfig()
    inputs: Final = ["hello"]
    logging_obj: Final = _logging_obj(inputs)
    response: Final = config.transform_embedding_response(
        model="embed-english-v3.0",
        raw_response=httpx.Response(200, json={"embeddings": {"float": [[0.1]]}}),
        model_response=EmbeddingResponse(),
        logging_obj=logging_obj,
        api_key="test-key",
        request_data={"texts": inputs},
        optional_params={},
        litellm_params={},
    )

    assert response.usage.prompt_tokens == len(litellm.encoding.encode("hello"))
    assert response.usage.total_tokens == response.usage.prompt_tokens


def test_cohere_embedding_usage_includes_text_and_image_token_details():
    config: Final = CohereEmbeddingConfig()
    usage: Final = config._calculate_usage(
        input=["hello"],
        encoding=litellm.encoding,
        meta={"billed_units": {"input_tokens": 5, "images": 2}},
    )

    assert usage.prompt_tokens == 7
    assert (usage.prompt_tokens_details.text_tokens, usage.prompt_tokens_details.image_tokens) == (5, 2)


def test_cohere_embedding_error_class_preserves_status_and_message():
    error: Final = CohereEmbeddingConfig().get_error_class(
        error_message="invalid embedding request",
        status_code=400,
        headers={},
    )

    assert error.status_code == 400
    assert "invalid embedding request" in str(error)


@pytest.mark.parametrize("sync_mode", [True, False])
async def test_embedding_with_extra_headers_are_sent_to_cohere(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, sync_mode: bool
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.cohere.ai/v2/embed").mock(
        return_value=httpx.Response(
            200,
            json={"embeddings": {"float": [[0.1, 0.2]]}, "meta": {"billed_units": {"input_tokens": 1}}},
        )
    )
    request: Final = {
        "model": "cohere/embed-english-v3.0",
        "input": ["hello"],
        "api_key": "test-key",
        "extra_headers": {"x-request-id": "test-request"},
    }

    response: Final = litellm.embedding(**request) if sync_mode else await litellm.aembedding(**request)

    assert route.calls.last.request.headers["x-request-id"] == "test-request"
    assert response.data[0]["embedding"] == [0.1, 0.2]


async def test_aembedding_sends_extra_headers_through_the_caller_supplied_async_client() -> None:
    requests: Final[list[httpx.Request]] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"embeddings": {"float": [[0.3, 0.4]]}, "meta": {"billed_units": {"input_tokens": 2}}},
        )

    response: Final = await litellm.aembedding(
        model="cohere/embed-english-v3.0",
        input=["hello world"],
        api_key="test-key",
        extra_headers={"my-test-param": "hello-world"},
        client=AsyncHTTPHandler(transport=httpx.MockTransport(handle_request)),
    )

    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.cohere.ai/v2/embed"
    assert requests[0].headers["my-test-param"] == "hello-world"
    assert response.data[0]["embedding"] == [0.3, 0.4]

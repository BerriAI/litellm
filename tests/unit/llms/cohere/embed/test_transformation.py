import datetime
import json
from collections.abc import Iterator
from typing import Final

import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
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


COHERE_V2_EMBED_URL: Final = "https://api.cohere.ai/v2/embed"


@pytest.fixture
def _cohere_httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    client_cache: Final = LLMClientCache()
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", client_cache)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "force_ipv4", False)
    yield
    client_cache.flush_cache()


def _cohere_v2_embed_response(vectors: list[list[float]], texts: list[str]) -> dict[str, object]:
    return {
        "id": "da6e531f-83f4-4bcb-8e9a-f1a2c6e7a1b3",
        "embeddings": {"float": vectors},
        "texts": texts,
        "meta": {"api_version": {"version": "2"}, "billed_units": {"input_tokens": 9}},
        "response_type": "embeddings_by_type",
    }


async def _embed(sync_mode: bool, **kwargs: object) -> EmbeddingResponse:
    if sync_mode:
        return litellm.embedding(model="cohere/embed-v4.0", api_key="cohere-test-key", **kwargs)
    return await litellm.aembedding(model="cohere/embed-v4.0", api_key="cohere-test-key", **kwargs)


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


@pytest.mark.usefixtures("_cohere_httpx_transport")
@pytest.mark.respx(assert_all_called=True)
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_cohere_embed_v4_multiple_texts(respx_mock: respx.MockRouter, sync_mode: bool) -> None:
    texts: Final = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming the world",
        "Python is a versatile programming language",
        "Natural language processing enables human-computer interaction",
    ]
    vectors: Final = [[0.011, -0.022], [0.033, -0.044], [0.055, -0.066], [0.077, -0.088]]
    route: Final = respx_mock.post(COHERE_V2_EMBED_URL).mock(
        return_value=httpx.Response(200, json=_cohere_v2_embed_response(vectors, texts))
    )

    response: Final = await _embed(sync_mode, input=texts, input_type="search_document")

    assert json.loads(route.calls.last.request.content) == {
        "model": "embed-v4.0",
        "texts": texts,
        "input_type": "search_document",
    }
    assert response.model == "embed-v4.0"
    assert response.data == [
        {"object": "embedding", "index": index, "embedding": vector} for index, vector in enumerate(vectors)
    ]
    assert response.usage.prompt_tokens == 9


@pytest.mark.usefixtures("_cohere_httpx_transport")
@pytest.mark.respx(assert_all_called=True)
@pytest.mark.parametrize("input_type", ["search_document", "search_query", "classification", "clustering"])
async def test_cohere_embed_v4_input_types(respx_mock: respx.MockRouter, input_type: str) -> None:
    text: Final = f"Test text for {input_type}"
    route: Final = respx_mock.post(COHERE_V2_EMBED_URL).mock(
        return_value=httpx.Response(200, json=_cohere_v2_embed_response([[0.5, -0.5]], [text]))
    )

    await _embed(False, input=[text], input_type=input_type)

    assert json.loads(route.calls.last.request.content) == {
        "model": "embed-v4.0",
        "texts": [text],
        "input_type": input_type,
    }


@pytest.mark.usefixtures("_cohere_httpx_transport")
@pytest.mark.respx(assert_all_called=True)
async def test_cohere_embed_v4_encoding_format(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(COHERE_V2_EMBED_URL).mock(
        return_value=httpx.Response(200, json=_cohere_v2_embed_response([[0.25, -0.75, 0.5]], ["Test encoding format"]))
    )

    response: Final = await _embed(True, input=["Test encoding format"], encoding_format="float")

    assert json.loads(route.calls.last.request.content) == {
        "model": "embed-v4.0",
        "texts": ["Test encoding format"],
        "input_type": "search_document",
        "embedding_types": ["float"],
    }
    assert response.data == [{"object": "embedding", "index": 0, "embedding": [0.25, -0.75, 0.5]}]


@pytest.mark.usefixtures("_cohere_httpx_transport")
@pytest.mark.respx(assert_all_called=True)
async def test_cohere_embed_v4_with_optional_params(respx_mock: respx.MockRouter) -> None:
    vector: Final = [round(index * 0.001, 3) for index in range(256)]
    route: Final = respx_mock.post(COHERE_V2_EMBED_URL).mock(
        return_value=httpx.Response(200, json=_cohere_v2_embed_response([vector], ["Test with optional parameters"]))
    )

    response: Final = await _embed(
        True,
        input=["Test with optional parameters"],
        input_type="search_query",
        dimensions=256,
        encoding_format="float",
    )

    assert json.loads(route.calls.last.request.content) == {
        "model": "embed-v4.0",
        "texts": ["Test with optional parameters"],
        "input_type": "search_query",
        "embedding_types": ["float"],
        "output_dimension": 256,
    }
    assert response.data == [{"object": "embedding", "index": 0, "embedding": vector}]


@pytest.mark.usefixtures("_cohere_httpx_transport")
@pytest.mark.respx(assert_all_called=True)
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_cohere_embed_v4_image_embedding(respx_mock: respx.MockRouter, sync_mode: bool) -> None:
    image: Final = (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    )
    route: Final = respx_mock.post(COHERE_V2_EMBED_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "5807ee2e-0cda-445a-9ec8-864c60a06606",
                "embeddings": {"float": [[0.125, -0.375]]},
                "texts": [],
                "images": [{"width": 1, "height": 1, "format": "image/png", "bit_depth": 8}],
                "meta": {"api_version": {"version": "2"}, "billed_units": {"images": 1}},
                "response_type": "embeddings_by_type",
            },
        )
    )

    response: Final = await _embed(sync_mode, input=[image], input_type="image")

    assert json.loads(route.calls.last.request.content) == {
        "model": "embed-v4.0",
        "images": [image],
        "input_type": "image",
    }
    assert response.data == [{"object": "embedding", "index": 0, "embedding": [0.125, -0.375]}]

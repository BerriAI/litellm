import datetime
from collections.abc import Mapping
from typing import Final

import httpx
import pytest
from pydantic import ValidationError

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.voyage.embedding.transformation_multimodal import VoyageMultimodalEmbeddingConfig
from litellm.types.utils import EmbeddingResponse

EMBEDDING_ROW = {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}


def _transform(payload: object) -> EmbeddingResponse:
    logging_obj = Logging(
        model="voyage/voyage-multimodal-3",
        messages=[],
        stream=False,
        call_type="embedding",
        start_time=datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        litellm_call_id="call-id",
        function_id="function-id",
    )
    return VoyageMultimodalEmbeddingConfig().transform_embedding_response(
        model="voyage-multimodal-3",
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


def test_voyage_multimodal_embedding_response_carries_provider_fields_and_usage():
    response = _transform(
        {
            "object": "list",
            "model": "voyage-multimodal-3",
            "data": [EMBEDDING_ROW],
            "usage": {"prompt_tokens": 8, "total_tokens": 9},
        }
    )

    assert response.model == "voyage-multimodal-3"
    assert response.object == "list"
    assert response.data == [EMBEDDING_ROW]
    assert _token_counts(response) == (9, 0, 9)


def test_voyage_multimodal_response_preserves_reported_usage_breakdown() -> None:
    response: Final = _transform(
        {"usage": {"total_tokens": 7, "text_tokens": 0, "image_pixels": 0, "video_pixels": "4000"}}
    )

    assert response.usage is not None
    details: Final = response.usage.prompt_tokens_details
    assert details is not None
    assert (details.text_tokens, details.image_pixels, details.video_pixels) == (0, 0, 4000)


@pytest.mark.parametrize(
    ("usage", "priced_text_tokens", "priced_pixels"),
    [
        ({"total_tokens": 14, "text_tokens": 9, "image_pixels": 9000, "video_pixels": 4000}, 9, 13000),
        ({"total_tokens": 6, "text_tokens": 0, "image_pixels": 9000}, 0, 9000),
        ({"total_tokens": 7, "text_tokens": 0, "image_pixels": 0, "video_pixels": 4000}, 0, 4000),
        ({"total_tokens": 8, "text_tokens": 8, "image_pixels": 0, "video_pixels": 0}, 8, 0),
        ({"total_tokens": 14, "text_tokens": 9}, 14, 0),
        ({"total_tokens": 14, "image_pixels": 9000}, 14, 0),
        ({"total_tokens": 14, "text_tokens": 9, "image_pixels": -1, "video_pixels": 4000}, 14, 0),
    ],
    ids=["mixed", "image-only", "video-only", "text-only", "no-pixels", "no-text-count", "malformed"],
)
def test_voyage_multimodal_embedding_prices_text_tokens_and_reported_pixels(
    monkeypatch: pytest.MonkeyPatch,
    usage: Mapping[str, int],
    priced_text_tokens: int,
    priced_pixels: int,
) -> None:
    model: Final = "voyage-multimodal-pixel-pricing-test"
    text_rate: Final = 0.01
    pixel_rate: Final = 0.00002
    monkeypatch.setitem(
        litellm.model_cost,
        f"voyage/{model}",
        {
            "litellm_provider": "voyage",
            "mode": "embedding",
            "input_cost_per_token": text_rate,
            "input_cost_per_pixel": pixel_rate,
        },
    )
    response: Final = _transform({"model": model, "data": [EMBEDDING_ROW], "usage": dict(usage)})

    assert _token_counts(response) == (usage["total_tokens"], 0, usage["total_tokens"])
    assert litellm.completion_cost(completion_response=response, model=f"voyage/{model}") == pytest.approx(
        priced_text_tokens * text_rate + priced_pixels * pixel_rate
    )


def test_voyage_multimodal_embedding_token_only_pricing_keeps_combined_token_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model: Final = "voyage-multimodal-token-pricing-test"
    text_rate: Final = 0.01
    monkeypatch.setitem(
        litellm.model_cost,
        f"voyage/{model}",
        {"litellm_provider": "voyage", "mode": "embedding", "input_cost_per_token": text_rate},
    )
    response: Final = _transform(
        {"model": model, "usage": {"total_tokens": 14, "text_tokens": 9, "image_pixels": 9000}}
    )

    assert litellm.completion_cost(completion_response=response, model=f"voyage/{model}") == pytest.approx(
        14 * text_rate
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"data": [EMBEDDING_ROW]}, (0, 0, 0)),
        ({"data": [EMBEDDING_ROW], "usage": {}}, (0, 0, 0)),
        ({"data": [EMBEDDING_ROW], "usage": {"prompt_tokens": 4}}, (0, 0, 0)),
        ({"data": [EMBEDDING_ROW], "usage": {"total_tokens": 6}}, (6, 0, 6)),
        ({"data": [EMBEDDING_ROW], "usage": {"total_tokens": None}}, (0, 0, 0)),
        ({"data": [EMBEDDING_ROW], "usage": {"total_tokens": ""}}, (0, 0, 0)),
        ({"data": [EMBEDDING_ROW], "usage": {"total_tokens": []}}, (0, 0, 0)),
        ({"data": [EMBEDDING_ROW], "usage": {"total_tokens": "12"}}, (12, 0, 12)),
        ({"data": [EMBEDDING_ROW], "usage": {"total_tokens": 12.0}}, (12, 0, 12)),
        (
            {"data": [EMBEDDING_ROW], "usage": {"prompt_tokens": 3, "total_tokens": 5, "completion_tokens": 2}},
            (5, 0, 5),
        ),
    ],
)
def test_voyage_multimodal_embedding_usage_tolerates_missing_null_and_loosely_typed_counts(payload, expected):
    response = _transform(payload)

    assert response.data == [EMBEDDING_ROW]
    assert _token_counts(response) == expected


@pytest.mark.parametrize("usage", [None, [], "tokens", 3])
def test_voyage_multimodal_embedding_usage_that_is_not_an_object_is_rejected(usage):
    with pytest.raises(ValidationError):
        _transform({"data": [EMBEDDING_ROW], "usage": usage})


@pytest.mark.parametrize("usage", [{"total_tokens": "many"}, {"total_tokens": 1.5}, {"total_tokens": [1]}])
def test_voyage_multimodal_embedding_token_count_that_is_not_an_integer_is_rejected(usage):
    with pytest.raises(ValidationError):
        _transform({"data": [EMBEDDING_ROW], "usage": usage})


@pytest.mark.parametrize(
    "api_base",
    [
        "https://gateway.example.com/voyage/v1",
        "https://gateway.example.com/voyage/v1/",
        "https://gateway.example.com/voyage/v1/multimodalembeddings/",
    ],
)
def test_get_complete_url_ignores_a_trailing_slash_on_api_base(api_base: str):
    assert (
        VoyageMultimodalEmbeddingConfig().get_complete_url(
            api_base=api_base, api_key=None, model="voyage-multimodal-3", optional_params={}, litellm_params={}
        )
        == "https://gateway.example.com/voyage/v1/multimodalembeddings"
    )

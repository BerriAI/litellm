import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.fal_ai.image_generation.transformation import FalAIImageGenerationConfig
from litellm.types.utils import ImageObject, ImageResponse


def _transform(payload: object, model_response: ImageResponse | None = None) -> ImageResponse:
    return FalAIImageGenerationConfig().transform_image_generation_response(
        model="fal-ai/some-new-model",
        raw_response=httpx.Response(200, json=payload),
        model_response=model_response or ImageResponse(),
        logging_obj=None,
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )


def test_transform_image_generation_response_maps_image_objects_and_bare_urls():
    response = _transform(
        {
            "images": [
                {"url": "https://fal.media/a.png", "width": 1024, "height": 768, "content_type": "image/png"},
                "https://fal.media/b.png",
                {"b64_json": "QUJD", "width": 0, "height": "768", "content_type": 5},
                7,
                None,
                ["https://fal.media/ignored.png"],
            ],
            "seed": 42,
            "timings": {"inference": 2.5},
        }
    )

    assert [(image.url, image.b64_json, image.provider_specific_fields) for image in response.data] == [
        ("https://fal.media/a.png", None, {"width": 1024, "height": 768, "content_type": "image/png"}),
        ("https://fal.media/b.png", None, None),
        (None, "QUJD", None),
    ]


def test_transform_image_generation_response_appends_to_images_already_on_the_response():
    model_response = ImageResponse(data=[ImageObject(url="https://fal.media/existing.png")])

    response = _transform({"images": ["https://fal.media/new.png"]}, model_response)

    assert response is model_response
    assert [image.url for image in response.data] == ["https://fal.media/existing.png", "https://fal.media/new.png"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"detail": "rate limit exceeded"},
        {"images": None},
        {"images": "https://fal.media/a.png"},
        {"images": {"url": "https://fal.media/a.png"}},
        {"images": 429},
        {"image": {"url": "https://fal.media/a.png"}},
    ],
)
def test_transform_image_generation_response_without_an_image_list_is_empty(payload: dict[str, object]):
    assert _transform(payload).data == []


@pytest.mark.parametrize(
    "payload", [7, True, "https://fal.media/a.png", [{"url": "https://fal.media/a.png"}], [{"images": []}]]
)
def test_transform_image_generation_response_rejects_non_object_bodies_without_echoing_them(payload: object):
    with pytest.raises(ValidationError) as exc_info:
        _transform(payload)

    assert [(error["type"], error["loc"]) for error in exc_info.value.errors()] == [("dict_type", ())]
    assert "input_value" not in str(exc_info.value)
    assert "fal.media" not in str(exc_info.value)


def test_transform_image_generation_response_reports_a_body_that_is_not_json_as_a_provider_error():
    with pytest.raises(BaseLLMException) as exc_info:
        FalAIImageGenerationConfig().transform_image_generation_response(
            model="fal-ai/some-new-model",
            raw_response=httpx.Response(502, content=b"<html>502 Bad Gateway</html>"),
            model_response=ImageResponse(),
            logging_obj=None,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

    assert exc_info.value.status_code == 502
    assert str(exc_info.value).startswith("Error transforming image generation response: ")

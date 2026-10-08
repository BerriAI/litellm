import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.fal_ai.image_generation.bria_transformation import FalAIBriaConfig
from litellm.types.utils import ImageObject, ImageResponse


def _transform(payload: object, model_response: ImageResponse | None = None) -> ImageResponse:
    return FalAIBriaConfig().transform_image_generation_response(
        model="bria/text-to-image/3.2",
        raw_response=httpx.Response(200, json=payload),
        model_response=model_response or ImageResponse(),
        logging_obj=None,
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )


def test_transform_image_generation_response_maps_the_single_image():
    response = _transform(
        {
            "image": {
                "url": "https://fal.media/a.png",
                "content_type": "image/png",
                "file_name": "a.png",
                "file_size": 123456,
                "width": 1024,
                "height": 1024,
            },
            "images": [{"url": "https://fal.media/ignored.png"}],
        }
    )

    assert [(image.url, image.b64_json, image.provider_specific_fields) for image in response.data] == [
        ("https://fal.media/a.png", None, None)
    ]


def test_transform_image_generation_response_keeps_an_image_without_a_url():
    response = _transform({"image": {"content_type": "image/png", "url": None}})

    assert [(image.url, image.b64_json) for image in response.data] == [(None, None)]


def test_transform_image_generation_response_appends_to_images_already_on_the_response():
    model_response = ImageResponse(data=[ImageObject(url="https://fal.media/existing.png")])

    response = _transform({"image": {"url": "https://fal.media/new.png"}}, model_response)

    assert response is model_response
    assert [image.url for image in response.data] == ["https://fal.media/existing.png", "https://fal.media/new.png"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"detail": "rate limit exceeded"},
        {"image": None},
        {"image": {}},
        {"image": "https://fal.media/a.png"},
        {"image": ["https://fal.media/a.png"]},
        {"image": 429},
        {"images": [{"url": "https://fal.media/a.png"}]},
    ],
)
def test_transform_image_generation_response_without_an_image_object_is_empty(payload: dict[str, object]):
    assert _transform(payload).data == []


@pytest.mark.parametrize(
    "payload", [7, True, "https://fal.media/a.png", [{"image": {"url": "https://fal.media/a.png"}}]]
)
def test_transform_image_generation_response_rejects_non_object_bodies_without_echoing_them(payload: object):
    with pytest.raises(ValidationError) as exc_info:
        _transform(payload)

    assert [(error["type"], error["loc"]) for error in exc_info.value.errors()] == [("dict_type", ())]
    assert "input_value" not in str(exc_info.value)
    assert "fal.media" not in str(exc_info.value)


@pytest.mark.parametrize("url", [7, ["Request timed out"], {"code": 429}])
def test_transform_image_generation_response_reports_a_url_of_the_wrong_type_with_its_value(url: object):
    with pytest.raises(ValidationError) as exc_info:
        _transform({"image": {"url": url}})

    assert [(error["type"], error["loc"], error["input"]) for error in exc_info.value.errors()] == [
        ("string_type", ("url",), url)
    ]

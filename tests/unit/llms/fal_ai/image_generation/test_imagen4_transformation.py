import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.fal_ai.image_generation.imagen4_transformation import FalAIImagen4Config
from litellm.types.utils import ImageObject, ImageResponse


def _transform(payload: object, model_response: ImageResponse | None = None) -> ImageResponse:
    return FalAIImagen4Config().transform_image_generation_response(
        model="fal-ai/imagen4/preview",
        raw_response=httpx.Response(200, json=payload),
        model_response=model_response or ImageResponse(),
        logging_obj=None,
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )


def test_transform_image_generation_response_maps_image_objects_bare_urls_and_seed():
    response = _transform(
        {
            "images": [
                {
                    "url": "https://fal.media/a.png",
                    "content_type": "image/png",
                    "file_name": "z9RV14K95DvU.png",
                    "file_size": 4404019,
                },
                "https://fal.media/b.png",
                {"content_type": "image/png"},
                7,
                None,
                ["https://fal.media/ignored.png"],
            ],
            "seed": 42,
        }
    )

    assert [(image.url, image.b64_json, image.provider_specific_fields) for image in response.data] == [
        ("https://fal.media/a.png", None, None),
        ("https://fal.media/b.png", None, None),
        (None, None, None),
    ]
    assert repr(response._hidden_params["seed"]) == "42"


@pytest.mark.parametrize("seed", [0, None, "42", 1.0, [1, 2], {"value": 42}])
def test_transform_image_generation_response_stores_the_seed_exactly_as_received(seed: object):
    response = _transform({"images": [], "seed": seed})

    assert repr(response._hidden_params["seed"]) == repr(seed)


def test_transform_image_generation_response_appends_to_images_already_on_the_response():
    model_response = ImageResponse(data=[ImageObject(url="https://fal.media/existing.png")])

    response = _transform({"images": ["https://fal.media/new.png"]}, model_response)

    assert response is model_response
    assert [image.url for image in response.data] == ["https://fal.media/existing.png", "https://fal.media/new.png"]
    assert "seed" not in response._hidden_params


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
    response = _transform(payload)

    assert response.data == []
    assert "seed" not in response._hidden_params


@pytest.mark.parametrize(
    "payload", [7, True, "https://fal.media/a.png", [{"url": "https://fal.media/a.png"}], [{"images": [], "seed": 1}]]
)
def test_transform_image_generation_response_rejects_non_object_bodies_without_echoing_them(payload: object):
    with pytest.raises(ValidationError) as exc_info:
        _transform(payload)

    assert [(error["type"], error["loc"]) for error in exc_info.value.errors()] == [("dict_type", ())]
    assert "input_value" not in str(exc_info.value)
    assert "fal.media" not in str(exc_info.value)


@pytest.mark.parametrize("position", [0, 403])
def test_transform_image_generation_response_keeps_earlier_images_when_a_url_has_the_wrong_type(position: int):
    model_response = ImageResponse()
    images = [{"url": f"https://fal.media/{index}.png"} for index in range(position)] + [{"url": 7}]

    with pytest.raises(ValidationError) as exc_info:
        _transform({"images": images, "seed": 42}, model_response)

    assert [(error["type"], error["loc"], error["input"]) for error in exc_info.value.errors()] == [
        ("string_type", ("url",), 7)
    ]
    assert [image.url for image in model_response.data] == [
        f"https://fal.media/{index}.png" for index in range(position)
    ]
    assert "seed" not in model_response._hidden_params

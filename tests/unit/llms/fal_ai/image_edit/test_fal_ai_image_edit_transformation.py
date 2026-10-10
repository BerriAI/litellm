import base64
import io
import json
import tempfile
from pathlib import Path

import httpx
import pytest

from litellm.llms.fal_ai.image_edit import FalAIImageEditConfig
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def test_fal_ai_resolves_to_image_edit_config():
    config = ProviderConfigManager.get_provider_image_edit_config(
        model="openai/gpt-image-2.5/flare/edit", provider=LlmProviders.FAL_AI
    )
    assert isinstance(config, FalAIImageEditConfig)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("openai/gpt-image-2.5/flare", "https://fal.run/openai/gpt-image-2.5/flare/edit"),
        ("openai/gpt-image-2.5/sunburst/edit", "https://fal.run/openai/gpt-image-2.5/sunburst/edit"),
        ("openai/gpt-image-2", "https://fal.run/openai/gpt-image-2/edit"),
    ],
)
def test_get_complete_url_appends_edit_suffix_once(model, expected):
    assert FalAIImageEditConfig().get_complete_url(model=model, api_base=None, litellm_params={}) == expected


def test_get_complete_url_respects_api_base():
    url = FalAIImageEditConfig().get_complete_url(
        model="openai/gpt-image-2.5/flare", api_base="https://proxy.internal/", litellm_params={}
    )
    assert url == "https://proxy.internal/openai/gpt-image-2.5/flare/edit"


def test_validate_environment_uses_fal_key_scheme():
    headers = FalAIImageEditConfig().validate_environment(headers={}, model="m", api_key="secret")
    assert headers["Authorization"] == "Key secret"


def test_validate_environment_requires_key(monkeypatch):
    monkeypatch.delenv("FAL_AI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="FAL_AI_API_KEY"):
        FalAIImageEditConfig().validate_environment(headers={}, model="m", api_key=None)


def test_map_openai_params_translates_to_fal_names():
    mapped = FalAIImageEditConfig().map_openai_params(
        image_edit_optional_params=ImageEditOptionalRequestParams(
            n=2, size="1024x1536", quality="xhigh", background="transparent"
        ),
        model="openai/gpt-image-2.5/flare/edit",
        drop_params=False,
    )
    assert mapped == {
        "num_images": 2,
        "image_size": {"width": 1024, "height": 1536},
        "quality": "xhigh",
        "background": "transparent",
    }


def test_transform_request_inlines_local_images_as_data_urls_and_keeps_remote_urls():
    body, files = FalAIImageEditConfig().transform_image_edit_request(
        model="openai/gpt-image-2.5/flare/edit",
        prompt="make it blue",
        image=[io.BytesIO(PNG_BYTES), "https://example.com/in.png"],
        image_edit_optional_request_params={"num_images": 1, "mask": io.BytesIO(PNG_BYTES)},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    expected_data_url = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()
    assert files == ()
    assert body["prompt"] == "make it blue"
    assert json.loads(json.dumps(body))["image_urls"] == [expected_data_url, "https://example.com/in.png"]
    assert body["mask_url"] == expected_data_url
    assert body["num_images"] == 1
    assert "mask" not in body


@pytest.mark.parametrize(
    "image_factory",
    [
        pytest.param(lambda path: ("red.png", PNG_BYTES), id="filename-bytes-tuple"),
        pytest.param(lambda path: ("red.png", PNG_BYTES, "image/png"), id="three-tuple-with-content-type"),
        pytest.param(lambda path: path, id="path"),
        pytest.param(lambda path: io.FileIO(str(path), "rb"), id="file-io"),
        pytest.param(
            lambda path: tempfile.SpooledTemporaryFile(suffix=".png"),
            id="spooled-temp-file",
        ),
    ],
)
def test_transform_request_reads_every_file_types_input(tmp_path, image_factory):
    path = Path(tmp_path) / "red.png"
    path.write_bytes(PNG_BYTES)
    image = image_factory(path)
    if isinstance(image, tempfile.SpooledTemporaryFile):
        image.write(PNG_BYTES)
        image.seek(3)
    body, _ = FalAIImageEditConfig().transform_image_edit_request(
        model="openai/gpt-image-2.5/flare/edit",
        prompt="make it blue",
        image=image,
        image_edit_optional_request_params={},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    expected_data_url = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()
    assert body["image_urls"][0] == expected_data_url


def test_transform_response_maps_fal_images():
    raw = httpx.Response(
        200,
        json={
            "images": [
                {
                    "url": "https://fal.media/out.png",
                    "width": 1024,
                    "height": 1536,
                    "content_type": "image/png",
                }
            ]
        },
    )
    response = FalAIImageEditConfig().transform_image_edit_response(
        model="openai/gpt-image-2.5/flare/edit", raw_response=raw, logging_obj=None
    )
    assert isinstance(response, ImageResponse)
    assert [image.url for image in response.data] == ["https://fal.media/out.png"]
    assert response.data[0].provider_specific_fields == {
        "width": 1024,
        "height": 1536,
        "content_type": "image/png",
    }


@pytest.mark.parametrize("image", [None, []])
def test_transform_request_requires_an_image(image):
    with pytest.raises(ValueError, match="input image"):
        FalAIImageEditConfig().transform_image_edit_request(
            model="openai/gpt-image-2.5/flare/edit",
            prompt="make it blue",
            image=image,
            image_edit_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

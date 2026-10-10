import base64
import io
import json
import tempfile
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import TypeAdapter

import litellm
from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.fal_ai.image_edit import FalAIImageEditConfig, get_fal_ai_image_edit_config
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


@pytest.mark.parametrize("model", ["fal-ai/nano-banana-2", "fal-ai/nano-banana-pro/edit"])
@pytest.mark.parametrize(
    "size,expected_aspect_ratio,expected_resolution",
    [
        ("1792x1024", "16:9", "1K"),
        ("1024x768", "4:3", "1K"),
        ("928x1152", "4:5", "1K"),
        ("512x512", "1:1", "0.5K"),
        ("767x767", "1:1", "0.5K"),
        ("768x768", "1:1", "1K"),
        ("1535x1535", "1:1", "1K"),
        ("1536x1536", "1:1", "2K"),
        ("2752x1536", "16:9", "2K"),
        ("3071x3071", "1:1", "2K"),
        ("3072x3072", "1:1", "4K"),
        ("5504x3072", "16:9", "4K"),
        ("auto", "auto", None),
        ("not-a-size", "1:1", None),
        (None, None, None),
    ],
)
def test_nano_banana_edit_maps_size_to_aspect_ratio_and_resolution(
    model: str, size: str | None, expected_aspect_ratio: str | None, expected_resolution: str | None
) -> None:
    resolution: Final = "1K" if expected_resolution == "0.5K" and "nano-banana-pro" in model else expected_resolution

    def respond(request: httpx.Request) -> httpx.Response:
        assert TypeAdapter(dict[str, object]).validate_json(request.content) == {
            "prompt": "Make the wall blue",
            "image_urls": ["https://example.com/room.png"],
            "num_images": 1,
            **({"aspect_ratio": expected_aspect_ratio} if expected_aspect_ratio else {}),
            **({"resolution": resolution} if resolution else {}),
        }
        return httpx.Response(200, json={"images": [{"url": "https://example.com/edited.png"}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        response: Final = litellm.image_edit(
            model=f"fal_ai/{model}",
            image="https://example.com/room.png",
            prompt="Make the wall blue",
            n=1,
            size=size,
            api_key="test-key",
            client=HTTPHandler(client=http_client),
        )
    assert isinstance(response, ImageResponse)
    assert response.data is not None
    assert [item.url for item in response.data] == ["https://example.com/edited.png"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["fal-ai/nano-banana-2/edit", "fal-ai/nano-banana-pro"])
@pytest.mark.parametrize("use_extra_body", [False, True])
@pytest.mark.parametrize("overrides", [("aspect_ratio", "resolution"), ("aspect_ratio",), ("resolution",)])
async def test_nano_banana_edit_forwards_native_controls_and_preserves_inline_images(
    model: str, use_extra_body: bool, overrides: tuple[str, ...]
) -> None:
    controls: Final = {
        **({"aspect_ratio": "4:3"} if "aspect_ratio" in overrides else {}),
        **({"resolution": "2K"} if "resolution" in overrides else {}),
        "system_prompt": "Preserve the room",
        "sync_mode": use_extra_body,
    }
    image_url: Final = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()
    output_url: Final = image_url if use_extra_body else "https://example.com/edited.png"

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"https://fal.run/{model.removesuffix('/edit')}/edit"
        assert request.headers["Authorization"] == "Key test-key"
        assert TypeAdapter(dict[str, object]).validate_json(request.content) == {
            "prompt": "Apply the swatch to the wall",
            "image_urls": [image_url, "https://example.com/swatch.png"],
            "num_images": 2,
            "aspect_ratio": "1:1",
            "resolution": "4K",
            **controls,
        }
        return httpx.Response(200, json={"images": [{"url": output_url}]})

    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(respond))
    async with client.client:
        response: Final = await litellm.aimage_edit(
            model=f"fal_ai/{model}",
            image=[PNG_BYTES, "https://example.com/swatch.png"],
            prompt="Apply the swatch to the wall",
            n=2,
            size="4096x4096",
            api_key="test-key",
            client=client,
            **(
                {"aspect_ratio": "1:1", "extra_body": {**controls, "_litellm_undeclared_sentinel": "internal"}}
                if use_extra_body
                else controls
            ),
        )
    assert response.data is not None
    assert [item.url for item in response.data] == [output_url]


@pytest.mark.parametrize("size,expected_ratio", [("4096x1024", "4:1"), ("1024x8192", "1:8")])
def test_nano_banana_2_edit_maps_extreme_sizes(size: str, expected_ratio: str) -> None:
    config: Final = get_fal_ai_image_edit_config("fal-ai/nano-banana-2/edit")
    assert config.map_openai_params(ImageEditOptionalRequestParams(size=size), "fal-ai/nano-banana-2/edit", False) == {
        "aspect_ratio": expected_ratio,
        "resolution": "2K",
    }


@pytest.mark.parametrize(
    "params",
    [
        ImageEditOptionalRequestParams(quality="high"),
        ImageEditOptionalRequestParams(background="transparent"),
        ImageEditOptionalRequestParams(mask="https://example.com/mask.png"),
    ],
)
def test_nano_banana_edit_rejects_gpt_specific_params(params: ImageEditOptionalRequestParams) -> None:
    model: Final = "fal-ai/nano-banana-2"
    with pytest.raises(litellm.UnsupportedParamsError, match="not supported"):
        ImageEditRequestUtils.get_optional_params_image_edit(
            model=model,
            image_edit_provider_config=get_fal_ai_image_edit_config(model),
            image_edit_optional_params=params,
        )

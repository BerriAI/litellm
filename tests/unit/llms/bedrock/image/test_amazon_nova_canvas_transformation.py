import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.bedrock.image_generation.amazon_nova_canvas_transformation import (
    AmazonNovaCanvasConfig,
)
from litellm.types.utils import ImageResponse


def test_transform_request_body_text_to_image():
    params = {
        "imageGenerationConfig": {
            "cfgScale": 7,
            "seed": 42,
            "quality": "standard",
            "width": 512,
            "height": 512,
            "numberOfImages": 1,
            "textToImageParams": {"negativeText": "blurry"},
        }
    }
    req = AmazonNovaCanvasConfig.transform_request_body("cat", params.copy())
    assert isinstance(req, dict)
    assert "textToImageParams" in req
    assert req["textToImageParams"]["text"] == "cat"
    assert req["imageGenerationConfig"]["width"] == 512


def test_transform_request_body_color_guided():
    params = {
        "taskType": "COLOR_GUIDED_GENERATION",
        "imageGenerationConfig": {
            "cfgScale": 7,
            "seed": 42,
            "quality": "standard",
            "width": 512,
            "height": 512,
            "numberOfImages": 1,
            "colorGuidedGenerationParams": {
                "colors": ["#FFFFFF"],
                "referenceImage": "img",
                "negativeText": "blurry",
            },
        },
    }
    req = AmazonNovaCanvasConfig.transform_request_body("cat", params.copy())
    assert "colorGuidedGenerationParams" in req
    assert req["colorGuidedGenerationParams"]["text"] == "cat"
    assert req["imageGenerationConfig"]["width"] == 512


def test_transform_request_body_inpainting():
    params = {
        "taskType": "INPAINTING",
        "imageGenerationConfig": {
            "cfgScale": 7,
            "seed": 42,
            "quality": "standard",
            "width": 512,
            "height": 512,
            "numberOfImages": 1,
            "inpaintingParams": {
                "maskImage": "mask",
                "inputImage": "input",
                "negativeText": "blurry",
            },
        },
    }
    req = AmazonNovaCanvasConfig.transform_request_body("cat", params.copy())
    assert "inpaintingParams" in req
    assert req["inpaintingParams"]["text"] == "cat"
    assert req["imageGenerationConfig"]["width"] == 512


def test_transform_response_dict_to_openai_response():
    response_dict = {"images": ["b64img1", "b64img2"]}
    model_response = ImageResponse()
    result = AmazonNovaCanvasConfig.transform_response_dict_to_openai_response(
        model_response, response_dict
    )
    assert hasattr(result, "data")
    assert len(result.data) == 2
    assert result.data[0].b64_json == "b64img1"


_NOVA_CANVAS_PROMPT: Final = "A serene mountain landscape at sunset with a lake reflection"
_NOVA_CANVAS_IMAGES: Final = ("b64-first-image", "b64-second-image")


def test_nova_canvas_image_gen_reports_positive_response_cost(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(
        url__regex=r"^https://bedrock-runtime\.us-east-1\.amazonaws\.com/model/amazon\.nova-canvas-v1(:|%3A)0/invoke$"
    ).mock(return_value=httpx.Response(200, json={"images": list(_NOVA_CANVAS_IMAGES)}))

    response: Final = litellm.image_generation(
        model="bedrock/amazon.nova-canvas-v1:0",
        prompt=_NOVA_CANVAS_PROMPT,
        aws_region_name="us-east-1",
        aws_access_key_id="fake-access-key",
        aws_secret_access_key="fake-secret-key",
    )

    assert route.call_count == 1
    sent: Final = json.loads(route.calls[0].request.content)
    assert sent["taskType"] == "TEXT_IMAGE"
    assert sent["textToImageParams"]["text"] == _NOVA_CANVAS_PROMPT
    assert [image.b64_json for image in response.data] == list(_NOVA_CANVAS_IMAGES)
    per_image: Final = litellm.model_cost["amazon.nova-canvas-v1:0"]["output_cost_per_image"]
    assert per_image > 0
    assert response._hidden_params["response_cost"] == pytest.approx(len(_NOVA_CANVAS_IMAGES) * per_image)  # pyright: ignore[reportPrivateUsage]  # cost is only surfaced on _hidden_params

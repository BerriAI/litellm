from typing import Final

import httpx
import pytest

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


def test_nova_canvas_image_gen_reports_positive_response_cost(respx_mock, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    respx_mock.post(url__regex=r".*amazonaws\.com.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "images": [
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                ]
            },
        )
    )
    response: Final = litellm.image_generation(
        model="bedrock/amazon.nova-canvas-v1:0",
        prompt="A serene mountain landscape at sunset with a lake reflection",
        aws_region_name="us-east-1",
    )
    assert response._hidden_params["response_cost"] > 0

import pytest
from litellm.llms.bedrock.image_generation.amazon_nova_canvas_transformation import AmazonNovaCanvasConfig
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
            "textToImageParams": {
                "negativeText": "blurry"
            }
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
                "negativeText": "blurry"
            }
        }
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
                "negativeText": "blurry"
            }
        }
    }
    req = AmazonNovaCanvasConfig.transform_request_body("cat", params.copy())
    assert "inpaintingParams" in req
    assert req["inpaintingParams"]["text"] == "cat"
    assert req["imageGenerationConfig"]["width"] == 512

def test_transform_response_dict_to_openai_response():
    response_dict = {"images": ["b64img1", "b64img2"]}
    model_response = ImageResponse()
    result = AmazonNovaCanvasConfig.transform_response_dict_to_openai_response(model_response, response_dict)
    assert hasattr(result, "data")
    assert len(result.data) == 2
    assert result.data[0].b64_json == "b64img1"
from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.bedrock.image_generation.amazon_nova_canvas_transformation import (
    AmazonNovaCanvasConfig,
)
from litellm.llms.bedrock.image_generation.amazon_stability1_transformation import (
    AmazonStabilityConfig,
)
from litellm.llms.bedrock.image_generation.amazon_stability3_transformation import (
    AmazonStability3Config,
)
from litellm.llms.bedrock.image_generation.cost_calculator import cost_calculator
from litellm.llms.bedrock.image_generation.image_handler import BedrockImageGeneration
from litellm.types.utils import ImageObject, ImageResponse


@pytest.fixture
def aws_test_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")


@pytest.mark.parametrize(
    "model,expected",
    [
        ("sd3-large", True),
        ("sd3-large-turbo", True),
        ("sd3-medium", True),
        ("sd3.5-large", True),
        ("sd3.5-large-turbo", True),
        ("gpt-4", False),
        (None, False),
        ("other-model", False),
    ],
)
def test_is_stability_3_model(model, expected):
    result = AmazonStability3Config.is_stability_3_model(model)
    assert result == expected


@pytest.mark.parametrize(
    "model,expected",
    [
        ("amazon.nova-canvas", True),
        ("sd3-large", False),
        ("sd3-large-turbo", False),
        ("sd3-medium", False),
        ("sd3.5-large", False),
        ("sd3.5-large-turbo", False),
        ("gpt-4", False),
        (None, False),
        ("other-model", False),
    ],
)
def test_is_nova_canvas_model(model, expected):
    result = AmazonNovaCanvasConfig.is_nova_model(model)
    assert result == expected


def test_transform_request_body():
    prompt = "A beautiful sunset"
    optional_params = {"size": "1024x1024"}

    result = AmazonStability3Config.transform_request_body(prompt, optional_params)

    assert result["prompt"] == prompt
    assert result["size"] == "1024x1024"


def test_map_openai_params():
    non_default_params = {"n": 2, "size": "1024x1024"}
    optional_params = {"cfg_scale": 7}

    result = AmazonStability3Config.map_openai_params(non_default_params, optional_params)

    assert result == optional_params
    assert "n" not in result


def test_transform_response_dict_to_openai_response():

    response_dict = {"images": ["base64_encoded_image_1", "base64_encoded_image_2"]}
    model_response = ImageResponse()

    result = AmazonStability3Config.transform_response_dict_to_openai_response(model_response, response_dict)

    assert isinstance(result, ImageResponse)
    assert len(result.data) == 2
    assert all(hasattr(img, "b64_json") for img in result.data)
    assert [img.b64_json for img in result.data] == response_dict["images"]


def test_transform_response_dict_to_openai_response_from_stability_3_models_with_no_null_finish_reason():

    response_dict = {"finish_reasons": ["Filter reason: prompt"]}
    model_response = ImageResponse()

    with pytest.raises(BedrockError) as exc_info:
        AmazonStability3Config.transform_response_dict_to_openai_response(model_response, response_dict)

    assert exc_info.value.status_code == 400
    assert exc_info.value.message == "Filter reason: prompt"


def test_amazon_stability_get_supported_openai_params():
    result = AmazonStabilityConfig.get_supported_openai_params()
    assert result == ["size"]


def test_amazon_stability_map_openai_params():

    non_default_params = {"size": "512x512"}
    optional_params = {"cfg_scale": 7}

    result = AmazonStabilityConfig.map_openai_params(non_default_params, optional_params)

    assert result["width"] == 512
    assert result["height"] == 512
    assert result["cfg_scale"] == 7


def test_amazon_stability_transform_response():

    response_dict = {
        "artifacts": [
            {"base64": "base64_encoded_image_1"},
            {"base64": "base64_encoded_image_2"},
        ]
    }
    model_response = ImageResponse()

    result = AmazonStabilityConfig.transform_response_dict_to_openai_response(model_response, response_dict)

    assert isinstance(result, ImageResponse)
    assert len(result.data) == 2
    assert all(hasattr(img, "b64_json") for img in result.data)
    assert [img.b64_json for img in result.data] == [
        "base64_encoded_image_1",
        "base64_encoded_image_2",
    ]


def test_get_request_body_stability3():
    handler = BedrockImageGeneration()
    prompt = "A beautiful sunset"
    optional_params = {}
    model = "stability.sd3-large"

    result = handler._get_request_body(model=model, prompt=prompt, optional_params=optional_params)

    assert result["prompt"] == prompt


def test_get_request_body_stability():
    handler = BedrockImageGeneration()
    prompt = "A beautiful sunset"
    optional_params = {"cfg_scale": 7}
    model = "stability.stable-diffusion-xl-v1"

    result = handler._get_request_body(model=model, prompt=prompt, optional_params=optional_params)

    assert result["text_prompts"][0]["text"] == prompt
    assert result["text_prompts"][0]["weight"] == 1
    assert result["cfg_scale"] == 7


def test_transform_request_body_nova_canvas():
    prompt = "A beautiful sunset"
    optional_params = {"size": "1024x1024"}

    result = AmazonNovaCanvasConfig.transform_request_body(prompt, optional_params)

    assert result["taskType"] == "TEXT_IMAGE"
    assert result["textToImageParams"]["text"] == prompt
    assert result["imageGenerationConfig"]["size"] == "1024x1024"


def test_map_openai_params_nova_canvas():
    non_default_params = {"n": 2, "size": "1024x1024"}
    optional_params = {"cfg_scale": 7}

    result = AmazonNovaCanvasConfig.map_openai_params(non_default_params, optional_params)

    assert result == optional_params
    assert "n" not in result


def test_transform_response_dict_to_openai_response_nova_canvas():

    response_dict = {"images": ["base64_encoded_image_1", "base64_encoded_image_2"]}
    model_response = ImageResponse()

    result = AmazonNovaCanvasConfig.transform_response_dict_to_openai_response(model_response, response_dict)

    assert isinstance(result, ImageResponse)
    assert len(result.data) == 2
    assert all(hasattr(img, "b64_json") for img in result.data)
    assert [img.b64_json for img in result.data] == response_dict["images"]


def test_amazon_nova_canvas_get_supported_openai_params():
    result = AmazonNovaCanvasConfig.get_supported_openai_params()
    assert result == ["n", "size", "quality"]


def test_get_request_body_nova_canvas_default():
    handler = BedrockImageGeneration()
    prompt = "A beautiful sunset"
    optional_params = {"cfg_scale": 7}
    model = "amazon.nova-canvas-v1"

    result = handler._get_request_body(model=model, prompt=prompt, optional_params=optional_params)

    assert result["taskType"] == "TEXT_IMAGE"
    assert result["textToImageParams"]["text"] == prompt
    assert result["imageGenerationConfig"]["cfg_scale"] == 7


def test_get_request_body_nova_canvas_text_image():
    handler = BedrockImageGeneration()
    prompt = "A beautiful sunset"
    optional_params = {"cfg_scale": 7, "taskType": "TEXT_IMAGE"}
    model = "amazon.nova-canvas-v1"

    result = handler._get_request_body(model=model, prompt=prompt, optional_params=optional_params)

    assert result["taskType"] == "TEXT_IMAGE"
    assert result["textToImageParams"]["text"] == prompt
    assert result["imageGenerationConfig"]["cfg_scale"] == 7


def test_get_request_body_nova_canvas_color_guided_generation():
    handler = BedrockImageGeneration()
    prompt = "A beautiful sunset"
    optional_params = {
        "cfg_scale": 7,
        "taskType": "COLOR_GUIDED_GENERATION",
        "colorGuidedGenerationParams": {"colors": ["#FF0000"]},
    }
    model = "amazon.nova-canvas-v1"

    result = handler._get_request_body(model=model, prompt=prompt, optional_params=optional_params)

    assert result["taskType"] == "COLOR_GUIDED_GENERATION"
    assert result["colorGuidedGenerationParams"]["text"] == prompt
    assert result["colorGuidedGenerationParams"]["colors"] == ["#FF0000"]
    assert result["imageGenerationConfig"]["cfg_scale"] == 7


def test_transform_request_body_with_invalid_task_type():
    text = "An image of a otter"
    optional_params = {"taskType": "INVALID_TASK"}

    with pytest.raises(NotImplementedError) as exc_info:
        AmazonNovaCanvasConfig.transform_request_body(text=text, optional_params=optional_params)
    assert "Task type INVALID_TASK is not supported" in str(exc_info.value)


def test_transform_response_dict_to_openai_response_stability3():
    handler = BedrockImageGeneration()
    model_response = ImageResponse()
    model = "stability.sd3-large"
    logging_obj = MagicMock()
    prompt = "A beautiful sunset"

    mock_response = MagicMock()
    mock_response.text = '{"images": ["base64_image_1", "base64_image_2"]}'
    mock_response.json.return_value = {"images": ["base64_image_1", "base64_image_2"]}

    result = handler._transform_response_dict_to_openai_response(
        model_response=model_response,
        model=model,
        logging_obj=logging_obj,
        prompt=prompt,
        response=mock_response,
        data={},
    )

    assert isinstance(result, ImageResponse)
    assert len(result.data) == 2
    assert all(hasattr(img, "b64_json") for img in result.data)
    assert [img.b64_json for img in result.data] == ["base64_image_1", "base64_image_2"]


def test_cost_calculator_stability3():

    image_response = ImageResponse(
        data=[
            ImageObject(b64_json="base64_image_1"),
            ImageObject(b64_json="base64_image_2"),
        ]
    )

    cost = cost_calculator(
        model="stability.sd3-large-v1:0",
        size="1024-x-1024",
        image_response=image_response,
    )

    print("cost", cost)

    assert isinstance(cost, float)
    assert cost > 0


def test_cost_calculator_stability1():

    image_response = ImageResponse(data=[ImageObject(b64_json="base64_image_1")])

    cost_default_steps = cost_calculator(
        model="stability.stable-diffusion-xl-v1",
        size="1024-x-1024",
        image_response=image_response,
        optional_params={"steps": 50},
    )

    cost_max_steps = cost_calculator(
        model="stability.stable-diffusion-xl-v1",
        size="1024-x-1024",
        image_response=image_response,
        optional_params={"steps": 51},
    )

    assert isinstance(cost_default_steps, float)
    assert isinstance(cost_max_steps, float)
    assert cost_default_steps > 0
    assert cost_max_steps > 0

    assert cost_max_steps > cost_default_steps


def test_cost_calculator_with_no_optional_params():
    image_response = ImageResponse(data=[ImageObject(b64_json="base64_image_1")])

    cost = cost_calculator(
        model="stability.stable-diffusion-xl-v0",
        size="512-x-512",
        image_response=image_response,
        optional_params=None,
    )

    assert isinstance(cost, float)
    assert cost > 0


def test_cost_calculator_basic():
    image_response = ImageResponse(data=[ImageObject(b64_json="base64_image_1")])

    cost = cost_calculator(
        model="stability.stable-diffusion-xl-v1",
        image_response=image_response,
        optional_params=None,
    )

    assert isinstance(cost, float)
    assert cost > 0


def test_bedrock_image_gen_with_aws_region_name(aws_test_credentials):
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    from litellm import image_generation

    client = HTTPHandler()

    with patch.object(client, "post") as mock_post:
        try:
            image_generation(
                model="bedrock/stability.stable-image-ultra-v1:1",
                prompt="A beautiful sunset",
                aws_region_name="us-west-2",
                client=client,
            )
        except Exception as e:
            print(e)
            raise e
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        print(kwargs)


def test_get_request_body_nova_canvas_inference_profile_arn():
    """Test that ARN format inference profiles are correctly handled"""
    handler = BedrockImageGeneration()
    prompt = "A beautiful sunset"
    optional_params = {}

    model = "arn:aws:bedrock:eu-west-1:000000000000:application-inference-profile/a0a0a0a0a0a0"

    nova_model = "us.amazon.nova-canvas-v1:0"

    bedrock_provider = handler.get_bedrock_invoke_provider(model=nova_model)

    result = handler._get_request_body(model=nova_model, prompt=prompt, optional_params=optional_params)

    assert result["taskType"] == "TEXT_IMAGE"
    assert result["textToImageParams"]["text"] == prompt


def test_get_request_body_nova_canvas_with_model_id_param():
    """Test that model_id parameter is filtered from request body"""
    handler = BedrockImageGeneration()
    prompt = "A beautiful sunset"

    optional_params = {"model_id": "amazon.nova-canvas-v1:0", "cfg_scale": 7}
    model = "amazon.nova-canvas-v1"

    result = handler._get_request_body(model=model, prompt=prompt, optional_params=optional_params)

    assert result["taskType"] == "TEXT_IMAGE"
    assert result["textToImageParams"]["text"] == prompt
    assert result["imageGenerationConfig"]["cfg_scale"] == 7

    assert "model_id" not in str(result)


def test_transform_request_body_nova_canvas_filter_model_id():
    """Test that model_id parameter is filtered in transform_request_body"""
    prompt = "A beautiful sunset"

    optional_params = {"model_id": "amazon.nova-canvas-v1:0", "size": "1024x1024"}

    result = AmazonNovaCanvasConfig.transform_request_body(prompt, optional_params)

    assert result["taskType"] == "TEXT_IMAGE"
    assert result["textToImageParams"]["text"] == prompt
    assert result["imageGenerationConfig"]["size"] == "1024x1024"

    assert "model_id" not in str(result)


def test_get_request_body_cross_region_inference_profile():
    """Test cross-region inference profile format support"""
    handler = BedrockImageGeneration()
    prompt = "A beautiful sunset"
    optional_params = {}

    model = "us.amazon.nova-canvas-v1:0"

    result = handler._get_request_body(model=model, prompt=prompt, optional_params=optional_params)

    assert result["taskType"] == "TEXT_IMAGE"
    assert result["textToImageParams"]["text"] == prompt


def test_extract_headers_from_optional_params_with_guardrails():
    """Test that guardrail parameters are correctly extracted from optional_params and converted to headers"""
    handler = BedrockImageGeneration()

    optional_params = {
        "guardrailIdentifier": "4cf5knqaeq15",
        "guardrailVersion": "1",
        "someOtherParam": "value",
    }

    headers = handler._extract_headers_from_optional_params(optional_params)

    assert headers["x-amz-bedrock-guardrail-identifier"] == "4cf5knqaeq15"
    assert headers["x-amz-bedrock-guardrail-version"] == "1"

    assert "guardrailIdentifier" not in optional_params
    assert "guardrailVersion" not in optional_params

    assert optional_params["someOtherParam"] == "value"

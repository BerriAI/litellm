import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import aimage_generation


@pytest.mark.parametrize(
    "model,expected_model_id",
    [
        ("runware/runware:400@1", "runware:400@1"),
        ("runware/bfl:7@1", "bfl:7@1"),
        ("runware/google:4@3", "google:4@3"),
        ("runware/alibaba:qwen-image@2.0", "alibaba:qwen-image@2.0"),
        ("runware/xai:grok-imagine@image", "xai:grok-imagine@image"),
        ("runware/runware:z-image@turbo", "runware:z-image@turbo"),
    ],
)
@pytest.mark.asyncio
async def test_runware_image_generation_basic(model, expected_model_id):
    """
    Test that runware image generation constructs correct request body and URL.

    Validates:
    - Correct API endpoint URL
    - Proper array-wrapped request body format
    - Correct Authorization header format (Bearer token)
    - Prompt mapped to positivePrompt
    - Model ID correctly stripped of runware/ prefix
    - taskType set to imageInference
    """
    captured_url = None
    captured_json_data = None
    captured_headers = None

    def capture_post_call(*args, **kwargs):
        nonlocal captured_url, captured_json_data, captured_headers

        captured_url = args[0] if args else kwargs.get("url")
        captured_json_data = kwargs.get("json")
        captured_headers = kwargs.get("headers")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "data": [
                {
                    "taskType": "imageInference",
                    "taskUUID": "test-uuid",
                    "imageURL": "https://example.com/generated-image.png",
                    "cost": 0.002,
                }
            ]
        }

        return mock_response

    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
        mock_post.side_effect = capture_post_call

        test_api_key = "test-runware-key-12345"
        test_prompt = "A cute baby sea otter"

        response = await aimage_generation(
            model=model,
            prompt=test_prompt,
            api_key=test_api_key,
        )

        # Validate response
        assert response is not None
        assert hasattr(response, "data")
        assert response.data is not None
        assert len(response.data) > 0
        assert response.data[0].url == "https://example.com/generated-image.png"

        # Validate URL
        assert captured_url is not None
        assert "api.runware.ai" in captured_url
        print(f"Validated URL: {captured_url}")

        # Validate headers
        assert captured_headers is not None
        assert "Authorization" in captured_headers
        assert captured_headers["Authorization"] == f"Bearer {test_api_key}"
        print(f"Validated headers: {captured_headers}")

        # Validate request body is array-wrapped
        assert captured_json_data is not None
        assert isinstance(captured_json_data, list)
        assert len(captured_json_data) == 1

        task = captured_json_data[0]
        assert task["positivePrompt"] == test_prompt
        assert task["taskType"] == "imageInference"
        assert task["model"] == expected_model_id
        assert "taskUUID" in task
        print(f"Validated request body: {captured_json_data}")


@pytest.mark.asyncio
async def test_runware_image_generation_with_size():
    """
    Test that size parameter is correctly parsed into width/height.
    """
    captured_json_data = None

    def capture_post_call(*args, **kwargs):
        nonlocal captured_json_data
        captured_json_data = kwargs.get("json")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "data": [
                {
                    "taskType": "imageInference",
                    "taskUUID": "test-uuid",
                    "imageURL": "https://example.com/image.png",
                }
            ]
        }
        return mock_response

    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
        mock_post.side_effect = capture_post_call

        await aimage_generation(
            model="runware/runware:400@1",
            prompt="A landscape",
            size="512x768",
            api_key="test-key",
        )

        task = captured_json_data[0]
        assert task["width"] == 512
        assert task["height"] == 768


@pytest.mark.asyncio
async def test_runware_image_generation_with_n():
    """
    Test that n parameter maps to numberResults.
    """
    captured_json_data = None

    def capture_post_call(*args, **kwargs):
        nonlocal captured_json_data
        captured_json_data = kwargs.get("json")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "data": [
                {
                    "taskType": "imageInference",
                    "taskUUID": "u1",
                    "imageURL": "https://example.com/1.png",
                },
                {
                    "taskType": "imageInference",
                    "taskUUID": "u2",
                    "imageURL": "https://example.com/2.png",
                },
            ]
        }
        return mock_response

    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
        mock_post.side_effect = capture_post_call

        response = await aimage_generation(
            model="runware/runware:400@1",
            prompt="Two cats",
            n=2,
            api_key="test-key",
        )

        task = captured_json_data[0]
        assert task["numberResults"] == 2
        assert len(response.data) == 2


@pytest.mark.asyncio
async def test_runware_image_generation_b64_response_format():
    """
    Test that response_format b64_json maps to outputType base64Data.
    """
    captured_json_data = None

    def capture_post_call(*args, **kwargs):
        nonlocal captured_json_data
        captured_json_data = kwargs.get("json")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "data": [
                {
                    "taskType": "imageInference",
                    "taskUUID": "test-uuid",
                    "imageBase64Data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                }
            ]
        }
        return mock_response

    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
        mock_post.side_effect = capture_post_call

        response = await aimage_generation(
            model="runware/runware:400@1",
            prompt="A cat",
            response_format="b64_json",
            api_key="test-key",
        )

        task = captured_json_data[0]
        assert task["outputType"] == "base64Data"
        assert response.data[0].b64_json is not None


@pytest.mark.asyncio
async def test_runware_image_generation_pass_through_params():
    """
    Test that Runware-specific parameters (negativePrompt, steps, CFGScale, seed,
    scheduler) are forwarded directly to the API request body.
    """
    captured_json_data = None

    def capture_post_call(*args, **kwargs):
        nonlocal captured_json_data
        captured_json_data = kwargs.get("json")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "data": [
                {
                    "taskType": "imageInference",
                    "taskUUID": "test-uuid",
                    "imageURL": "https://example.com/image.png",
                }
            ]
        }
        return mock_response

    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
        mock_post.side_effect = capture_post_call

        await aimage_generation(
            model="runware/runware:400@1",
            prompt="A landscape",
            api_key="test-key",
            negativePrompt="blurry, low quality",
            steps=30,
            CFGScale=7.5,
            seed=42,
            scheduler="DPM++ 2M Karras",
        )

        task = captured_json_data[0]
        assert task["negativePrompt"] == "blurry, low quality"
        assert task["steps"] == 30
        assert task["CFGScale"] == 7.5
        assert task["seed"] == 42
        assert task["scheduler"] == "DPM++ 2M Karras"


def test_runware_request_transformation_unit():
    """
    Unit test for RunwareImageGenerationConfig.transform_image_generation_request
    """
    from litellm.llms.runware.image_generation.transformation import (
        RunwareImageGenerationConfig,
    )

    config = RunwareImageGenerationConfig()

    result = config.transform_image_generation_request(
        model="runware:400@1",
        prompt="A beautiful sunset",
        optional_params={"size": "512x768", "n": 3, "response_format": "b64_json"},
        litellm_params={},
        headers={},
    )

    assert isinstance(result, list)
    assert len(result) == 1

    task = result[0]
    assert task["taskType"] == "imageInference"
    assert task["positivePrompt"] == "A beautiful sunset"
    assert task["model"] == "runware:400@1"
    assert task["width"] == 512
    assert task["height"] == 768
    assert task["numberResults"] == 3
    assert task["outputType"] == "base64Data"
    assert task["includeCost"] is True
    assert "taskUUID" in task


def test_runware_response_transformation_unit():
    """
    Unit test for RunwareImageGenerationConfig.transform_image_generation_response
    """
    from unittest.mock import MagicMock

    from litellm.llms.runware.image_generation.transformation import (
        RunwareImageGenerationConfig,
    )
    from litellm.types.utils import ImageResponse

    config = RunwareImageGenerationConfig()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {
        "data": [
            {
                "taskType": "imageInference",
                "taskUUID": "abc-123",
                "imageURL": "https://cdn.runware.ai/image1.png",
                "cost": 0.002,
            },
            {
                "taskType": "imageInference",
                "taskUUID": "abc-456",
                "imageURL": "https://cdn.runware.ai/image2.png",
                "cost": 0.002,
            },
        ]
    }

    model_response = ImageResponse(created=1234567890, data=[])

    result = config.transform_image_generation_response(
        model="runware:400@1",
        raw_response=mock_response,
        model_response=model_response,
        logging_obj=MagicMock(),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert len(result.data) == 2
    assert result.data[0].url == "https://cdn.runware.ai/image1.png"
    assert result.data[1].url == "https://cdn.runware.ai/image2.png"
    # Verify multi-image cost is summed
    assert result._hidden_params["runware_cost"] == 0.004


def test_runware_cost_calculator():
    """
    Unit test for Runware cost calculator with Runware-reported cost.
    """
    from litellm.llms.runware.cost_calculator import cost_calculator
    from litellm.types.utils import ImageObject, ImageResponse

    response = ImageResponse(
        created=1234567890,
        data=[
            ImageObject(url="https://example.com/image.png"),
        ],
    )
    response._hidden_params = {"runware_cost": 0.005}

    cost = cost_calculator(
        model="runware:400@1",
        image_response=response,
    )

    assert cost == 0.005


def test_runware_cost_calculator_multi_image():
    """
    Unit test for Runware cost calculator with multiple images summed.
    """
    from litellm.llms.runware.cost_calculator import cost_calculator
    from litellm.types.utils import ImageObject, ImageResponse

    response = ImageResponse(
        created=1234567890,
        data=[
            ImageObject(url="https://example.com/image1.png"),
            ImageObject(url="https://example.com/image2.png"),
        ],
    )
    response._hidden_params = {"runware_cost": 0.010}

    cost = cost_calculator(
        model="runware:400@1",
        image_response=response,
    )

    assert cost == 0.010


def test_runware_validate_environment():
    """
    Test that validate_environment sets correct auth headers.
    """
    from litellm.llms.runware.image_generation.transformation import (
        RunwareImageGenerationConfig,
    )

    config = RunwareImageGenerationConfig()
    headers = config.validate_environment(
        headers={},
        model="runware:400@1",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="my-test-key",
    )

    assert headers["Authorization"] == "Bearer my-test-key"
    assert headers["Content-Type"] == "application/json"


def test_runware_validate_environment_missing_key():
    """
    Test that validate_environment raises when no API key is available.
    """
    from litellm.llms.runware.image_generation.transformation import (
        RunwareImageGenerationConfig,
    )

    config = RunwareImageGenerationConfig()
    with patch(
        "litellm.llms.runware.image_generation.transformation.get_secret_str",
        return_value=None,
    ):
        with pytest.raises(ValueError, match="RUNWARE_API_KEY is not set"):
            config.validate_environment(
                headers={},
                model="runware:400@1",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )

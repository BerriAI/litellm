import logging
import os
import sys
import traceback

from dotenv import load_dotenv
from openai.types.image import Image

logging.basicConfig(level=logging.DEBUG)
load_dotenv()
import asyncio
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest

import litellm
from litellm.llms.bedrock.image.amazon_stability3_transformation import (
    AmazonStability3Config,
)
from litellm.llms.bedrock.image.amazon_stability1_transformation import (
    AmazonStabilityConfig,
)
from litellm.types.llms.bedrock import (
    AmazonStability3TextToImageRequest,
    AmazonStability3TextToImageResponse,
)
from litellm.types.utils import ImageResponse
from unittest.mock import MagicMock, patch
from litellm.llms.bedrock.image.image_handler import (
    BedrockImageGeneration,
    BedrockImagePreparedRequest,
)


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
    result = AmazonStability3Config._is_stability_3_model(model)
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

    result = AmazonStability3Config.map_openai_params(
        non_default_params, optional_params
    )

    assert result == optional_params
    assert "n" not in result  # OpenAI params should not be included


def test_transform_response_dict_to_openai_response():
    # Create a mock response
    response_dict = {"images": ["base64_encoded_image_1", "base64_encoded_image_2"]}
    model_response = ImageResponse()

    result = AmazonStability3Config.transform_response_dict_to_openai_response(
        model_response, response_dict
    )

    assert isinstance(result, ImageResponse)
    assert len(result.data) == 2
    assert all(hasattr(img, "b64_json") for img in result.data)
    assert [img.b64_json for img in result.data] == response_dict["images"]


def test_amazon_stability_get_supported_openai_params():
    result = AmazonStabilityConfig.get_supported_openai_params()
    assert result == ["size"]


def test_amazon_stability_map_openai_params():
    # Test with size parameter
    non_default_params = {"size": "512x512"}
    optional_params = {"cfg_scale": 7}

    result = AmazonStabilityConfig.map_openai_params(
        non_default_params, optional_params
    )

    assert result["width"] == 512
    assert result["height"] == 512
    assert result["cfg_scale"] == 7


def test_amazon_stability_transform_response():
    # Create a mock response
    response_dict = {
        "artifacts": [
            {"base64": "base64_encoded_image_1"},
            {"base64": "base64_encoded_image_2"},
        ]
    }
    model_response = ImageResponse()

    result = AmazonStabilityConfig.transform_response_dict_to_openai_response(
        model_response, response_dict
    )

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

    result = handler._get_request_body(
        model=model, prompt=prompt, optional_params=optional_params
    )

    assert result["prompt"] == prompt


def test_get_request_body_stability():
    handler = BedrockImageGeneration()
    prompt = "A beautiful sunset"
    optional_params = {"cfg_scale": 7}
    model = "stability.stable-diffusion-xl"

    result = handler._get_request_body(
        model=model, prompt=prompt, optional_params=optional_params
    )

    assert result["text_prompts"][0]["text"] == prompt
    assert result["text_prompts"][0]["weight"] == 1
    assert result["cfg_scale"] == 7


def test_transform_response_dict_to_openai_response_stability3():
    handler = BedrockImageGeneration()
    model_response = ImageResponse()
    model = "stability.sd3-large"
    logging_obj = MagicMock()
    prompt = "A beautiful sunset"

    # Mock response for Stability AI SD3
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

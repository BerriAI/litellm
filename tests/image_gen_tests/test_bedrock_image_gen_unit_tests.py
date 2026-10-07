import logging
import traceback

from dotenv import load_dotenv
from openai.types.image import Image


from litellm.llms.bedrock.image_generation.amazon_nova_canvas_transformation import (
    AmazonNovaCanvasConfig,
)

logging.basicConfig(level=logging.DEBUG)
load_dotenv()
import asyncio

import pytest
from litellm.llms.bedrock.image_generation.cost_calculator import cost_calculator
from litellm.types.utils import ImageResponse, ImageObject

import litellm
from litellm.llms.bedrock.image_generation.amazon_stability3_transformation import (
    AmazonStability3Config,
)
from litellm.llms.bedrock.image_generation.amazon_stability1_transformation import (
    AmazonStabilityConfig,
)
from litellm.types.llms.bedrock import (
    AmazonStability3TextToImageRequest,
    AmazonStability3TextToImageResponse,
)
from unittest.mock import MagicMock, patch
from litellm.llms.bedrock.image_generation.image_handler import (
    BedrockImageGeneration,
    BedrockImagePreparedRequest,
)
from litellm.llms.bedrock.common_utils import BedrockError




















































# Test cases for issue #14373 - Bedrock Application Inference Profiles with Nova Canvas








def test_amazon_nova_canvas_image_gen():
    """Test Amazon Nova Canvas image generation with cost tracking."""
    from litellm import image_generation

    model_id = "bedrock/amazon.nova-canvas-v1:0"

    response = litellm.image_generation(
        model=model_id,
        prompt="A serene mountain landscape at sunset with a lake reflection",
        aws_region_name="us-east-1",
    )

    print(f"response cost: {response._hidden_params['response_cost']}")

    assert response._hidden_params["response_cost"] > 0

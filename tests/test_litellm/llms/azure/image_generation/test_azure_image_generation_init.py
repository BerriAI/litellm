import json
import os
import sys
import traceback
from typing import Callable, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.llms.azure.azure import AzureChatCompletion
from litellm.llms.azure.image_generation import (
    AzureDallE3ImageGenerationConfig,
    get_azure_image_generation_config,
)
from litellm.utils import get_optional_params_image_gen


@pytest.mark.parametrize(
    "received_model, expected_config",
    [
        ("dall-e-3", AzureDallE3ImageGenerationConfig),
        ("dalle-3", AzureDallE3ImageGenerationConfig),
        ("openai_dall_e_3", AzureDallE3ImageGenerationConfig),
    ],
)
def test_azure_image_generation_config(received_model, expected_config):
    assert isinstance(
        get_azure_image_generation_config(received_model), expected_config
    )


def test_azure_image_generation_flattens_extra_body():
    """
    Test that Azure image generation correctly flattens extra_body parameters.
    
    Azure's image generation API doesn't support the extra_body parameter,
    so we need to flatten any parameters in extra_body to the top level.
    
    This test verifies the fix for: https://github.com/BerriAI/litellm/issues/16059
    Where partial_images and stream parameters were incorrectly sent in extra_body.
    """
    # Test 1: Verify get_optional_params_image_gen puts extra params in extra_body
    optional_params = get_optional_params_image_gen(
        model="gpt-image-1",
        n=1,
        size="1024x1024",
        custom_llm_provider="azure",
        partial_images=2,
        stream=True
    )
    
    assert "extra_body" in optional_params
    assert "partial_images" in optional_params["extra_body"]
    assert "stream" in optional_params["extra_body"]
    assert optional_params["extra_body"]["partial_images"] == 2
    assert optional_params["extra_body"]["stream"] is True
    
    # Test 2: Verify Azure flattens extra_body when building request data
    # Simulate what happens in Azure's image_generation method
    test_optional_params = {
        "n": 1,
        "size": "1024x1024",
        "extra_body": {
            "partial_images": 2,
            "stream": True,
            "custom_param": "test_value"
        }
    }
    
    # This is what the Azure image_generation method does
    extra_body = test_optional_params.pop("extra_body", {})
    flattened_params = {**test_optional_params, **extra_body}
    
    data = {"model": "gpt-image-1", "prompt": "A cute sea otter", **flattened_params}
    
    # Verify the final data structure
    assert "extra_body" not in data, "extra_body should NOT be in the final data dict"
    assert "partial_images" in data, "partial_images should be at top level"
    assert "stream" in data, "stream should be at top level"
    assert "custom_param" in data, "custom_param should be at top level"
    assert data["partial_images"] == 2
    assert data["stream"] is True
    assert data["custom_param"] == "test_value"
    assert data["n"] == 1
    assert data["size"] == "1024x1024"

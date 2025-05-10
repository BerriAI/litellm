import json
import os
import sys

import httpx
import pytest
import respx

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm.utils import get_optional_params_image_gen


def test_get_optional_params_image_gen():
    from litellm.llms.azure.image_generation import AzureGPTImageGenerationConfig

    provider_config = AzureGPTImageGenerationConfig()
    optional_params = get_optional_params_image_gen(
        model="gpt-image-1",
        response_format="b64_json",
        n=3,
        custom_llm_provider="azure",
        drop_params=True,
        provider_config=provider_config,
    )
    assert optional_params is not None
    assert "response_format" not in optional_params
    assert optional_params["n"] == 3

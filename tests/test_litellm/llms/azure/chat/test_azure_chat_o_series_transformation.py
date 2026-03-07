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
from litellm.llms.azure.chat.o_series_transformation import AzureOpenAIO1Config


@pytest.mark.asyncio
async def test_azure_chat_o_series_transformation():
    provider_config = AzureOpenAIO1Config()
    model = "o_series/web-interface-o1-mini"
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    optional_params = {}
    litellm_params = {}
    headers = {}

    response = await provider_config.async_transform_request(
        model, messages, optional_params, litellm_params, headers
    )
    print(response)
    assert response["model"] == "web-interface-o1-mini"


def test_azure_o_series_strips_output_config():
    """Test that Azure O-series strips output_config with effort parameter.

    See: https://github.com/BerriAI/litellm/issues/22797
    """
    config = AzureOpenAIO1Config()
    request = config.transform_request(
        model="o3",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={
            "output_config": {"effort": "high"},
            "max_completion_tokens": 100,
        },
        litellm_params={},
        headers={},
    )
    assert "output_config" not in request
    assert request["max_completion_tokens"] == 100


def test_azure_o_series_works_without_output_config():
    """Test that Azure O-series requests work normally when output_config is not present."""
    config = AzureOpenAIO1Config()
    request = config.transform_request(
        model="o3",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={
            "max_completion_tokens": 100,
        },
        litellm_params={},
        headers={},
    )
    assert "output_config" not in request
    assert request["max_completion_tokens"] == 100

"""
Unit tests for BurnCloud configuration.

These tests validate the BurnCloudConfig class which extends OpenAIGPTConfig.
BurnCloud is an OpenAI-compatible provider with a few customizations.
"""

import os
import sys
from typing import Dict, List, Optional
from unittest.mock import patch

import pytest
from litellm.llms.burncloud.chat.transformation import BurnCloudConfig

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.novita.chat.transformation import NovitaConfig


class TestBurnCloudConfig:
    """Test class for BurnCloudConfig functionality"""

    def test_validate_environment(self):
        """Test that validate_environment adds correct headers"""
        config = BurnCloudConfig()
        headers = {}
        api_key = "fake-burncloud-key"

        result = config.validate_environment(
            headers=headers,
            model="burncloud/deepseek-v3",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base="https://ai.burncloud.com/v1",
        )

        # Verify headers
        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"
        assert result["X-BurnCloud-Source"] == "litellm"

    def test_missing_api_key(self):
        """Test error handling when API key is missing"""
        config = BurnCloudConfig()

        with pytest.raises(ValueError) as excinfo:
            config.validate_environment(
                headers={},
                model="burncloud/deepseek-v3",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base="https://ai.burncloud.com/v1",
            )

        assert "Missing BurnCloud API Key" in str(excinfo.value)

    def test_inheritance(self):
        """Test proper inheritance from OpenAIGPTConfig"""
        config = BurnCloudConfig()

        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

        assert isinstance(config, OpenAIGPTConfig)
        assert hasattr(config, "get_supported_openai_params")

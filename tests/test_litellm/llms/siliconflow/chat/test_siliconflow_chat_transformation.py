"""
Unit tests for SiliconFlow configuration.

These tests validate the SiliconFlowConfig class which extends OpenAIGPTConfig.
SiliconFlow is an OpenAI-compatible provider with a few customizations.
"""

import os
import sys
from typing import Dict, List, Optional
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.siliconflow.chat.transformation import SiliconFlowConfig


class TestSiliconFlowConfig:
    """Test class for SiliconFlowConfig functionality"""

    def test_validate_environment(self):
        """Test that validate_environment adds correct headers"""
        config = SiliconFlowConfig()
        headers = {}
        api_key = "fake-siliconflow-key"

        result = config.validate_environment(
            headers=headers,
            model="siliconflow/deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base="https://api.siliconflow.com/v1",
        )

        # Verify headers
        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

    def test_missing_api_key(self):
        """Test error handling when API key is missing"""
        config = SiliconFlowConfig()

        with pytest.raises(ValueError) as excinfo:
            config.validate_environment(
                headers={},
                model="siliconflow/deepseek-ai/DeepSeek-V3",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base="https://api.siliconflow.com/v1",
            )

        assert "Missing SiliconFlow API Key" in str(excinfo.value)

    def test_inheritance(self):
        """Test proper inheritance from OpenAIGPTConfig"""
        config = SiliconFlowConfig()

        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

        assert isinstance(config, OpenAIGPTConfig)
        assert hasattr(config, "get_supported_openai_params")

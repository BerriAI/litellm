"""
Unit tests for Novita AI configuration.

These tests validate the NovitaConfig class which extends OpenAIGPTConfig.
Novita AI is an OpenAI-compatible provider with a few customizations.
"""

import os
import sys
from typing import Dict, List, Optional
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.novita.chat.transformation import NovitaConfig


class TestNovitaConfig:
    """Test class for NovitaConfig functionality"""

    def test_validate_environment(self):
        """Test that validate_environment adds correct headers"""
        config = NovitaConfig()
        headers = {}
        api_key = "fake-novita-key"

        result = config.validate_environment(
            headers=headers,
            model="novita/meta-llama/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base="https://api.novita.ai/v3/openai",
        )

        # Verify headers
        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"
        assert result["X-Novita-Source"] == "litellm"

    def test_missing_api_key(self):
        """Test error handling when API key is missing"""
        config = NovitaConfig()

        with pytest.raises(ValueError) as excinfo:
            config.validate_environment(
                headers={},
                model="novita/meta-llama/llama-3.3-70b-instruct",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base="https://api.novita.ai/v3/openai",
            )

        assert "Missing Novita AI API Key" in str(excinfo.value)

    def test_inheritance(self):
        """Test proper inheritance from OpenAIGPTConfig"""
        config = NovitaConfig()

        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

        assert isinstance(config, OpenAIGPTConfig)
        assert hasattr(config, "get_supported_openai_params")

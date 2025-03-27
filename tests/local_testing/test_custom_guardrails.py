import asyncio
import os
import sys
from typing import Literal, Optional, Union

import pytest
from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.proxy.proxy_server import UserAPIKeyAuth
from litellm.integrations.custom_guardrail import CustomGuardrail

sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2

# Test Constants
TEST_API_BASE = "http://127.0.0.1:8000/api/scan"
TEST_API_JWT = "token"
TEST_THRESHOLD = 1

class CustomGuardrailMock(CustomGuardrail):
    """Mock implementation of CustomGuardrail for testing purposes"""
    def __init__(
        self,
        custom_config: dict,
        **kwargs,
    ) -> None:
        self.custom_config = custom_config

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal["completion", "text_completion", "embeddings"],
    ) -> Optional[Union[Exception, str, dict]]:
        """Mock pre-call hook that always succeeds"""
        return None

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal["completion", "embeddings", "image_generation", "moderation", "audio_transcription"],
    ) -> None:
        """Mock moderation hook that always succeeds"""
        return None

class TestCustomGuardrails:
    """Test suite for custom guardrails functionality"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment before each test"""
        litellm.set_verbose = True
        yield
        # Reset callbacks after each test
        litellm.callbacks = []

    def get_test_guardrail_config(self, guardrail_class: str = "test_custom_guardrails.CustomGuardrailMock"):
        """Helper method to generate test guardrail configuration"""
        return [{
            "guardrail_name": "custom-guardrail",
            "litellm_params": {
                "guardrail": guardrail_class,
                "guard_name": "custom_guard",
                "mode": "pre_call",
                "custom_config": {
                    "api_base": TEST_API_BASE,
                    "api_jwt": TEST_API_JWT,
                    "threshold": TEST_THRESHOLD,
                }
            },
        }]

    def test_unsupported_guardrail(self):
        """Test initialization with unsupported guardrail class"""
        with pytest.raises(Exception) as exc_info:
            init_guardrails_v2(
                all_guardrails=self.get_test_guardrail_config("FakeCustomGuardrail"),
                config_file_path="blabla.yml",
            )
        assert "Unsupported guardrail" in str(exc_info.value)

    def test_missing_config_file(self):
        """Test initialization with missing config file"""
        with pytest.raises(Exception) as exc_info:
            init_guardrails_v2(
                all_guardrails=self.get_test_guardrail_config(),
                config_file_path="",
            )
        assert "GuardrailsAIException - Please pass the config_file_path to initialize_guardrails_v2" in str(exc_info.value)

    def test_successful_initialization(self):
        """Test successful guardrail initialization and configuration"""
        init_guardrails_v2(
            all_guardrails=self.get_test_guardrail_config(),
            config_file_path="local_testing/test_custom_guardrails.py",
        )

        # Verify guardrail was properly initialized
        custom_guardrails = [
            callback for callback in litellm.callbacks 
            if isinstance(callback, CustomGuardrail)
        ]
        assert len(custom_guardrails) == 1

        # Verify configuration was properly set
        custom_guardrail = custom_guardrails[0]
        assert custom_guardrail.custom_config.get("api_base") == TEST_API_BASE
        assert custom_guardrail.custom_config.get("api_jwt") == TEST_API_JWT
        assert custom_guardrail.custom_config.get("threshold") == TEST_THRESHOLD

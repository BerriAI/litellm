from typing import Dict, Literal, Optional, Union

import pytest
from litellm import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2
from litellm.proxy.proxy_server import UserAPIKeyAuth

# Test Constants
TEST_API_BASE = "http://127.0.0.1:8000/api/scan"
TEST_API_JWT = "token"
TEST_THRESHOLD = 1

class CustomGuardrailMock(CustomGuardrail):
    """Mock implementation of CustomGuardrail for testing purposes"""
    def __init__(self, **kwargs) -> None:
        # Initialize with message_logging=True for parent class
        super().__init__(message_logging=True)
        # Store all kwargs as optional_params
        self.optional_params = kwargs

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: Dict,
        call_type: Literal["completion", "text_completion", "embeddings"],
    ) -> Optional[Union[Exception, str, Dict]]:
        """Mock pre-call hook that always succeeds"""
        return None

    async def async_moderation_hook(
        self,
        data: Dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal["completion", "embeddings", "image_generation", "moderation", "audio_transcription"],
    ) -> None:
        """Mock moderation hook that always succeeds"""
        return None

class TestCustomGuardrails:
    """Test suite for custom guardrails functionality"""
    
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Setup test environment before each test"""
        import litellm
        litellm.set_verbose = True
        yield
        # Reset callbacks after each test
        litellm.callbacks = []

    def get_test_guardrail_config(self, guardrail_class: str = "test_custom_guardrails.CustomGuardrailMock") -> list[Dict]:
        """Helper method to generate test guardrail configuration"""
        return [{
            "guardrail_name": "custom_guardrail",
            "litellm_params": {
                "guardrail": guardrail_class,
                "guard_name": "custom_guard",
                "mode": "pre_call",
                "api_base": TEST_API_BASE,
                "api_jwt": TEST_API_JWT,
                "threshold": TEST_THRESHOLD,
            },
        }]

    def test_unsupported_guardrail(self) -> None:
        """Test initialization with unsupported guardrail class"""
        with pytest.raises(ValueError) as exc_info:
            init_guardrails_v2(
                all_guardrails=self.get_test_guardrail_config("FakeCustomGuardrail"),
                config_file_path="test_config.yml",
            )
        assert "Unsupported guardrail" in str(exc_info.value)

    def test_missing_config_file(self) -> None:
        """Test initialization with missing config file"""
        with pytest.raises(Exception) as exc_info:
            init_guardrails_v2(
                all_guardrails=self.get_test_guardrail_config(),
                config_file_path="",
            )
        assert "GuardrailsAIException - Please pass the config_file_path" in str(exc_info.value)

    def test_successful_initialization(self) -> None:
        """Test successful guardrail initialization and configuration"""
        import litellm
        
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
        assert custom_guardrail.optional_params.get("api_base") == TEST_API_BASE
        assert custom_guardrail.optional_params.get("api_jwt") == TEST_API_JWT
        assert custom_guardrail.optional_params.get("threshold") == TEST_THRESHOLD

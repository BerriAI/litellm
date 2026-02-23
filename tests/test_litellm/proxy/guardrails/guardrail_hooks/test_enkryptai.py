"""
Tests for EnkryptAI guardrail integration

This test file tests the EnkryptAI guardrail implementation.
"""
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm import ModelResponse
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.enkryptai import EnkryptAIGuardrails
from litellm.types.utils import Choices, Message


@pytest.fixture
def enkryptai_guardrail():
    """Create an EnkryptAIGuardrail instance for testing"""
    return EnkryptAIGuardrails(
        api_key="test-api-key",
        api_base="https://api.test.enkryptai.com",
        guardrail_name="test-enkryptai-guardrail",
        event_hook="pre_call",
        default_on=True,
        detectors={
            "nsfw": {"enabled": True},
            "toxicity": {"enabled": True},
        }
    )


@pytest.fixture
def mock_user_api_key_dict():
    """Create a mock UserAPIKeyAuth object"""
    return UserAPIKeyAuth(
        user_id="test-user-id",
        user_email="test@example.com",
        key_name="test-key",
        key_alias=None,
        team_id=None,
        team_alias=None,
        user_role=None,
        api_key="test-api-key",
        permissions={},
        models=[],
        spend=0.0,
        max_budget=None,
        soft_budget=None,
        tpm_limit=None,
        rpm_limit=None,
        metadata={},
        max_parallel_requests=None,
        allowed_cache_controls=[],
        model_spend={},
        model_max_budget={},
    )


@pytest.fixture
def mock_request_data():
    """Create mock request data"""
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, how are you?"},
        ],
        "litellm_call_id": "test-call-id",
        "metadata": {"requester_ip_address": "192.168.1.1"},
    }


class TestEnkryptAIGuardrailConfiguration:
    """Test configuration and initialization of EnkryptAI guardrail"""

    def test_init_with_config(self):
        """Test initializing EnkryptAI guardrail with configuration"""
        guardrail = EnkryptAIGuardrails(
            api_key="test-key",
            api_base="https://api.test.enkryptai.com",
            policy_name="test-policy",
            detectors={"toxicity": {"enabled": True}}
        )
        assert guardrail.api_key == "test-key"
        assert guardrail.api_base == "https://api.test.enkryptai.com"
        assert guardrail.policy_name == "test-policy"
        assert guardrail.optional_params.get("detectors") == {"toxicity": {"enabled": True}}

    def test_init_with_env_vars(self):
        """Test initialization with environment variables"""
        with patch.dict(
            os.environ,
            {
                "ENKRYPTAI_API_KEY": "env-api-key",
                "ENKRYPTAI_API_BASE": "https://env.api.enkryptai.com",
            },
        ):
            guardrail = EnkryptAIGuardrails()
            assert guardrail.api_key == "env-api-key"
            assert guardrail.api_base == "https://env.api.enkryptai.com"

    def test_init_with_params_override_env(self):
        """Test that constructor params override environment variables"""
        with patch.dict(
            os.environ,
            {
                "ENKRYPTAI_API_KEY": "env-api-key",
            },
        ):
            guardrail = EnkryptAIGuardrails(
                api_key="param-api-key",
            )
            assert guardrail.api_key == "param-api-key"

    def test_init_without_api_key_raises_error(self):
        """Test that initialization without API key raises ValueError"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="EnkryptAI API key is required"):
                EnkryptAIGuardrails()


class TestEnkryptAIGuardrailHooks:
    """Test the guardrail hook methods"""

    @pytest.mark.asyncio
    async def test_pre_call_hook_allowed(
        self, enkryptai_guardrail, mock_user_api_key_dict, mock_request_data
    ):
        """Test pre-call hook when content is allowed"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "summary": {"nsfw": 0, "toxicity": []},
            "violations": [],
            "detected": False,
        }
        mock_response.raise_for_status = MagicMock()
        mock_response.status_code = 200

        with patch.object(
            enkryptai_guardrail.async_handler, "post", return_value=mock_response
        ) as mock_post:
            result = await enkryptai_guardrail.async_pre_call_hook(
                user_api_key_dict=mock_user_api_key_dict,
                cache=MagicMock(),
                data=mock_request_data,
                call_type="completion",
            )

            assert result == mock_request_data
            # Should be called twice - once for system message, once for user message
            assert mock_post.call_count == 2

            # Verify last API call details (for user message)
            call_args = mock_post.call_args
            assert call_args[1]["url"].endswith("/guardrails/policy/detect")
            assert call_args[1]["headers"]["apikey"] == "test-api-key"
            assert call_args[1]["json"]["text"] == "Hello, how are you?"

    @pytest.mark.asyncio
    async def test_pre_call_hook_blocked(
        self, enkryptai_guardrail, mock_user_api_key_dict, mock_request_data
    ):
        """Test pre-call hook when content is blocked"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "summary": {"toxicity": ["high"]},
            "violations": ["toxicity"],
            "detected": True,
        }
        mock_response.raise_for_status = MagicMock()
        mock_response.status_code = 200

        with patch.object(
            enkryptai_guardrail.async_handler, "post", return_value=mock_response
        ):
            with pytest.raises(ValueError) as exc_info:
                await enkryptai_guardrail.async_pre_call_hook(
                    user_api_key_dict=mock_user_api_key_dict,
                    cache=MagicMock(),
                    data=mock_request_data,
                    call_type="completion",
                )

            assert "Guardrail failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_pre_call_hook_with_policy_header(
        self, mock_user_api_key_dict, mock_request_data
    ):
        """Test pre-call hook with policy header"""
        guardrail = EnkryptAIGuardrails(
            api_key="test-key",
            policy_name="my-policy",
            guardrail_name="test-guardrail",
            event_hook="pre_call",
            default_on=True,
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "violations": [],
            "detected": False,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            guardrail.async_handler, "post", return_value=mock_response
        ) as mock_post:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=mock_user_api_key_dict,
                cache=MagicMock(),
                data=mock_request_data,
                call_type="completion",
            )

            # Verify policy header was sent
            call_args = mock_post.call_args
            assert call_args[1]["headers"]["x-enkrypt-policy"] == "my-policy"

    @pytest.mark.asyncio
    async def test_post_call_success_hook(
        self, enkryptai_guardrail, mock_user_api_key_dict, mock_request_data
    ):
        """Test post-call success hook"""
        # Create a mock ModelResponse
        response = ModelResponse(
            id="test-response-id",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="I'm doing well, thank you!", role="assistant"
                    ),
                )
            ],
            created=1234567890,
            model="gpt-3.5-turbo",
            object="chat.completion",
            system_fingerprint=None,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        mock_api_response = MagicMock()
        mock_api_response.json.return_value = {
            "violations": [],
            "detected": False,
        }
        mock_api_response.raise_for_status = MagicMock()

        # Update guardrail to use post_call event hook
        enkryptai_guardrail.event_hook = "post_call"

        with patch.object(
            enkryptai_guardrail.async_handler, "post", return_value=mock_api_response
        ) as mock_post:
            # Method doesn't return anything, just verify it doesn't raise an exception
            await enkryptai_guardrail.async_post_call_success_hook(
                data=mock_request_data,
                user_api_key_dict=mock_user_api_key_dict,
                response=response,
            )

            mock_post.assert_called_once()

            # Verify API call details
            call_args = mock_post.call_args
            assert call_args[1]["json"]["text"] == "I'm doing well, thank you!"

    @pytest.mark.asyncio
    async def test_moderation_hook(
        self, enkryptai_guardrail, mock_user_api_key_dict, mock_request_data
    ):
        """Test moderation hook (during_call)"""
        # Update guardrail to use during_call event hook
        enkryptai_guardrail.event_hook = "during_call"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "violations": [],
            "detected": False,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            enkryptai_guardrail.async_handler, "post", return_value=mock_response
        ):
            result = await enkryptai_guardrail.async_moderation_hook(
                data=mock_request_data,
                user_api_key_dict=mock_user_api_key_dict,
                call_type="completion",
            )

            assert result == mock_request_data

    @pytest.mark.asyncio
    async def test_api_failure_handling(
        self, enkryptai_guardrail, mock_user_api_key_dict, mock_request_data
    ):
        """Test API failure handling"""
        with patch.object(
            enkryptai_guardrail.async_handler,
            "post",
            side_effect=httpx.HTTPStatusError(
                "API Error", request=MagicMock(), response=MagicMock(status_code=500)
            ),
        ):
            with pytest.raises(httpx.HTTPStatusError):
                await enkryptai_guardrail.async_pre_call_hook(
                    user_api_key_dict=mock_user_api_key_dict,
                    cache=MagicMock(),
                    data=mock_request_data,
                    call_type="completion",
                )

    @pytest.mark.asyncio
    async def test_monitor_mode(
        self, mock_user_api_key_dict, mock_request_data
    ):
        """Test monitor mode (block_on_violation=False)"""
        guardrail = EnkryptAIGuardrails(
            api_key="test-key",
            block_on_violation=False,
            guardrail_name="test-guardrail",
            event_hook="pre_call",
            default_on=True,
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "violations": ["toxicity"],
            "detected": True,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            guardrail.async_handler, "post", return_value=mock_response
        ):
            # Should not raise exception in monitor mode
            result = await guardrail.async_pre_call_hook(
                user_api_key_dict=mock_user_api_key_dict,
                cache=MagicMock(),
                data=mock_request_data,
                call_type="completion",
            )

            assert result == mock_request_data

    def test_determine_guardrail_status(self, enkryptai_guardrail):
        """Test _determine_guardrail_status helper method"""
        # Test success status (no attacks detected)
        response_json = {"summary": {"nsfw": 0, "toxicity": []}}
        status = enkryptai_guardrail._determine_guardrail_status(response_json)
        assert status == "success"

        # Test intervened status (attacks detected)
        response_json = {"summary": {"toxicity": ["high"], "nsfw": 1}}
        status = enkryptai_guardrail._determine_guardrail_status(response_json)
        assert status == "guardrail_intervened"

        # Test failed status (invalid response)
        response_json = "invalid"
        status = enkryptai_guardrail._determine_guardrail_status(response_json)
        assert status == "guardrail_failed_to_respond"




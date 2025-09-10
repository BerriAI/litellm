import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm import ModelResponse
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.noma import (
    NomaGuardrail,
    initialize_guardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.noma.noma import NomaBlockedMessage
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2
from litellm.types.utils import Choices, Message


@pytest.fixture
def noma_guardrail():
    """Create a NomaGuardrail instance for testing"""
    return NomaGuardrail(
        api_key="test-api-key",
        api_base="https://api.test.noma.security/",
        application_id="test-app",
        monitor_mode=False,
        block_failures=True,
        guardrail_name="test-noma-guardrail",
        event_hook="pre_call",
        default_on=True,
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


class TestNomaGuardrailConfiguration:
    """Test configuration and initialization of Noma guardrail"""

    def test_init_with_config(self):
        """Test initializing Noma guardrail via init_guardrails_v2"""
        with patch.dict(
            os.environ,
            {
                "NOMA_API_KEY": "test-api-key",
                "NOMA_API_BASE": "https://api.test.noma.security/",
            },
        ):
            init_guardrails_v2(
                all_guardrails=[
                    {
                        "guardrail_name": "noma-pre-guard",
                        "litellm_params": {
                            "guardrail": "noma",
                            "mode": "pre_call",
                            "application_id": "test-app",
                            "monitor_mode": False,
                            "block_failures": True,
                        },
                    }
                ],
                config_file_path="",
            )

    def test_init_with_env_vars(self):
        """Test initialization with environment variables"""
        with patch.dict(
            os.environ,
            {
                "NOMA_API_KEY": "env-api-key",
                "NOMA_API_BASE": "https://env.api.noma.security/",
                "NOMA_APPLICATION_ID": "env-app-id",
                "NOMA_MONITOR_MODE": "true",
                "NOMA_BLOCK_FAILURES": "false",
            },
        ):
            guardrail = NomaGuardrail()
            assert guardrail.api_key == "env-api-key"
            assert guardrail.api_base == "https://env.api.noma.security/"
            assert guardrail.application_id == "env-app-id"
            assert guardrail.monitor_mode is True
            assert guardrail.block_failures is False

    def test_init_with_params_override_env(self):
        """Test that constructor params override environment variables"""
        with patch.dict(
            os.environ,
            {
                "NOMA_API_KEY": "env-api-key",
                "NOMA_MONITOR_MODE": "true",
            },
        ):
            guardrail = NomaGuardrail(
                api_key="param-api-key",
                monitor_mode=False,
            )
            assert guardrail.api_key == "param-api-key"
            assert guardrail.monitor_mode is False

    def test_initialize_guardrail_function(self):
        """Test the initialize_guardrail function"""
        from litellm.types.guardrails import Guardrail, LitellmParams

        litellm_params = LitellmParams(
            guardrail="noma",
            mode="pre_call",
            api_key="test-key",
            api_base="https://test.api/",
            application_id="test-app",
            monitor_mode=True,
            block_failures=False,
        )

        guardrail = Guardrail(
            guardrail_name="test-guardrail",
            litellm_params=litellm_params,
        )

        with patch("litellm.logging_callback_manager.add_litellm_callback") as mock_add:
            result = initialize_guardrail(litellm_params, guardrail)

            assert isinstance(result, NomaGuardrail)
            assert result.api_key == "test-key"
            assert result.api_base == "https://test.api/"
            assert result.application_id == "test-app"
            assert result.monitor_mode is True
            assert result.block_failures is False
            mock_add.assert_called_once_with(result)


class TestNomaBlockedMessage:
    """Test the NomaBlockedMessage exception class"""

    def test_blocked_message_basic(self):
        """Test basic blocked message creation"""
        response = {
            "verdict": False,
            "prompt": {
                "harmfulContent": {"result": True, "confidence": 0.9},
                "code": {"result": False, "confidence": 0.1},
            },
        }

        exception = NomaBlockedMessage(response)
        assert exception.status_code == 400
        assert exception.detail["error"] == "Request blocked by Noma guardrail"
        assert "harmfulContent" in exception.detail["details"]["prompt"]
        assert "code" not in exception.detail["details"]["prompt"]

    def test_blocked_message_with_sensitive_data(self):
        """Test blocked message with sensitive data detection"""
        response = {
            "verdict": False,
            "prompt": {
                "sensitiveData": {
                    "email": {"result": True, "entities": ["test@example.com"]},
                    "phone": {"result": False},
                },
            },
        }

        exception = NomaBlockedMessage(response)
        assert "email" in exception.detail["details"]["prompt"]["sensitiveData"]
        assert "phone" not in exception.detail["details"]["prompt"]["sensitiveData"]

    def test_blocked_message_with_topics(self):
        """Test blocked message with topic guardrails"""
        response = {
            "verdict": False,
            "prompt": {
                "bannedTopics": {
                    "violence": {"result": True, "confidence": 0.95},
                    "politics": {"result": False, "confidence": 0.2},
                },
            },
        }

        exception = NomaBlockedMessage(response)
        assert "violence" in exception.detail["details"]["prompt"]["bannedTopics"]
        assert "politics" not in exception.detail["details"]["prompt"]["bannedTopics"]


class TestNomaGuardrailHooks:
    """Test the guardrail hook methods"""

    @pytest.mark.asyncio
    async def test_pre_call_hook_allowed(
        self, noma_guardrail, mock_user_api_key_dict, mock_request_data
    ):
        """Test pre-call hook when content is allowed"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"verdict": True}
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            noma_guardrail.async_handler, "post", return_value=mock_response
        ) as mock_post:
            result = await noma_guardrail.async_pre_call_hook(
                user_api_key_dict=mock_user_api_key_dict,
                cache=MagicMock(),
                data=mock_request_data,
                call_type="completion",
            )

            assert result == mock_request_data
            mock_post.assert_called_once()

            # Verify API call details
            call_args = mock_post.call_args
            assert call_args[0][0].endswith("/ai-dr/v1/prompt/scan/aggregate")
            assert call_args[1]["headers"]["X-Noma-AIDR-Application-ID"] == "test-app"
            assert call_args[1]["headers"]["Authorization"] == "Bearer test-api-key"
            assert call_args[1]["json"]["request"]["text"] == "Hello, how are you?"

    @pytest.mark.asyncio
    async def test_pre_call_hook_blocked(
        self, noma_guardrail, mock_user_api_key_dict, mock_request_data
    ):
        """Test pre-call hook when content is blocked"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "verdict": False,
            "originalResponse": {
                "prompt": {"harmfulContent": {"result": True, "confidence": 0.9}}
            },
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            noma_guardrail.async_handler, "post", return_value=mock_response
        ):
            with pytest.raises(NomaBlockedMessage) as exc_info:
                await noma_guardrail.async_pre_call_hook(
                    user_api_key_dict=mock_user_api_key_dict,
                    cache=MagicMock(),
                    data=mock_request_data,
                    call_type="completion",
                )

            assert exc_info.value.status_code == 400
            assert "harmfulContent" in exc_info.value.detail["details"]["prompt"]

    @pytest.mark.asyncio
    async def test_pre_call_hook_monitor_mode(
        self, mock_user_api_key_dict, mock_request_data
    ):
        """Test pre-call hook in monitor mode (logs but doesn't block)"""
        guardrail = NomaGuardrail(
            api_key="test-key",
            monitor_mode=True,
            guardrail_name="test-guardrail",
            event_hook="pre_call",
            default_on=True,
        )

        with patch.object(
            guardrail, "_create_background_noma_check"
        ) as mock_create_background:
            # Should return immediately without waiting for API call
            result = await guardrail.async_pre_call_hook(
                user_api_key_dict=mock_user_api_key_dict,
                cache=MagicMock(),
                data=mock_request_data,
                call_type="completion",
            )

            assert result == mock_request_data
            # Verify background task was created
            mock_create_background.assert_called_once()

    @pytest.mark.asyncio
    async def test_pre_call_hook_monitor_mode_background_task_failure(
        self, mock_user_api_key_dict, mock_request_data
    ):
        """Test pre-call hook in monitor mode when background task creation fails"""
        guardrail = NomaGuardrail(
            api_key="test-key",
            monitor_mode=True,
            guardrail_name="test-guardrail",
            event_hook="pre_call",
            default_on=True,
        )

        with patch.object(
            guardrail, "_create_background_noma_check", side_effect=Exception("Task creation failed")
        ):
            # Should still return successfully even if background task creation fails
            result = await guardrail.async_pre_call_hook(
                user_api_key_dict=mock_user_api_key_dict,
                cache=MagicMock(),
                data=mock_request_data,
                call_type="completion",
            )

            assert result == mock_request_data

    @pytest.mark.asyncio
    async def test_post_call_success_hook(
        self, noma_guardrail, mock_user_api_key_dict, mock_request_data
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
        mock_api_response.json.return_value = {"verdict": True}
        mock_api_response.raise_for_status = MagicMock()

        # Update guardrail to use post_call event hook
        noma_guardrail.event_hook = "post_call"

        with patch.object(
            noma_guardrail.async_handler, "post", return_value=mock_api_response
        ) as mock_post:
            result = await noma_guardrail.async_post_call_success_hook(
                data=mock_request_data,
                user_api_key_dict=mock_user_api_key_dict,
                response=response,
            )

            assert result == response
            mock_post.assert_called_once()

            # Verify API call details
            call_args = mock_post.call_args
            assert (
                call_args[1]["json"]["response"]["text"] == "I'm doing well, thank you!"
            )
            assert call_args[1]["json"]["context"]["requestId"] == "test-response-id"

    @pytest.mark.asyncio
    async def test_moderation_hook(
        self, noma_guardrail, mock_user_api_key_dict, mock_request_data
    ):
        """Test moderation hook (during_call)"""
        # Update guardrail to use during_call event hook
        noma_guardrail.event_hook = "during_call"

        mock_response = MagicMock()
        mock_response.json.return_value = {"verdict": True}
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            noma_guardrail.async_handler, "post", return_value=mock_response
        ):
            result = await noma_guardrail.async_moderation_hook(
                data=mock_request_data,
                user_api_key_dict=mock_user_api_key_dict,
                call_type="completion",
            )

            assert result == mock_request_data

    @pytest.mark.asyncio
    async def test_api_failure_handling(
        self, noma_guardrail, mock_user_api_key_dict, mock_request_data
    ):
        with patch.object(
            noma_guardrail.async_handler,
            "post",
            side_effect=httpx.HTTPStatusError(
                "API Error", request=MagicMock(), response=MagicMock(status_code=500)
            ),
        ):
            with pytest.raises(httpx.HTTPStatusError):
                await noma_guardrail.async_pre_call_hook(
                    user_api_key_dict=mock_user_api_key_dict,
                    cache=MagicMock(),
                    data=mock_request_data,
                    call_type="completion",
                )

    @pytest.mark.asyncio
    async def test_api_failure_no_block(
        self, mock_user_api_key_dict, mock_request_data
    ):
        guardrail = NomaGuardrail(
            api_key="test-key",
            block_failures=False,
            guardrail_name="test-guardrail",
            event_hook="pre_call",
            default_on=True,
        )

        with patch.object(
            guardrail.async_handler,
            "post",
            side_effect=httpx.HTTPStatusError(
                "API Error", request=MagicMock(), response=MagicMock(status_code=500)
            ),
        ):
            result = await guardrail.async_pre_call_hook(
                user_api_key_dict=mock_user_api_key_dict,
                cache=MagicMock(),
                data=mock_request_data,
                call_type="completion",
            )

            assert result == mock_request_data

    def test_extract_user_message(self, noma_guardrail):
        data = {
            "messages": [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "First user message"},
                {"role": "assistant", "content": "Assistant response"},
                {"role": "user", "content": "Second user message"},
            ]
        }

        import asyncio

        message = asyncio.run(noma_guardrail._extract_user_message(data))
        assert message == "Second user message"

        data = {"messages": [{"role": "system", "content": "System prompt"}]}
        message = asyncio.run(noma_guardrail._extract_user_message(data))
        assert message is None

        data = {"messages": []}
        message = asyncio.run(noma_guardrail._extract_user_message(data))
        assert message is None

        data = {}
        message = asyncio.run(noma_guardrail._extract_user_message(data))
        assert message is None


class TestBackgroundProcessing:
    """Test the new background processing functionality"""

    @pytest.fixture
    def monitor_mode_guardrail(self):
        """Create a guardrail with monitor mode enabled"""
        return NomaGuardrail(
            api_key="test-api-key",
            api_base="https://api.test.noma.security/",
            application_id="test-app",
            monitor_mode=True,  # Enable monitor mode
            block_failures=True,
            guardrail_name="test-noma-guardrail",
            event_hook="pre_call",
            default_on=True,
        )

    @pytest.mark.asyncio
    async def test_process_user_message_check_monitor_mode(
        self, monitor_mode_guardrail, mock_user_api_key_dict, mock_request_data
    ):
        """Test shared helper method in monitor mode"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "verdict": False,
            "originalResponse": {"prompt": {"harmfulContent": {"result": True}}},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            monitor_mode_guardrail.async_handler, "post", return_value=mock_response
        ) as mock_post:
            with patch.object(
                monitor_mode_guardrail, "_handle_verdict_background"
            ) as mock_handle_verdict:
                result = await monitor_mode_guardrail._process_user_message_check(
                    mock_request_data, mock_user_api_key_dict
                )

                assert result == "Hello, how are you?"
                mock_post.assert_called_once()
                mock_handle_verdict.assert_called_once_with(
                    "user", "Hello, how are you?", mock_response.json.return_value
                )

    @pytest.mark.asyncio
    async def test_process_user_message_check_non_monitor_mode(
        self, noma_guardrail, mock_user_api_key_dict, mock_request_data
    ):
        """Test shared helper method in non-monitor mode"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"verdict": True}
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            noma_guardrail.async_handler, "post", return_value=mock_response
        ) as mock_post:
            with patch.object(
                noma_guardrail, "_check_verdict"
            ) as mock_check_verdict:
                result = await noma_guardrail._process_user_message_check(
                    mock_request_data, mock_user_api_key_dict
                )

                assert result == "Hello, how are you?"
                mock_post.assert_called_once()
                mock_check_verdict.assert_called_once_with(
                    "user", "Hello, how are you?", mock_response.json.return_value
                )

    @pytest.mark.asyncio
    async def test_process_llm_response_check_monitor_mode(
        self, monitor_mode_guardrail, mock_user_api_key_dict, mock_request_data
    ):
        """Test LLM response processing in monitor mode"""
        from litellm.types.utils import Choices, Message
        
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
        mock_api_response.json.return_value = {"verdict": True}
        mock_api_response.raise_for_status = MagicMock()

        with patch.object(
            monitor_mode_guardrail.async_handler, "post", return_value=mock_api_response
        ) as mock_post:
            with patch.object(
                monitor_mode_guardrail, "_handle_verdict_background"
            ) as mock_handle_verdict:
                result = await monitor_mode_guardrail._process_llm_response_check(
                    mock_request_data, response, mock_user_api_key_dict
                )

                assert result == "I'm doing well, thank you!"
                mock_post.assert_called_once()
                mock_handle_verdict.assert_called_once_with(
                    "assistant", "I'm doing well, thank you!", mock_api_response.json.return_value
                )

    @pytest.mark.asyncio
    async def test_check_user_message_background(
        self, monitor_mode_guardrail, mock_user_api_key_dict, mock_request_data
    ):
        """Test background user message check method"""
        with patch.object(
            monitor_mode_guardrail, "_process_user_message_check"
        ) as mock_process:
            await monitor_mode_guardrail._check_user_message_background(
                mock_request_data, mock_user_api_key_dict
            )

            mock_process.assert_called_once_with(mock_request_data, mock_user_api_key_dict)

    @pytest.mark.asyncio
    async def test_check_user_message_background_exception_handling(
        self, monitor_mode_guardrail, mock_user_api_key_dict, mock_request_data
    ):
        """Test background user message check handles exceptions gracefully"""
        with patch.object(
            monitor_mode_guardrail, "_process_user_message_check",
            side_effect=Exception("API failed")
        ):
            # Should not raise exception, just log error
            await monitor_mode_guardrail._check_user_message_background(
                mock_request_data, mock_user_api_key_dict
            )

    @pytest.mark.asyncio
    async def test_check_llm_response_background(
        self, monitor_mode_guardrail, mock_user_api_key_dict, mock_request_data
    ):
        """Test background LLM response check method"""
        from litellm.types.utils import Choices, Message
        
        response = ModelResponse(
            id="test-response-id",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="Test response", role="assistant"),
                )
            ],
            created=1234567890,
            model="gpt-3.5-turbo",
            object="chat.completion",
            system_fingerprint=None,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        with patch.object(
            monitor_mode_guardrail, "_process_llm_response_check"
        ) as mock_process:
            await monitor_mode_guardrail._check_llm_response_background(
                mock_request_data, response, mock_user_api_key_dict
            )

            mock_process.assert_called_once_with(
                mock_request_data, response, mock_user_api_key_dict
            )

    @pytest.mark.asyncio
    async def test_handle_verdict_background_blocked(self, monitor_mode_guardrail):
        """Test background verdict handling for blocked content"""
        response_json = {
            "verdict": False,
            "originalResponse": {"prompt": {"harmfulContent": {"result": True}}},
        }

        with patch("litellm._logging.verbose_proxy_logger.warning") as mock_warning:
            await monitor_mode_guardrail._handle_verdict_background(
                "user", "test message", response_json
            )
            
            mock_warning.assert_called_once()
            assert "blocked user message" in mock_warning.call_args[0][0]

    @pytest.mark.asyncio
    async def test_handle_verdict_background_allowed(self, monitor_mode_guardrail):
        """Test background verdict handling for allowed content"""
        response_json = {"verdict": True}

        with patch("litellm._logging.verbose_proxy_logger.info") as mock_info:
            await monitor_mode_guardrail._handle_verdict_background(
                "assistant", "test response", response_json
            )
            
            mock_info.assert_called_once()
            assert "allowed assistant message" in mock_info.call_args[0][0]

    @pytest.mark.asyncio
    async def test_create_background_noma_check(self, monitor_mode_guardrail):
        """Test background task creation"""
        async def dummy_coroutine():
            return "completed"

        with patch("asyncio.create_task") as mock_create_task:
            monitor_mode_guardrail._create_background_noma_check(dummy_coroutine())
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_background_noma_check_exception(self, monitor_mode_guardrail):
        """Test background task creation with exception handling"""
        async def dummy_coroutine():
            return "completed"

        with patch("asyncio.create_task", side_effect=Exception("Task creation failed")):
            # Should not raise exception, just log error
            monitor_mode_guardrail._create_background_noma_check(dummy_coroutine())

    @pytest.mark.asyncio
    async def test_moderation_hook_monitor_mode(
        self, monitor_mode_guardrail, mock_user_api_key_dict, mock_request_data
    ):
        """Test moderation hook in monitor mode"""
        # Update event hook to during_call
        monitor_mode_guardrail.event_hook = "during_call"

        with patch.object(
            monitor_mode_guardrail, "_create_background_noma_check"
        ) as mock_create_background:
            result = await monitor_mode_guardrail.async_moderation_hook(
                data=mock_request_data,
                user_api_key_dict=mock_user_api_key_dict,
                call_type="completion",
            )

            assert result == mock_request_data
            mock_create_background.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_call_success_hook_monitor_mode(
        self, monitor_mode_guardrail, mock_user_api_key_dict, mock_request_data
    ):
        """Test post-call success hook in monitor mode"""
        from litellm.types.utils import Choices, Message
        
        # Update event hook to post_call
        monitor_mode_guardrail.event_hook = "post_call"

        response = ModelResponse(
            id="test-response-id",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="Test response", role="assistant"),
                )
            ],
            created=1234567890,
            model="gpt-3.5-turbo",
            object="chat.completion",
            system_fingerprint=None,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        with patch.object(
            monitor_mode_guardrail, "_create_background_noma_check"
        ) as mock_create_background:
            result = await monitor_mode_guardrail.async_post_call_success_hook(
                data=mock_request_data,
                user_api_key_dict=mock_user_api_key_dict,
                response=response,
            )

            assert result == response
            mock_create_background.assert_called_once()


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_guardrail_flow(self):
        """Test full guardrail flow with multiple hooks"""
        with patch.dict(
            os.environ,
            {
                "NOMA_API_KEY": "test-api-key",
                "NOMA_API_BASE": "https://api.test.noma.security/",
            },
        ):
            init_guardrails_v2(
                all_guardrails=[
                    {
                        "guardrail_name": "noma-pre-guard",
                        "litellm_params": {
                            "guardrail": "noma",
                            "mode": "pre_call",
                            "application_id": "test-app",
                        },
                    },
                    {
                        "guardrail_name": "noma-post-guard",
                        "litellm_params": {
                            "guardrail": "noma",
                            "mode": "post_call",
                            "application_id": "test-app",
                        },
                    },
                ],
                config_file_path="",
            )

            custom_loggers = (
                litellm.logging_callback_manager.get_custom_loggers_for_type(
                    callback_type=litellm.integrations.custom_guardrail.CustomGuardrail
                )
            )
            assert len(custom_loggers) >= 2

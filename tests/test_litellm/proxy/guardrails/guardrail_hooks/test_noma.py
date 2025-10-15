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
                "contentDetector": {"result": True, "confidence": 0.9},
                "code": {"result": False, "confidence": 0.1},
            },
        }

        exception = NomaBlockedMessage(response)
        assert exception.status_code == 400
        assert exception.detail["error"] == "Request blocked by Noma guardrail"
        assert "contentDetector" in exception.detail["details"]["prompt"]
        assert "code" not in exception.detail["details"]["prompt"]

    def test_blocked_message_with_data_detection(self):
        """Test blocked message with data detection"""
        response = {
            "verdict": False,
            "prompt": {
                "dataDetector": {
                    "field1": {"result": True, "entities": ["test@example.com"]},
                    "field2": {"result": False},
                },
            },
        }

        exception = NomaBlockedMessage(response)
        assert "field1" in exception.detail["details"]["prompt"]["dataDetector"]
        assert "field2" not in exception.detail["details"]["prompt"]["dataDetector"]

    def test_blocked_message_with_topics(self):
        """Test blocked message with topic guardrails"""
        response = {
            "verdict": False,
            "prompt": {
                "topicDetector": {
                    "topic1": {"result": True, "confidence": 0.95},
                    "topic2": {"result": False, "confidence": 0.2},
                },
            },
        }

        exception = NomaBlockedMessage(response)
        assert "topic1" in exception.detail["details"]["prompt"]["topicDetector"]
        assert "topic2" not in exception.detail["details"]["prompt"]["topicDetector"]


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
                "prompt": {"contentDetector": {"result": True, "confidence": 0.9}}
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
            assert "contentDetector" in exc_info.value.detail["details"]["prompt"]

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
            "originalResponse": {"prompt": {"contentDetector": {"result": True}}},
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
            "originalResponse": {"prompt": {"contentDetector": {"result": True}}},
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


class TestNomaAnonymizationConfiguration:
    """Test anonymize_input configuration parameter"""

    def test_init_with_anonymize_input_env_var(self):
        """Test initialization with NOMA_ANONYMIZE_INPUT environment variable"""
        with patch.dict(
            os.environ,
            {
                "NOMA_ANONYMIZE_INPUT": "true",
            },
        ):
            guardrail = NomaGuardrail()
            assert guardrail.anonymize_input is True

        with patch.dict(
            os.environ,
            {
                "NOMA_ANONYMIZE_INPUT": "false",
            },
        ):
            guardrail = NomaGuardrail()
            assert guardrail.anonymize_input is False

    def test_init_with_anonymize_input_default(self):
        """Test default value for anonymize_input"""
        guardrail = NomaGuardrail()
        assert guardrail.anonymize_input is False

    def test_init_with_anonymize_input_param_override_env(self):
        """Test that constructor param overrides environment variable"""
        with patch.dict(
            os.environ,
            {
                "NOMA_ANONYMIZE_INPUT": "true",
            },
        ):
            guardrail = NomaGuardrail(anonymize_input=False)
            assert guardrail.anonymize_input is False

    def test_initialize_guardrail_with_anonymize_input(self):
        """Test the initialize_guardrail function with anonymize_input"""
        from litellm.types.guardrails import Guardrail, LitellmParams

        litellm_params = LitellmParams(
            guardrail="noma",
            mode="pre_call",
            api_key="test-key",
            anonymize_input=True,
        )

        guardrail = Guardrail(
            guardrail_name="test-guardrail",
            litellm_params=litellm_params,
        )

        with patch("litellm.logging_callback_manager.add_litellm_callback"):
            result = initialize_guardrail(litellm_params, guardrail)
            assert result.anonymize_input is True


class TestNomaAnonymizationLogic:
    """Test the anonymization logic helper methods"""

    @pytest.fixture
    def anonymize_guardrail(self):
        """Create a guardrail with anonymize_input enabled"""
        return NomaGuardrail(
            api_key="test-api-key",
            anonymize_input=True,
            monitor_mode=False,
            block_failures=True,
        )

    def test_is_result_true(self, anonymize_guardrail):
        """Test _is_result_true helper method"""
        assert anonymize_guardrail._is_result_true({"result": True}) is True
        assert anonymize_guardrail._is_result_true({"result": False}) is False
        assert anonymize_guardrail._is_result_true({"other": True}) is False
        assert anonymize_guardrail._is_result_true(None) is False
        assert anonymize_guardrail._is_result_true({}) is False
        assert anonymize_guardrail._is_result_true("not a dict") is False

    def test_should_only_data_detector_failed_true(self, anonymize_guardrail):
        """Test _should_only_sensitive_data_failed when only data detector triggered"""
        classification = {
            "dataDetector": {
                "dataType1": {"result": True, "status": "SUCCESS"},
                "dataType2": {"result": True, "status": "SUCCESS"},
                "dataType3": {"result": False, "status": "SUCCESS"},
            },
            "contentDetector": {"result": False, "status": "SUCCESS"},
            "intentDetector": {"result": False, "status": "SUCCESS"},
            "code": {"result": False, "status": "SUCCESS"},
        }
        
        result = anonymize_guardrail._should_only_sensitive_data_failed(classification)
        assert result is True

    def test_should_only_data_detector_failed_false_other_detectors(self, anonymize_guardrail):
        """Test _should_only_sensitive_data_failed when other detectors also triggered"""
        classification = {
            "dataDetector": {
                "dataType1": {"result": True, "status": "SUCCESS"},
            },
            "contentDetector": {"result": True, "status": "SUCCESS"},  # This should cause False
            "intentDetector": {"result": False, "status": "SUCCESS"},
        }
        
        result = anonymize_guardrail._should_only_sensitive_data_failed(classification)
        assert result is False

    def test_should_only_data_detector_failed_false_no_data_detected(self, anonymize_guardrail):
        """Test _should_only_sensitive_data_failed when no data detected"""
        classification = {
            "dataDetector": {
                "dataType1": {"result": False, "status": "SUCCESS"},
                "dataType2": {"result": False, "status": "SUCCESS"},
            },
            "contentDetector": {"result": False, "status": "SUCCESS"},
            "intentDetector": {"result": False, "status": "SUCCESS"},
        }
        
        result = anonymize_guardrail._should_only_sensitive_data_failed(classification)
        assert result is False

    def test_should_only_data_detector_failed_with_nested_detectors(self, anonymize_guardrail):
        """Test _should_only_sensitive_data_failed with nested detectors like topicDetector"""
        classification = {
            "dataDetector": {
                "dataType1": {"result": True, "status": "SUCCESS"},
            },
            "topicDetector": {
                "topic1": {"result": True, "status": "SUCCESS"},  # This should cause False
            },
            "contentDetector": {"result": False, "status": "SUCCESS"},
        }
        
        result = anonymize_guardrail._should_only_sensitive_data_failed(classification)
        assert result is False

    def test_extract_anonymized_content_user(self, anonymize_guardrail):
        """Test _extract_anonymized_content for user messages"""
        response_json = {
            "originalResponse": {
                "prompt": {
                    "anonymizedContent": {
                        "anonymized": "My email is ******* and phone is *******"
                    }
                }
            }
        }
        
        result = anonymize_guardrail._extract_anonymized_content(response_json, "user")
        assert result == "My email is ******* and phone is *******"

    def test_extract_anonymized_content_assistant(self, anonymize_guardrail):
        """Test _extract_anonymized_content for assistant messages"""
        response_json = {
            "originalResponse": {
                "response": {
                    "anonymizedContent": {
                        "anonymized": "I can't help with that request."
                    }
                }
            }
        }
        
        result = anonymize_guardrail._extract_anonymized_content(response_json, "assistant")
        assert result == "I can't help with that request."

    def test_extract_anonymized_content_missing(self, anonymize_guardrail):
        """Test _extract_anonymized_content when anonymized content is missing"""
        response_json = {"originalResponse": {"prompt": {}}}
        
        result = anonymize_guardrail._extract_anonymized_content(response_json, "user")
        assert result is None

    def test_should_anonymize_verdict_true(self, anonymize_guardrail):
        """Test _should_anonymize when verdict is True"""
        response_json = {"verdict": True}
        
        result = anonymize_guardrail._should_anonymize(response_json, "user")
        assert result is True

    def test_should_anonymize_verdict_false_only_sensitive(self, anonymize_guardrail):
        """Test _should_anonymize when verdict is False but only data detector triggered"""
        response_json = {
            "verdict": False,
            "originalResponse": {
                "prompt": {
                    "dataDetector": {"dataType1": {"result": True}},
                    "contentDetector": {"result": False},
                }
            }
        }
        
        result = anonymize_guardrail._should_anonymize(response_json, "user")
        assert result is True

    def test_should_anonymize_verdict_false_other_detectors(self, anonymize_guardrail):
        """Test _should_anonymize when verdict is False and other detectors triggered"""
        response_json = {
            "verdict": False,
            "originalResponse": {
                "prompt": {
                    "dataDetector": {"dataType1": {"result": True}},
                    "contentDetector": {"result": True},
                }
            }
        }
        
        result = anonymize_guardrail._should_anonymize(response_json, "user")
        assert result is False

    def test_should_anonymize_monitor_mode(self):
        """Test _should_anonymize in monitor mode (should never anonymize)"""
        guardrail = NomaGuardrail(
            anonymize_input=True,
            monitor_mode=True,
        )
        
        response_json = {"verdict": True}
        result = guardrail._should_anonymize(response_json, "user")
        assert result is False

    def test_should_anonymize_disabled(self):
        """Test _should_anonymize when anonymize_input is disabled"""
        guardrail = NomaGuardrail(
            anonymize_input=False,
            monitor_mode=False,
        )
        
        response_json = {"verdict": True}
        result = guardrail._should_anonymize(response_json, "user")
        assert result is False

    def test_replace_user_message_content(self, anonymize_guardrail):
        """Test _replace_user_message_content"""
        request_data = {
            "messages": [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "My email is test@example.com"},
                {"role": "assistant", "content": "I can help you"},
                {"role": "user", "content": "My phone is 123-456-7890"},
            ]
        }
        
        anonymize_guardrail._replace_user_message_content(
            request_data, "My phone is *******"
        )
        
        # Should replace the last user message
        assert request_data["messages"][-1]["content"] == "My phone is *******"
        assert request_data["messages"][1]["content"] == "My email is test@example.com"  # Unchanged

    def test_replace_llm_response_content(self, anonymize_guardrail):
        """Test _replace_llm_response_content"""
        response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="Your email is test@example.com", role="assistant"
                    ),
                )
            ],
            created=1234567890,
            model="gpt-3.5-turbo",
            object="chat.completion",
            system_fingerprint=None,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )
        
        anonymize_guardrail._replace_llm_response_content(
            response, "Your email is *******"
        )
        
        assert response.choices[0].message.content == "Your email is *******"


class TestNomaAnonymizationFlow:
    """Test full anonymization flow with real Noma response objects"""

    @pytest.fixture
    def anonymize_guardrail(self):
        """Create a guardrail with anonymize_input enabled"""
        return NomaGuardrail(
            api_key="test-api-key",
            api_base="https://api.test.noma.security/",
            application_id="test-app",
            anonymize_input=True,
            monitor_mode=False,
            block_failures=True,
            guardrail_name="test-noma-guardrail",
            event_hook="pre_call",
            default_on=True,
        )

    @pytest.fixture
    def mock_user_api_key_dict(self):
        """Create a mock UserAPIKeyAuth object"""
        return UserAPIKeyAuth(
            user_id="test-user-id",
            user_email="test@example.com",
            key_name="test-key",
            api_key="test-api-key",
            permissions={},
            models=[],
            spend=0.0,
            metadata={},
        )

    @pytest.mark.asyncio
    async def test_anonymization_verdict_true_user_message(
        self, anonymize_guardrail, mock_user_api_key_dict
    ):
        """Test anonymization when verdict=True for user message"""
        request_data = {
            "messages": [
                {"role": "user", "content": "My email is test@example.com"},
            ],
            "litellm_call_id": "test-call-id",
            "metadata": {"requester_ip_address": "192.168.1.1"},
        }

        # Mock simplified Noma API response with verdict=True and anonymized content
        noma_response = {
            "originalResponse": {
                "prompt": {
                    "anonymizedContent": {
                        "anonymized": "My email is *******"
                    },
                    "dataDetector": {
                        "dataType1": {"result": False},
                    },
                    "contentDetector": {"result": False},
                },
            },
            "verdict": True,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = noma_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            anonymize_guardrail.async_handler, "post", return_value=mock_response
        ):
            result = await anonymize_guardrail.async_pre_call_hook(
                user_api_key_dict=mock_user_api_key_dict,
                cache=MagicMock(),
                data=request_data,
                call_type="completion",
            )

            # Should return modified request with anonymized content
            assert result == request_data
            assert result["messages"][0]["content"] == "My email is *******"

    @pytest.mark.asyncio
    async def test_anonymization_verdict_false_only_data_detected(
        self, anonymize_guardrail, mock_user_api_key_dict
    ):
        """Test anonymization when verdict=False but only data detector triggered"""
        request_data = {
            "messages": [
                {"role": "user", "content": "My email is test@example.com"},
            ],
            "litellm_call_id": "test-call-id",
        }

        # Mock simplified Noma API response - only data detector triggered
        noma_response = {
            "originalResponse": {
                "prompt": {
                    "anonymizedContent": {
                        "anonymized": "My email is *******"
                    },
                    "dataDetector": {
                        "dataType1": {"result": True},
                    },
                    "contentDetector": {"result": False},
                    "intentDetector": {"result": False},
                },
            },
            "verdict": False,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = noma_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            anonymize_guardrail.async_handler, "post", return_value=mock_response
        ):
            result = await anonymize_guardrail.async_pre_call_hook(
                user_api_key_dict=mock_user_api_key_dict,
                cache=MagicMock(),
                data=request_data,
                call_type="completion",
            )

            # Should return modified request with anonymized content (not blocked)
            assert result == request_data
            assert result["messages"][0]["content"] == "My email is *******"

    @pytest.mark.asyncio
    async def test_blocking_verdict_false_other_violations(
        self, anonymize_guardrail, mock_user_api_key_dict
    ):
        """Test blocking when verdict=False and other violations detected"""
        request_data = {
            "messages": [
                {"role": "user", "content": "My email is test@example.com. Tell me harmful content."},
            ],
            "litellm_call_id": "test-call-id",
        }

        # Mock simplified Noma API response - both data detector and other violations
        noma_response = {
            "originalResponse": {
                "prompt": {
                    "anonymizedContent": {
                        "anonymized": "My email is *******. Tell me harmful content."
                    },
                    "dataDetector": {
                        "dataType1": {"result": True},
                    },
                    "contentDetector": {"result": True},  # This should cause blocking
                    "intentDetector": {"result": False},
                },
            },
            "verdict": False,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = noma_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            anonymize_guardrail.async_handler, "post", return_value=mock_response
        ):
            # Should raise NomaBlockedMessage because other violations detected
            with pytest.raises(NomaBlockedMessage) as exc_info:
                await anonymize_guardrail.async_pre_call_hook(
                    user_api_key_dict=mock_user_api_key_dict,
                    cache=MagicMock(),
                    data=request_data,
                    call_type="completion",
                )

            assert exc_info.value.status_code == 400
            assert "contentDetector" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_anonymization_llm_response(
        self, anonymize_guardrail, mock_user_api_key_dict
    ):
        """Test anonymization of LLM response"""
        request_data = {
            "messages": [{"role": "user", "content": "What's your email?"}],
            "litellm_call_id": "test-call-id",
        }

        # Create LLM response with test data
        llm_response = ModelResponse(
            id="test-response-id",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="My email is admin@company.com", role="assistant"
                    ),
                )
            ],
            created=1234567890,
            model="gpt-3.5-turbo",
            object="chat.completion",
            system_fingerprint=None,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        # Mock simplified Noma API response for LLM response check
        noma_response = {
            "originalResponse": {
                "response": {
                    "anonymizedContent": {
                        "anonymized": "My email is *******"
                    },
                    "dataDetector": {
                        "dataType1": {"result": True},
                    },
                    "contentDetector": {"result": False},
                },
            },
            "verdict": False,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = noma_response
        mock_response.raise_for_status = MagicMock()

        # Update guardrail to use post_call event hook
        anonymize_guardrail.event_hook = "post_call"

        with patch.object(
            anonymize_guardrail.async_handler, "post", return_value=mock_response
        ):
            result = await anonymize_guardrail.async_post_call_success_hook(
                data=request_data,
                user_api_key_dict=mock_user_api_key_dict,
                response=llm_response,
            )

            # Should return modified response with anonymized content
            assert result == llm_response
            assert result.choices[0].message.content == "My email is *******"

    @pytest.mark.asyncio
    async def test_no_anonymization_when_disabled(
        self, mock_user_api_key_dict
    ):
        """Test that no anonymization occurs when anonymize_input=False"""
        guardrail = NomaGuardrail(
            api_key="test-api-key",
            anonymize_input=False,  # Disabled
            monitor_mode=False,
            block_failures=True,
        )

        request_data = {
            "messages": [
                {"role": "user", "content": "My email is test@example.com"},
            ],
        }

        noma_response = {
            "originalResponse": {
                "prompt": {
                    "anonymizedContent": {
                        "anonymized": "My email is *******"
                    },
                    "dataDetector": {
                        "dataType1": {"result": True},
                    },
                    "contentDetector": {"result": False},
                },
            },
            "verdict": False,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = noma_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            guardrail.async_handler, "post", return_value=mock_response
        ):
            # Should raise NomaBlockedMessage because anonymization is disabled
            with pytest.raises(NomaBlockedMessage):
                await guardrail.async_pre_call_hook(
                    user_api_key_dict=mock_user_api_key_dict,
                    cache=MagicMock(),
                    data=request_data,
                    call_type="completion",
                )

    @pytest.mark.asyncio
    async def test_no_anonymization_in_monitor_mode(
        self, mock_user_api_key_dict
    ):
        """Test that no anonymization occurs in monitor mode"""
        guardrail = NomaGuardrail(
            api_key="test-api-key",
            anonymize_input=True,
            monitor_mode=True,  # Monitor mode
            block_failures=True,
        )

        request_data = {
            "messages": [
                {"role": "user", "content": "My email is test@example.com"},
            ],
        }

        with patch.object(
            guardrail, "_create_background_noma_check"
        ) as mock_create_background:
            result = await guardrail.async_pre_call_hook(
                user_api_key_dict=mock_user_api_key_dict,
                cache=MagicMock(),
                data=request_data,
                call_type="completion",
            )

            # Should return original data unchanged
            assert result == request_data
            assert request_data["messages"][0]["content"] == "My email is test@example.com"
            mock_create_background.assert_called_once()

    @pytest.mark.asyncio
    async def test_anonymization_no_anonymized_content_available(
        self, anonymize_guardrail, mock_user_api_key_dict
    ):
        """Test behavior when anonymized content is not available"""
        request_data = {
            "messages": [
                {"role": "user", "content": "My email is test@example.com"},
            ],
        }

        noma_response = {
            "originalResponse": {
                "prompt": {
                    "dataDetector": {
                        "dataType1": {"result": True},
                    },
                    "contentDetector": {"result": False},
                },
            },
            "verdict": False,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = noma_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            anonymize_guardrail.async_handler, "post", return_value=mock_response
        ):
            # Should raise NomaBlockedMessage because no anonymized content available
            with pytest.raises(NomaBlockedMessage):
                await anonymize_guardrail.async_pre_call_hook(
                    user_api_key_dict=mock_user_api_key_dict,
                    cache=MagicMock(),
                    data=request_data,
                    call_type="completion",
                )

    @pytest.mark.asyncio
    async def test_anonymization_llm_response_no_anonymized_content_available(
        self, anonymize_guardrail, mock_user_api_key_dict
    ):
        """Test behavior when LLM response has no anonymized content available"""
        request_data = {
            "messages": [{"role": "user", "content": "What's your email?"}],
            "litellm_call_id": "test-call-id",
        }

        # Create LLM response with test data
        llm_response = ModelResponse(
            id="test-response-id",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="My email is admin@company.com", role="assistant"
                    ),
                )
            ],
            created=1234567890,
            model="gpt-3.5-turbo",
            object="chat.completion",
            system_fingerprint=None,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        # Mock Noma API response with no anonymized content available
        noma_response = {
            "originalResponse": {
                "response": {
                    "dataDetector": {
                        "dataType1": {"result": True},
                    },
                    "contentDetector": {"result": False},
                },
            },
            "verdict": False,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = noma_response
        mock_response.raise_for_status = MagicMock()

        # Update guardrail to use post_call event hook
        anonymize_guardrail.event_hook = "post_call"

        with patch.object(
            anonymize_guardrail.async_handler, "post", return_value=mock_response
        ):
            # Should raise NomaBlockedMessage because no anonymized content available for LLM response
            with pytest.raises(NomaBlockedMessage):
                await anonymize_guardrail.async_post_call_success_hook(
                    data=request_data,
                    user_api_key_dict=mock_user_api_key_dict,
                    response=llm_response,
                )

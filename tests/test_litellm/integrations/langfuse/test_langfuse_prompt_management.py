import os
from datetime import datetime
from unittest.mock import patch, MagicMock

from litellm.integrations.langfuse.langfuse_prompt_management import (
    LangfusePromptManagement,
)
import litellm
from litellm.litellm_core_utils.litellm_logging import Logging


class TestLangfusePromptManagement:
    def test_get_prompt_from_id(self):
        langfuse_prompt_management = LangfusePromptManagement()
        with patch.object(
            langfuse_prompt_management, "should_run_prompt_management"
        ) as mock_should_run_prompt_management, patch.object(
            langfuse_prompt_management, "_get_prompt_from_id"
        ) as mock_get_prompt_from_id:
            mock_should_run_prompt_management.return_value = True
            langfuse_prompt_management.get_chat_completion_prompt(
                model="langfuse/langfuse-model",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                non_default_params={},
                prompt_id="test-chat-prompt",
                prompt_variables={},
                dynamic_callback_params={},
                prompt_version=4,
            )

            mock_get_prompt_from_id.assert_called_once()
            assert mock_get_prompt_from_id.call_args.kwargs["prompt_version"] == 4

    def test_log_failure_event_runs_async_logger(self):
        langfuse_prompt_management = LangfusePromptManagement()
        with patch(
            "litellm.integrations.langfuse.langfuse_prompt_management.run_async_function"
        ) as mock_run_async:
            kwargs = {"standard_callback_dynamic_params": {}}
            start_time, end_time = 1, 2

            langfuse_prompt_management.log_failure_event(
                kwargs=kwargs,
                response_obj=None,
                start_time=start_time,
                end_time=end_time,
            )

            mock_run_async.assert_called_once()
            assert (
                mock_run_async.call_args[0][0]
                == langfuse_prompt_management.async_log_failure_event
            )

    def test_langfuse_otel_model_does_not_match_langfuse_callback(self):
        """Test that langfuse_otel/gemini-2.5-pro matches langfuse_otel, not langfuse."""
        logging_obj = Logging(
            model="test-model",
            messages=[],
            stream=False,
            call_type="completion",
            start_time=datetime.now(),
            litellm_call_id="test-call-id",
            function_id="test-function-id",
        )
        
        with patch('litellm.litellm_core_utils.litellm_logging._init_custom_logger_compatible_class') as mock_init:
            # Return LangfuseOtelLogger for langfuse_otel
            from litellm.integrations.langfuse.langfuse_otel import LangfuseOtelLogger
            mock_otel_logger = MagicMock(spec=LangfuseOtelLogger)
            
            def init_side_effect(logging_integration, **kwargs):
                if logging_integration == "langfuse_otel":
                    return mock_otel_logger
                elif logging_integration == "langfuse":
                    return MagicMock(spec=LangfusePromptManagement)
                return None
            
            mock_init.side_effect = init_side_effect
            
            # Test langfuse_otel model - should match langfuse_otel, not langfuse
            result = logging_obj.get_custom_logger_for_prompt_management(
                model="langfuse_otel/gemini-2.5-pro",
                non_default_params={},
            )
            
            # Verify it was called with langfuse_otel, not langfuse
            assert mock_init.called, "Mock should have been called"
            # Extract logging_integration from keyword arguments
            called_with = [call.kwargs.get('logging_integration') for call in mock_init.call_args_list]
            assert "langfuse_otel" in called_with, f"Expected langfuse_otel in calls, got {called_with}"
            # Verify langfuse_otel was called first (not langfuse)
            assert called_with[0] == "langfuse_otel", f"Expected first call to be langfuse_otel, got {called_with[0]}"

    def test_langfuse_model_matches_langfuse_callback(self):
        """Test that langfuse/gemini-2.5-pro correctly matches langfuse callback."""
        logging_obj = Logging(
            model="test-model",
            messages=[],
            stream=False,
            call_type="completion",
            start_time=datetime.now(),
            litellm_call_id="test-call-id",
            function_id="test-function-id",
        )
        
        with patch('litellm.litellm_core_utils.litellm_logging._init_custom_logger_compatible_class') as mock_init:
            mock_langfuse_logger = MagicMock(spec=LangfusePromptManagement)
            mock_init.return_value = mock_langfuse_logger
            
            result = logging_obj.get_custom_logger_for_prompt_management(
                model="langfuse/gemini-2.5-pro",
                non_default_params={},
            )
            
            # Verify it was called with langfuse
            mock_init.assert_called()
            call_kwargs = mock_init.call_args.kwargs
            assert call_kwargs.get('logging_integration') == "langfuse"

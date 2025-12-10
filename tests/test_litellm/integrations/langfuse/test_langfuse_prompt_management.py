import os
from unittest.mock import patch

from litellm.integrations.langfuse.langfuse_prompt_management import (
    LangfusePromptManagement,
)


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

    def test_trace_id_propagation_flag_from_env(self):
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_SECRET_KEY": "secret",
                "LANGFUSE_PUBLIC_KEY": "public",
                "LANGFUSE_PROPAGATE_TRACE_ID": "True",
            },
            clear=True,
        ):
            pm = LangfusePromptManagement()
            assert pm.langfuse_propagate_trace_id is True

        with patch.dict(
            os.environ,
            {
                "LANGFUSE_SECRET_KEY": "secret",
                "LANGFUSE_PUBLIC_KEY": "public",
                "LANGFUSE_PROPAGATE_TRACE_ID": "False",
            },
            clear=True,
        ):
            pm2 = LangfusePromptManagement()
            assert pm2.langfuse_propagate_trace_id is False

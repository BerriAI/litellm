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

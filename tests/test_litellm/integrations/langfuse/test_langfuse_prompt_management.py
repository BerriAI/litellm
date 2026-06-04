import os
from unittest.mock import MagicMock, patch

from litellm.integrations.langfuse.langfuse_prompt_management import (
    LangfusePromptManagement,
    langfuse_client_init,
)


class TestLangfusePromptManagement:
    def setup_method(self):
        # Mock langfuse package to avoid triggering real import.
        # The real langfuse import fails on Python 3.14 due to pydantic v1 incompatibility.
        # This also prevents test-ordering issues when earlier tests remove sys.modules["langfuse"].
        self._mock_langfuse = MagicMock()
        self._mock_langfuse.version.__version__ = "3.0.0"
        self._langfuse_patcher = patch.dict(
            "sys.modules", {"langfuse": self._mock_langfuse}
        )
        self._langfuse_patcher.start()

    def teardown_method(self):
        self._langfuse_patcher.stop()

    def test_get_prompt_from_id(self):
        langfuse_prompt_management = LangfusePromptManagement()
        with (
            patch.object(
                langfuse_prompt_management, "should_run_prompt_management"
            ) as mock_should_run_prompt_management,
            patch.object(
                langfuse_prompt_management, "_get_prompt_from_id"
            ) as mock_get_prompt_from_id,
        ):
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

    def test_langfuse_client_init_passes_httpx_client(self):
        mock_langfuse_class = MagicMock()
        with (
            patch(
                "litellm.integrations.langfuse.langfuse_prompt_management.resolve_langfuse_credentials",
                return_value=("pk-1234", "sk-1234", "https://localhost"),
            ),
            patch(
                "litellm.integrations.langfuse.langfuse_prompt_management.LangFuseLogger._get_langfuse_flush_interval",
                return_value=1,
            ),
            patch.dict("sys.modules", {"langfuse": self._mock_langfuse}),
            patch(
                "litellm.llms.custom_httpx.http_handler._get_httpx_client"
            ) as mock_get_httpx,
        ):
            mock_http_handler = MagicMock()
            mock_http_handler.client = MagicMock()
            mock_get_httpx.return_value = mock_http_handler

            self._mock_langfuse.Langfuse = mock_langfuse_class

            langfuse_client_init(
                langfuse_public_key="pk-1234",
                langfuse_secret="sk-1234",
                langfuse_host="https://localhost",
            )

            mock_langfuse_class.assert_called_once()
            call_kwargs = mock_langfuse_class.call_args[1]
            assert "httpx_client" in call_kwargs
            assert call_kwargs["httpx_client"] is mock_http_handler.client

        langfuse_client_init.cache_clear()

from unittest.mock import MagicMock

from litellm.integrations.langfuse.langfuse_handler import LangFuseHandler


def test_missing_dynamic_params_returns_global_logger() -> None:
    global_logger = MagicMock()
    dynamic_logger_cache = MagicMock()

    logger = LangFuseHandler.get_langfuse_logger_for_request(
        standard_callback_dynamic_params=None,
        in_memory_dynamic_logger_cache=dynamic_logger_cache,
        globalLangfuseLogger=global_logger,
    )

    assert logger is global_logger
    dynamic_logger_cache.get_cache.assert_not_called()


def test_missing_dynamic_params_are_not_dynamic_credentials() -> None:
    assert (
        LangFuseHandler._dynamic_langfuse_credentials_are_passed(
            standard_callback_dynamic_params=None
        )
        is False
    )

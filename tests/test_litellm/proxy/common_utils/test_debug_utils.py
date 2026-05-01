import json
import logging

import pytest

from litellm._logging import (
    verbose_logger,
    verbose_proxy_logger,
    verbose_router_logger,
)
from litellm.proxy.common_utils.debug_utils import init_verbose_loggers


VERBOSE_LOGGERS = (
    verbose_logger,
    verbose_router_logger,
    verbose_proxy_logger,
)


@pytest.mark.parametrize(
    ("litellm_log", "expected_level"),
    (("INFO", logging.INFO), ("DEBUG", logging.DEBUG)),
)
def test_litellm_log_env_sets_all_verbose_loggers(
    monkeypatch: pytest.MonkeyPatch, litellm_log: str, expected_level: int
):
    original_levels = {logger: logger.level for logger in VERBOSE_LOGGERS}
    try:
        for logger in VERBOSE_LOGGERS:
            logger.setLevel(logging.WARNING)

        monkeypatch.setenv(
            "WORKER_CONFIG",
            json.dumps({"debug": False, "detailed_debug": False}),
        )
        monkeypatch.setenv("LITELLM_LOG", litellm_log)

        init_verbose_loggers()

        assert [logger.level for logger in VERBOSE_LOGGERS] == [expected_level] * len(
            VERBOSE_LOGGERS
        )
    finally:
        for logger, level in original_levels.items():
            logger.setLevel(level)


@pytest.mark.asyncio
async def test_initialize_litellm_log_info_sets_all_verbose_loggers(
    monkeypatch: pytest.MonkeyPatch,
):
    from litellm.proxy import proxy_server
    from litellm.proxy.common_utils import banner

    original_levels = {logger: logger.level for logger in VERBOSE_LOGGERS}
    try:
        for logger in VERBOSE_LOGGERS:
            logger.setLevel(logging.WARNING)

        monkeypatch.setenv("LITELLM_LOG", "INFO")
        monkeypatch.setenv("LITELLM_DONT_SHOW_FEEDBACK_BOX", "true")
        monkeypatch.setattr(banner, "show_banner", lambda: None)

        await proxy_server.initialize(debug=False, detailed_debug=False)

        assert [logger.level for logger in VERBOSE_LOGGERS] == [logging.INFO] * len(
            VERBOSE_LOGGERS
        )
    finally:
        for logger, level in original_levels.items():
            logger.setLevel(level)

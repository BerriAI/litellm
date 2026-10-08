import logging
from typing import Final

import pytest

import litellm
from litellm.proxy.hooks.batch_redis_get import PROXY_BatchRedisRequests


def test_print_verbose_logs_at_debug_instead_of_printing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    batch_redis: Final = PROXY_BatchRedisRequests()
    monkeypatch.setattr(litellm, "set_verbose", True)
    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    capsys.readouterr()
    batch_redis.print_verbose("batch redis verbose statement")
    captured: Final = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    caplog.set_level(logging.DEBUG, logger="LiteLLM Proxy")
    batch_redis.print_verbose("batch redis verbose statement")
    assert any(
        record.levelno == logging.DEBUG
        and record.name == "LiteLLM Proxy"
        and "batch redis verbose statement" in record.getMessage()
        for record in caplog.records
    )

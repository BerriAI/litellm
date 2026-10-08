import logging
from typing import Final

import pytest
from enterprise.enterprise_hooks.blocked_user_list import ENTERPRISE_BlockedUserList

import litellm


def test_print_verbose_logs_at_debug_instead_of_printing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    blocked_user_list: Final = ENTERPRISE_BlockedUserList(prisma_client=None)
    monkeypatch.setattr(litellm, "set_verbose", True)
    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    capsys.readouterr()
    blocked_user_list.print_verbose("blocked user list verbose statement")
    captured: Final = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    caplog.set_level(logging.DEBUG, logger="LiteLLM Proxy")
    blocked_user_list.print_verbose("blocked user list verbose statement")
    assert any(
        record.levelno == logging.DEBUG
        and record.name == "LiteLLM Proxy"
        and "blocked user list verbose statement" in record.getMessage()
        for record in caplog.records
    )

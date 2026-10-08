import logging
from typing import Final

import pytest
from enterprise.enterprise_hooks.google_text_moderation import ENTERPRISE_GoogleTextModeration

import litellm


def test_print_verbose_logs_at_debug_instead_of_printing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    moderation: Final = ENTERPRISE_GoogleTextModeration.__new__(ENTERPRISE_GoogleTextModeration)
    monkeypatch.setattr(litellm, "set_verbose", True)
    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    capsys.readouterr()
    moderation.print_verbose("google text moderation verbose statement")
    captured: Final = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    caplog.set_level(logging.DEBUG, logger="LiteLLM Proxy")
    moderation.print_verbose("google text moderation verbose statement")
    assert any(
        record.levelno == logging.DEBUG
        and record.name == "LiteLLM Proxy"
        and "google text moderation verbose statement" in record.getMessage()
        for record in caplog.records
    )

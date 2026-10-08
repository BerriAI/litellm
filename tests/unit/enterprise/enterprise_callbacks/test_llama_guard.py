import logging
from typing import Final

import pytest
from litellm_enterprise.enterprise_callbacks.llama_guard import _ENTERPRISE_LlamaGuard

import litellm


def test_print_verbose_logs_at_debug_instead_of_printing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    llama_guard: Final = _ENTERPRISE_LlamaGuard(model_name="llamaguard-7b")
    monkeypatch.setattr(litellm, "set_verbose", True)
    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    capsys.readouterr()
    llama_guard.print_verbose("llama guard verbose statement")
    captured: Final = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    caplog.set_level(logging.DEBUG, logger="LiteLLM Proxy")
    llama_guard.print_verbose("llama guard verbose statement")
    assert any(
        record.levelno == logging.DEBUG
        and record.name == "LiteLLM Proxy"
        and "llama guard verbose statement" in record.getMessage()
        for record in caplog.records
    )

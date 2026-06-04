"""print_verbose() must not leak secrets to stdout when callers pass dicts like
model_call_details that contain api_key fields."""

import logging

import litellm
from litellm.utils import print_verbose as utils_print_verbose

SECRET_VALUE = "sk-ant-test-DO-NOT-LEAK-12345"


def _payload():
    return {
        "model": "claude-3-5-sonnet-20241022",
        "api_key": SECRET_VALUE,
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_utils_print_verbose_redacts_stdout_when_set_verbose(capsys, monkeypatch):
    monkeypatch.setattr(litellm, "set_verbose", True)
    utils_print_verbose(_payload())
    out = capsys.readouterr().out
    assert SECRET_VALUE not in out, f"api_key leaked to stdout: {out!r}"


def test_utils_print_verbose_routes_through_logger_redacted(caplog, monkeypatch):
    monkeypatch.setattr(litellm, "set_verbose", False)
    caplog.set_level(logging.DEBUG, logger="LiteLLM")
    utils_print_verbose(_payload())
    leaked = [r.getMessage() for r in caplog.records if SECRET_VALUE in r.getMessage()]
    assert not leaked, f"api_key leaked through verbose_logger: {leaked!r}"

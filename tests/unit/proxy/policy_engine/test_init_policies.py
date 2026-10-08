import logging

import pytest

from litellm.proxy.policy_engine.init_policies import _print_policies_on_startup

_POLICIES = {
    "pii-policy": {
        "description": "Blocks PII before it reaches the model",
        "guardrails": {"add": ["pii-guard"], "remove": ["lenient-guard"]},
        "condition": {"model": "gpt-*"},
    }
}
_ATTACHMENTS = [
    {"policy": "pii-policy", "teams": ["team-alpha"]},
    {"policy": "pii-policy", "scope": "*"},
]


def _records(caplog: pytest.LogCaptureFixture, level: int) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == "LiteLLM Proxy" and record.levelno == level]


def test_startup_report_is_one_info_record_not_stdout(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    caplog.set_level(logging.WARNING, logger="LiteLLM Proxy")
    _print_policies_on_startup(_POLICIES, _ATTACHMENTS)
    assert capsys.readouterr() == ("", "")

    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    _print_policies_on_startup(_POLICIES, _ATTACHMENTS)
    [record] = _records(caplog, logging.INFO)
    message = record.getMessage()
    assert "LiteLLM Policy Engine: Loaded 1 policies" in message
    assert "  - pii-policy\n" in message
    assert "description: Blocks PII before it reaches the model" in message
    assert "guardrails.add: ['pii-guard']" in message
    assert "guardrails.remove: ['lenient-guard']" in message
    assert "condition.model: gpt-*" in message
    assert "Policy Attachments: 2 attachment(s)" in message
    assert "  - pii-policy -> teams=['team-alpha']" in message
    assert "  - pii-policy -> scope=* (global)" in message
    assert _records(caplog, logging.WARNING) == []


def test_missing_attachments_is_its_own_warning_record_not_stdout(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    caplog.set_level(logging.ERROR, logger="LiteLLM Proxy")
    _print_policies_on_startup(_POLICIES, None)
    assert capsys.readouterr() == ("", "")

    caplog.set_level(logging.WARNING, logger="LiteLLM Proxy")
    _print_policies_on_startup(_POLICIES, None)
    [record] = _records(caplog, logging.WARNING)
    assert "No policy_attachments configured" in record.getMessage()
    assert "Policy Attachments" not in record.getMessage()

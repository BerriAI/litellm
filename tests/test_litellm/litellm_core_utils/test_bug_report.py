from __future__ import annotations

from typing import cast
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from litellm._version import version
from litellm.exceptions import APIConnectionError, BadRequestError, InternalServerError
from litellm.litellm_core_utils.bug_report import (
    DISABLE_ENV_VAR,
    ISSUE_URL_BASE,
    MAX_URL_LENGTH,
    bug_report_enabled,
    bug_report_issue_url,
    bug_report_notice,
    build_bug_report,
    should_report_bug,
    strip_bug_report_notice,
)
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


def test_build_bug_report_keeps_only_redacted_litellm_frames():
    with pytest.raises(BadRequestError) as raised:
        get_llm_provider(cast(str, None))
    report = build_bug_report(raised.value, surface="sdk")

    assert report.litellm_frames
    assert all(frame.startswith("litellm/") for frame in report.litellm_frames)
    assert all("test_bug_report.py" not in frame for frame in report.litellm_frames)


def test_issue_url_redacts_message_and_prefills_sdk_fields():
    report = build_bug_report(
        RuntimeError("failed with key sk-abcdefghijklmnopqrstuvwxyz1234567890"),
        surface="sdk",
    )
    query = parse_qs(urlparse(bug_report_issue_url(report)).query)

    assert ISSUE_URL_BASE in bug_report_issue_url(report)
    assert "sk-abcdef" not in str(query)
    assert query["title"][0].startswith("[Bug]: RuntimeError:")
    assert query["version"] == [version]
    assert query["template"] == ["bug_report.yml"]
    assert query["domain"] == ["Python SDK: the litellm package itself"]
    assert query["deployment"] == ["pip / Python SDK"]
    assert "RuntimeError" in query["description"][0]
    assert "Python:" in query["description"][0]


def test_issue_url_is_bounded_for_long_messages():
    report = build_bug_report(RuntimeError("x" * 20_000), surface="proxy")
    query = parse_qs(urlparse(bug_report_issue_url(report)).query)

    assert len(bug_report_issue_url(report)) <= MAX_URL_LENGTH
    assert "RuntimeError" in query["description"][0]


def test_issue_url_is_bounded_for_long_non_ascii_context():
    report = build_bug_report(
        ValueError("故障" * 10_000),
        surface="sdk",
        model="模型" * 300,
        custom_llm_provider="供給" * 300,
        call_type="呼出" * 300,
    )

    assert len(bug_report_issue_url(report)) <= MAX_URL_LENGTH


def test_issue_url_builds_without_a_traceback():
    exc = RuntimeError("no traceback")
    assert exc.__traceback__ is None
    report = build_bug_report(exc, surface="proxy")

    assert report.litellm_frames == ()
    assert bug_report_issue_url(report).startswith(ISSUE_URL_BASE)


def test_bug_report_can_be_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(DISABLE_ENV_VAR, "true")

    assert bug_report_enabled() is False
    assert should_report_bug(RuntimeError("boom")) is False


@pytest.mark.parametrize(
    "exc",
    [
        InternalServerError(message="upstream 500", llm_provider="openai", model="gpt-4"),
        APIConnectionError(
            message="connection reset",
            llm_provider="openai",
            model="gpt-4",
            request=httpx.Request(method="POST", url="https://api.openai.com/v1/"),
        ),
        BadRequestError(message="bad input", llm_provider="openai", model="gpt-4"),
        "not an exception",
    ],
)
def test_should_report_bug_skips_provider_and_network_errors(exc: object):
    assert should_report_bug(exc) is False


def test_should_report_bug_accepts_plain_python_errors():
    assert should_report_bug(KeyError("missing")) is True


def test_proxy_provider_uses_translation_domain():
    report = build_bug_report(
        RuntimeError("proxy failure"),
        surface="proxy",
        call_type="/v1/chat/completions",
        model="gpt-4",
        custom_llm_provider="openai",
    )
    query = parse_qs(urlparse(bug_report_issue_url(report)).query)

    assert query["domain"] == ["LLM translation: a specific provider's request or response"]


def test_strip_bug_report_notice():
    report = build_bug_report(RuntimeError("boom"), surface="sdk")
    notice = bug_report_notice(report)

    assert strip_bug_report_notice(f"boom\n\n{notice}") == "boom\n"
    assert strip_bug_report_notice("boom") == "boom"

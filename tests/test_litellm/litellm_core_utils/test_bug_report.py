from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

import litellm
from litellm._version import version
from litellm.exceptions import APIConnectionError, BadRequestError, InternalServerError
from litellm.litellm_core_utils.bug_report import (
    DISABLE_ENV_VAR,
    ISSUE_URL_BASE,
    MAX_FRAMES,
    MAX_URL_LENGTH,
    allowlisted,
    bug_report_enabled,
    bug_report_issue_url,
    bug_report_notice,
    build_bug_report,
    should_report_bug,
    strip_bug_report_notice,
)
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


def test_build_bug_report_keeps_only_litellm_frames():
    with pytest.raises(BadRequestError) as raised:
        get_llm_provider(cast(str, None))
    report = build_bug_report(raised.value, surface="sdk")

    assert report.litellm_frames
    assert all(frame.startswith("litellm/") for frame in report.litellm_frames)
    assert all("test_bug_report.py" not in frame for frame in report.litellm_frames)


def test_issue_url_never_contains_the_exception_message():
    secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    prompt = "my social security number is 123-45-6789"
    report = build_bug_report(RuntimeError(f"{secret} {prompt}"), surface="sdk")
    url = bug_report_issue_url(report)
    query = parse_qs(urlparse(url).query)

    assert url.startswith(ISSUE_URL_BASE)
    assert secret not in url and "123-45-6789" not in url and "social" not in url
    assert query["title"] == ["[Bug]: RuntimeError in litellm"]
    assert query["version"] == [version]
    assert query["template"] == ["bug_report.yml"]
    assert query["domain"] == ["Python SDK: the litellm package itself"]
    assert query["deployment"] == ["pip / Python SDK"]
    assert "Exception: `RuntimeError`" in query["description"][0]
    assert "Python:" in query["description"][0]


def test_issue_url_drops_unknown_provider_and_call_type():
    report = build_bug_report(
        ValueError("boom"),
        surface="proxy",
        custom_llm_provider="acme-internal-gateway",
    )
    query = parse_qs(urlparse(bug_report_issue_url(report)).query)

    assert report.custom_llm_provider is None
    assert "acme" not in bug_report_issue_url(report)
    assert "Provider: unknown" in query["description"][0]
    assert "Endpoint / call: unknown" in query["description"][0]


def test_allowlisted_only_passes_exact_members():
    allowed = frozenset({"/v1/chat/completions"})

    assert allowlisted("/v1/chat/completions", allowed) == "/v1/chat/completions"
    assert allowlisted("/v1/chat/completions/../../admin", allowed) is None
    assert allowlisted(None, allowed) is None
    assert allowlisted(["/v1/chat/completions"], allowed) is None
    assert allowlisted({"provider": "openai"}, allowed) is None


def test_build_bug_report_survives_unhashable_provider_from_request_data():
    report = build_bug_report(
        KeyError("missing"),
        surface="proxy",
        custom_llm_provider={"name": "openai"},
    )

    assert report.custom_llm_provider is None


def test_frames_are_capped_and_url_is_bounded():
    namespace: dict[str, object] = {}
    exec(
        compile(
            "def recurse(depth):\n    if depth == 0:\n        raise RuntimeError('deep')\n    recurse(depth - 1)\n",
            str(Path(litellm.__file__).with_name("fake_deep_module.py")),
            "exec",
        ),
        namespace,
    )
    recurse = cast(Callable[[int], None], namespace["recurse"])

    with pytest.raises(RuntimeError) as raised:
        recurse(200)
    report = build_bug_report(raised.value, surface="proxy")

    assert len(report.litellm_frames) == MAX_FRAMES
    assert all(frame.startswith("litellm/fake_deep_module.py:") for frame in report.litellm_frames)
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


def test_proxy_known_provider_uses_translation_domain():
    report = build_bug_report(
        RuntimeError("proxy failure"),
        surface="proxy",
        call_type="/v1/chat/completions",
        custom_llm_provider="openai",
    )
    query = parse_qs(urlparse(bug_report_issue_url(report)).query)

    assert report.custom_llm_provider == "openai"
    assert query["domain"] == ["LLM translation: a specific provider's request or response"]
    assert "Endpoint / call: /v1/chat/completions" in query["description"][0]


def test_strip_bug_report_notice():
    report = build_bug_report(RuntimeError("boom"), surface="sdk")
    notice = bug_report_notice(report)

    assert strip_bug_report_notice(f"boom\n\n{notice}") == "boom\n"
    assert strip_bug_report_notice("boom") == "boom"

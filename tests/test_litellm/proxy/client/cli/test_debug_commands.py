import json
import os
import time
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from litellm.proxy.client.cli import cli
from litellm.proxy.client.cli.commands import debug as debug_module
from litellm.proxy.client.cli.commands.debug import (
    SLASH_COMMAND_NAME,
    detect_claude_session_id,
    install_slash_command,
)

SESSION = "e96634a3-fa28-4083-b354-55542e2dca01"

OK_ROW = {
    "request_id": "req-ok",
    "startTime": "2026-09-02T10:00:00",
    "endTime": "2026-09-02T10:00:02",
    "model": "claude-opus-4-1",
    "custom_llm_provider": "anthropic",
    "status": "success",
    "spend": 0.0125,
    "prompt_tokens": 100,
    "completion_tokens": 20,
    "metadata": {"status": "success"},
}
FAILED_ROW = {
    "request_id": "req-failed",
    "startTime": "2026-09-02T10:01:00",
    "endTime": "2026-09-02T10:01:01",
    "model": "claude-opus-4-1",
    "custom_llm_provider": "anthropic",
    "status": "failure",
    "spend": 0.0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    # query_raw hands metadata back as a JSON string on some paths
    "metadata": json.dumps(
        {
            "status": "failure",
            "error_information": {
                "error_code": "400",
                "error_class": "BadRequestError",
                "error_message": "`prompt` is required when `stop` is not true.",
            },
        }
    ),
}


def _fake_http(rows, payloads):
    calls = []

    class FakeHTTP:
        def __init__(self, *_args, **_kwargs):
            pass

        def request(self, method, uri, **kwargs):
            calls.append(uri)
            if uri == "/spend/logs/session/ui":
                assert kwargs["params"]["session_id"] == SESSION
                return {"data": rows, "total": len(rows), "page": 1, "page_size": 100, "total_pages": 1}
            request_id = uri.rsplit("/", 1)[1]
            return payloads.get(request_id)

    return FakeHTTP, calls


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("LITELLM_PROXY_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "sk-test")
    monkeypatch.setattr(debug_module, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(debug_module, "CLAUDE_DIR", tmp_path / "claude")


def test_report_includes_spend_error_and_bodies_for_failed_turn(tmp_path):
    payloads = {
        "req-failed": {
            "proxy_server_request": {"body": {"model": "claude-opus-4-1", "messages": [{"role": "user"}]}},
            "response": {"error": {"message": "`prompt` is required"}},
        },
        "req-ok": {"proxy_server_request": {"body": {"model": "claude-opus-4-1"}}, "response": {"id": "msg_1"}},
    }
    FakeHTTP, calls = _fake_http([FAILED_ROW, OK_ROW], payloads)
    with patch.object(debug_module, "HTTPClient", FakeHTTP):
        result = CliRunner().invoke(cli, ["debug", "claude", "--session-id", SESSION, "--recent-bodies", "0"])

    assert result.exit_code == 0, result.output
    assert "turns: 2, failed: 1" in result.output
    assert "total spend: $0.012500" in result.output
    assert "### 1. ok claude-opus-4-1" in result.output
    assert "### 2. FAILED claude-opus-4-1" in result.output
    assert "`400` BadRequestError" in result.output
    assert "`prompt` is required when `stop` is not true." in result.output
    assert '"messages"' in result.output
    assert "msg_1" not in result.output
    assert calls == ["/spend/logs/session/ui", "/spend/logs/ui/req-failed"]
    saved = tmp_path / "reports" / f"claude-{SESSION}.md"
    assert result.stdout.startswith(saved.read_text())
    assert "### 2. FAILED" in saved.read_text()


def test_recent_bodies_fetches_latest_turns_even_when_successful():
    payloads = {"req-ok": {"proxy_server_request": {"body": {"x": 1}}, "response": {"id": "msg_1"}}}
    FakeHTTP, calls = _fake_http([OK_ROW], payloads)
    with patch.object(debug_module, "HTTPClient", FakeHTTP):
        result = CliRunner().invoke(cli, ["debug", "claude", "--session-id", SESSION, "--no-save"])

    assert result.exit_code == 0, result.output
    assert "msg_1" in result.output
    assert calls == ["/spend/logs/session/ui", "/spend/logs/ui/req-ok"]


def test_bodies_are_truncated_to_max_chars():
    payloads = {"req-ok": {"proxy_server_request": {"body": "a" * 5000}, "response": None}}
    FakeHTTP, _ = _fake_http([OK_ROW], payloads)
    with patch.object(debug_module, "HTTPClient", FakeHTTP):
        result = CliRunner().invoke(
            cli, ["debug", "claude", "--session-id", SESSION, "--no-save", "--max-body-chars", "200"]
        )

    assert result.exit_code == 0, result.output
    assert "truncated" in result.output
    assert "a" * 300 not in result.output


def test_no_rows_is_a_clear_error():
    FakeHTTP, _ = _fake_http([], {})
    with patch.object(debug_module, "HTTPClient", FakeHTTP):
        result = CliRunner().invoke(cli, ["debug", "claude", "--session-id", SESSION])

    assert result.exit_code != 0
    assert "No spend logs found for session" in result.output


def test_no_session_id_anywhere_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)
    result = CliRunner().invoke(cli, ["debug", "claude"])
    assert result.exit_code != 0
    assert "Could not find a Claude Code session" in result.output


def test_detect_session_id_prefers_env_then_newest_transcript(tmp_path):
    project = tmp_path / "projects" / "-Users-me-repo"
    project.mkdir(parents=True)
    old = project / "old-session.jsonl"
    new = project / "new-session.jsonl"
    old.write_text("{}")
    new.write_text("{}")
    now = time.time()
    os.utime(old, (now - 100, now - 100))
    os.utime(new, (now, now))

    assert detect_claude_session_id({}, tmp_path) == "new-session"
    assert detect_claude_session_id({"CLAUDE_SESSION_ID": "from-env"}, tmp_path) == "from-env"
    assert detect_claude_session_id({}, tmp_path / "missing") is None


def test_install_slash_command_writes_runnable_command_file(tmp_path):
    path = install_slash_command(tmp_path)
    assert path == tmp_path / "commands" / f"{SLASH_COMMAND_NAME}.md"
    body = path.read_text()
    assert body.startswith("---\n")
    assert "allowed-tools: Bash(lite debug claude:*)" in body
    assert "!`lite debug claude $ARGUMENTS`" in body

    result = CliRunner().invoke(cli, ["debug", "install-claude-command"])
    assert result.exit_code == 0, result.output
    assert "/debug-lite" in result.output

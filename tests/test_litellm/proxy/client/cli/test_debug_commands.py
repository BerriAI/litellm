import json
import os
import time

import pytest
import responses
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


PROXY = "http://localhost:4000"


def _mock_proxy(rows, payloads):
    responses.get(
        f"{PROXY}/spend/logs/session/ui",
        json={"data": rows, "total": len(rows), "page": 1, "page_size": 100, "total_pages": 1},
        match=[responses.matchers.query_param_matcher({"session_id": SESSION}, strict_match=False)],
    )
    for request_id, payload in payloads.items():
        responses.get(f"{PROXY}/spend/logs/ui/{request_id}", json=payload)


def _called_paths():
    return [c.request.path_url.split("?")[0] for c in responses.calls]


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("LITELLM_PROXY_URL", PROXY)
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "sk-test")
    monkeypatch.setattr(debug_module, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(debug_module, "CLAUDE_DIR", tmp_path / "claude")


@responses.activate
def test_report_includes_spend_error_and_bodies_for_failed_turn(tmp_path):
    payloads = {
        "req-failed": {
            "proxy_server_request": {"body": {"model": "claude-opus-4-1", "messages": [{"role": "user"}]}},
            "response": {"error": {"message": "`prompt` is required"}},
        },
        "req-ok": {"proxy_server_request": {"body": {"model": "claude-opus-4-1"}}, "response": {"id": "msg_1"}},
    }
    _mock_proxy([FAILED_ROW, OK_ROW], payloads)
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
    assert _called_paths() == ["/spend/logs/session/ui", "/spend/logs/ui/req-failed"]
    saved = tmp_path / "reports" / f"claude-{SESSION}.md"
    assert result.stdout.startswith(saved.read_text())
    assert "### 2. FAILED" in saved.read_text()


@responses.activate
def test_recent_bodies_fetches_latest_turns_even_when_successful():
    _mock_proxy([OK_ROW], {"req-ok": {"proxy_server_request": {"body": {"x": 1}}, "response": {"id": "msg_1"}}})
    result = CliRunner().invoke(cli, ["debug", "claude", "--session-id", SESSION, "--no-save"])

    assert result.exit_code == 0, result.output
    assert "msg_1" in result.output
    assert _called_paths() == ["/spend/logs/session/ui", "/spend/logs/ui/req-ok"]


@responses.activate
def test_bodies_are_truncated_to_max_chars():
    _mock_proxy([OK_ROW], {"req-ok": {"proxy_server_request": {"body": "a" * 5000}, "response": None}})
    result = CliRunner().invoke(
        cli, ["debug", "claude", "--session-id", SESSION, "--no-save", "--max-body-chars", "200"]
    )

    assert result.exit_code == 0, result.output
    assert "truncated" in result.output
    assert "a" * 300 not in result.output


@responses.activate
def test_no_rows_is_a_clear_error():
    _mock_proxy([], {})
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

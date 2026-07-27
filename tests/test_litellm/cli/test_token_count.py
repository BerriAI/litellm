"""Tests for the ``litellm token-count`` CLI subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from litellm.cli.token_count import TokenCount, cli, count_tokens


def test_count_tokens_with_text(monkeypatch):
    """--text passes the raw string to token_counter."""
    captured: dict = {}

    def fake_token_counter(**kw):
        captured.update(kw)
        return 7

    monkeypatch.setattr("litellm.token_counter", fake_token_counter)
    result = count_tokens(model="gpt-4o", text="hello world")
    assert isinstance(result, TokenCount)
    assert result.model == "gpt-4o"
    assert result.mode == "text"
    assert result.count == 7
    assert captured["model"] == "gpt-4o"
    assert captured["text"] == "hello world"


def test_count_tokens_with_file(monkeypatch, tmp_path: Path):
    """--file reads from disk and counts the file's tokens."""
    p = tmp_path / "prompt.txt"
    p.write_text("file contents here", encoding="utf-8")
    captured: dict = {}

    def fake_token_counter(**kw):
        captured.update(kw)
        return 11

    monkeypatch.setattr("litellm.token_counter", fake_token_counter)
    result = count_tokens(model="claude-sonnet-4-6", file=str(p))
    assert result.mode == "file"
    assert result.count == 11
    assert captured["text"] == "file contents here"


def test_count_tokens_with_messages(monkeypatch):
    """--messages parses the JSON list and forwards to token_counter."""
    captured: dict = {}

    def fake_token_counter(**kw):
        captured.update(kw)
        return 19

    monkeypatch.setattr("litellm.token_counter", fake_token_counter)
    messages = '[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]'
    result = count_tokens(model="gpt-4o", messages=messages)
    assert result.mode == "messages"
    assert result.count == 19
    assert captured["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_count_tokens_rejects_mutually_exclusive_input(monkeypatch):
    """Passing two input sources at once is a usage error."""
    monkeypatch.setattr("litellm.token_counter", lambda **kw: 0)
    with pytest.raises(ValueError) as exc:
        count_tokens(model="gpt-4o", text="hi", file="/tmp/x")
    assert "mutually exclusive" in str(exc.value).lower()


def test_count_tokens_requires_some_input():
    """No input source at all is a usage error."""
    with pytest.raises(ValueError) as exc:
        count_tokens(model="gpt-4o")
    assert "--text" in str(exc.value)


def test_count_tokens_requires_model():
    """Empty model is rejected at the helper level."""
    with pytest.raises(ValueError) as exc:
        count_tokens(model="", text="hi")
    assert "model is required" in str(exc.value)


def test_count_tokens_rejects_invalid_messages_json(monkeypatch):
    """Malformed JSON in --messages surfaces as ValueError."""
    monkeypatch.setattr("litellm.token_counter", lambda **kw: 0)
    with pytest.raises(ValueError) as exc:
        count_tokens(model="gpt-4o", messages="{not json")
    assert "json" in str(exc.value).lower()


def test_count_tokens_rejects_non_list_messages(monkeypatch):
    """A JSON object (not a list) for --messages is rejected."""
    monkeypatch.setattr("litellm.token_counter", lambda **kw: 0)
    with pytest.raises(ValueError) as exc:
        count_tokens(model="gpt-4o", messages='{"role": "user"}')
    assert "list" in str(exc.value).lower()


def test_count_tokens_missing_file(monkeypatch, tmp_path: Path):
    """A --file path that does not exist surfaces as ValueError."""
    missing = tmp_path / "does-not-exist.txt"
    monkeypatch.setattr("litellm.token_counter", lambda **kw: 0)
    with pytest.raises(ValueError) as exc:
        count_tokens(model="gpt-4o", file=str(missing))
    assert "does not exist" in str(exc.value)


def test_count_tokens_wraps_token_counter_failure(monkeypatch):
    """Underlying token_counter failures surface as RuntimeError."""

    def _raise(**kw):
        raise RuntimeError("simulated tokenizer failure")

    monkeypatch.setattr("litellm.token_counter", _raise)
    with pytest.raises(RuntimeError) as exc:
        count_tokens(model="gpt-4o", text="hi")
    assert "simulated tokenizer failure" in str(exc.value)


def test_to_jsonable_round_trip():
    result = TokenCount(model="gpt-4o", mode="text", count=42)
    assert json.loads(json.dumps(result.to_jsonable())) == {
        "model": "gpt-4o",
        "mode": "text",
        "count": 42,
    }


def test_cli_text_runs_and_prints_count(monkeypatch):
    monkeypatch.setattr("litellm.token_counter", lambda **kw: 5)
    runner = CliRunner()
    result = runner.invoke(cli, ["--model", "gpt-4o", "--text", "hello world"])
    assert result.exit_code == 0, result.output
    assert "count=5" in result.output
    assert "model=gpt-4o" in result.output
    assert "mode=text" in result.output


def test_cli_json_output_is_valid_json(monkeypatch):
    monkeypatch.setattr("litellm.token_counter", lambda **kw: 12)
    runner = CliRunner()
    result = runner.invoke(cli, ["--model", "gpt-4o", "--text", "hi", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    assert payload == {"model": "gpt-4o", "mode": "text", "count": 12}


def test_cli_file_reads_from_disk(monkeypatch, tmp_path: Path):
    p = tmp_path / "p.txt"
    p.write_text("disk contents", encoding="utf-8")
    monkeypatch.setattr("litellm.token_counter", lambda **kw: 9)
    runner = CliRunner()
    result = runner.invoke(cli, ["--model", "gpt-4o", "--file", str(p)])
    assert result.exit_code == 0, result.output
    assert "count=9" in result.output
    assert "mode=file" in result.output


def test_cli_messages_parses_json(monkeypatch):
    monkeypatch.setattr("litellm.token_counter", lambda **kw: 3)
    runner = CliRunner()
    messages = '[{"role":"user","content":"hi"}]'
    result = runner.invoke(cli, ["--model", "gpt-4o", "--messages", messages, "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    assert payload == {"model": "gpt-4o", "mode": "messages", "count": 3}


def test_cli_mutually_exclusive_flags_exit_2(monkeypatch):
    monkeypatch.setattr("litellm.token_counter", lambda **kw: 0)
    runner = CliRunner()
    result = runner.invoke(cli, ["--model", "gpt-4o", "--text", "hi", "--file", "/tmp/x"])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output.lower()


def test_cli_missing_input_exits_2(monkeypatch):
    monkeypatch.setattr("litellm.token_counter", lambda **kw: 0)
    runner = CliRunner()
    result = runner.invoke(cli, ["--model", "gpt-4o"])
    assert result.exit_code == 2
    assert "--text" in result.output


def test_cli_missing_model_exits_2(monkeypatch):
    monkeypatch.setattr("litellm.token_counter", lambda **kw: 0)
    runner = CliRunner()
    result = runner.invoke(cli, ["--text", "hi"])
    assert result.exit_code == 2


def test_cli_invalid_messages_json_exits_2(monkeypatch):
    monkeypatch.setattr("litellm.token_counter", lambda **kw: 0)
    runner = CliRunner()
    result = runner.invoke(cli, ["--model", "gpt-4o", "--messages", "{not json"])
    assert result.exit_code == 2
    assert "json" in result.output.lower()


def test_cli_non_list_messages_exits_2(monkeypatch):
    monkeypatch.setattr("litellm.token_counter", lambda **kw: 0)
    runner = CliRunner()
    result = runner.invoke(cli, ["--model", "gpt-4o", "--messages", '{"role":"user"}'])
    assert result.exit_code == 2
    assert "list" in result.output.lower()


def test_cli_missing_file_exits_2(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("litellm.token_counter", lambda **kw: 0)
    missing = tmp_path / "absent.txt"
    runner = CliRunner()
    result = runner.invoke(cli, ["--model", "gpt-4o", "--file", str(missing)])
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_cli_does_not_echo_input_text(monkeypatch):
    """Privacy property: the input text must never appear in stdout."""
    monkeypatch.setattr("litellm.token_counter", lambda **kw: 1)
    secret = "patient-zero-secret-marker-XYZ-9182"
    runner = CliRunner()
    result = runner.invoke(cli, ["--model", "gpt-4o", "--text", secret, "--json"])
    assert result.exit_code == 0, result.output
    assert secret not in result.output, (
        "token-count printed the input text to stdout; the CLI should only print the count"
    )
    payload = json.loads(result.output.strip())
    assert payload["count"] == 1
    assert payload["mode"] == "text"


def test_cli_does_not_echo_messages_payload(monkeypatch):
    """Privacy property: the messages JSON must never appear in stdout."""
    monkeypatch.setattr("litellm.token_counter", lambda **kw: 2)
    secret = "ssn-123-45-6789-dont-leak"
    messages = json.dumps([{"role": "user", "content": secret}])
    runner = CliRunner()
    result = runner.invoke(cli, ["--model", "gpt-4o", "--messages", messages, "--json"])
    assert result.exit_code == 0, result.output
    assert secret not in result.output
    assert messages not in result.output

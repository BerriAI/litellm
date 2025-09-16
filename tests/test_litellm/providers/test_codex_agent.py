import importlib
import stat
import types
from pathlib import Path
from unittest.mock import MagicMock, patch


def _fake_popen(stdout_lines):
    proc = MagicMock()
    # Minimal stdio stubs
    proc.stdin = types.SimpleNamespace(
        write=lambda *_: None,
        flush=lambda: None,
        close=lambda: None,
    )
    # Simulate line-by-line stdout
    lines = iter([l + "\n" for l in stdout_lines] + [""])
    proc.stdout = types.SimpleNamespace()
    proc.stdout.readline = lambda: next(lines, "")
    proc.stderr = types.SimpleNamespace()
    proc.wait = lambda timeout=None: 0
    proc.poll = lambda: 0
    proc.pid = 12345
    return proc


@patch("subprocess.Popen")
def test_codex_agent_completion_returns_joined_output(mock_popen, tmp_path, monkeypatch):
    # Enable provider and ensure registration runs during import
    monkeypatch.setenv("LITELLM_ENABLE_CODEX_AGENT", "1")

    # Provide a fake executable binary path that passes X_OK
    fake_bin = tmp_path / "codex"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("LITELLM_CODEX_BINARY_PATH", str(fake_bin))

    # Mock spawned process to stream two lines
    mock_popen.return_value = _fake_popen(["hello", "world"])

    # Import (or reload) litellm after env is set so the provider registers
    import litellm as _ll
    importlib.reload(_ll)

    # Call completion (non-stream) and verify OpenAI-style ModelResponse
    resp = _ll.completion(
        model="codex-agent",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "extra_body": {
                "codex_cli_model": "gpt-5",
                "codex_json": True,
            }
        },
    )
    text = resp.choices[0].message.content
    assert "hello" in text and "world" in text
    assert resp.choices[0].finish_reason == "stop"

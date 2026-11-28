import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest

import litellm

_LITE: Final = (
    f"import sys; sys.path.insert(0, {str(Path(litellm.__file__).resolve().parents[1])!r}); "
    "from litellm.proxy.client.cli.main import cli; cli()"
)
_RECORD_INVOCATION: Final = """
import hashlib, json, os, sys
key = os.environ.get("OPENAI_API_KEY", "")
with open(os.environ["STAND_IN_LOG"], "a") as log:
    log.write(json.dumps({"argv": sys.argv[1:], "key_sha256": hashlib.sha256(key.encode()).hexdigest()}) + "\\n")
if sys.argv[1:] == ["service", "restart"]:
    print("http://127.0.0.1:4096")
    print("restart notice", file=sys.stderr)
    sys.exit(int(os.environ["STAND_IN_RESTART_EXIT"]))
"""

_Launch = Callable[..., list[dict[str, object]]]


def _sha256(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@pytest.fixture
def launch_lite_opencode(tmp_path: Path) -> _Launch:
    bin_dir: Final = tmp_path / "bin"
    bin_dir.mkdir()
    log: Final = tmp_path / "invocations.jsonl"
    recorder: Final = tmp_path / "record.py"
    recorder.write_text(_RECORD_INVOCATION)
    opencode: Final = bin_dir / "opencode"
    opencode.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{recorder}" "$@"\n')
    opencode.chmod(0o700)

    def launch(key: str, *args: str, restart_exit: int = 0) -> list[dict[str, object]]:
        log.write_text("")
        lite: Final = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                _LITE,
                "--base-url",
                "http://127.0.0.1:9",
                "--api-key",
                key,
                "opencode",
                "--skip-verify",
                *args,
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
            env={
                **os.environ,
                "HOME": str(tmp_path),
                "XDG_CONFIG_HOME": str(tmp_path / "config"),
                "OPENAI_API_KEY": "sk-stale",
                "STAND_IN_LOG": str(log),
                "STAND_IN_RESTART_EXIT": str(restart_exit),
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            },
        )
        assert lite.returncode == 0, lite.stderr
        assert "4096" not in lite.stdout and "restart notice" not in lite.stderr
        return [json.loads(line) for line in log.read_text().splitlines()]

    return launch


def _restart_then(key: str, *args: str) -> list[dict[str, object]]:
    return [
        {"argv": ["service", "restart"], "key_sha256": _sha256(key)},
        {"argv": list(args), "key_sha256": _sha256(key)},
    ]


def _launch_only(key: str, *args: str) -> list[dict[str, object]]:
    return [{"argv": list(args), "key_sha256": _sha256(key)}]


@pytest.mark.skipif(sys.platform.startswith("win"), reason="the stand-in opencode is a POSIX shell script")
def test_lite_opencode_restarts_the_background_server_only_when_the_launch_key_changed(
    launch_lite_opencode: _Launch,
):
    assert launch_lite_opencode("sk-first", "run", "hello") == _restart_then("sk-first", "run", "hello")
    assert launch_lite_opencode("sk-first", "service", "stop") == _launch_only("sk-first", "service", "stop")
    assert launch_lite_opencode("sk-renewed", "run", "hello") == _restart_then("sk-renewed", "run", "hello")


@pytest.mark.skipif(sys.platform.startswith("win"), reason="the stand-in opencode is a POSIX shell script")
def test_a_failed_restart_is_retried_on_the_next_launch(launch_lite_opencode: _Launch):
    assert launch_lite_opencode("sk-key", "run", "x", restart_exit=1) == _restart_then("sk-key", "run", "x")
    assert launch_lite_opencode("sk-key", "run", "x") == _restart_then("sk-key", "run", "x")
    assert launch_lite_opencode("sk-key", "run", "x") == _launch_only("sk-key", "run", "x")

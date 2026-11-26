import hashlib
import json
import os
import subprocess
import sys
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
"""


def _invocations_of_lite_opencode(tmp_path: Path, *args: str) -> list[dict[str, object]]:
    bin_dir: Final = tmp_path / "bin"
    bin_dir.mkdir()
    log: Final = tmp_path / "invocations.jsonl"
    recorder: Final = tmp_path / "record.py"
    recorder.write_text(_RECORD_INVOCATION)
    opencode: Final = bin_dir / "opencode"
    opencode.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{recorder}" "$@"\n')
    opencode.chmod(0o700)
    lite: Final = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            _LITE,
            "--base-url",
            "http://127.0.0.1:9",
            "--api-key",
            "sk-current",
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
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        },
    )
    assert lite.returncode == 0, lite.stderr
    assert "4096" not in lite.stdout and "restart notice" not in lite.stderr
    return [json.loads(line) for line in log.read_text().splitlines()]


@pytest.mark.skipif(sys.platform.startswith("win"), reason="the stand-in opencode is a POSIX shell script")
@pytest.mark.parametrize("user_args", [("run", "hello"), ("service", "stop")], ids=["run", "service_stop"])
def test_lite_opencode_restarts_the_background_server_with_the_launch_key_before_opening_it(
    tmp_path: Path, user_args: tuple[str, ...]
):
    launch_key: Final = hashlib.sha256(b"sk-current").hexdigest()
    assert _invocations_of_lite_opencode(tmp_path, *user_args) == [
        {"argv": ["service", "restart"], "key_sha256": launch_key},
        {"argv": list(user_args), "key_sha256": launch_key},
    ]

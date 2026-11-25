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


def _launch_lite_opencode(tmp_path: Path, version_line: str, *args: str) -> list[str]:
    bin_dir: Final = tmp_path / "bin"
    bin_dir.mkdir()
    argv_file: Final = tmp_path / "argv"
    opencode: Final = bin_dir / "opencode"
    opencode.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = "--version" ]; then echo "{version_line}"; exit 0; fi\n'
        f'printf "%s\\n" "$@" > "{argv_file}"\n'
    )
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
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        },
    )
    assert lite.returncode == 0, lite.stderr
    return argv_file.read_text().splitlines()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="the stand-in opencode is a POSIX shell script")
@pytest.mark.parametrize(
    ("version_line", "expected"),
    [
        ("opencode v2.4.1", ["--standalone", "run", "hello"]),
        ("1.19.6", ["run", "hello"]),
    ],
    ids=["opencode_2_gets_a_private_server", "opencode_1_has_no_shared_server"],
)
def test_lite_opencode_runs_opencode_2_with_its_own_server_so_the_launch_key_is_used(
    tmp_path: Path, version_line: str, expected: list[str]
):
    assert _launch_lite_opencode(tmp_path, version_line, "run", "hello") == expected


@pytest.mark.skipif(sys.platform.startswith("win"), reason="the stand-in opencode is a POSIX shell script")
def test_lite_opencode_service_stop_launches_without_the_flag_opencode_rejects_there(tmp_path: Path):
    assert _launch_lite_opencode(tmp_path, "opencode v2.4.1", "service", "stop") == ["service", "stop"]

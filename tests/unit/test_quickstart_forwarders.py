import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Final

import pytest

ROOT: Final = Path(__file__).resolve().parents[2]
URL: Final = "https://raw.githubusercontent.com/BerriAI/litellm/main/litellm/scripts/quickstart"


@pytest.mark.parametrize("piped", [False, True])
@pytest.mark.parametrize("args", [(), ("stop",), ("--yes", "status", "two words", "", "*")])
@pytest.mark.parametrize("exit_code", [0, 37])
def test_sh_forwards_arguments_and_exit_code(
    tmp_path: Path, piped: bool, args: tuple[str, ...], exit_code: int
) -> None:
    result: Final = _run_sh(tmp_path, piped, args, exit_code, False)
    assert result.returncode == exit_code, result.stderr
    assert result.stdout.splitlines() == ["ran", *args]
    assert (tmp_path / "url").read_text() == URL + ".sh"
    assert tuple((tmp_path / "downloads").iterdir()) == ()


def _run_sh(
    tmp_path: Path, piped: bool, args: tuple[str, ...], exit_code: int, fail: bool
) -> subprocess.CompletedProcess[str]:
    stub: Final = tmp_path / "new.sh"
    stub.write_text('printf "%s\\n" ran\nfor arg do printf "%s\\n" "$arg"; done\nexit "$STUB_EXIT"\n')
    curl: Final = tmp_path / "curl"
    curl.write_text('#!/bin/sh\nprintf "%s" "$2" > "$URL_LOG"\ncp "$STUB" "$4"\nexit "$CURL_EXIT"\n')
    curl.chmod(0o755)
    downloads: Final = tmp_path / "downloads"
    downloads.mkdir()
    script: Final = ROOT / "scripts/quickstart.sh"
    return subprocess.run(
        ["sh", "-s", "--", *args] if piped else ["sh", str(script), *args],
        input=script.read_text() if piped else None,
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "STUB": str(stub),
            "STUB_EXIT": str(exit_code),
            "CURL_EXIT": "22" if fail else "0",
            "URL_LOG": str(tmp_path / "url"),
            "TMPDIR": str(downloads),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_sh_never_runs_partial_download(tmp_path: Path) -> None:
    result: Final = _run_sh(tmp_path, True, (), 0, True)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "could not download" in result.stderr
    assert tuple((tmp_path / "downloads").iterdir()) == ()


def test_forwarders_are_not_accepted_as_offline_install_scripts() -> None:
    assert not re.search(r'^main "\$@"$', (ROOT / "scripts/quickstart.sh").read_text(), re.MULTILINE)
    assert not re.search(
        r"^\$code = Invoke-LiteLLMQuickstart", (ROOT / "scripts/quickstart.ps1").read_text(), re.MULTILINE
    )


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is not installed")
@pytest.mark.parametrize("mode", ["file", "iex", "scriptblock"])
@pytest.mark.parametrize("exit_code", [0, 37])
@pytest.mark.parametrize("fail", [False, True])
def test_ps1_forwards_arguments_and_exit_code(tmp_path: Path, mode: str, exit_code: int, fail: bool) -> None:
    script: Final = ROOT / "scripts/quickstart.ps1"
    stub: Final = tmp_path / "new.ps1"
    stub.write_text(
        'param([switch]$Yes)\nWrite-Output "Yes=$Yes"\n$args | ForEach-Object { "arg=$_" }\n'
        f"$global:LASTEXITCODE = {exit_code}\n"
    )
    invocation: Final = {
        "file": '& $env:FORWARDER -Yes stop "two words" "" "*"; exit $LASTEXITCODE',
        "iex": "Get-Content -Raw $env:FORWARDER | Invoke-Expression",
        "scriptblock": '& ([scriptblock]::Create((Get-Content -Raw $env:FORWARDER))) -Yes stop "two words" "" "*"',
    }[mode]
    command: Final = (
        "function Invoke-RestMethod { param($Uri, [switch]$UseBasicParsing, $ErrorAction) "
        "[IO.File]::WriteAllText($env:URL_LOG, $Uri); "
        'if ($env:FAIL_DOWNLOAD -eq "1") { throw "download failed" }; '
        "Get-Content -Raw $env:STUB }; " + invocation
    )
    result: Final = subprocess.run(
        ["pwsh", "-NoProfile", "-Command", command],
        env={
            **os.environ,
            "FORWARDER": str(script),
            "STUB": str(stub),
            "URL_LOG": str(tmp_path / "url"),
            "FAIL_DOWNLOAD": "1" if fail else "0",
        },
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == (1 if fail else exit_code), result.stderr
    assert result.stdout.splitlines() == (
        [] if fail else ["Yes=False"] if mode == "iex" else ["Yes=True", "arg=stop", "arg=two words", "arg=", "arg=*"]
    )
    assert (tmp_path / "url").read_text() == URL + ".ps1"

"""Verify console commands from installed SDK profiles."""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Final


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_cli(profile: str) -> str:
    import litellm

    if profile in ("cli", "proxy"):
        from litellm.proxy.proxy_cli import run_server

        _require(litellm.run_server is run_server, "public server command identity changed")
    for command, extra in (("litellm", "proxy"), ("lite", "cli"), ("litellm-proxy", "cli")):
        executable: Final = Path(sys.executable).parent / command
        result: Final = subprocess.run((str(executable), "--help"), capture_output=True, text=True, timeout=30)
        if profile == "proxy" or (profile == "cli" and command != "litellm"):
            _require(result.returncode == 0, f"{command} failed: {result.stderr}")
            _require("Usage:" in result.stdout, f"{command} did not display help")
        else:
            _require(result.returncode != 0, f"{command} succeeded without CLI dependencies")
            _require(f"litellm[{extra}]" in result.stderr, f"{command} omitted installation guidance")
            _require("Traceback" not in result.stderr, f"{command} printed a traceback")
    return "console commands preserve help or explain their required extra"


if __name__ == "__main__":
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("core", "cli", "proxy"), required=True)
    profile: Final = parser.parse_args().profile
    if profile == "core":
        _require(importlib.util.find_spec("click") is None, "core still installs click")
    print(check_cli(profile))

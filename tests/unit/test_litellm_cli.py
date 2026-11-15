import sys
from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm.cli.main import cli

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_PYPROJECT: Final = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_lite_keeps_an_sdk_that_was_already_imported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setitem(sys.modules, "litellm", litellm)
    monkeypatch.setattr(litellm, "success_callback", ["kept"])
    monkeypatch.setattr(sys, "argv", ["lite", "--help"])

    with pytest.raises(SystemExit) as exited:
        cli()

    assert exited.value.code == 0
    assert "LiteLLM Proxy CLI - Manage your LiteLLM proxy server" in capsys.readouterr().out
    assert sys.modules["litellm"] is litellm
    assert litellm.success_callback == ["kept"]

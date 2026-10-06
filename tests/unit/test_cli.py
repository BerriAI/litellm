import sys
from collections.abc import Callable
from typing import Final

import pytest

from litellm._cli import cli, litellm_proxy_cli, run_server


@pytest.mark.parametrize(
    "command,extra,module",
    (
        (run_server, "proxy", "click"),
        (run_server, "proxy", "dotenv"),
        (run_server, "proxy", "pydantic_settings"),
        (cli, "cli", "click"),
        (cli, "cli", "rich"),
        (litellm_proxy_cli, "cli", "filelock"),
    ),
)
def test_console_missing_extra_exits_with_guidance(
    command: Callable[[], None], extra: str, module: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(SystemExit) as caught:
        command()
    assert f"litellm[{extra}]" in str(caught.value)


@pytest.mark.parametrize("command", (run_server, cli, litellm_proxy_cli))
def test_console_with_dependencies_displays_help(
    command: Callable[[], None], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["litellm", "--help"])
    with pytest.raises(SystemExit) as caught:
        command()
    assert caught.value.code == 0
    output: Final = capsys.readouterr()
    assert "Usage:" in output.out
    assert "Traceback" not in output.err


def test_server_help_preserves_plain_legacy_installation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for module in ("fastapi", "uvicorn", "litellm_proxy_extras"):
        monkeypatch.setitem(sys.modules, module, None)
    monkeypatch.setattr(sys, "argv", ["litellm", "--help"])
    with pytest.raises(SystemExit) as caught:
        run_server()
    assert caught.value.code == 0
    assert "Usage:" in capsys.readouterr().out

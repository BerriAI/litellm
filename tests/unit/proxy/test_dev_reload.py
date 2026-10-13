import importlib
from pathlib import Path
from typing import Final

import pytest
from uvicorn import config as uvicorn_config

from litellm.proxy.dev_reload import ReloadOptions, load_app_or_exit, pool_config

UNCHANGED_LOGGING: Final = {"version": 1, "incremental": True}


def _options(app: str) -> ReloadOptions:
    return {
        "app": app,
        "log_config": UNCHANGED_LOGGING,
        "reload": True,
        "reload_dirs": (),
        "reload_includes": ("*.py",),
    }


@pytest.mark.parametrize(
    ("module", "source"),
    (
        ("reload_syntax_error_app", "def app(\n"),
        ("reload_name_error_app", "app = undefined_name\n"),
        ("reload_runtime_error_app", "raise RuntimeError('broken edit')\n"),
        ("reload_missing_attribute_app", "value = 1\n"),
    ),
)
def test_a_broken_app_exits_the_worker_as_a_startup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: str, source: str
) -> None:
    (tmp_path / f"{module}.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(SystemExit) as exit_info:
        pool_config(_options(f"{module}:app"), workers=2).load()
    assert exit_info.value.code == uvicorn_config.STARTUP_FAILURE


def test_a_working_app_loads_as_the_module_defines_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "reload_working_app.py").write_text("async def app(scope, receive, send):\n    return None\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert load_app_or_exit("reload_working_app:app") is importlib.import_module("reload_working_app").app

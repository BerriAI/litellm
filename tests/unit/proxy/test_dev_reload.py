import importlib
import logging
import logging.handlers
import traceback
from pathlib import Path
from typing import Final

import pytest
from uvicorn import config as uvicorn_config

from litellm.proxy.dev_reload import (
    RELOAD_OPTIONS,
    RELOAD_WORKER_HEALTHCHECK_SECONDS,
    ReloadOptions,
    load_app_or_exit,
    load_worker_app,
    multi_worker_reload_supported,
    pool_config,
)
from litellm.proxy.proxy_cli import ProxyInitializationHelpers

UNCHANGED_LOGGING: Final = {"version": 1, "incremental": True}
POOL_WORKERS: Final = 2
REQUIRES_UVICORN_STARTUP_FAILURE: Final = pytest.mark.skipif(
    not multi_worker_reload_supported(), reason="uvicorn<0.50.0 has no startup failure exit code"
)


def _options(app: str) -> ReloadOptions:
    return {
        "app": app,
        "log_config": UNCHANGED_LOGGING,
        "reload": True,
        "reload_dirs": (),
        "reload_includes": ("*.py",),
    }


@REQUIRES_UVICORN_STARTUP_FAILURE
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


@REQUIRES_UVICORN_STARTUP_FAILURE
@pytest.mark.parametrize(
    ("module", "source"),
    (
        ("reload_import_error_app", "import os\nfrom os import missing_name\n"),
        ("reload_value_error_app", "import os\nraise ValueError('broken edit')\n"),
    ),
)
def test_a_broken_import_logs_the_line_that_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: str, source: str
) -> None:
    app_file: Final = tmp_path / f"{module}.py"
    app_file.write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    uvicorn_logger: Final = logging.getLogger("uvicorn.error")
    captured: Final = logging.handlers.BufferingHandler(capacity=POOL_WORKERS)
    uvicorn_logger.addHandler(captured)
    try:
        with pytest.raises(SystemExit):
            load_app_or_exit(f"{module}:app")
    finally:
        uvicorn_logger.removeHandler(captured)
    (record,) = captured.buffer
    assert record.exc_info is not None
    raised_at: Final = frozenset((frame.filename, frame.lineno) for frame in traceback.extract_tb(record.exc_info[2]))
    assert (str(app_file), 2) in raised_at, record.exc_text


def test_the_reload_pool_defaults_the_worker_healthcheck_and_keeps_an_explicit_one() -> None:
    explicit: Final = RELOAD_WORKER_HEALTHCHECK_SECONDS + 7
    assert pool_config(_options("app:app"), workers=2).timeout_worker_healthcheck == RELOAD_WORKER_HEALTHCHECK_SECONDS
    assert (
        pool_config({**_options("app:app"), "timeout_worker_healthcheck": explicit}, workers=2).timeout_worker_healthcheck
        == explicit
    )


def test_the_reload_pool_accepts_and_keeps_every_uvicorn_arg_the_proxy_cli_builds(tmp_path: Path) -> None:
    cli_args: Final = ProxyInitializationHelpers.get_default_unvicorn_init_args(
        "127.0.0.1", 4123, keepalive_timeout=61, timeout_worker_healthcheck=RELOAD_WORKER_HEALTHCHECK_SECONDS + 7
    )
    options: Final = RELOAD_OPTIONS.validate_python(
        {
            **cli_args,
            "log_config": UNCHANGED_LOGGING,
            "reload": True,
            "reload_dirs": [str(tmp_path)],
            "reload_includes": ["*.py"],
        }
    )
    config: Final = pool_config(options, workers=POOL_WORKERS)
    server_args: Final = {key: value for key, value in cli_args.items() if key not in ("app", "log_config")}
    assert {key: getattr(config, key) for key in server_args} == server_args


def test_the_reload_pool_passes_a_uvicorn_arg_it_does_not_list_through_to_the_workers() -> None:
    config: Final = pool_config(RELOAD_OPTIONS.validate_python({**_options("app:app"), "backlog": 123}), workers=POOL_WORKERS)
    assert (config.backlog, config.reload) == (123, False)


def test_the_reload_pool_applies_child_process_overrides_to_its_workers() -> None:
    config: Final = pool_config(
        _options("app:app"),
        workers=POOL_WORKERS,
        worker_overrides={"app": "litellm_cli.main:proxy_worker_app", "factory": True, "log_config": None},
    )
    assert config.app.args == ("litellm_cli.main:proxy_worker_app", True)
    assert config.factory
    assert config.log_config is None


def test_load_worker_app_calls_the_overridden_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module: Final = "reload_worker_app_factory"
    (tmp_path / f"{module}.py").write_text(
        "def app(scope, receive, send):\n    return None\n\ndef factory():\n    return app\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    app: Final = importlib.import_module(module).app
    assert load_worker_app(f"{module}:factory", worker_app_factory=True) is app


def test_windows_keeps_single_worker_reload_whatever_the_uvicorn_version() -> None:
    assert not multi_worker_reload_supported("win32")


def test_a_working_app_loads_as_the_module_defines_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "reload_working_app.py").write_text("async def app(scope, receive, send):\n    return None\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert load_app_or_exit("reload_working_app:app") is importlib.import_module("reload_working_app").app

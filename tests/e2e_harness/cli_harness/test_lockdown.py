from __future__ import annotations

from pathlib import Path
from typing import Final

from cli_harness.lockdown import isolated_home, locked_down_env

RUNNER_ENV: Final = {
    "PATH": "/usr/bin",
    "LANG": "C.UTF-8",
    "HOME": "/home/runner",
    "OPENAI_API_KEY": "sk-runner-secret",
    "AWS_SECRET_ACCESS_KEY": "runner-aws-secret",
    "DATABASE_URL": "postgresql://runner",
}


def test_cli_sees_runtime_vars_and_its_overrides_but_no_runner_credentials(tmp_path: Path) -> None:
    env: Final = locked_down_env(home=tmp_path, overrides={"OPENAI_BASE_URL": "http://gateway"}, source=RUNNER_ENV)
    assert dict(env) == {
        "PATH": "/usr/bin",
        "LANG": "C.UTF-8",
        "HOME": str(tmp_path),
        "OPENAI_BASE_URL": "http://gateway",
    }


def test_isolated_home_starts_empty_and_is_gone_after_the_cli_exits() -> None:
    with isolated_home("cli-harness-test-") as home:
        assert home.is_dir() and not any(home.iterdir())
        (home / ".config").mkdir()
    assert not home.exists()

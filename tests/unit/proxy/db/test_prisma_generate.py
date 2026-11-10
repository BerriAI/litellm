import importlib
import os
import subprocess
import sys
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pytest

from litellm.proxy.db.prisma_generate import (
    SHIPPED_SCHEMA_PATH,
    ClientGenerated,
    ClientGenerationFailed,
    ClientPresent,
    PrismaClientStatus,
    ensure_prisma_client,
    env_with_scripts_dir,
)

STUB_INIT: Final = """
try:
    from .client import Prisma
except ModuleNotFoundError:
    def __getattr__(name):
        raise RuntimeError("The Client hasn't been generated yet")
"""


@dataclass
class FakeRunner:
    on_run: Callable[[], None]
    calls: list[tuple[tuple[str, ...], Mapping[str, str]]] = field(default_factory=list)  # mutable-ok: records calls

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        env: Mapping[str, str],
        stdout: int | None,
        stderr: int | None,
    ) -> "subprocess.CompletedProcess[str]":
        self.calls.append((tuple(argv), env))
        self.on_run()
        return subprocess.CompletedProcess(tuple(argv), 0, "", "")


def _ensure(package_dir: Path, runner: FakeRunner, package_name: str = "prisma") -> PrismaClientStatus:
    return ensure_prisma_client(
        package_dir=package_dir,
        schema_path=SHIPPED_SCHEMA_PATH,
        run=runner,
        timeout=5.0,
        env={"PATH": "/usr/bin"},
        package_name=package_name,
    )


def _write_client(package_dir: Path) -> Callable[[], None]:
    def run() -> None:
        _ = (package_dir / "client.py").write_text("class Prisma:\n    pass\n")

    return run


def _raise(error: BaseException) -> Callable[[], None]:
    def run() -> None:
        raise error

    return run


@pytest.fixture
def stub_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[tuple[str, Path], None, None]:
    name: Final = "prisma_stub_for_generate_test"
    package_dir: Final = tmp_path / name
    package_dir.mkdir()
    _ = (package_dir / "__init__.py").write_text(STUB_INIT)
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    try:
        yield name, package_dir
    finally:
        _ = sys.modules.pop(name, None)
        _ = sys.modules.pop(f"{name}.client", None)


def test_a_missing_client_is_generated_from_the_shipped_schema(stub_package: tuple[str, Path]) -> None:
    name, package_dir = stub_package
    runner: Final = FakeRunner(on_run=_write_client(package_dir))

    status: Final = _ensure(package_dir, runner, package_name=name)

    assert status == ClientGenerated()
    assert [argv for argv, _ in runner.calls] == [("prisma", "generate", "--schema", str(SHIPPED_SCHEMA_PATH))]
    assert SHIPPED_SCHEMA_PATH.is_file()


def test_an_existing_client_is_left_alone(tmp_path: Path) -> None:
    _write_client(tmp_path)()
    runner: Final = FakeRunner(on_run=_raise(AssertionError("must not run prisma generate")))

    assert _ensure(tmp_path, runner) == ClientPresent()
    assert runner.calls == []


def test_a_package_imported_before_generation_exposes_the_new_client(stub_package: tuple[str, Path]) -> None:
    name, package_dir = stub_package
    stub: Final = importlib.import_module(name)
    with pytest.raises(RuntimeError):
        _ = hasattr(stub, "Prisma")

    assert _ensure(package_dir, FakeRunner(on_run=_write_client(package_dir)), package_name=name) == ClientGenerated()
    assert "Prisma" in vars(sys.modules[name])


@pytest.mark.parametrize(
    ("error", "reason_part"),
    [
        (subprocess.CalledProcessError(1, "prisma"), "exited 1"),
        (subprocess.TimeoutExpired("prisma", 5.0), "timed out"),
        (FileNotFoundError("prisma"), "could not run"),
    ],
)
def test_a_failed_generate_is_reported_not_raised(tmp_path: Path, error: BaseException, reason_part: str) -> None:
    status: Final = _ensure(tmp_path, FakeRunner(on_run=_raise(error)))

    assert isinstance(status, ClientGenerationFailed)
    assert reason_part in status.reason


def test_a_generate_that_writes_no_client_is_a_failure(tmp_path: Path) -> None:
    status: Final = _ensure(tmp_path, FakeRunner(on_run=lambda: None))

    assert isinstance(status, ClientGenerationFailed)


def test_a_missing_prisma_package_is_a_failure() -> None:
    runner: Final = FakeRunner(on_run=_raise(AssertionError("must not run prisma generate")))
    status: Final = ensure_prisma_client(
        package_dir=None, schema_path=SHIPPED_SCHEMA_PATH, run=runner, timeout=5.0, env={}
    )

    assert isinstance(status, ClientGenerationFailed)
    assert runner.calls == []


@pytest.mark.parametrize(
    ("env", "expected_path"),
    [
        ({"PATH": "/usr/bin", "HOME": "/root"}, os.pathsep.join(("/opt/litellm/.venv/bin", "/usr/bin"))),
        ({"HOME": "/root"}, "/opt/litellm/.venv/bin"),
    ],
)
def test_the_venv_scripts_dir_comes_first_on_path(env: dict[str, str], expected_path: str) -> None:
    result: Final = env_with_scripts_dir(env, "/opt/litellm/.venv/bin")

    assert dict(result) == {**env, "PATH": expected_path}

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

from ...shared.parity.ledger import RustTestIdentity, RustTestScope

CARGO_CONFIG: Final = ConfigDict(extra="ignore", frozen=True, strict=True)
CommandRunner: TypeAlias = Callable[[tuple[str, ...], Path], str]


class CargoPackage(BaseModel):
    model_config = CARGO_CONFIG

    id: str
    name: str


class CargoMetadata(BaseModel):
    model_config = CARGO_CONFIG

    packages: tuple[CargoPackage, ...]


class CargoMessage(BaseModel):
    model_config = CARGO_CONFIG

    reason: str


class CargoTarget(BaseModel):
    model_config = CARGO_CONFIG

    name: str
    kind: tuple[str, ...]


class CargoProfile(BaseModel):
    model_config = CARGO_CONFIG

    test: bool


class CargoArtifact(BaseModel):
    model_config = CARGO_CONFIG

    reason: Literal["compiler-artifact"]
    package_id: str
    target: CargoTarget
    profile: CargoProfile
    executable: str | None


def run_command(command: tuple[str, ...], cwd: Path) -> str:
    result: Final = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ValueError(
            f"Rust inventory command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr}\n{result.stdout}"
        )
    return result.stdout


def _build_command(scope: RustTestScope) -> tuple[str, ...]:
    selector: Final = ("--lib",) if scope.target.kind == "lib" else (f"--{scope.target.kind}", scope.target.name)
    features: Final = ("--features", ",".join(scope.features)) if scope.features else ()
    defaults: Final = () if scope.default_features else ("--no-default-features",)
    return (
        "cargo",
        "test",
        "--package",
        scope.target.package,
        *selector,
        *features,
        *defaults,
        "--locked",
        "--no-run",
        "--message-format=json",
        "--color",
        "never",
    )


def _test_names(output: str) -> frozenset[str]:
    lines: Final = tuple(line for line in output.splitlines() if line)
    invalid: Final = tuple(line for line in lines if not line.endswith((": test", ": benchmark")))
    if invalid:
        raise ValueError(f"Unrecognized libtest inventory output: {invalid!r}")
    names: Final = tuple(line.removesuffix(": test") for line in lines if line.endswith(": test"))
    if len(names) != len(frozenset(names)):
        raise ValueError("Duplicate test names in libtest inventory")
    return frozenset(names)


def _scope_tests(
    scope: RustTestScope, metadata: CargoMetadata, cwd: Path, command_runner: CommandRunner
) -> frozenset[RustTestIdentity]:
    package_ids: Final = tuple(package.id for package in metadata.packages if package.name == scope.target.package)
    if len(package_ids) != 1:
        raise ValueError(f"Expected one Cargo package for {scope.target.package}, found {len(package_ids)}")
    output: Final = command_runner(_build_command(scope), cwd)
    artifacts: Final = tuple(
        CargoArtifact.model_validate_json(line)
        for line in output.splitlines()
        if CargoMessage.model_validate_json(line).reason == "compiler-artifact"
    )
    executables: Final = frozenset(
        artifact.executable
        for artifact in artifacts
        if artifact.package_id == package_ids[0]
        and artifact.target.name == scope.target.name
        and scope.target.kind in artifact.target.kind
        and artifact.profile.test
        and artifact.executable is not None
    )
    if len(executables) != 1:
        raise ValueError(f"Expected one test executable for {scope.target.key}, found {len(executables)}")
    executable: Final = next(iter(executables))
    names: Final = _test_names(command_runner((executable, "--list", "--format", "terse"), cwd))
    ignored: Final = _test_names(command_runner((executable, "--list", "--ignored", "--format", "terse"), cwd))
    identities: Final = frozenset(RustTestIdentity(target=scope.target, name=name) for name in names)
    scoped: Final = frozenset(identity for identity in identities if scope.contains(identity))
    ignored_scoped: Final = tuple(sorted(identity.key for identity in scoped if identity.name in ignored))
    if ignored_scoped:
        raise ValueError(f"Ignored Rust tests cannot satisfy the ledger: {', '.join(ignored_scoped)}")
    empty_modules: Final = tuple(
        module for module in scope.modules if not any(identity.name.startswith(f"{module}::") for identity in scoped)
    )
    if empty_modules:
        raise ValueError(f"No compiled tests in {scope.target.key} modules: {', '.join(empty_modules)}")
    return scoped


def enumerate_rust_tests(
    repo_root: Path, scopes: tuple[RustTestScope, ...], *, command_runner: CommandRunner = run_command
) -> frozenset[RustTestIdentity]:
    if not scopes:
        return frozenset()
    cwd: Final = repo_root / "litellm-rust"
    metadata: Final = CargoMetadata.model_validate_json(
        command_runner(("cargo", "metadata", "--format-version", "1", "--no-deps", "--locked"), cwd)
    )
    return frozenset(identity for scope in scopes for identity in _scope_tests(scope, metadata, cwd, command_runner))

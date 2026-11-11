from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

CommandRunner: TypeAlias = Callable[[tuple[str, ...], Path], str]
_MODEL_CONFIG: Final = ConfigDict(extra="forbid", frozen=True, strict=True)


class RustTarget(BaseModel):
    model_config = _MODEL_CONFIG

    package: str
    name: str
    kind: Literal["lib", "bin", "test"]

    @property
    def key(self) -> str:
        return f"{self.package}/{self.kind}/{self.name}"


class RustTestIdentity(BaseModel):
    model_config = _MODEL_CONFIG

    target: RustTarget
    name: str

    @property
    def key(self) -> str:
        return f"{self.target.key}::{self.name}"


class RustTestScope(BaseModel):
    model_config = _MODEL_CONFIG

    target: RustTarget
    modules: Annotated[tuple[str, ...], Field(min_length=1)]
    features: tuple[str, ...] = ()
    default_features: bool = True

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        duplicate_features: Final = tuple(
            feature for feature, values in groupby(sorted(self.features)) if sum(1 for _ in values) > 1
        )
        duplicate_modules: Final = tuple(
            module for module, values in groupby(sorted(self.modules)) if sum(1 for _ in values) > 1
        )
        if duplicate_features:
            raise ValueError(f"Rust features contain duplicates: {', '.join(duplicate_features)}")
        if duplicate_modules:
            raise ValueError(f"Rust modules contain duplicates: {', '.join(duplicate_modules)}")
        if any(not module or module.endswith("::") for module in self.modules):
            raise ValueError("Rust modules must be non-empty and omit the trailing :: separator")
        overlaps: Final = tuple(
            f"{outer} includes {inner}"
            for outer in self.modules
            for inner in self.modules
            if inner.startswith(f"{outer}::")
        )
        if overlaps:
            raise ValueError(f"Rust modules overlap: {', '.join(overlaps)}")
        return self

    def contains(self, identity: RustTestIdentity) -> bool:
        return identity.target == self.target and any(
            identity.name.startswith(f"{module}::") for module in self.modules
        )


class _CargoPackage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    id: str
    name: str


class _CargoMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    packages: tuple[_CargoPackage, ...]


class _CargoMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    reason: str


class _CargoTarget(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    name: str
    kind: tuple[str, ...]


class _CargoProfile(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    test: bool


class _CargoArtifact(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    reason: Literal["compiler-artifact"]
    package_id: str
    target: _CargoTarget
    profile: _CargoProfile
    executable: str | None


@dataclass(frozen=True, slots=True)
class RustReport:
    tests: tuple[str, ...]
    exit_code: int
    output: str


def run_command(command: tuple[str, ...], cwd: Path) -> str:
    try:
        result: Final = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"Rust inventory command failed: {error}") from error
    if result.returncode != 0:
        raise ValueError(
            f"Rust inventory command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr}\n{result.stdout}"
        )
    return result.stdout


def run_rust_tests(manifest: Path, package: str | None, test_filter: str, *, collect_only: bool = False) -> RustReport:
    command: Final = (
        "cargo",
        "test",
        "--manifest-path",
        str(manifest),
        *(("--package", package) if package else ()),
        "--lib",
        test_filter,
        "--",
        *(("--list",) if collect_only else ("--format=pretty",)),
    )
    try:
        result: Final = subprocess.run(command, capture_output=True, text=True, check=False, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as error:
        return RustReport((), 1, str(error))
    tests: Final = (
        tuple(line.removesuffix(": test") for line in result.stdout.splitlines() if line.endswith(": test"))
        if collect_only
        else tuple(
            line.removeprefix("test ").removesuffix(" ... ok")
            for line in result.stdout.splitlines()
            if line.startswith("test ") and line.endswith(" ... ok")
        )
    )
    return RustReport(tests, result.returncode, result.stdout + result.stderr)


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
    scope: RustTestScope,
    metadata: _CargoMetadata,
    cwd: Path,
    command_runner: CommandRunner,
) -> frozenset[RustTestIdentity]:
    package_ids: Final = tuple(package.id for package in metadata.packages if package.name == scope.target.package)
    if len(package_ids) != 1:
        raise ValueError(f"Expected one Cargo package for {scope.target.package}, found {len(package_ids)}")
    output: Final = command_runner(_build_command(scope), cwd)
    artifacts: Final = tuple(
        _CargoArtifact.model_validate_json(line)
        for line in output.splitlines()
        if _CargoMessage.model_validate_json(line).reason == "compiler-artifact"
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
        raise ValueError(f"Ignored Rust tests cannot satisfy the mapping: {', '.join(ignored_scoped)}")
    empty_modules: Final = tuple(
        module for module in scope.modules if not any(identity.name.startswith(f"{module}::") for identity in scoped)
    )
    if empty_modules:
        raise ValueError(f"No compiled tests in {scope.target.key} modules: {', '.join(empty_modules)}")
    return scoped


def enumerate_rust_tests(
    repo_root: Path,
    scopes: tuple[RustTestScope, ...],
    *,
    command_runner: CommandRunner = run_command,
) -> frozenset[RustTestIdentity]:
    if not scopes:
        return frozenset()
    cwd: Final = repo_root / "litellm-rust"
    metadata: Final = _CargoMetadata.model_validate_json(
        command_runner(("cargo", "metadata", "--format-version", "1", "--no-deps", "--locked"), cwd)
    )
    return frozenset(identity for scope in scopes for identity in _scope_tests(scope, metadata, cwd, command_runner))

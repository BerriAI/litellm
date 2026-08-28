# /// script
# requires-python = ">=3.10"
# dependencies = ["pip==26.2.1", "pyperf==2.10.0", "psutil==7.2.2", "pydantic==2.13.4"]
# ///
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.parser import BytesParser
from functools import partial
from pathlib import Path
from typing import Final, Literal, TextIO

from bench_sdk_runtime import PROBE, command, probe, provider, require, runtime_environment, startup, summary
from pydantic import TypeAdapter


@dataclass(frozen=True, slots=True)
class Wheel:
    filename: str
    name: str
    version: str
    sha256: str
    compressed_bytes: int
    uncompressed_bytes: int
    tags: tuple[str, ...]
    extras: tuple[str, ...]
    native_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Source:
    kind: Literal["local", "package", "wheel"]
    value: str


@dataclass(frozen=True, slots=True)
class Options:
    source: Source
    output: Path
    extras: str
    samples: int
    install_samples: int
    warmups: int
    timeout: int
    constraints: Path | None
    wheelhouse: Path | None


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        hasher: Final = hashlib.sha256()
        for chunk in iter(partial(stream.read, 1024 * 1024), b""):
            hasher.update(chunk)
        return hasher.hexdigest()


def wheel_info(path: Path) -> Wheel:
    with zipfile.ZipFile(path) as archive:
        metadata_files: Final = tuple(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        require(len(metadata_files) == 1, f"Expected one wheel METADATA file in {path}")
        metadata: Final = BytesParser().parsebytes(archive.read(metadata_files[0]))
        wheel_metadata: Final = BytesParser().parsebytes(
            archive.read(metadata_files[0].removesuffix("METADATA") + "WHEEL")
        )
        return Wheel(
            path.name,
            re.sub(r"[-_.]+", "-", str(metadata["Name"])).lower(),
            str(metadata["Version"]),
            digest(path),
            path.stat().st_size,
            sum(item.file_size for item in archive.infolist()),
            tuple(wheel_metadata.get_all("Tag", ())),
            tuple(metadata.get_all("Provides-Extra", ())),
            tuple(name for name in archive.namelist() if name.endswith((".so", ".pyd", ".dylib"))),
        )


def file_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink())


def git_metadata(source: Path) -> dict[str, object]:
    if not source.is_dir():
        return {"path": str(source), "commit": None, "dirty": None}
    head: Final = subprocess.run(
        ("git", "-C", str(source), "rev-parse", "HEAD"),
        capture_output=True,
        text=True,
    )
    if head.returncode:
        return {"path": str(source), "commit": None, "dirty": None}
    status: Final = subprocess.run(
        ("git", "-C", str(source), "status", "--porcelain"),
        capture_output=True,
        text=True,
        check=True,
    )
    return {"path": str(source), "commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}


def lock_text(wheels: Sequence[Wheel], extras: str) -> str:
    return "".join(
        f"{wheel.name}{f'[{extras}]' if wheel.name == 'litellm' and extras else ''}"
        f"=={wheel.version} --hash=sha256:{wheel.sha256}\n"
        for wheel in sorted(wheels, key=lambda item: item.name)
    )


def snapshot(source: Path, destination: Path) -> None:
    files: Final = subprocess.run(
        ("git", "-C", str(source), "ls-files", "--cached", "--others", "--exclude-standard", "-z"),
        capture_output=True,
        check=False,
    )
    if files.returncode:
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns(".git", ".venv", "venv", "__pycache__", "target", "dist", "build"),
        )
        return
    destination.mkdir()
    for original, copied in (
        (source / relative, destination / relative)
        for relative in frozenset(os.fsdecode(name) for name in files.stdout.split(b"\0") if name)
    ):
        if not original.exists() and not original.is_symlink():
            continue
        copied.parent.mkdir(parents=True, exist_ok=True)
        if original.is_symlink():
            require(original.resolve().is_relative_to(source.resolve()), f"Symlink escapes local source: {original}")
            copied.symlink_to(
                os.path.relpath(destination / original.resolve().relative_to(source.resolve()), copied.parent)
            )
            continue
        require(original.is_file(), f"Local snapshot needs a populated source tree, not a Git submodule: {original}")
        shutil.copy2(original, copied)


def prepare(
    source: Source,
    output: Path,
    work: Path,
    extras: str,
    constraints: Path | None,
    wheelhouse: Path | None,
    environment: Mapping[str, str],
    log: TextIO,
) -> tuple[Wheel, ...]:
    destination: Final = output / "wheelhouse"
    destination.mkdir()
    root_wheels: Final = work / "root-wheel"
    root_wheels.mkdir()
    offline: Final = ("--no-index", "--find-links", str(wheelhouse)) if wheelhouse else ()
    if source.kind == "local":
        copied: Final = work / "source"
        snapshot(Path(source.value).resolve(), copied)
        command(
            (
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-cache-dir",
                "--wheel-dir",
                str(root_wheels),
                str(copied),
            ),
            work,
            environment,
            log,
            timeout=1800,
        )
    elif source.kind == "package":
        command(
            (
                sys.executable,
                "-m",
                "pip",
                "download",
                "--no-deps",
                "--only-binary=:all:",
                "--dest",
                str(root_wheels),
                *offline,
                f"litellm=={source.value}",
            ),
            work,
            environment,
            log,
        )
    else:
        wheel: Final = Path(source.value).resolve()
        shutil.copy2(wheel, root_wheels / wheel.name)
    roots: Final = tuple(root_wheels.glob("*.whl"))
    require(len(roots) == 1 and wheel_info(roots[0]).name == "litellm", "Source must produce one LiteLLM wheel")
    require(not extras or set(extras.split(",")).issubset(wheel_info(roots[0]).extras), "Unknown installation extra")
    command(
        (
            sys.executable,
            "-m",
            "pip",
            "download",
            "--only-binary=:all:",
            "--dest",
            str(destination),
            *(("--constraint", str(constraints)) if constraints else ()),
            *offline,
            f"{roots[0]}{f'[{extras}]' if extras else ''}",
        ),
        work,
        environment,
        log,
    )
    wheels: Final = tuple(wheel_info(path) for path in sorted(destination.glob("*.whl")))
    require(len({wheel.name for wheel in wheels}) == len(wheels), "Multiple wheels for one distribution")
    (output / "requirements.lock").write_text(lock_text(wheels, extras))
    (output / "constraints.txt").write_text(
        "".join(f"{wheel.name}=={wheel.version}\n" for wheel in wheels if wheel.name != "litellm")
    )
    return wheels


def install_sample(
    index: int,
    output: Path,
    work: Path,
    environment: Mapping[str, str],
    log: TextIO,
) -> tuple[Path, float, int, int]:
    target: Final = work / f"target-{index}"
    command((sys.executable, "-I", "-m", "venv", "--without-pip", str(target)), work, environment, log)
    before: Final = file_bytes(target)
    started: Final = time.perf_counter_ns()
    command(
        (
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(target),
            "install",
            "--no-index",
            "--no-cache-dir",
            "--only-binary=:all:",
            "--require-hashes",
            "--compile",
            "--find-links",
            str(output / "wheelhouse"),
            "--report",
            str(output / f"install-{index}.json"),
            "-r",
            str(output / "requirements.lock"),
        ),
        work,
        environment,
        log,
    )
    elapsed: Final = (time.perf_counter_ns() - started) / 1e9
    after: Final = file_bytes(target)
    return target / "bin" / "python", elapsed, before, after


def positive_int(value: str) -> int:
    parsed: Final = int(value)
    require(parsed > 0, "Counts and timeouts must be positive")
    return parsed


def arguments() -> Options:
    parser: Final = argparse.ArgumentParser(
        description="Benchmark a local LiteLLM build, a pinned published package, or an existing wheel"
    )
    sources: Final = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument(
        "--local",
        dest="source",
        type=lambda value: Source("local", value),
        nargs="?",
        const=Source("local", "."),
        help="Build a private copy of a local checkout (omit path for cwd)",
    )
    sources.add_argument(
        "--package",
        dest="source",
        type=lambda value: Source("package", value),
        help="Download exactly this published LiteLLM version; never build from source",
    )
    sources.add_argument(
        "--wheel",
        dest="source",
        type=lambda value: Source("wheel", value),
        help="Use this existing wheel; never build from source",
    )
    parser.add_argument("--output", type=Path, required=True, help="New artifact directory; existing paths are refused")
    parser.add_argument("--extras", default="", help="Comma-separated installation extras, e.g. proxy")
    parser.add_argument("--samples", type=positive_int, default=10, help="Fresh-process timing samples (default: 10)")
    parser.add_argument(
        "--install-samples", type=positive_int, default=3, help="Pristine offline installs (default: 3)"
    )
    parser.add_argument("--warmups", type=positive_int, default=1, help="Untimed workflow warmups (default: 1)")
    parser.add_argument("--timeout", type=positive_int, default=120, help="Seconds allowed per runtime probe")
    parser.add_argument("--constraints", type=Path, help="Pinned dependency constraints for implementation comparisons")
    parser.add_argument("--wheelhouse", type=Path, help="Resolve dependencies offline from an archived wheelhouse")
    return TypeAdapter(Options).validate_python(vars(parser.parse_args()))


def benchmark(options: Options, output: Path, work: Path, log: TextIO) -> dict[str, object]:
    source: Final = options.source
    if source.kind == "package":
        require(bool(re.fullmatch(r"[0-9][A-Za-z0-9.!+_-]*", source.value)), "--package requires an exact version")
    elif source.kind == "local":
        require(Path(source.value).is_dir(), "--local must name a source directory")
    else:
        require(Path(source.value).is_file() and source.value.endswith(".whl"), "--wheel must name a wheel file")
    require(bool(re.fullmatch(r"[A-Za-z0-9_,-]*", options.extras)), "Invalid extras")
    provenance: Final = {
        "kind": source.kind,
        "requested": source.value,
        "built_from_source": source.kind == "local",
        **(git_metadata(Path(source.value).resolve()) if source.kind != "package" else {}),
    }
    harness: Final = {
        **git_metadata(Path(__file__).resolve().parent),
        "files": {name: digest(Path(__file__).with_name(name)) for name in ("bench_sdk.py", "bench_sdk_runtime.py")},
        "tools": {name: importlib.metadata.version(name) for name in ("pip", "pyperf", "psutil", "pydantic")},
    }
    home: Final = work / "home"
    home.mkdir()
    runtime: Final = work / "runtime"
    runtime.mkdir()
    environment: Final = {
        **{key: value for key, value in os.environ.items() if not key.startswith(("PIP_", "PYTHON"))},
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
        "PIP_CACHE_DIR": str(work / "pip-cache"),
        "CARGO_TARGET_DIR": str(work / "cargo-target"),
    }
    sys.stderr.write(f"Source: {source.kind} ({source.value})\n")
    sys.stderr.write(
        "Building a private source copy, then resolving binary dependencies...\n"
        if source.kind == "local"
        else "Resolving binary wheels only; source builds disabled...\n"
    )
    wheels: Final = prepare(
        source,
        output,
        work,
        options.extras,
        options.constraints.resolve() if options.constraints else None,
        options.wheelhouse.resolve() if options.wheelhouse else None,
        environment,
        log,
    )
    sys.stderr.write("Measuring pristine offline installs...\n")
    installs: Final = tuple(
        install_sample(index, output, work, environment, log) for index in range(options.install_samples)
    )
    python: Final = installs[-1][0]
    command((sys.executable, "-m", "pip", "--python", str(python), "check"), work, environment, log)
    inventory: Final = command(
        (sys.executable, "-m", "pip", "--python", str(python), "inspect"), work, environment, log
    )
    (output / "installed.json").write_text(inventory)
    runtime_env: Final = runtime_environment(home)
    sys.stderr.write("Measuring fresh-process imports and local responses...\n")
    with provider() as (base_url, requests):
        for _ in range(options.warmups):
            probe(python, base_url, runtime, runtime_env, log, requests, timeout=options.timeout)
        samples: Final = tuple(
            probe(python, base_url, runtime, runtime_env, log, requests, timeout=options.timeout)[0]
            for _ in range(options.samples)
        )
        _, diagnostics = probe(
            python, base_url, runtime, runtime_env, log, requests, diagnostic=True, timeout=options.timeout
        )
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    startup_metrics: Final = startup(python, output, runtime, runtime_env, log, options.samples, options.timeout)
    with (output / "importtime.log").open("w") as profile:
        command(
            (str(python), "-I", "-B", "-X", "importtime", "-c", PROBE, "import_exit"),
            runtime,
            runtime_env,
            profile,
            options.timeout,
        )
    log.flush()
    root: Final = next(wheel for wheel in wheels if wheel.name == "litellm")
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": provenance,
        "harness": harness,
        "environment": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "extras": options.extras,
            "runtime_variables": runtime_env,
            "bytecode": "pip --compile, probes -B (read existing pyc, never write)",
            "filesystem_cache": "warmup runs; OS page cache not flushed",
            "network": "Python socket audit guard allows loopback only; not an OS sandbox",
            "scenario": "synchronous non-streaming completion; fixed loopback provider; retries disabled",
            "warmups": options.warmups,
            "constraints_sha256": digest(options.constraints) if options.constraints else None,
        },
        "artifacts": {"root": asdict(root), "wheels": tuple(asdict(wheel) for wheel in wheels)},
        "sizes": {
            "root_wheel_bytes": root.compressed_bytes,
            "root_uncompressed_bytes": root.uncompressed_bytes,
            "resolved_wheelhouse_bytes": sum({wheel.sha256: wheel.compressed_bytes for wheel in wheels}.values()),
            "dependency_wheel_bytes": sum(wheel.compressed_bytes for wheel in wheels if wheel.name != "litellm"),
            "environment_before_bytes": tuple(before for _, _, before, _ in installs),
            "environment_after_bytes": tuple(after for _, _, _, after in installs),
            "installed_delta_bytes": tuple(after - before for _, _, before, after in installs),
        },
        "timings": {
            "offline_install": summary(tuple(elapsed for _, elapsed, _, _ in installs), "seconds"),
            **{
                key.removesuffix("_ns"): summary(tuple(sample[key] / 1e9 for sample in samples), "seconds")
                for key in samples[0]
            },
            **startup_metrics,
        },
        "raw_workflow_samples_ns": samples,
        "memory": tuple({key: value for key, value in stage.items() if key != "modules"} for stage in diagnostics),
        "diagnostics": "diagnostics.json",
    }


def main() -> None:
    options: Final = arguments()
    require(sys.platform in ("linux", "darwin"), "This benchmark currently supports Linux and macOS")
    output: Final = options.output.resolve()
    require(not output.exists(), f"Output already exists: {output}")
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="litellm-sdk-bench-") as temporary, (output / "run.log").open("w") as log:
        result: Final = benchmark(options, output, Path(temporary), log)
        temporary_result: Final = output / "result.json.tmp"
        temporary_result.write_text(json.dumps(result, indent=2) + "\n")
        temporary_result.replace(output / "result.json")
    sys.stdout.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

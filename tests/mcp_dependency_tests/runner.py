# /// script
# requires-python = ">=3.12"
# dependencies = ["packaging==26.0"]
# ///

import argparse
import email
from email.message import Message
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import tomllib
from typing import Final
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

HERE: Final = Path(__file__).resolve().parent
ROOT: Final = HERE.parents[1]
PROFILES: Final = ("core", "mcp", "proxy")
MODES: Final = ("minimum", "locked")
COMPANIONS: Final = ("litellm-enterprise", "litellm-proxy-extras")


def wheel_metadata(wheel: Path) -> Message:
    with zipfile.ZipFile(wheel) as archive:
        names: Final = tuple(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        if len(names) != 1:
            raise ValueError("expected exactly one wheel METADATA file")
        return email.message_from_bytes(archive.read(names[0]))


def wheel_project(wheel: Path) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    metadata: Final = wheel_metadata(wheel)
    if metadata["Name"] != "litellm":
        raise ValueError("expected a litellm wheel")
    return (
        str(metadata["Requires-Python"]),
        tuple(str(value) for value in metadata.get_all("Requires-Dist", [])),
        tuple(str(value) for value in metadata.get_all("Provides-Extra", [])),
    )


def companions(wheel: Path, profile: str) -> tuple[Path, ...]:
    if profile != "proxy":
        return ()
    paths: Final = tuple(tuple(wheel.parent.glob(f"{name.replace('-', '_')}-*.whl")) for name in COMPANIONS)
    if any(len(matches) != 1 for matches in paths):
        raise ValueError("build exactly one enterprise and proxy-extras companion wheel beside the litellm wheel")
    return tuple(matches[0] for matches in paths)


def project_text(wheel: Path, profile: str, root: Path = ROOT) -> str:
    python_range, requirements, extras = wheel_project(wheel)
    if profile != "core" and profile not in extras:
        raise ValueError(f"wheel does not provide extra {profile}")
    policy: Final = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["uv"]
    candidate: Final = tomllib.loads((HERE / "candidate.toml").read_text())
    additions: Final = tuple(candidate["dependencies"]) if profile != "core" else ()
    overrides: Final = tuple(policy.get("override-dependencies", ())) + (
        tuple(candidate["overrides"]) if profile != "core" else ()
    )
    local_requirements: Final = tuple(
        f"{wheel_metadata(path)['Name']} @ file://__WHEEL_DIR__/{path.name}" for path in companions(wheel, profile)
    )
    local_metadata: Final = tuple(
        {
            field: tuple(str(value) for value in wheel_metadata(path).get_all(field, []))
            for field in ("Name", "Version", "Requires-Python", "Requires-Dist", "Provides-Extra")
        }
        for path in companions(wheel, profile)
    )
    return "\n".join(
        (
            "[project]",
            'name = "litellm-dependency-candidate"',
            'version = "0"',
            f"requires-python = {json.dumps(python_range)}",
            f"dependencies = {json.dumps(requirements + additions + local_requirements)}",
            "[project.optional-dependencies]",
            *(f"{json.dumps(extra)} = []" for extra in extras),
            "[tool.uv]",
            f"constraint-dependencies = {json.dumps(policy.get('constraint-dependencies', []))}",
            f"override-dependencies = {json.dumps(overrides)}",
            "[tool.mcp-dependency-gate]",
            f"exclude-newer = {json.dumps(candidate['exclude-newer'])}",
            f"companion-metadata = {json.dumps(json.dumps(local_metadata, sort_keys=True))}",
            "",
        )
    )


def fingerprint(project: str, profile: str, mode: str) -> str:
    return hashlib.sha256(f"{profile}\n{mode}\n{project}".encode()).hexdigest()


def run(command: tuple[str, ...], cwd: Path) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def lock(wheel: Path, profile: str, mode: str, snapshots: Path) -> None:
    project: Final = project_text(wheel, profile)
    cutoff: Final = tomllib.loads((HERE / "candidate.toml").read_text())["exclude-newer"]
    snapshots.mkdir(parents=True, exist_ok=True)
    destination: Final = snapshots / f"{profile}-{mode}.txt"
    with tempfile.TemporaryDirectory(prefix="mcp-lock-") as temporary:
        work: Final = Path(temporary)
        (work / "pyproject.toml").write_text(project.replace("file://__WHEEL_DIR__", wheel.parent.as_uri()))
        run(
            (
                "uv",
                "pip",
                "compile",
                str(work / "pyproject.toml"),
                *(("--extra", profile) if profile != "core" else ()),
                "--universal",
                "--python-version",
                "3.10",
                "--generate-hashes",
                "--no-header",
                "--no-annotate",
                "--resolution",
                "lowest-direct" if mode == "minimum" else "highest",
                "--exclude-newer",
                cutoff,
                "--output-file",
                str(work / "requirements.txt"),
                *(argument for name in COMPANIONS for argument in ("--no-emit-package", name)),
            ),
            work,
        )
        locked: Final = (work / "requirements.txt").read_text()
        destination.write_text(
            f"# inputs-sha256: {fingerprint(project, profile, mode)}\n# exclude-newer: {cutoff}\n" + locked
        )


def validate_snapshot(snapshot: str, project: str, profile: str, mode: str) -> None:
    if not snapshot.startswith(f"# inputs-sha256: {fingerprint(project, profile, mode)}\n"):
        raise ValueError("snapshot is stale for this wheel/policy; regenerate with lock")


def locked_versions(snapshot: str, environment: dict[str, str]) -> dict[str, str]:
    requirements: Final = tuple(
        Requirement(line.split("\\", 1)[0].strip())
        for line in snapshot.splitlines()
        if line and not line[0].isspace() and not line.startswith("#")
    )
    return {
        canonicalize_name(requirement.name): next(iter(requirement.specifier)).version
        for requirement in requirements
        if requirement.marker is None or requirement.marker.evaluate(environment)
    }


def verify_inventory(snapshot: str, report: dict[str, object], local_versions: dict[str, str]) -> None:
    environment: Final = report["environment"]
    installed: Final = report["installed"]
    if not isinstance(environment, dict) or not isinstance(installed, dict):
        raise ValueError("invalid environment inventory")
    expected: Final = locked_versions(snapshot, environment) | local_versions
    if installed != expected:
        raise ValueError(f"installed packages do not match snapshot: expected {expected}, got {installed}")


def check(wheel: Path, profile: str, mode: str, snapshots: Path, python: str, environment: Path) -> None:
    snapshot: Final = snapshots / f"{profile}-{mode}.txt"
    text: Final = snapshot.read_text()
    validate_snapshot(text, project_text(wheel, profile), profile, mode)
    if environment.exists():
        raise ValueError("use a new environment path; existing environments are never modified")
    environment.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mcp-install-") as temporary:
        work: Final = Path(temporary)
        pinned_python: Final = tomllib.loads((HERE / "candidate.toml").read_text())["python"][python]
        run(("uv", "venv", str(environment), "--python", pinned_python), work)
        executable: Final = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run(("uv", "pip", "sync", "--python", str(executable), "--require-hashes", str(snapshot)), work)
        local_wheels: Final = (wheel,) + companions(wheel, profile)
        run(
            ("uv", "pip", "install", "--python", str(executable), "--no-deps", *(str(path) for path in local_wheels)),
            work,
        )
        run((str(executable), "-I", str(HERE / "check_environment.py"), profile, str(environment)), work)
        report: Final = json.loads((environment / "report.json").read_text())
        verify_inventory(
            text,
            report,
            {
                canonicalize_name(str(wheel_metadata(path)["Name"])): str(wheel_metadata(path)["Version"])
                for path in local_wheels
            },
        )
        if profile == "core":
            run((str(executable), "-I", str(ROOT / "tests/base_sdk_tests/check_base_sdk_install.py")), work)
    print(f"PASS {profile}/{mode} on Python {python}: {environment}")


def main() -> None:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("action", choices=("lock", "check"))
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--snapshots", type=Path, default=HERE / "locks")
    parser.add_argument("--python", choices=("3.10", "3.11", "3.12", "3.13", "3.14"), default="3.12")
    parser.add_argument("--environment", type=Path)
    args: Final = parser.parse_args()
    if args.action == "lock":
        lock(args.wheel.resolve(), args.profile, args.mode, args.snapshots.resolve())
    else:
        if args.environment is None:
            parser.error("check requires --environment")
        check(
            args.wheel.resolve(),
            args.profile,
            args.mode,
            args.snapshots.resolve(),
            args.python,
            args.environment.resolve(),
        )


if __name__ == "__main__":
    main()

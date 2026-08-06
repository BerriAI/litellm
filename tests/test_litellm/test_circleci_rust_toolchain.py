"""Static guardrails for how CircleCI provisions Rust.

The root package builds `litellm-rust` through maturin, so any job that runs
`uv sync` or `uv build` compiles the bridge. The `cimg/python` images ship no
Rust toolchain, and when cargo is missing maturin's `puccinialin` helper
quietly provisions one itself: it fetches `rustup-init` from the unversioned
`https://static.rust-lang.org/rustup/dist/<triple>/` path with no checksum and
installs a floating `stable` toolchain. uv suppresses build-backend output on a
successful sync, so that happens with nothing in the job log to show for it,
and the compiler a job builds with changes whenever upstream publishes.

Two invariants are pinned here:

  1. No step list (job or reusable command) reaches a `uv sync` / `uv build`
     without a Rust toolchain already provisioned ahead of it. That is the
     `install_rust` command on Linux and an inline pinned rustup install in the
     Windows job, so the check accepts either. A new job that syncs without one
     falls back to the unpinned path, which is exactly the regression a static
     check catches at PR time and a green CI run does not.
  2. `install_rust` itself pins what it downloads: an explicit rustup version in
     the URL, a verified SHA-256, and an exact toolchain version rather than a
     channel name.

The Windows job predates `install_rust` and provisions its toolchain inline, so
invariant 2 is scoped to `install_rust`; invariant 1 covers both.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / ".circleci" / "config.yml"

BUILDS_WORKSPACE = re.compile(r"\buv\s+(?:sync|build)\b")
RUSTUP_ARCHIVE_URL = re.compile(r"https://static\.rust-lang\.org/rustup/archive/\d+\.\d+\.\d+/")
EXACT_TOOLCHAIN = re.compile(r"--default-toolchain\s+\"?\d+\.\d+\.\d+\"?")


def _config() -> dict[str, object]:
    return yaml.safe_load(CONFIG.read_text())


def _step_text(step: object) -> str:
    """Flatten one step into the shell text it runs, or '' for a command reference."""
    if isinstance(step, dict):
        run = step.get("run")
        if isinstance(run, str):
            return run
        if isinstance(run, dict):
            command = run.get("command")
            return command if isinstance(command, str) else ""
    return ""


def _without_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _provisions_rust(step: object) -> bool:
    if step == "install_rust":
        return True
    text = _step_text(step)
    return "rustup-init" in text and ("sha256sum" in text or "SHA256" in text)


def _step_lists() -> dict[str, list[object]]:
    config = _config()
    lists: dict[str, list[object]] = {}
    for kind in ("jobs", "commands"):
        section = config.get(kind)
        if not isinstance(section, dict):
            continue
        for name, body in section.items():
            steps = body.get("steps") if isinstance(body, dict) else None
            if isinstance(steps, list):
                lists[f"{kind[:-1]} {name}"] = steps
    return lists


def _first_unprovisioned_build(steps: list[object]) -> str | None:
    """Return the shell text of the first workspace build reached without Rust, if any."""
    rust_ready = False
    for step in steps:
        if _provisions_rust(step):
            rust_ready = True
        text = _step_text(step)
        if BUILDS_WORKSPACE.search(_without_comments(text)) and not rust_ready:
            return text
    return None


def test_step_lists_exist() -> None:
    lists = _step_lists()
    assert "command install_rust" in lists
    building = {
        name
        for name, steps in lists.items()
        if any(BUILDS_WORKSPACE.search(_without_comments(_step_text(s))) for s in steps)
    }
    assert len(building) > 10, f"expected many workspace-building step lists, found {sorted(building)}"


def test_no_workspace_build_without_a_provisioned_rust_toolchain() -> None:
    offenders = {
        name: build for name, steps in _step_lists().items() if (build := _first_unprovisioned_build(steps)) is not None
    }
    assert not offenders, (
        "these CircleCI step lists run `uv sync`/`uv build` with no Rust toolchain provisioned first, "
        "so maturin will download an unpinned rustup and a floating toolchain instead: "
        f"{ {name: build.strip().splitlines()[0] for name, build in offenders.items()} }"
    )


@pytest.fixture(name="install_rust_command")
def _install_rust_command() -> str:
    steps = _step_lists()["command install_rust"]
    return "\n".join(_step_text(step) for step in steps)


def test_install_rust_pins_the_rustup_version_in_the_url(install_rust_command: str) -> None:
    assert RUSTUP_ARCHIVE_URL.search(install_rust_command), (
        "install_rust must download rustup-init from a version-pinned /rustup/archive/<x.y.z>/ URL; "
        "the /rustup/dist/ path always serves whatever rustup is current"
    )
    assert "/rustup/dist/" not in install_rust_command


def test_install_rust_verifies_the_installer_checksum(install_rust_command: str) -> None:
    assert "sha256sum -c" in install_rust_command
    assert re.search(r"RUSTUP_SHA256=[0-9a-f]{64}\b", install_rust_command), (
        "install_rust must compare the downloaded installer against a hardcoded SHA-256 "
        "taken from rust-lang's published .sha256 sidecar"
    )
    checksum_index = install_rust_command.index("sha256sum -c")
    execute_index = install_rust_command.index("/tmp/rustup-init -y")
    assert checksum_index < execute_index, "the checksum must be verified before the installer is executed"


def test_install_rust_pins_an_exact_toolchain_version(install_rust_command: str) -> None:
    assert EXACT_TOOLCHAIN.search(install_rust_command), (
        "install_rust must pin an exact toolchain version (e.g. 1.97.1); a channel name like "
        "stable/beta/nightly makes the compiler drift with whatever upstream published that day"
    )

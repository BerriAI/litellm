import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "with_dashboard_node.sh"


def _floor() -> str:
    pkg = json.loads((ROOT / "ui" / "litellm-dashboard" / "package.json").read_text())
    return pkg["engines"]["node"].removeprefix(">=")


def _bump_major(version: str, delta: int) -> str:
    major, minor, patch = version.split(".")
    return f"{int(major) + delta}.{minor}.{patch}"


def _fake_node(bin_dir: Path, version: str) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    node = bin_dir / "node"
    node.write_text(f'#!/bin/sh\necho "v{version}"\n')
    node.chmod(0o755)
    return bin_dir


def _run(bin_dirs: list[Path], home: Path) -> subprocess.CompletedProcess[str]:
    path = os.pathsep.join([*(str(b) for b in bin_dirs), "/usr/bin", "/bin"])
    home.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [str(SCRIPT), "sh", "-c", "node --version"],
        capture_output=True,
        text=True,
        env={"PATH": path, "HOME": str(home)},
    )


def test_node_meeting_the_floor_runs_the_command_as_is(tmp_path):
    bins = _fake_node(tmp_path / "bin", _floor())
    proc = _run([bins], tmp_path / "home")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"v{_floor()}"


def test_node_above_the_floor_runs_the_command_as_is(tmp_path):
    above = _bump_major(_floor(), 1)
    bins = _fake_node(tmp_path / "bin", above)
    proc = _run([bins], tmp_path / "home")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"v{above}"


def test_old_node_without_any_manager_fails_with_instructions(tmp_path):
    bins = _fake_node(tmp_path / "bin", _bump_major(_floor(), -1))
    proc = _run([bins], tmp_path / "home")
    assert proc.returncode == 1
    assert "does not meet" in proc.stderr
    assert _floor() in proc.stderr
    assert "nvm" in proc.stderr


def test_missing_node_without_any_manager_fails_with_instructions(tmp_path):
    proc = _run([], tmp_path / "home")
    assert proc.returncode == 1
    assert "missing" in proc.stderr


def test_old_node_switches_via_nvm_when_present(tmp_path):
    old = _fake_node(tmp_path / "old-bin", _bump_major(_floor(), -1))
    new = _fake_node(tmp_path / "new-bin", "99.0.0")
    home = tmp_path / "home"
    nvm_dir = home / ".nvm"
    nvm_dir.mkdir(parents=True)
    (nvm_dir / "nvm.sh").write_text(
        f'nvm() {{ [ "$1" = use ] && PATH="{new}:$PATH"; return 0; }}\n'
    )
    proc = _run([old], home)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "v99.0.0"
    assert "via nvm" in proc.stderr


def test_old_node_switches_via_fnm_when_nvm_is_absent(tmp_path):
    old = _fake_node(tmp_path / "old-bin", _bump_major(_floor(), -1))
    new = _fake_node(tmp_path / "new-bin", "99.0.0")
    fnm = tmp_path / "old-bin" / "fnm"
    fnm.write_text(
        f'#!/bin/sh\n[ "$1" = env ] && echo \'export PATH="{new}:$PATH"\'\nexit 0\n'
    )
    fnm.chmod(0o755)
    proc = _run([old], tmp_path / "home")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "v99.0.0"
    assert "via fnm" in proc.stderr


def _nvm_home(tmp_path, nvm_sh: str) -> Path:
    home = tmp_path / "home"
    nvm_dir = home / ".nvm"
    nvm_dir.mkdir(parents=True)
    (nvm_dir / "nvm.sh").write_text(nvm_sh)
    return home


def test_failing_nvm_load_stops_before_running_the_command(tmp_path):
    old = _fake_node(tmp_path / "old-bin", _bump_major(_floor(), -1))
    proc = _run([old], _nvm_home(tmp_path, "false\n"))
    assert proc.returncode == 1
    assert "could not load nvm" in proc.stderr
    assert proc.stdout == ""


def test_failing_nvm_install_stops_before_running_the_command(tmp_path):
    old = _fake_node(tmp_path / "old-bin", _bump_major(_floor(), -1))
    proc = _run([old], _nvm_home(tmp_path, 'nvm() { [ "$1" = install ] && return 1; return 0; }\n'))
    assert proc.returncode == 1
    assert "nvm install" in proc.stderr
    assert proc.stdout == ""


def test_failing_nvm_use_stops_before_running_the_command(tmp_path):
    old = _fake_node(tmp_path / "old-bin", _bump_major(_floor(), -1))
    proc = _run([old], _nvm_home(tmp_path, 'nvm() { [ "$1" = use ] && return 1; return 0; }\n'))
    assert proc.returncode == 1
    assert "nvm use" in proc.stderr
    assert proc.stdout == ""


def test_no_command_is_a_usage_error(tmp_path):
    proc = subprocess.run(
        [str(SCRIPT)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert proc.returncode == 2
    assert "usage" in proc.stderr

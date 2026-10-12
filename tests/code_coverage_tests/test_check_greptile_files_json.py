import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

GUARD: Final = Path(__file__).resolve().parent / "check_greptile_files_json.py"
FILES_JSON: Final = Path(".greptile/files.json")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _write_agents_md(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# conventions\n")


def _repo_with(tmp_path: Path, files: tuple[str, ...]) -> Path:
    guard_copy: Final = tmp_path / "tests" / "code_coverage_tests" / GUARD.name
    guard_copy.parent.mkdir(parents=True)
    shutil.copy(GUARD, guard_copy)
    for relative_path in files:
        _write_agents_md(tmp_path / relative_path)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    return guard_copy


def _run(guard: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-I", str(guard), *args], capture_output=True, text=True)


def test_write_scopes_each_nested_agents_md_to_its_own_directory(tmp_path: Path) -> None:
    guard: Final = _repo_with(tmp_path, ("AGENTS.md", "tests/AGENTS.md", "litellm/_v2/cache/AGENTS.md"))

    assert _run(guard, "--write").returncode == 0

    written: Final = json.loads((tmp_path / FILES_JSON).read_text())
    assert [(entry["path"], entry["scope"]) for entry in written["files"]] == [
        ("litellm/_v2/cache/AGENTS.md", ["litellm/_v2/cache/**"]),
        ("tests/AGENTS.md", ["tests/**"]),
    ]
    assert _run(guard).returncode == 0


def test_new_agents_md_missing_from_files_json_fails(tmp_path: Path) -> None:
    guard: Final = _repo_with(tmp_path, ("tests/AGENTS.md",))
    assert _run(guard, "--write").returncode == 0
    _write_agents_md(tmp_path / "ui" / "AGENTS.md")
    _git(tmp_path, "add", "-A")

    result: Final = _run(guard)

    assert result.returncode == 1
    assert '+      "path": "ui/AGENTS.md",' in result.stdout


def test_deleted_agents_md_left_in_files_json_fails(tmp_path: Path) -> None:
    guard: Final = _repo_with(tmp_path, ("tests/AGENTS.md", "ui/AGENTS.md"))
    assert _run(guard, "--write").returncode == 0
    _git(tmp_path, "rm", "-q", "-f", "ui/AGENTS.md")

    result: Final = _run(guard)

    assert result.returncode == 1
    assert '-      "path": "ui/AGENTS.md",' in result.stdout


def test_missing_files_json_fails(tmp_path: Path) -> None:
    guard: Final = _repo_with(tmp_path, ("tests/AGENTS.md",))

    assert _run(guard).returncode == 1

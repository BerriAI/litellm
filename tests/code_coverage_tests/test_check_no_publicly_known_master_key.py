import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

GUARD: Final = Path(__file__).resolve().parent / "check_no_publicly_known_master_key.py"
RETIRED_KEY: Final = "sk-" + "1234"
BUNDLE_CHUNK: Final = "litellm/proxy/_experimental/out/_next/static/chunks/chunk.js"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo_with(tmp_path: Path, files: dict[str, str]) -> Path:
    guard_copy: Final = tmp_path / "tests" / "code_coverage_tests" / GUARD.name
    guard_copy.parent.mkdir(parents=True)
    shutil.copy(GUARD, guard_copy)
    for relative_path, contents in files.items():
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    return guard_copy


def _run(guard: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(guard)], capture_output=True, text=True)


def test_generated_dashboard_bundle_is_not_flagged(tmp_path: Path) -> None:
    guard: Final = _repo_with(tmp_path, {BUNDLE_CHUNK: f'api_key="{RETIRED_KEY}"\n', "docs/a.md": "clean\n"})

    result: Final = _run(guard)

    assert result.returncode == 0, result.stdout
    assert BUNDLE_CHUNK not in result.stdout


@pytest.mark.parametrize(
    "source_path",
    ["litellm/proxy/proxy_server.py", "ui/litellm-dashboard/src/app/page.tsx", "litellm/proxy/_experimental/x.py"],
)
def test_source_files_are_still_flagged(tmp_path: Path, source_path: str) -> None:
    guard: Final = _repo_with(
        tmp_path,
        {BUNDLE_CHUNK: f'api_key="{RETIRED_KEY}"\n', source_path: f'key = "{RETIRED_KEY}"\n'},
    )

    result: Final = _run(guard)

    assert result.returncode == 1
    assert f"{source_path}:1" in result.stdout
    assert BUNDLE_CHUNK not in result.stdout

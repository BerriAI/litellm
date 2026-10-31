import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

CHECKER: Final = Path(__file__).resolve().parents[2] / "scripts/check_schema_compatibility.py"
SCHEMA: Final = "model Item {\n id String @id\n label String?\n}\n"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(root), *args), capture_output=True, text=True, check=True, timeout=30
    ).stdout.strip()


@pytest.mark.parametrize("baseline_option", ("--base-ref", "--base-commit"))
@pytest.mark.parametrize(("schema", "expected"), ((SCHEMA, 0), ("model Item {\n id String @id\n}\n", 1)))
def test_schema_check_preserves_break_detection_with_or_without_history(
    tmp_path: Path, schema: str, expected: int, baseline_option: str
) -> None:
    source: Final = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "--initial-branch=main")
    _git(source, "config", "user.name", "Schema test")
    _git(source, "config", "user.email", "schema-test@example.invalid")
    _git(source, "config", "commit.gpgsign", "false")
    (source / "schema.prisma").write_text(SCHEMA)
    _git(source, "add", "schema.prisma")
    _git(source, "commit", "-m", "baseline")
    baseline: Final = _git(source, "rev-parse", "HEAD")
    (source / "schema.prisma").write_text(schema)
    _git(source, "add", "schema.prisma")
    _git(source, "commit", "--allow-empty", "-m", "candidate")
    checkout: Final = tmp_path / "checkout"
    _git(tmp_path, "clone", "--depth=1", source.as_uri(), str(checkout))
    _git(checkout, "fetch", "--no-tags", "--depth=1", "origin", baseline)
    if baseline_option == "--base-ref":
        _git(checkout, "fetch", "--unshallow", "origin", "main")

    result: Final = subprocess.run(
        (sys.executable, "-I", str(CHECKER), baseline_option, baseline),
        cwd=checkout,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == expected, result.stdout + result.stderr
    assert _git(checkout, "rev-parse", "--is-shallow-repository") == (
        "true" if baseline_option == "--base-commit" else "false"
    )
    expected_output: Final = (
        f"Schema compatibility passed against baseline {baseline}"
        if expected == 0
        else "Breaking: Field Item.label was removed"
    )
    assert expected_output in result.stdout + result.stderr

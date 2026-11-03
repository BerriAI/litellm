import importlib.util
import sys
from pathlib import Path

_CHECKER_PATH = Path(__file__).resolve().parents[1] / "code_coverage_tests" / "check_migrations_lock_timeout.py"
_SPEC = importlib.util.spec_from_file_location("check_migrations_lock_timeout", _CHECKER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
checker = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = checker
_SPEC.loader.exec_module(checker)


def _scan(tmp_path: Path, sql: str, name: str = "20261010000000_fixture") -> tuple[str, ...]:
    directory = tmp_path / name
    directory.mkdir()
    (directory / "migration.sql").write_text(sql, encoding="utf-8")
    return checker.scan_migration(directory)


def test_missing_lock_timeout_line_fails(tmp_path: Path) -> None:
    violations = _scan(tmp_path, 'ALTER TABLE "Foo" ADD COLUMN "bar" TEXT;')

    assert len(violations) == 1
    assert "SET LOCAL lock_timeout = '10s';" in violations[0]


def test_lock_timeout_line_first_passes(tmp_path: Path) -> None:
    assert _scan(tmp_path, "SET LOCAL lock_timeout = '10s';\nALTER TABLE foo ADD bar int;") == ()


def test_multiline_lock_timeout_statement_first_passes(tmp_path: Path) -> None:
    sql = "SET LOCAL\nlock_timeout = '10s';\nALTER TABLE foo ADD bar int;"

    assert _scan(tmp_path, sql) == ()


def test_comments_before_lock_timeout_line_pass(tmp_path: Path) -> None:
    sql = "-- migration setup\n\n  -- lock bound\nset local LOCK_TIMEOUT = '2s';\nALTER TABLE foo ADD bar int;"

    assert _scan(tmp_path, sql) == ()


def test_lock_timeout_line_after_another_statement_fails(tmp_path: Path) -> None:
    sql = "ALTER TABLE foo ADD bar int;\nSET LOCAL lock_timeout = '10s';"

    assert _scan(tmp_path, sql)


def test_session_lock_timeout_without_local_fails(tmp_path: Path) -> None:
    assert _scan(tmp_path, "SET lock_timeout = '10s';\nALTER TABLE foo ADD bar int;")


def test_grandfathered_migration_without_lock_timeout_passes(tmp_path: Path) -> None:
    name = "20261009000000_add_credential_display_name"

    assert _scan(tmp_path, "ALTER TABLE foo ADD bar int;", name) == ()


def test_concurrently_migration_without_lock_timeout_passes(tmp_path: Path) -> None:
    sql = "create index concurrently foo_bar_idx on foo (bar);"

    assert _scan(tmp_path, sql) == ()

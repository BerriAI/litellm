import importlib.util
import sys
import types
from pathlib import Path

if "testing.postgresql" not in sys.modules:
    _testing = types.ModuleType("testing")
    _testing_postgresql = types.ModuleType("testing.postgresql")
    _testing.postgresql = _testing_postgresql
    sys.modules.setdefault("testing", _testing)
    sys.modules.setdefault("testing.postgresql", _testing_postgresql)

REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "run_migration", REPO_ROOT / "ci_cd" / "run_migration.py"
)
run_migration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_migration)


def test_prepare_isolated_scratch_does_not_touch_repo_root_migrations(tmp_path):
    """Regression for #33135: scratch dir must not collide with repo migrations/.

    ``create_migration`` used ``schema_path.parent / "migrations"`` as its
    scratch directory and ``rmtree``'d it, wiping the tracked repo-root
    ``migrations/`` folder (Dockerfile, run.py). The scratch copy must live in
    an isolated temp dir instead.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    schema_path = repo_root / "schema.prisma"
    schema_path.write_text("// schema")

    # tracked repo-root migrations/ that must survive untouched
    tracked_migrations = repo_root / "migrations"
    tracked_migrations.mkdir()
    dockerfile = tracked_migrations / "Dockerfile"
    dockerfile.write_text("FROM scratch")
    run_py = tracked_migrations / "run.py"
    run_py.write_text("print('hi')")

    # the real migrations source that gets copied into the scratch dir
    source_migrations = repo_root / "litellm_proxy_extras" / "migrations"
    source_migrations.mkdir(parents=True)
    (source_migrations / "20240101000000_init").mkdir()
    (source_migrations / "20240101000000_init" / "migration.sql").write_text(
        "SELECT 1;"
    )

    temp_root, temp_schema_path = run_migration._prepare_isolated_scratch(
        schema_path, source_migrations
    )
    try:
        assert temp_root.resolve() != tracked_migrations.resolve()
        assert repo_root.resolve() not in temp_root.resolve().parents

        assert temp_schema_path.read_text() == "// schema"
        assert (temp_root / "migrations" / "20240101000000_init" / "migration.sql").read_text() == "SELECT 1;"

        assert dockerfile.read_text() == "FROM scratch"
        assert run_py.read_text() == "print('hi')"
        assert sorted(p.name for p in tracked_migrations.iterdir()) == [
            "Dockerfile",
            "run.py",
        ]
    finally:
        import shutil

        shutil.rmtree(temp_root, ignore_errors=True)

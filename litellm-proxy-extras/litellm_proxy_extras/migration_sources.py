import shutil
from itertools import chain
from pathlib import Path
from typing import Final

PACKAGE_DIR: Final = Path(__file__).parent
MIGRATION_SOURCES: Final = (PACKAGE_DIR / "migrations", PACKAGE_DIR / "spend_logs" / "migrations")


def shipped_migrations() -> tuple[Path, ...]:
    """Every packaged migration directory, in the name order `prisma migrate deploy` applies them.

    Migrations that touch LiteLLM_SpendLogs live under `spend_logs/migrations/` so that path can
    carry its own code owner; Prisma only reads one flat directory, so `materialize_prisma_dir`
    merges the sources back together before any Prisma command runs."""
    candidates: Final = chain.from_iterable(source.iterdir() for source in MIGRATION_SOURCES)
    return tuple(sorted((path for path in candidates if (path / "migration.sql").is_file()), key=lambda p: p.name))


def materialize_prisma_dir(target: Path) -> Path:
    """Write the flat Prisma layout (schema.prisma plus one migrations/ directory) into `target`."""
    migrations: Final = target / "migrations"
    migrations.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PACKAGE_DIR / "schema.prisma", target / "schema.prisma")
    shutil.copy2(PACKAGE_DIR / "migrations" / "migration_lock.toml", migrations / "migration_lock.toml")
    for migration in shipped_migrations():
        shutil.copytree(migration, migrations / migration.name, dirs_exist_ok=True)
    return target

#!/usr/bin/env python3

import re
import sys
from itertools import chain
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = REPO_ROOT / "litellm-proxy-extras" / "litellm_proxy_extras" / "migrations"
GRANDFATHERED_THROUGH: Final = "20261009000000_add_credential_display_name"
LOCK_TIMEOUT_LINE: Final = "SET LOCAL lock_timeout = '10s';"
LOCK_TIMEOUT_RE: Final = re.compile(
    r"SET\s+LOCAL\s+lock_timeout\s*=\s*'\d+(?:ms|s)';",
    re.IGNORECASE,
)


def first_statement(sql: str) -> str:
    executable_sql: Final = "\n".join(
        line for line in sql.splitlines() if line.strip() and not line.lstrip().startswith("--")
    )
    match: Final = re.search(r"(?s).*?;", executable_sql)
    return match.group().strip() if match is not None else executable_sql.strip()


def scan_migration(directory: Path) -> tuple[str, ...]:
    if directory.name <= GRANDFATHERED_THROUGH:
        return ()

    sql: Final = (directory / "migration.sql").read_text(encoding="utf-8")
    if "CONCURRENTLY" in sql.upper():
        return ()

    if LOCK_TIMEOUT_RE.fullmatch(first_statement(sql)):
        return ()

    return (
        f"{directory.name}/migration.sql: first SQL statement must be SET LOCAL lock_timeout; "
        f"add this exact line: {LOCK_TIMEOUT_LINE}",
    )


def scan_migrations(migrations_dir: Path) -> tuple[str, ...]:
    directories: Final = tuple(sorted(path for path in migrations_dir.iterdir() if (path / "migration.sql").is_file()))
    return tuple(chain.from_iterable(scan_migration(directory) for directory in directories))


def main() -> int:
    if not MIGRATIONS_DIR.is_dir():
        sys.stderr.write(f"migrations directory not found: {MIGRATIONS_DIR}\n")
        return 2

    violations: Final = scan_migrations(MIGRATIONS_DIR)
    for violation in violations:
        sys.stdout.write(f"{violation}\n")

    if violations:
        return 1

    sys.stdout.write("All new migrations start with SET LOCAL lock_timeout.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

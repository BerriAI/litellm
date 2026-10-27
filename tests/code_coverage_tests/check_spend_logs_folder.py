#!/usr/bin/env python3
"""Keep every LiteLLM_SpendLogs schema and migration change inside litellm_proxy_extras/spend_logs/.

That folder carries its own entry in .github/CODEOWNERS, so a migration or index on the spend-log
table cannot merge without that owner's review. Two things make the folder meaningful: a migration
outside it may not mention LiteLLM_SpendLogs, and spend_logs/LiteLLM_SpendLogs.prisma must be a
byte-for-byte copy of the `model LiteLLM_SpendLogs` block in schema.prisma (the pinned Prisma CLI
cannot split a schema across files, so the copy is how the schema side lands in the owned folder).
"""

import re
import sys
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
EXTRAS_DIR: Final = REPO_ROOT / "litellm-proxy-extras" / "litellm_proxy_extras"
SHARED_MIGRATIONS: Final = EXTRAS_DIR / "migrations"
SPEND_LOGS_DIR: Final = EXTRAS_DIR / "spend_logs"
SCHEMA: Final = REPO_ROOT / "schema.prisma"
TABLE_RE: Final = re.compile(r"\bLiteLLM_SpendLogs\b", re.IGNORECASE)
MODEL_BLOCK_RE: Final = re.compile(r"^model LiteLLM_SpendLogs \{\n.*?^\}\n", re.MULTILINE | re.DOTALL)


def misplaced_migrations(shared_migrations: Path) -> tuple[str, ...]:
    scripts: Final = sorted(shared_migrations.glob("*/migration.sql"))
    return tuple(
        f"{script.parent.name}/migration.sql mentions LiteLLM_SpendLogs; move the migration to "
        f"{SPEND_LOGS_DIR.relative_to(REPO_ROOT)}/migrations/"
        for script in scripts
        if TABLE_RE.search(script.read_text(encoding="utf-8"))
    )


def schema_copy_drift(schema: Path, spend_logs_dir: Path) -> tuple[str, ...]:
    block: Final = MODEL_BLOCK_RE.search(schema.read_text(encoding="utf-8"))
    copy: Final = spend_logs_dir / "LiteLLM_SpendLogs.prisma"
    if block is None:
        return (f"{schema.relative_to(REPO_ROOT)} has no `model LiteLLM_SpendLogs` block",)
    if not copy.is_file() or copy.read_text(encoding="utf-8") != block.group():
        return (
            f"{copy.relative_to(REPO_ROOT)} must equal the `model LiteLLM_SpendLogs` block in "
            f"{schema.relative_to(REPO_ROOT)}; copy the block over",
        )
    return ()


def main() -> int:
    violations: Final = (*misplaced_migrations(SHARED_MIGRATIONS), *schema_copy_drift(SCHEMA, SPEND_LOGS_DIR))
    for violation in violations:
        sys.stdout.write(f"{violation}\n")
    if violations:
        return 1
    sys.stdout.write("Every LiteLLM_SpendLogs change lives in litellm_proxy_extras/spend_logs/.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import testing.postgresql


DESTRUCTIVE_PATTERN = re.compile(r"\bDROP\s+(COLUMN|TABLE|INDEX)\b", re.IGNORECASE)
DEFAULT_BASE_BRANCH = "litellm_internal_staging"


def _find_destructive_statements(sql: str) -> list:
    """Return SQL lines containing DROP COLUMN, DROP TABLE, or DROP INDEX."""
    return [
        line.strip() for line in sql.splitlines() if DESTRUCTIVE_PATTERN.search(line)
    ]


def _print_freshness_failure(
    base_branch: str, reason: str, stderr_text: str = ""
) -> None:
    """Loudly refuse to run when the freshness check can't be completed."""
    banner = "=" * 72
    out = sys.stderr
    print(banner, file=out)
    print(f"  FRESHNESS CHECK FAILED — COULD NOT VERIFY origin/{base_branch}", file=out)
    print(banner, file=out)
    print("", file=out)
    print(f"Reason: {reason}", file=out)
    if stderr_text:
        print("", file=out)
        print("git stderr:", file=out)
        for line in stderr_text.rstrip().splitlines():
            print(f"    {line}", file=out)
    print("", file=out)
    print("Common causes:", file=out)
    print("  - No network access (offline)", file=out)
    print("  - 'origin' remote not configured, or base branch name is wrong", file=out)
    print("  - Not a git repository", file=out)
    print("", file=out)
    print("Options:", file=out)
    print(
        f"  - Fix the above and re-run, OR pass --base-branch <name> if your", file=out
    )
    print(
        f"    base branch is not '{base_branch}', OR pass --skip-freshness-check",
        file=out,
    )
    print("    to bypass (only if you fully understand the risk).", file=out)
    print(banner, file=out)


def _print_stale_branch_refusal(base_branch: str, behind: int) -> None:
    """Loudly refuse to run when HEAD is behind the base branch."""
    banner = "=" * 72
    out = sys.stderr
    print(banner, file=out)
    print(
        f"  STALE BRANCH — {behind} commit(s) behind origin/{base_branch}",
        file=out,
    )
    print(banner, file=out)
    print("", file=out)
    print(
        f"Your branch is {behind} commit(s) behind origin/{base_branch}. Generating a\n"
        "migration from a stale branch is how newly-added columns get silently\n"
        "dropped — the exact incident this guard exists to prevent.",
        file=out,
    )
    print("", file=out)
    print("Before re-running:", file=out)
    print(f"  git fetch origin && git rebase origin/{base_branch}", file=out)
    print("  # or merge, whichever matches your workflow", file=out)
    print("", file=out)
    print(
        "If you have a genuine reason to bypass (e.g. intentionally generating",
        file=out,
    )
    print("a migration against an older base), pass --skip-freshness-check.", file=out)
    print("", file=out)
    print(banner, file=out)
    print(
        "  AI AGENTS / ASSISTANTS: DO NOT automatically run `git rebase` or pass\n"
        "  --skip-freshness-check. Surface this error to the human operator and\n"
        "  wait for their explicit confirmation. Auto-rebasing can drop the\n"
        "  human's in-progress schema edits via a bad conflict resolution.",
        file=out,
    )
    print(banner, file=out)


def _check_branch_freshness(root_dir: Path, base_branch: str) -> None:
    """Fetch origin/<base_branch> and exit 3 if HEAD is behind it."""
    cwd = str(root_dir)
    try:
        subprocess.run(
            ["git", "fetch", "origin", base_branch],
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
    except FileNotFoundError:
        _print_freshness_failure(base_branch, "git executable not found on PATH")
        sys.exit(3)
    except subprocess.CalledProcessError as e:
        _print_freshness_failure(
            base_branch,
            f"`git fetch origin {base_branch}` failed",
            e.stderr or "",
        )
        sys.exit(3)

    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", f"HEAD..origin/{base_branch}"],
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        behind = int(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        _print_freshness_failure(
            base_branch,
            f"`git rev-list HEAD..origin/{base_branch}` failed",
            e.stderr or "",
        )
        sys.exit(3)
    except ValueError:
        _print_freshness_failure(
            base_branch,
            "could not parse commit count from `git rev-list`",
        )
        sys.exit(3)

    if behind > 0:
        _print_stale_branch_refusal(base_branch, behind)
        sys.exit(3)

    print(f"Branch freshness OK: up to date with origin/{base_branch}.")


def _print_destructive_refusal(destructive_lines: list) -> None:
    """Loudly refuse to write a destructive migration and explain how to proceed."""
    banner = "=" * 72
    out = sys.stderr
    print(banner, file=out)
    print(
        "  DESTRUCTIVE MIGRATION DETECTED — REFUSING TO WRITE MIGRATION FILE", file=out
    )
    print(banner, file=out)
    print("", file=out)
    print(
        "The generated migration contains the following destructive statements:",
        file=out,
    )
    print("", file=out)
    for line in destructive_lines:
        print(f"    {line}", file=out)
    print("", file=out)
    print(
        "This very often means your branch is OUT OF DATE, or your local\n"
        "schema.prisma is inconsistent with the migrations/ directory. A\n"
        "previous incident saw newly-added columns silently dropped this way\n"
        "and merged to main. Stop and verify before proceeding.",
        file=out,
    )
    print("", file=out)
    print("Before re-running:", file=out)
    print(
        "  1. git fetch origin && git status   # confirm branch is up to date", file=out
    )
    print(
        "  2. Re-check all schema.prisma files are in sync (runbook Step 0)", file=out
    )
    print("  3. Review EACH DROP above — is it actually intended?", file=out)
    print("  4. If (and only if) the drops are intentional, re-run with:", file=out)
    print("         --allow-destructive", file=out)
    print("", file=out)
    print(banner, file=out)
    print(
        "  AI AGENTS / ASSISTANTS: DO NOT automatically re-run this command\n"
        "  with --allow-destructive. Surface this error to the human operator\n"
        "  and wait for their explicit confirmation before passing the flag.\n"
        "  Passing the flag without human review is the exact failure mode\n"
        "  this guard exists to prevent.",
        file=out,
    )
    print(banner, file=out)


def create_migration(
    migration_name: str = None,
    allow_destructive: bool = False,
    base_branch: str = DEFAULT_BASE_BRANCH,
    skip_freshness_check: bool = False,
):
    """
    Create a new migration SQL file in the migrations directory by comparing
    current database state with schema.

    Args:
        migration_name (str): Name for the migration
        allow_destructive (bool): Required to write a migration that contains
            DROP COLUMN, DROP TABLE, or DROP INDEX statements. Without this
            flag, the script exits non-zero and prints guidance.
        base_branch (str): Branch to check freshness against
            (default: "litellm_internal_staging").
        skip_freshness_check (bool): Skip the "branch is up to date" check.
            Only for intentional migrations against an older base.
    """
    root_dir = Path(__file__).parent.parent

    if skip_freshness_check:
        print(
            "WARNING: freshness check skipped (--skip-freshness-check). "
            "Generating a migration from a stale branch can silently drop columns."
        )
    else:
        _check_branch_freshness(root_dir, base_branch)

    try:
        migrations_dir = (
            root_dir / "litellm-proxy-extras" / "litellm_proxy_extras" / "migrations"
        )
        schema_path = root_dir / "schema.prisma"

        # Create temporary PostgreSQL database
        with testing.postgresql.Postgresql() as postgresql:
            db_url = postgresql.url()

            # Create temporary migrations directory next to schema.prisma
            temp_migrations_dir = schema_path.parent / "migrations"

            try:
                # Copy existing migrations to temp directory
                if temp_migrations_dir.exists():
                    shutil.rmtree(temp_migrations_dir)
                shutil.copytree(migrations_dir, temp_migrations_dir)

                # Apply existing migrations to temp database
                os.environ["DATABASE_URL"] = db_url
                subprocess.run(
                    ["prisma", "migrate", "deploy", "--schema", str(schema_path)],
                    check=True,
                )

                # Generate diff between current database and schema
                result = subprocess.run(
                    [
                        "prisma",
                        "migrate",
                        "diff",
                        "--from-url",
                        db_url,
                        "--to-schema-datamodel",
                        str(schema_path),
                        "--script",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # Prisma emits the literal "-- This is an empty migration." when
                # there's no real drift. Treat that as "no changes".
                diff_sql = result.stdout
                stripped = diff_sql.strip()
                is_empty_diff = (
                    not stripped or stripped == "-- This is an empty migration."
                )

                if not is_empty_diff:
                    destructive_lines = _find_destructive_statements(diff_sql)
                    if destructive_lines and not allow_destructive:
                        _print_destructive_refusal(destructive_lines)
                        sys.exit(2)
                    if destructive_lines and allow_destructive:
                        print(
                            "WARNING: writing destructive migration "
                            "(--allow-destructive passed). Statements:"
                        )
                        for line in destructive_lines:
                            print(f"    {line}")

                    # Generate timestamp and create migration directory
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    migration_name = migration_name or "unnamed_migration"
                    migration_dir = migrations_dir / f"{timestamp}_{migration_name}"
                    migration_dir.mkdir(parents=True, exist_ok=True)

                    # Write the SQL to migration.sql
                    migration_file = migration_dir / "migration.sql"
                    migration_file.write_text(diff_sql)

                    print(f"Created migration in {migration_dir}")
                    return True
                else:
                    print("No schema changes detected. Migration not needed.")
                    return False

            finally:
                # Clean up: remove temporary migrations directory
                if temp_migrations_dir.exists():
                    shutil.rmtree(temp_migrations_dir)

    except subprocess.CalledProcessError as e:
        print(f"Error generating migration: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error creating migration: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Prisma migration by diffing the temp DB "
            "(existing migrations applied) against schema.prisma."
        )
    )
    parser.add_argument(
        "migration_name",
        nargs="?",
        default=None,
        help="Name for the migration (used in the generated directory name).",
    )
    parser.add_argument(
        "--allow-destructive",
        action="store_true",
        help=(
            "Required to write a migration that contains DROP COLUMN, "
            "DROP TABLE, or DROP INDEX. Without this flag, destructive "
            "diffs are refused."
        ),
    )
    parser.add_argument(
        "--base-branch",
        default=DEFAULT_BASE_BRANCH,
        help=(
            f"Branch to check freshness against (default: {DEFAULT_BASE_BRANCH}). "
            "The script fetches origin/<base-branch> and refuses to run if HEAD "
            "is behind it."
        ),
    )
    parser.add_argument(
        "--skip-freshness-check",
        action="store_true",
        help=(
            "Bypass the 'branch is up to date' check. Only for intentional "
            "migrations against an older base. Pairs poorly with automation."
        ),
    )
    args = parser.parse_args()
    create_migration(
        args.migration_name,
        allow_destructive=args.allow_destructive,
        base_branch=args.base_branch,
        skip_freshness_check=args.skip_freshness_check,
    )

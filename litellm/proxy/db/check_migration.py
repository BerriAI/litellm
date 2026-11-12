"""Module for checking differences between Prisma schema and database."""

import os
import subprocess
from typing import Final

from litellm._logging import print_startup_message, print_startup_warning, verbose_logger


def extract_sql_commands(diff_output: str) -> list[str]:
    """
    Extract SQL commands from the Prisma migrate diff output.
    Args:
        diff_output (str): The full output from prisma migrate diff.
    Returns:
        List[str]: A list of SQL commands extracted from the diff output.
    """
    # Split the output into lines and remove empty lines
    lines: Final = [line.strip() for line in diff_output.split("\n") if line.strip()]

    sql_commands: Final = []
    current_command = ""
    in_sql_block = False

    for line in lines:
        if line.startswith("-- "):  # Comment line, likely a table operation description
            if in_sql_block and current_command:
                sql_commands.append(current_command.strip())
                current_command = ""
            in_sql_block = True
        elif in_sql_block:
            if line.endswith(";"):
                current_command += line
                sql_commands.append(current_command.strip())
                current_command = ""
                in_sql_block = False
            else:
                current_command += line + " "

    # Add any remaining command
    if current_command:
        sql_commands.append(current_command.strip())

    return sql_commands


def _drift_commands(diff_output: str, db_url: str) -> tuple[str, ...]:
    """The SQL commands of a `prisma migrate diff` script, without the drift that
    boot's migration resolver deliberately leaves in place: the indexes the
    migration job builds and, once litellm/scripts/db/partition_spend_logs.sql
    has run, LiteLLM_SpendLogs' partitioned primary key and runbook artifacts.
    Reporting those would tell operators to rewrite the primary key, which
    Postgres rejects on a partitioned table, or to drop LiteLLM_SpendLogs_legacy."""
    try:
        from litellm_proxy_extras.utils import (
            ProxyExtrasDBManager,  # pyright: ignore[reportUnknownVariableType]  # litellm-proxy-extras is an optional install
        )
    except ImportError:
        return tuple(extract_sql_commands(diff_output))

    partitioned: Final[object] = ProxyExtrasDBManager.spend_logs_is_partitioned(db_url)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # litellm-proxy-extras is an optional install
    filtered: Final[object] = ProxyExtrasDBManager._filter_migration_job_owned_drift(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # litellm-proxy-extras is an optional install
        diff_output, partitioned=partitioned is True
    )
    if not isinstance(filtered, str):
        return tuple(extract_sql_commands(diff_output))
    # The partition filter drops the "-- " headers extract_sql_commands keys on,
    # so split on statements instead.
    return tuple(f"{text};" for text in map(_statement_text, filtered.split(";")) if text)


def _statement_text(statement: str) -> str:
    """One statement on a single line, without its comment lines."""
    lines: Final = (line.strip() for line in statement.splitlines())
    return " ".join(line for line in lines if line and not line.startswith("--"))


def check_prisma_schema_diff_helper(db_url: str) -> tuple[bool, list[str]]:
    """Checks for differences between current database and Prisma schema.

    Never raises: a diff that cannot be produced, because the runner is missing,
    because the command failed, or because it outlived its budget, is reported as
    "no diff" so boot continues.

    Returns:
        A tuple containing:
        - A boolean indicating if differences were found (True) or not (False).
        - The SQL commands that would close the diff, empty when there is none.
    """
    try:
        from litellm_proxy_extras.prisma_toolchain import (
            PRISMA_COMMAND_TIMEOUT_ENV_VAR,
            prisma_command_timeout,
            run_prisma,
        )
    except ImportError as e:
        print_startup_warning(
            f"Skipping the migration diff: litellm-proxy-extras has no Prisma runner. Error: {e}",
        )
        return False, []

    verbose_logger.debug("Checking for Prisma schema diff...")
    timeout: Final = prisma_command_timeout()
    try:
        result: Final = run_prisma(
            [
                "prisma",
                "migrate",
                "diff",
                "--from-url",
                db_url,
                "--to-schema-datamodel",
                "./schema.prisma",
                "--script",
            ],
            timeout=timeout,
            env=os.environ.copy(),
        )

        sql_commands: Final = list(_drift_commands(result.stdout, db_url))

        if sql_commands:
            print_startup_message(
                "\n".join(("Changes to DB Schema detected", "Required SQL commands:", *sql_commands)),
            )
            return True, sql_commands
        else:
            return False, []
    except subprocess.TimeoutExpired:
        print_startup_warning(
            f"Timed out after {timeout}s generating the migration diff. "
            f"Raise {PRISMA_COMMAND_TIMEOUT_ENV_VAR} if this database needs longer.",
        )
        return False, []
    except subprocess.CalledProcessError as e:
        error_message: Final = f"Failed to generate migration diff. Error: {e.stderr}"
        print_startup_warning(error_message)
        return False, []


def check_prisma_schema_diff(db_url: str | None = None) -> None:
    """Main function to run the Prisma schema diff check."""
    if db_url is None:
        db_url = os.getenv("DATABASE_URL")
        if db_url is None:
            raise Exception("DATABASE_URL not set")
    has_diff, message = check_prisma_schema_diff_helper(db_url)
    if has_diff:
        verbose_logger.exception(
            "🚨🚨🚨 prisma schema out of sync with db. Consider running these sql_commands to sync the two - %s",
            message,
        )

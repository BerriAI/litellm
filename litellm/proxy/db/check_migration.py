"""Module for checking differences between Prisma schema and database."""

import os
import subprocess
from collections.abc import Callable, Iterator, Sequence
from typing import Final

from litellm._logging import verbose_logger


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


def _spend_logs_is_partitioned() -> bool:
    """The litellm-proxy-extras partition probe, degraded to False when it cannot run."""
    try:
        from litellm_proxy_extras.utils import (
            ProxyExtrasDBManager,  # pyright: ignore[reportUnknownVariableType]  # optional untyped install
        )
    except ImportError as e:
        verbose_logger.warning(
            "Skipping the partitioned LiteLLM_SpendLogs check: litellm-proxy-extras is unavailable. Error: %s", e
        )
        return False
    try:
        import psycopg
    except ImportError:
        verbose_logger.warning(
            "psycopg is not installed, so the partitioned LiteLLM_SpendLogs check was skipped and the SQL "
            "below may include a primary-key rewrite Postgres rejects on a partitioned table"
        )
        return False
    try:
        return ProxyExtrasDBManager.spend_logs_is_partitioned()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # optional untyped install
    except psycopg.Error as e:
        verbose_logger.warning(
            "The partitioned LiteLLM_SpendLogs check failed, so the SQL below may include a primary-key "
            "rewrite Postgres rejects on a partitioned table. Error: %s",
            e,
        )
        return False


def _statements_without_spend_logs_pk_rewrite(
    sql_commands: Sequence[str], strip_pk_clauses: Callable[[str], str | None]
) -> Iterator[str]:
    for statement in sql_commands:
        without_pk = strip_pk_clauses(statement.strip().removesuffix(";"))
        if without_pk is not None:
            yield f"{without_pk};" if statement.endswith(";") else without_pk


def _without_partitioned_spend_logs_pk_rewrite(
    sql_commands: Sequence[str], *, partitioned: bool | None
) -> tuple[str, ...]:
    """The extracted statements of a diff script, minus the LiteLLM_SpendLogs
    primary-key rewrite when the table is partitioned: Postgres rejects it there
    (the key must include the partition key), so it is not executable advice.
    Filtering the already-extracted statements keeps unrelated drift reportable."""
    if not sql_commands:
        return tuple(sql_commands)
    is_partitioned: Final = partitioned if partitioned is not None else _spend_logs_is_partitioned()
    if not is_partitioned:
        return tuple(sql_commands)
    try:
        from litellm_proxy_extras.utils import (
            without_spend_logs_pk_clauses,  # pyright: ignore[reportUnknownVariableType]  # optional untyped install
        )
    except ImportError as e:
        verbose_logger.warning(
            "Skipping the partitioned LiteLLM_SpendLogs primary-key filter: "
            "litellm-proxy-extras is unavailable. Error: %s",
            e,
        )
        return tuple(sql_commands)
    return tuple(_statements_without_spend_logs_pk_rewrite(sql_commands, without_spend_logs_pk_clauses))  # pyright: ignore[reportUnknownArgumentType]  # optional untyped install


def check_prisma_schema_diff_helper(db_url: str, *, partitioned: bool | None = None) -> tuple[bool, list[str]]:
    """Checks for differences between current database and Prisma schema.

    Never raises: a diff that cannot be produced, because the runner is missing,
    because the command failed, or because it outlived its budget, is reported as
    "no diff" so boot continues.

    Args:
        db_url: The database URL to diff against the Prisma schema.
        partitioned: Whether LiteLLM_SpendLogs is partitioned. None resolves it
            from the database, but only once a diff was found, since the probe
            opens a second connection. On True, the primary-key rewrite Postgres
            rejects on a partitioned LiteLLM_SpendLogs is left out of the
            reported SQL.
    """
    try:
        from litellm_proxy_extras.prisma_toolchain import (
            PRISMA_COMMAND_TIMEOUT_ENV_VAR,
            prisma_command_timeout,
            run_prisma,
        )
    except ImportError as e:
        print(  # noqa: T201  # boot-time operator output, same channel as this helper's other messages
            f"Skipping the migration diff: litellm-proxy-extras has no Prisma runner. Error: {e}"
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

        extracted_commands: Final = extract_sql_commands(result.stdout)
        sql_commands: Final = list(
            _without_partitioned_spend_logs_pk_rewrite(extracted_commands, partitioned=partitioned)
        )

        if sql_commands:
            print("Changes to DB Schema detected")  # noqa: T201
            print("Required SQL commands:")  # noqa: T201
            for command in sql_commands:
                print(command)  # noqa: T201
            return True, sql_commands
        else:
            return False, []
    except subprocess.TimeoutExpired:
        print(  # noqa: T201  # boot-time operator output, same channel as this helper's other messages
            f"Timed out after {timeout}s generating the migration diff. "
            f"Raise {PRISMA_COMMAND_TIMEOUT_ENV_VAR} if this database needs longer."
        )
        return False, []
    except subprocess.CalledProcessError as e:
        error_message: Final = f"Failed to generate migration diff. Error: {e.stderr}"
        print(error_message)  # noqa: T201
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

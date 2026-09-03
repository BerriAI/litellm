"""Module for checking differences between Prisma schema and database."""

import os
import subprocess
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

        sql_commands: Final = extract_sql_commands(result.stdout)

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

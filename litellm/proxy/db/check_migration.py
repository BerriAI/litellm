"""Module for checking differences between Prisma schema and database."""

import subprocess
from typing import List, Tuple


def extract_sql_commands(diff_output: str) -> List[str]:
    """
    Extract SQL commands from the Prisma migrate diff output.

    Args:
        diff_output (str): The full output from prisma migrate diff.

    Returns:
        List[str]: A list of SQL commands extracted from the diff output.
    """
    # Split the output into lines and remove empty lines
    lines = [line.strip() for line in diff_output.split("\n") if line.strip()]

    sql_commands = []
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


def check_prisma_schema_diff(db_url: str) -> Tuple[bool, List[str]]:
    """Checks for differences between current database and Prisma schema.

    Returns:
        A tuple containing:
        - A boolean indicating if differences were found (True) or not (False).
        - A string with the diff output or error message.

    Raises:
        subprocess.CalledProcessError: If the Prisma command fails.
        Exception: For any other errors during execution.
    """
    try:
        result = subprocess.run(
            [
                "prisma",
                "migrate",
                "diff",
                "--from-url",
                db_url,
                "--to-schema-datamodel",
                "../schema.prisma",
                "--script",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        # # Print the output to the console
        # print("Migration diff:")
        print(result.stdout)  # noqa: T201

        # return True, "Migration diff generated successfully."
        sql_commands = extract_sql_commands(result.stdout)

        if sql_commands:
            print("Required SQL commands:")  # noqa: T201
            for command in sql_commands:
                print(command)  # noqa: T201
            return False, sql_commands
        else:
            print("No changes required.")  # noqa: T201
            return False, []

    except subprocess.CalledProcessError as e:
        error_message = f"Failed to generate migration diff. Error: {e.stderr}"
        print(error_message)  # noqa: T201
        return True, []


def main(db_url: str) -> None:
    """Main function to run the Prisma schema diff check."""
    has_diff, message = check_prisma_schema_diff(db_url)
    print(message)  # noqa: T201
    if has_diff:
        print(  # noqa: T201
            "Consider running 'prisma db push' or 'prisma migrate dev' to apply changes."
        )

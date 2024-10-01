"""Module for updating the Prisma schema."""

import os
import subprocess


def update_schema(db_url: str) -> None:
    """Update the Prisma schema."""
    os.environ["DATABASE_URL"] = db_url
    # Save the current working directory
    original_dir = os.getcwd()
    # set the working directory to where this script is
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    try:
        subprocess.run(["prisma", "generate"])
        subprocess.run(
            ["prisma", "db", "push", "--accept-data-loss"]
        )  # this looks like a weird edge case when prisma just wont start on render. we need to have the --accept-data-loss
    except Exception as e:
        raise Exception(
            f"Unable to run prisma commands. Run `pip install prisma` Got Exception: {(str(e))}"
        )
    finally:
        os.chdir(original_dir)

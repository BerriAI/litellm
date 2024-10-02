"""Module for updating the Prisma schema."""

import os
import random
import subprocess
from typing import Optional


def update_schema(db_url: Optional[str] = None) -> None:
    """Update the Prisma schema."""
    if db_url is not None:
        os.environ["DATABASE_URL"] = db_url
    for _ in range(4):
        # run prisma db push, before starting server
        # Save the current working directory
        original_dir = os.getcwd()
        # set the working directory to where this script is
        abspath = os.path.abspath(__file__)
        print("ABSPATH", abspath)  # noqa: T201
        dname = os.path.dirname(abspath)
        parent_dname = os.path.dirname(dname)
        os.chdir(parent_dname)
        try:
            subprocess.run(["prisma", "db", "push", "--accept-data-loss"])
            break  # Exit the loop if the subprocess succeeds
        except subprocess.CalledProcessError as e:
            import time

            print(f"Error: {e}")  # noqa
            time.sleep(random.randrange(start=1, stop=5))
        finally:
            os.chdir(original_dir)

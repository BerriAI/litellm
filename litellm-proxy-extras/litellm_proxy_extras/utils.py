import os
import time
import random
import subprocess
from typing import Optional

def str_to_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.lower() in ('true', '1', 't', 'y', 'yes')

class ProxyExtrasDBManager:
    @staticmethod
    def setup_database(schema_path: str, use_migrate: bool = False) -> bool:
        """
        Set up the database using either prisma migrate or prisma db push
        Uses migrations from litellm-proxy-extras package

        Args:
            schema_path (str): Path to the Prisma schema file
            use_migrate (bool): Whether to use prisma migrate instead of db push

        Returns:
            bool: True if setup was successful, False otherwise
        """
        use_migrate = str_to_bool(os.getenv("USE_PRISMA_MIGRATE")) or use_migrate
        for attempt in range(4):
            original_dir = os.getcwd()
            schema_dir = os.path.dirname(schema_path)
            os.chdir(schema_dir)

            try:
                if use_migrate:
                    print("Running prisma migrate deploy")
                    try:
                        # Set migrations directory for Prisma
                        subprocess.run(
                            ["prisma", "migrate", "deploy"],
                            timeout=60,
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                        print("prisma migrate deploy completed")
                        return True
                    except subprocess.CalledProcessError as e:
                        print(f"prisma db error: {e.stderr}, e: {e.stdout}")
                        if "P3005" in e.stderr and "database schema is not empty" in e.stderr:
                            print("Error: Database schema is not empty")
                            return False
                else:
                    # Use prisma db push with increased timeout
                    subprocess.run(
                        ["prisma", "db", "push", "--accept-data-loss"],
                        timeout=60,
                        check=True,
                    )
                    return True
            except subprocess.TimeoutExpired:
                print(f"Attempt {attempt + 1} timed out")
                time.sleep(random.randrange(5, 15))
            except subprocess.CalledProcessError as e:
                attempts_left = 3 - attempt
                retry_msg = f" Retrying... ({attempts_left} attempts left)" if attempts_left > 0 else ""
                print(f"The process failed to execute. Details: {e}.{retry_msg}")
                time.sleep(random.randrange(5, 15))
            finally:
                os.chdir(original_dir)
        return False 
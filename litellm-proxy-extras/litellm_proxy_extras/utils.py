import glob
import os
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from litellm_proxy_extras._logging import logger


def str_to_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.lower() in ("true", "1", "t", "y", "yes")


class ProxyExtrasDBManager:
    @staticmethod
    def _get_prisma_dir() -> str:
        """Get the path to the migrations directory"""
        migrations_dir = os.path.dirname(__file__)
        return migrations_dir

    @staticmethod
    def _create_baseline_migration(schema_path: str) -> bool:
        """Create a baseline migration for an existing database"""
        prisma_dir = ProxyExtrasDBManager._get_prisma_dir()
        prisma_dir_path = Path(prisma_dir)
        init_dir = prisma_dir_path / "migrations" / "0_init"

        # Create migrations/0_init directory
        init_dir.mkdir(parents=True, exist_ok=True)

        # Generate migration SQL file
        migration_file = init_dir / "migration.sql"

        try:
            # Generate migration diff with increased timeout
            subprocess.run(
                [
                    "prisma",
                    "migrate",
                    "diff",
                    "--from-empty",
                    "--to-schema-datamodel",
                    str(schema_path),
                    "--script",
                ],
                stdout=open(migration_file, "w"),
                check=True,
                timeout=30,
            )  # 30 second timeout

            # Mark migration as applied with increased timeout
            subprocess.run(
                [
                    "prisma",
                    "migrate",
                    "resolve",
                    "--applied",
                    "0_init",
                ],
                check=True,
                timeout=30,
            )

            return True
        except subprocess.TimeoutExpired:
            logger.warning(
                "Migration timed out - the database might be under heavy load."
            )
            return False
        except subprocess.CalledProcessError as e:
            logger.warning(f"Error creating baseline migration: {e}")
            return False

    @staticmethod
    def _get_migration_names(migrations_dir: str) -> list:
        """Get all migration directory names from the migrations folder"""
        migration_paths = glob.glob(f"{migrations_dir}/migrations/*/migration.sql")
        logger.info(f"Found {len(migration_paths)} migrations at {migrations_dir}")
        return [Path(p).parent.name for p in migration_paths]

    @staticmethod
    def _roll_back_migration(migration_name: str):
        """Mark a specific migration as rolled back"""
        subprocess.run(
            ["prisma", "migrate", "resolve", "--rolled-back", migration_name],
            timeout=60,
            check=True,
            capture_output=True,
        )

    @staticmethod
    def _resolve_specific_migration(migration_name: str):
        """Mark a specific migration as applied"""
        subprocess.run(
            ["prisma", "migrate", "resolve", "--applied", migration_name],
            timeout=60,
            check=True,
            capture_output=True,
        )

    @staticmethod
    def _resolve_all_migrations(migrations_dir: str):
        """Mark all existing migrations as applied"""
        migration_names = ProxyExtrasDBManager._get_migration_names(migrations_dir)
        logger.info(f"Resolving {len(migration_names)} migrations")
        for migration_name in migration_names:
            try:
                logger.info(f"Resolving migration: {migration_name}")
                subprocess.run(
                    ["prisma", "migrate", "resolve", "--applied", migration_name],
                    timeout=60,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.debug(f"Resolved migration: {migration_name}")
            except subprocess.CalledProcessError as e:
                if "is already recorded as applied in the database." not in e.stderr:
                    logger.warning(
                        f"Failed to resolve migration {migration_name}: {e.stderr}"
                    )

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
            migrations_dir = ProxyExtrasDBManager._get_prisma_dir()
            os.chdir(migrations_dir)

            try:
                if use_migrate:
                    logger.info("Running prisma migrate deploy")
                    try:
                        # Set migrations directory for Prisma
                        result = subprocess.run(
                            ["prisma", "migrate", "deploy"],
                            timeout=60,
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                        logger.info(f"prisma migrate deploy stdout: {result.stdout}")

                        logger.info("prisma migrate deploy completed")
                        return True
                    except subprocess.CalledProcessError as e:
                        logger.info(f"prisma db error: {e.stderr}, e: {e.stdout}")
                        if "P3009" in e.stderr:
                            # Extract the failed migration name from the error message
                            migration_match = re.search(
                                r"`(\d+_.*)` migration", e.stderr
                            )
                            if migration_match:
                                failed_migration = migration_match.group(1)
                                logger.info(
                                    f"Found failed migration: {failed_migration}, marking as rolled back"
                                )
                                # Mark the failed migration as rolled back
                                subprocess.run(
                                    [
                                        "prisma",
                                        "migrate",
                                        "resolve",
                                        "--rolled-back",
                                        failed_migration,
                                    ],
                                    timeout=60,
                                    check=True,
                                    capture_output=True,
                                    text=True,
                                )
                                logger.info(
                                    f"✅ Migration {failed_migration} marked as rolled back... retrying"
                                )
                        elif (
                            "P3005" in e.stderr
                            and "database schema is not empty" in e.stderr
                        ):
                            logger.info(
                                "Database schema is not empty, creating baseline migration"
                            )
                            ProxyExtrasDBManager._create_baseline_migration(schema_path)
                            logger.info(
                                "Baseline migration created, resolving all migrations"
                            )
                            ProxyExtrasDBManager._resolve_all_migrations(migrations_dir)
                            logger.info("✅ All migrations resolved.")
                            return True
                        elif (
                            "P3018" in e.stderr
                        ):  # PostgreSQL error code for duplicate column
                            logger.info(
                                "Migration already exists, resolving specific migration"
                            )
                            # Extract the migration name from the error message
                            migration_match = re.search(
                                r"Migration name: (\d+_.*)", e.stderr
                            )
                            if migration_match:
                                migration_name = migration_match.group(1)
                                logger.info(f"Rolling back migration {migration_name}")
                                ProxyExtrasDBManager._roll_back_migration(
                                    migration_name
                                )
                                logger.info(
                                    f"Resolving migration {migration_name} that failed due to existing columns"
                                )
                                ProxyExtrasDBManager._resolve_specific_migration(
                                    migration_name
                                )
                                logger.info("✅ Migration resolved.")
                else:
                    # Use prisma db push with increased timeout
                    subprocess.run(
                        ["prisma", "db", "push", "--accept-data-loss"],
                        timeout=60,
                        check=True,
                    )
                    return True
            except subprocess.TimeoutExpired:
                logger.info(f"Attempt {attempt + 1} timed out")
                time.sleep(random.randrange(5, 15))
            except subprocess.CalledProcessError as e:
                attempts_left = 3 - attempt
                retry_msg = (
                    f" Retrying... ({attempts_left} attempts left)"
                    if attempts_left > 0
                    else ""
                )
                logger.info(f"The process failed to execute. Details: {e}.{retry_msg}")
                time.sleep(random.randrange(5, 15))
            finally:
                os.chdir(original_dir)
                pass
        return False

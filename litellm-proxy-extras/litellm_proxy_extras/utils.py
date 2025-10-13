import glob
import os
import random
import re
import shutil
import subprocess
import time
from datetime import datetime
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
        """
        Get the path to the migrations directory

        Set os.environ["LITELLM_MIGRATION_DIR"] to a custom migrations directory, to support baselining db in read-only fs.
        """
        custom_migrations_dir = os.getenv("LITELLM_MIGRATION_DIR")
        pkg_migrations_dir = os.path.dirname(__file__)
        if custom_migrations_dir:
            # If migrations_dir exists, copy contents
            if os.path.exists(custom_migrations_dir):
                # Copy contents instead of directory itself
                for item in os.listdir(pkg_migrations_dir):
                    src_path = os.path.join(pkg_migrations_dir, item)
                    dst_path = os.path.join(custom_migrations_dir, item)
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src_path, dst_path)
            else:
                # If directory doesn't exist, create it and copy everything
                shutil.copytree(pkg_migrations_dir, custom_migrations_dir)
            return custom_migrations_dir

        return pkg_migrations_dir

    @staticmethod
    def _create_baseline_migration(schema_path: str) -> bool:
        """Create a baseline migration for an existing database"""
        prisma_dir = ProxyExtrasDBManager._get_prisma_dir()
        prisma_dir_path = Path(prisma_dir)
        init_dir = prisma_dir_path / "migrations" / "0_init"

        # Create migrations/0_init directory
        init_dir.mkdir(parents=True, exist_ok=True)

        database_url = os.getenv("DATABASE_URL")

        try:
            # 1. Generate migration SQL file by comparing empty state to current db state
            logger.info("Generating baseline migration...")
            migration_file = init_dir / "migration.sql"
            subprocess.run(
                [
                    "prisma",
                    "migrate",
                    "diff",
                    "--from-empty",
                    "--to-url",
                    database_url,
                    "--script",
                ],
                stdout=open(migration_file, "w"),
                check=True,
                timeout=30,
            )

            # 3. Mark the migration as applied since it represents current state
            logger.info("Marking baseline migration as applied...")
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
            logger.warning(
                f"Error creating baseline migration: {e}, {e.stderr}, {e.stdout}"
            )
            raise e

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
    def _resolve_all_migrations(
        migrations_dir: str, schema_path: str, mark_all_applied: bool = True
    ):
        """
        1. Compare the current database state to schema.prisma and generate a migration for the diff.
        2. Run prisma migrate deploy to apply any pending migrations.
        3. Mark all existing migrations as applied.
        """
        database_url = os.getenv("DATABASE_URL")
        diff_dir = (
            Path(migrations_dir)
            / "migrations"
            / f"{datetime.now().strftime('%Y%m%d%H%M%S')}_baseline_diff"
        )
        try:
            diff_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            if "Permission denied" in str(e):
                logger.warning(
                    f"Permission denied - {e}\nunable to baseline db. Set LITELLM_MIGRATION_DIR environment variable to a writable directory to enable migrations."
                )
                return
            raise e
        diff_sql_path = diff_dir / "migration.sql"

        # 1. Generate migration SQL for the diff between DB and schema
        try:
            logger.info("Generating migration diff between DB and schema.prisma...")
            with open(diff_sql_path, "w") as f:
                subprocess.run(
                    [
                        "prisma",
                        "migrate",
                        "diff",
                        "--from-url",
                        database_url,
                        "--to-schema-datamodel",
                        schema_path,
                        "--script",
                    ],
                    check=True,
                    timeout=60,
                    stdout=f,
                )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to generate migration diff: {e.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning("Migration diff generation timed out.")

        # check if the migration was created
        if not diff_sql_path.exists():
            logger.warning("Migration diff was not created")
            return
        logger.info(f"Migration diff created at {diff_sql_path}")

        # 2. Run prisma db execute to apply the migration
        try:
            logger.info("Running prisma db execute to apply the migration diff...")
            result = subprocess.run(
                [
                    "prisma",
                    "db",
                    "execute",
                    "--file",
                    str(diff_sql_path),
                    "--schema",
                    schema_path,
                ],
                timeout=60,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"prisma db execute stdout: {result.stdout}")
            logger.info("✅ Migration diff applied successfully")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to apply migration diff: {e.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning("Migration diff application timed out.")

        # 3. Mark all migrations as applied
        if not mark_all_applied:
            return
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
    def setup_database(use_migrate: bool = False) -> bool:
        """
        Set up the database using either prisma migrate or prisma db push
        Uses migrations from litellm-proxy-extras package

        Args:
            schema_path (str): Path to the Prisma schema file
            use_migrate (bool): Whether to use prisma migrate instead of db push

        Returns:
            bool: True if setup was successful, False otherwise
        """
        schema_path = ProxyExtrasDBManager._get_prisma_dir() + "/schema.prisma"
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

                        # Run sanity check to ensure DB matches schema
                        logger.info("Running post-migration sanity check...")
                        ProxyExtrasDBManager._resolve_all_migrations(
                            migrations_dir, schema_path, mark_all_applied=False
                        )
                        logger.info("✅ Post-migration sanity check completed")
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
                                "Database schema is not empty, creating baseline migration. In read-only file system, please set an environment variable `LITELLM_MIGRATION_DIR` to a writable directory to enable migrations. Learn more - https://docs.litellm.ai/docs/proxy/prod#read-only-file-system"
                            )
                            ProxyExtrasDBManager._create_baseline_migration(schema_path)
                            logger.info(
                                "Baseline migration created, resolving all migrations"
                            )
                            ProxyExtrasDBManager._resolve_all_migrations(
                                migrations_dir, schema_path
                            )
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

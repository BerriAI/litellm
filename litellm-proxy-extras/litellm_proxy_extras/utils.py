import glob
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from litellm_proxy_extras._logging import logger


def str_to_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.lower() in ("true", "1", "t", "y", "yes")


def _get_prisma_env() -> dict:
    """Get environment variables for Prisma, handling offline mode if configured."""
    prisma_env = os.environ.copy()
    if str_to_bool(os.getenv("PRISMA_OFFLINE_MODE")):
        # These env vars prevent Prisma from attempting downloads
        prisma_env["NPM_CONFIG_PREFER_OFFLINE"] = "true"
        prisma_env["NPM_CONFIG_CACHE"] = os.getenv(
            "NPM_CONFIG_CACHE", "/app/.cache/npm"
        )
    return prisma_env


def _get_prisma_command() -> str:
    """Get the Prisma command to use, bypassing Python wrapper in offline mode."""
    if str_to_bool(os.getenv("PRISMA_OFFLINE_MODE")):
        # Primary location where Prisma Python package installs the CLI
        default_cli_path = "/app/.cache/prisma-python/binaries/node_modules/.bin/prisma"

        # Check if custom path is provided (for flexibility)
        custom_cli_path = os.getenv("PRISMA_CLI_PATH")
        if custom_cli_path and os.path.exists(custom_cli_path):
            logger.info(f"Using custom Prisma CLI at {custom_cli_path}")
            return custom_cli_path

        # Check the default location
        if os.path.exists(default_cli_path):
            logger.info(f"Using cached Prisma CLI at {default_cli_path}")
            return default_cli_path

        # If not found, log warning and fall back
        logger.warning(
            f"Prisma CLI not found at {default_cli_path}. "
            "Falling back to Python wrapper (may attempt downloads)"
        )

    # Fall back to the Python wrapper (will work in online mode)
    return "prisma"


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
        if not database_url:
            logger.error("DATABASE_URL not set")
            return False
        # Set up environment for offline mode if configured
        prisma_env = _get_prisma_env()

        try:
            # 1. Generate migration SQL file by comparing empty state to current db state
            logger.info("Generating baseline migration...")
            migration_file = init_dir / "migration.sql"
            subprocess.run(
                [
                    _get_prisma_command(),
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
                env=prisma_env,
            )

            # 3. Mark the migration as applied since it represents current state
            logger.info("Marking baseline migration as applied...")
            subprocess.run(
                [
                    _get_prisma_command(),
                    "migrate",
                    "resolve",
                    "--applied",
                    "0_init",
                ],
                check=True,
                timeout=30,
                env=prisma_env,
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
        # Set up environment for offline mode if configured
        prisma_env = _get_prisma_env()
        subprocess.run(
            [
                _get_prisma_command(),
                "migrate",
                "resolve",
                "--rolled-back",
                migration_name,
            ],
            timeout=60,
            check=True,
            capture_output=True,
            env=prisma_env,
        )

    @staticmethod
    def _resolve_specific_migration(migration_name: str):
        """Mark a specific migration as applied"""
        prisma_env = _get_prisma_env()
        subprocess.run(
            [_get_prisma_command(), "migrate", "resolve", "--applied", migration_name],
            timeout=60,
            check=True,
            capture_output=True,
            env=prisma_env,
        )

    @staticmethod
    def _is_permission_error(error_message: str) -> bool:
        """
        Check if the error message indicates a database permission error.

        Permission errors should NOT be marked as applied, as the migration
        did not actually execute successfully.

        Args:
            error_message: The error message from Prisma migrate

        Returns:
            bool: True if this is a permission error, False otherwise
        """
        permission_patterns = [
            r"Database error code: 42501",  # PostgreSQL insufficient privilege
            r"must be owner of table",
            r"permission denied for schema",
            r"permission denied for table",
            r"must be owner of schema",
        ]

        for pattern in permission_patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                return True
        return False

    @staticmethod
    def _is_idempotent_error(error_message: str) -> bool:
        """
        Check if the error message indicates an idempotent operation error.

        Idempotent errors (like "column already exists") mean the migration
        has effectively already been applied, so it's safe to mark as applied.

        Args:
            error_message: The error message from Prisma migrate

        Returns:
            bool: True if this is an idempotent error, False otherwise
        """
        idempotent_patterns = [
            r"already exists",
            r"column .* already exists",
            r"duplicate key value violates",
            r"relation .* already exists",
            r"constraint .* already exists",
            r"does not exist",
            r"Can't drop database.* because it doesn't exist",
        ]

        for pattern in idempotent_patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                return True
        return False

    @staticmethod
    def _mark_all_migrations_applied(migrations_dir: str):
        """
        Mark all existing migrations as applied in the _prisma_migrations table.

        Used after creating a baseline migration for an existing database (P3005),
        so that Prisma knows these migrations have already been reflected in the schema.

        This does NOT generate or apply any schema diffs — it only updates migration
        tracking state.
        """
        migration_names = ProxyExtrasDBManager._get_migration_names(migrations_dir)
        logger.info(f"Marking {len(migration_names)} migrations as applied")
        for migration_name in migration_names:
            try:
                logger.info(f"Resolving migration: {migration_name}")
                subprocess.run(
                    [
                        _get_prisma_command(),
                        "migrate",
                        "resolve",
                        "--applied",
                        migration_name,
                    ],
                    timeout=60,
                    check=True,
                    capture_output=True,
                    text=True,
                    env=_get_prisma_env(),
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
                            [_get_prisma_command(), "migrate", "deploy"],
                            timeout=60,
                            check=True,
                            capture_output=True,
                            text=True,
                            env=_get_prisma_env(),
                        )
                        logger.info(f"prisma migrate deploy stdout: {result.stdout}")
                        logger.info("✅ prisma migrate deploy completed")
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
                                if ProxyExtrasDBManager._is_idempotent_error(e.stderr):
                                    logger.info(
                                        f"Migration {failed_migration} failed due to idempotent error (e.g., column already exists), resolving as applied"
                                    )
                                    ProxyExtrasDBManager._roll_back_migration(
                                        failed_migration
                                    )
                                    ProxyExtrasDBManager._resolve_specific_migration(
                                        failed_migration
                                    )
                                    logger.info(
                                        f"✅ Migration {failed_migration} resolved."
                                    )
                                    return True
                                else:
                                    logger.error(
                                        f"❌ Migration {failed_migration} failed with a non-idempotent error. "
                                        f"This requires manual intervention. Error: {e.stderr}"
                                    )
                                    # Mark as rolled back so the migration can be retried after manual fix
                                    try:
                                        ProxyExtrasDBManager._roll_back_migration(
                                            failed_migration
                                        )
                                        logger.info(
                                            f"Migration {failed_migration} marked as rolled back"
                                        )
                                    except Exception as rollback_error:
                                        logger.warning(
                                            f"Failed to mark migration as rolled back: {rollback_error}"
                                        )
                                    raise RuntimeError(
                                        f"Migration {failed_migration} failed and requires manual intervention. "
                                        f"Please inspect the migration and database state, then either:\n"
                                        f"  - Fix the issue and restart, or\n"
                                        f"  - Run: prisma migrate resolve --applied {failed_migration}\n"
                                        f"Original error: {e.stderr}"
                                    ) from e
                        elif (
                            "P3005" in e.stderr
                            and "database schema is not empty" in e.stderr
                        ):
                            logger.info(
                                "Database schema is not empty, creating baseline migration. In read-only file system, please set an environment variable `LITELLM_MIGRATION_DIR` to a writable directory to enable migrations. Learn more - https://docs.litellm.ai/docs/proxy/prod#read-only-file-system"
                            )
                            ProxyExtrasDBManager._create_baseline_migration(schema_path)
                            logger.info(
                                "Baseline migration created, marking all existing migrations as applied"
                            )
                            ProxyExtrasDBManager._mark_all_migrations_applied(
                                migrations_dir
                            )
                            # Now run prisma migrate deploy to apply any truly pending migrations
                            logger.info(
                                "Running prisma migrate deploy for any pending migrations..."
                            )
                            result = subprocess.run(
                                [_get_prisma_command(), "migrate", "deploy"],
                                timeout=60,
                                check=True,
                                capture_output=True,
                                text=True,
                                env=_get_prisma_env(),
                            )
                            logger.info(
                                f"prisma migrate deploy stdout: {result.stdout}"
                            )
                            logger.info("✅ All migrations applied.")
                            return True
                        elif "P3018" in e.stderr:
                            # Check if this is a permission error or idempotent error
                            if ProxyExtrasDBManager._is_permission_error(e.stderr):
                                # Permission errors should NOT be marked as applied
                                # Extract migration name for logging
                                migration_match = re.search(
                                    r"Migration name: (\d+_.*)", e.stderr
                                )
                                migration_name = (
                                    migration_match.group(1)
                                    if migration_match
                                    else "unknown"
                                )

                                logger.error(
                                    f"❌ Migration {migration_name} failed due to insufficient permissions. "
                                    f"Please check database user privileges. Error: {e.stderr}"
                                )

                                # Mark as rolled back and exit with error
                                if migration_match:
                                    try:
                                        ProxyExtrasDBManager._roll_back_migration(
                                            migration_name
                                        )
                                        logger.info(
                                            f"Migration {migration_name} marked as rolled back"
                                        )
                                    except Exception as rollback_error:
                                        logger.warning(
                                            f"Failed to mark migration as rolled back: {rollback_error}"
                                        )

                                # Re-raise the error to prevent silent failures
                                raise RuntimeError(
                                    f"Migration failed due to permission error. Migration {migration_name} "
                                    f"was NOT applied. Please grant necessary database permissions and retry."
                                ) from e

                            elif ProxyExtrasDBManager._is_idempotent_error(e.stderr):
                                # Idempotent errors mean the migration has effectively been applied
                                logger.info(
                                    "Migration failed due to idempotent error (e.g., column already exists), "
                                    "resolving as applied"
                                )
                                # Extract the migration name from the error message
                                migration_match = re.search(
                                    r"Migration name: (\d+_.*)", e.stderr
                                )
                                if migration_match:
                                    migration_name = migration_match.group(1)
                                    logger.info(
                                        f"Rolling back migration {migration_name}"
                                    )
                                    ProxyExtrasDBManager._roll_back_migration(
                                        migration_name
                                    )
                                    logger.info(
                                        f"Resolving migration {migration_name} that failed "
                                        f"due to existing schema objects"
                                    )
                                    ProxyExtrasDBManager._resolve_specific_migration(
                                        migration_name
                                    )
                                    logger.info("✅ Migration resolved.")
                            else:
                                # Unknown P3018 error - log and re-raise for safety
                                logger.warning(
                                    f"P3018 error encountered but could not classify "
                                    f"as permission or idempotent error. "
                                    f"Error: {e.stderr}"
                                )
                                raise
                else:
                    # Use prisma db push with increased timeout
                    subprocess.run(
                        [_get_prisma_command(), "db", "push", "--accept-data-loss"],
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

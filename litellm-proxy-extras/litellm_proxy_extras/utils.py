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
from litellm.caching.redis_cache import RedisCache


def str_to_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.lower() in ("true", "1", "t", "y", "yes")


class MigrationLockManager:
    """Redis-based lock manager for database migrations"""

    MIGRATION_LOCK_KEY = "migration_lock"
    LOCK_TTL_SECONDS = 300  # 5 minutes TTL

    def __init__(self, redis_cache: Optional[RedisCache] = None):
        self.redis_cache = redis_cache
        self.lock_acquired = False
        self.pod_id = f"pod_{os.getpid()}_{int(time.time())}"

    def _get_redis_lock_key(self) -> str:
        """Get Redis lock key for migration"""
        return f"migration_lock:{self.MIGRATION_LOCK_KEY}"

    def acquire_lock(self) -> bool:
        """Acquire migration lock"""
        if self.redis_cache is None:
            logger.warning(
                "Redis cache is not available, running migration without lock protection"
            )
            self.lock_acquired = True
            return True

        try:
            lock_key = self._get_redis_lock_key()

            # Redis SET with NX (only if not exists) and EX (expiration)
            acquired = self.redis_cache.set_cache(
                key=lock_key, value=self.pod_id, nx=True, ttl=self.LOCK_TTL_SECONDS
            )

            if acquired:
                self.lock_acquired = True
                logger.info(f"Migration lock acquired by pod {self.pod_id}")
                return True
            else:
                logger.info("Migration lock is already held by another pod")
                return False

        except Exception as e:
            logger.warning(f"Failed to acquire migration lock: {e}")
            return False

    def wait_for_lock_release(
        self, check_interval: int = 5, max_wait: int = 300
    ) -> bool:
        """Wait for another process to release the lock"""
        if self.redis_cache is None:
            logger.warning("Redis cache is not available, cannot wait for lock")
            return False

        logger.info(f"Waiting for migration lock to be released (max {max_wait}s)...")
        start_time = time.time()

        while time.time() - start_time < max_wait:
            # Try to acquire lock using the public acquire_lock method
            if self.acquire_lock():
                logger.info(
                    f"Migration lock acquired after waiting by pod {self.pod_id}"
                )
                return True

            time.sleep(check_interval)

        logger.warning(f"Failed to acquire migration lock within {max_wait} seconds")
        return False

    def release_lock(self):
        """Release migration lock"""
        if not self.lock_acquired or self.redis_cache is None:
            return

        try:
            lock_key = self._get_redis_lock_key()

            # Verify current pod owns the lock
            current_value = self.redis_cache.get_cache(lock_key)
            if current_value and str(current_value) == self.pod_id:
                self.redis_cache.delete_cache(lock_key)
                logger.info(f"Migration lock released by pod {self.pod_id}")
            else:
                logger.warning(f"Pod {self.pod_id} cannot release lock (not owner)")

        except Exception as e:
            logger.warning(f"Failed to release migration lock: {e}")
        finally:
            self.lock_acquired = False

    def __enter__(self):
        """Context manager entry - acquire lock when entering with statement"""
        self.acquire_lock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release lock when exiting with statement"""
        self.release_lock()

def _get_prisma_env() -> dict:
    """Get environment variables for Prisma, handling offline mode if configured."""
    prisma_env = os.environ.copy()
    if str_to_bool(os.getenv("PRISMA_OFFLINE_MODE")):
        # These env vars prevent Prisma from attempting downloads
        prisma_env["NPM_CONFIG_PREFER_OFFLINE"] = "true"
        prisma_env["NPM_CONFIG_CACHE"] = os.getenv("NPM_CONFIG_CACHE", "/app/.cache/npm")
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
                env=prisma_env
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
                env=prisma_env
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
            [_get_prisma_command(), "migrate", "resolve", "--rolled-back", migration_name],
            timeout=60,
            check=True,
            capture_output=True,
            env=prisma_env
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
            env=prisma_env
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
        ]

        for pattern in idempotent_patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                return True
        return False

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
        if not database_url:
            logger.error("DATABASE_URL not set")
            return
        
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
                        _get_prisma_command(),
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
                    env=_get_prisma_env()
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
                    _get_prisma_command(),
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
                env=_get_prisma_env()
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
                    [_get_prisma_command(), "migrate", "resolve", "--applied", migration_name],
                    timeout=60,
                    check=True,
                    capture_output=True,
                    text=True,
                    env=_get_prisma_env()
                )
                logger.debug(f"Resolved migration: {migration_name}")
            except subprocess.CalledProcessError as e:
                if "is already recorded as applied in the database." not in e.stderr:
                    logger.warning(
                        f"Failed to resolve migration {migration_name}: {e.stderr}"
                    )

    @staticmethod
    def setup_database(
        use_migrate: bool = False, redis_cache: Optional[RedisCache] = None
    ) -> bool:
        """
        Set up the database using either prisma migrate or prisma db push
        Uses migrations from litellm-proxy-extras package.
        In multi-instance environment, use redis lock to prevent concurrent execution.

        Args:
            schema_path (str): Path to the Prisma schema file
            use_migrate (bool): Whether to use prisma migrate instead of db push
            redis_cache: Redis cache instance for distributed locking

        Returns:
            bool: True if setup was successful, False otherwise
        """
        schema_path = ProxyExtrasDBManager._get_prisma_dir() + "/schema.prisma"

        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            logger.error("DATABASE_URL environment variable is not set")
            return False

        # Use MigrationLockManager to prevent concurrent migration execution
        with MigrationLockManager(redis_cache) as lock_manager:
            # Lock is already acquired in __enter__, check if it was successful
            if not lock_manager.lock_acquired:
                # Cannot acquire lock, another process is running migration
                logger.info(
                    "Another pod is running migration, waiting for completion..."
                )

                # Wait for other process to complete migration
                if not lock_manager.wait_for_lock_release():
                    logger.error("Failed to acquire migration lock after waiting")
                    return False

            # Successfully acquired lock, proceed with migration
            logger.info("Acquired migration lock, proceeding with migration")
            return ProxyExtrasDBManager._execute_migration(use_migrate, schema_path)

    @staticmethod
    def _execute_migration(use_migrate: bool, schema_path: str) -> bool:
        """Execute the actual migration"""
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
                            env=_get_prisma_env()
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
                                        _get_prisma_command(),
                                        "migrate",
                                        "resolve",
                                        "--rolled-back",
                                        failed_migration,
                                    ],
                                    timeout=60,
                                    check=True,
                                    capture_output=True,
                                    text=True,
                                    env=_get_prisma_env()
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

import glob
import os
import random
import re
import shutil
import subprocess
import tempfile
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


_MIGRATION_TS_RE = re.compile(r"^(\d{14})_")


def _migration_timestamp(name: str) -> int:
    """Extract the leading `YYYYMMDDHHMMSS` timestamp from a migration name.

    Returns 0 if the name doesn't match the Prisma pattern — unexpected-format
    entries sort as "oldest" and are treated as historical.
    """
    m = _MIGRATION_TS_RE.match(name)
    return int(m.group(1)) if m else 0


def _max_migration_timestamp(names) -> int:
    """Max timestamp in a set/list of migration names (0 if empty)."""
    if not names:
        return 0
    return max(_migration_timestamp(n) for n in names)


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
        # Prefer DIRECT_URL for schema introspection — pooler URLs (e.g. neon -pooler)
        # do not support the extended query protocol required by prisma migrate diff.
        diff_url = os.getenv("DIRECT_URL") or database_url

        diff_dir = Path(tempfile.mkdtemp(prefix="litellm_migration_diff_"))
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
                        diff_url,
                        "--to-schema-datamodel",
                        schema_path,
                        "--script",
                    ],
                    check=True,
                    timeout=60,
                    stdout=f,
                    env=_get_prisma_env(),
                )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to generate migration diff: {e.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning("Migration diff generation timed out.")

        # check if the migration was created
        if not diff_sql_path.exists():
            logger.warning(
                "Migration diff was not created (prisma migrate diff failed — "
                "likely a pooler URL). Falling back to direct SQL execution of "
                "each migration file."
            )
            # Fall back: run each migration SQL file directly via prisma db execute.
            # This works with pooler URLs (no schema introspection needed) and is
            # safe to re-run because migrations use IF NOT EXISTS / IF EXISTS guards.
            migration_files = sorted(Path(migrations_dir).glob("*/migration.sql"))
            for mig_file in migration_files:
                try:
                    subprocess.run(
                        [
                            _get_prisma_command(),
                            "db",
                            "execute",
                            "--file",
                            str(mig_file),
                            "--schema",
                            schema_path,
                        ],
                        timeout=60,
                        check=True,
                        capture_output=True,
                        text=True,
                        env=_get_prisma_env(),
                    )
                    logger.info(f"Applied migration: {mig_file.parent.name}")
                except subprocess.CalledProcessError as e:
                    logger.warning(
                        f"Failed to apply migration {mig_file.parent.name}: {e.stderr}"
                    )
                except subprocess.TimeoutExpired:
                    logger.warning(f"Migration {mig_file.parent.name} timed out.")
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
                env=_get_prisma_env(),
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
    def _strip_prisma_query_params(url: str) -> str:
        """Remove Prisma-specific query params (connection_limit, pool_timeout,
        schema, etc.) from DATABASE_URL so psycopg can parse it."""
        from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

        parsed = urlparse(url)
        if not parsed.query:
            return url
        libpq_params = {
            "sslmode",
            "sslcert",
            "sslkey",
            "sslrootcert",
            "sslpassword",
            "application_name",
            "connect_timeout",
            "client_encoding",
            "options",
            "service",
            "gssencmode",
            "krbsrvname",
            "target_session_attrs",
        }
        kept = [(k, v) for k, v in parse_qsl(parsed.query) if k in libpq_params]
        return urlunparse(parsed._replace(query=urlencode(kept)))

    @staticmethod
    def _warn_if_db_ahead_of_head(migrations_dir: str) -> None:
        """
        Log a warning if _prisma_migrations contains applied migrations with
        timestamps newer than every migration this build ships.

        This is informational only for the v2 resolver — it tells the operator
        the DB was likely migrated by a newer deployment, which is usually a
        signal that this (older) version shouldn't run against it. We do NOT
        block startup: many users have weird _prisma_migrations state from
        prior thrashing bugs, and blocking them would be a breaking change.

        Safe no-op if psycopg isn't installed or DB isn't reachable.
        """
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return

        try:
            import psycopg
        except ImportError:
            return

        cleaned_url = ProxyExtrasDBManager._strip_prisma_query_params(database_url)
        known = set(ProxyExtrasDBManager._get_migration_names(migrations_dir))

        try:
            # autocommit=True keeps the SELECT outside a transaction. Without
            # it, psycopg3's `with conn` calls COMMIT on clean exit — which
            # fails after `UndefinedTable` (fresh DB) leaves the transaction
            # in an aborted state.
            with psycopg.connect(
                cleaned_url, connect_timeout=10, autocommit=True
            ) as conn:
                try:
                    rows = conn.execute(
                        "SELECT migration_name FROM _prisma_migrations "
                        "WHERE finished_at IS NOT NULL AND rolled_back_at IS NULL"
                    ).fetchall()
                except psycopg.errors.UndefinedTable:
                    return
        except (psycopg.OperationalError, psycopg.DatabaseError):
            # Swallow connection failures AND any other DB-layer error
            # (e.g. InsufficientPrivilege if the runtime user lacks SELECT
            # on _prisma_migrations). This is an informational check —
            # never block startup on it.
            return

        applied = {r[0] for r in rows}
        unknown = applied - known
        if not unknown:
            return

        head_newest_ts = _max_migration_timestamp(known)
        hostile = {
            name for name in unknown if _migration_timestamp(name) > head_newest_ts
        }
        if not hostile:
            return

        sorted_hostile = sorted(hostile)
        logger.warning(
            "Database has %d migration(s) applied that are NEWER than any "
            "migration this LiteLLM version ships. This usually means the "
            "database was migrated by a newer LiteLLM deployment. Some API "
            "endpoints may fail because this proxy's Prisma client does not "
            "know about those schema changes. Consider upgrading this "
            "deployment. Unknown: %s",
            len(hostile),
            ", ".join(sorted_hostile[:5]) + (" ..." if len(sorted_hostile) > 5 else ""),
        )

    @staticmethod
    def _setup_database_v2(use_migrate: bool) -> bool:
        """
        v2 migration resolver (opt-in via --use_v2_migration_resolver).

        Runs `prisma migrate deploy` and handles standard recovery paths
        (P3005 baseline, P3009/P3018 idempotent errors). Critically, it does
        NOT call `_resolve_all_migrations` — the diff-and-force recovery that
        caused schema thrashing when two LiteLLM versions contended for the
        same DB during rolling deploys.

        Ahead-of-HEAD state (DB has migrations newer than this build ships)
        is logged as a warning, not a fatal error — users whose DBs got into
        weird shapes from the old thrashing should still be able to start.
        """
        schema_path = ProxyExtrasDBManager._get_prisma_dir() + "/schema.prisma"
        migrations_dir = ProxyExtrasDBManager._get_prisma_dir()

        if not use_migrate:
            # Preserve `prisma db push` path unchanged.
            original_dir = os.getcwd()
            os.chdir(migrations_dir)
            try:
                subprocess.run(
                    [_get_prisma_command(), "db", "push", "--accept-data-loss"],
                    timeout=60,
                    check=True,
                    env=_get_prisma_env(),
                )
                return True
            except (
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ) as e:
                # Re-raise as RuntimeError so proxy_cli.py's
                # `except RuntimeError` catches it and exits cleanly.
                raise RuntimeError(f"prisma db push failed.\n\nDetail: {e}") from e
            finally:
                os.chdir(original_dir)

        # Informational — never blocks.
        ProxyExtrasDBManager._warn_if_db_ahead_of_head(migrations_dir)

        original_dir = os.getcwd()
        os.chdir(migrations_dir)
        try:
            for attempt in range(4):
                try:
                    result = subprocess.run(
                        [_get_prisma_command(), "migrate", "deploy"],
                        timeout=60,
                        check=True,
                        capture_output=True,
                        text=True,
                        env=_get_prisma_env(),
                    )
                    logger.info(f"prisma migrate deploy stdout: {result.stdout}")
                    return True

                except subprocess.TimeoutExpired:
                    logger.info(
                        f"prisma migrate deploy attempt {attempt + 1} timed out, retrying"
                    )
                    time.sleep(random.randrange(5, 15))
                    continue

                except subprocess.CalledProcessError as e:
                    stderr = e.stderr or ""

                    if "P3005" in stderr and "database schema is not empty" in stderr:
                        logger.info(
                            "Schema exists but no migrations ledger — creating baseline"
                        )
                        ProxyExtrasDBManager._create_baseline_migration(schema_path)
                        continue

                    if "P3009" in stderr:
                        migration_match = re.search(r"`(\d+_\S+?)`", stderr)
                        if (
                            migration_match
                            and ProxyExtrasDBManager._is_idempotent_error(stderr)
                        ):
                            name = migration_match.group(1)
                            logger.info(
                                f"Migration {name} failed idempotently — marking applied and retrying"
                            )
                            try:
                                ProxyExtrasDBManager._roll_back_migration(name)
                            except (
                                subprocess.CalledProcessError,
                                subprocess.TimeoutExpired,
                            ):
                                pass  # may already be rolled-back
                            try:
                                ProxyExtrasDBManager._resolve_specific_migration(name)
                            except (
                                subprocess.CalledProcessError,
                                subprocess.TimeoutExpired,
                            ) as resolve_err:
                                # We're already inside the outer
                                # `except CalledProcessError` handler —
                                # re-raising CalledProcessError from here
                                # would escape as itself, bypassing
                                # proxy_cli.py's `except RuntimeError`.
                                raise RuntimeError(
                                    f"Failed to mark migration {name} as applied "
                                    f"after idempotent recovery. Manual "
                                    f"intervention may be required.\n\n"
                                    f"Detail: {resolve_err}"
                                ) from resolve_err
                            continue
                        raise RuntimeError(
                            "Database migration failed and cannot be auto-recovered. "
                            f"Manual intervention required.\n\nPrisma error:\n{stderr}"
                        ) from e

                    if "P3018" in stderr:
                        if ProxyExtrasDBManager._is_permission_error(stderr):
                            raise RuntimeError(
                                "Database migration failed due to insufficient "
                                "permissions. Please grant the required privileges "
                                f"and retry.\n\nPrisma error:\n{stderr}"
                            ) from e

                        migration_match = re.search(
                            r"Migration name: (\d+_\S+)", stderr
                        )
                        if (
                            migration_match
                            and ProxyExtrasDBManager._is_idempotent_error(stderr)
                        ):
                            name = migration_match.group(1)
                            logger.info(
                                f"Migration {name} SQL hit idempotent error — marking applied and retrying"
                            )
                            try:
                                ProxyExtrasDBManager._roll_back_migration(name)
                            except (
                                subprocess.CalledProcessError,
                                subprocess.TimeoutExpired,
                            ):
                                pass  # may already be rolled-back
                            try:
                                ProxyExtrasDBManager._resolve_specific_migration(name)
                            except (
                                subprocess.CalledProcessError,
                                subprocess.TimeoutExpired,
                            ) as resolve_err:
                                raise RuntimeError(
                                    f"Failed to mark migration {name} as applied "
                                    f"after idempotent recovery. Manual "
                                    f"intervention may be required.\n\n"
                                    f"Detail: {resolve_err}"
                                ) from resolve_err
                            continue

                        raise RuntimeError(
                            "Database migration failed and cannot be auto-recovered. "
                            f"Manual intervention required.\n\nPrisma error:\n{stderr}"
                        ) from e

                    raise RuntimeError(
                        "Database migration failed and cannot be auto-recovered. "
                        f"Manual intervention required.\n\nPrisma error:\n{stderr}"
                    ) from e

            raise RuntimeError(
                "Database migration failed after 4 attempts (retry loop "
                "exhausted by timeouts or repeated idempotent-recovery "
                "continues). Check database connectivity, load, and "
                "_prisma_migrations ledger state."
            )
        finally:
            os.chdir(original_dir)

    @staticmethod
    def setup_database(
        use_migrate: bool = False, use_v2_resolver: bool = False
    ) -> bool:
        """
        Set up the database using either prisma migrate or prisma db push
        Uses migrations from litellm-proxy-extras package

        Args:
            use_migrate: Whether to use prisma migrate instead of db push
            use_v2_resolver: Opt into the v2 migration resolver (safer during
                rolling deploys; does not run the diff-and-force recovery
                that causes schema thrashing). Defaults to False for
                backwards compatibility.

        Returns:
            bool: True if setup was successful, False otherwise
        """
        if use_v2_resolver:
            logger.info("Using v2 migration resolver (--use_v2_migration_resolver)")
            return ProxyExtrasDBManager._setup_database_v2(use_migrate=use_migrate)

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

                        logger.info("prisma migrate deploy completed")

                        # Skip sanity check when deploy reports no pending migrations —
                        # DB already matches schema, no drift to correct.
                        if "No pending migrations to apply" in result.stdout:
                            logger.info(
                                "No pending migrations — skipping post-migration sanity check"
                            )
                            return True

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
                                if ProxyExtrasDBManager._is_idempotent_error(e.stderr):
                                    logger.info(
                                        f"Migration {failed_migration} failed due to idempotent error (e.g., column already exists), resolving as applied"
                                    )
                                    try:
                                        ProxyExtrasDBManager._roll_back_migration(
                                            failed_migration
                                        )
                                    except (
                                        subprocess.CalledProcessError,
                                        subprocess.TimeoutExpired,
                                    ) as rollback_err:
                                        logger.warning(
                                            f"Failed to roll back migration {failed_migration}: {rollback_err}. "
                                            f"It may already be in a rolled-back state."
                                        )
                                    try:
                                        ProxyExtrasDBManager._resolve_specific_migration(
                                            failed_migration
                                        )
                                        logger.info(
                                            f"✅ Migration {failed_migration} resolved, retrying to apply remaining migrations"
                                        )
                                    except (
                                        subprocess.CalledProcessError,
                                        subprocess.TimeoutExpired,
                                    ) as resolve_err:
                                        logger.warning(
                                            f"Failed to resolve migration {failed_migration}: {resolve_err}"
                                        )
                                    # Apply any schema drift not covered by the marked-as-applied migration
                                    ProxyExtrasDBManager._resolve_all_migrations(
                                        migrations_dir,
                                        schema_path,
                                        mark_all_applied=False,
                                    )
                                else:
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
                                        env=_get_prisma_env(),
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
                                    try:
                                        logger.info(
                                            f"Rolling back migration {migration_name}"
                                        )
                                        ProxyExtrasDBManager._roll_back_migration(
                                            migration_name
                                        )
                                    except (
                                        subprocess.CalledProcessError,
                                        subprocess.TimeoutExpired,
                                    ) as rollback_err:
                                        logger.warning(
                                            f"Failed to roll back migration {migration_name}: {rollback_err}. "
                                            f"It may already be in a rolled-back state."
                                        )
                                    try:
                                        logger.info(
                                            f"Resolving migration {migration_name} that failed "
                                            f"due to existing schema objects"
                                        )
                                        ProxyExtrasDBManager._resolve_specific_migration(
                                            migration_name
                                        )
                                        logger.info(
                                            f"✅ Migration {migration_name} resolved, "
                                            f"retrying to apply remaining migrations"
                                        )
                                    except (
                                        subprocess.CalledProcessError,
                                        subprocess.TimeoutExpired,
                                    ) as resolve_err:
                                        logger.warning(
                                            f"Failed to resolve migration {migration_name}: {resolve_err}"
                                        )
                                    # Apply any schema drift not covered by the marked-as-applied migration
                                    ProxyExtrasDBManager._resolve_all_migrations(
                                        migrations_dir,
                                        schema_path,
                                        mark_all_applied=False,
                                    )
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

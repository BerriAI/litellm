"""Optional post-migration step that raises Postgres REPLICA IDENTITY to FULL.

Logical-replication consumers (Neon / lakehouse sync and similar) need FULL
replica identity to reconstruct the old row of an UPDATE or DELETE. Prisma
leaves every table it creates at the Postgres default, so the setting has to be
re-applied by hand after each migration run. Setting
``LITELLM_SET_REPLICA_IDENTITY_FULL`` makes every migration run re-assert it.

The statement goes through the Prisma CLI rather than a Postgres driver because
``litellm-proxy-extras`` has no runtime dependencies, while the CLI is already
required for the migrations themselves.
"""

import subprocess
import tempfile
from pathlib import Path

from litellm_proxy_extras._logging import logger
from litellm_proxy_extras.prisma_toolchain import prisma_command_timeout

REPLICA_IDENTITY_FULL_ENV_VAR = "LITELLM_SET_REPLICA_IDENTITY_FULL"

REPLICA_IDENTITY_FULL_SQL = r"""
DO $$
DECLARE
    target regclass;
BEGIN
    SET LOCAL lock_timeout = '5s';
    FOR target IN
        SELECT c.oid::regclass
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relkind = 'r'
          AND c.relreplident <> 'f'
          AND n.nspname = ANY (current_schemas(false))
          AND c.relname LIKE 'LiteLLM\_%'
    LOOP
        BEGIN
            EXECUTE format('ALTER TABLE %s REPLICA IDENTITY FULL', target);
        EXCEPTION WHEN lock_not_available THEN
            RAISE WARNING 'REPLICA IDENTITY FULL skipped for %: table busy, retrying next run', target;
        END;
    END LOOP;
END
$$;
"""


def apply_replica_identity_full(
    schema_path: str,
    prisma_command: str,
    prisma_env: dict[str, str],
) -> bool:
    """Set REPLICA IDENTITY FULL on every LiteLLM table that is not already FULL.

    Never raises. Replication metadata is not needed to serve requests, so
    every failure mode is reported and stepped over rather than taking down a
    migration run that already succeeded: a database that refuses the ALTER
    (most often because the runtime user does not own the tables), a missing
    or unrunnable Prisma CLI, a read-only temp directory, or a timeout.

    Returns True when the statement was applied, False when it failed.
    """
    logger.info("Applying REPLICA IDENTITY FULL to LiteLLM tables")
    try:
        with tempfile.TemporaryDirectory(prefix="litellm_replica_identity_") as tmp_dir:
            sql_path = Path(tmp_dir) / "replica_identity_full.sql"
            sql_path.write_text(REPLICA_IDENTITY_FULL_SQL)
            subprocess.run(
                [
                    prisma_command,
                    "db",
                    "execute",
                    "--file",
                    str(sql_path),
                    "--schema",
                    schema_path,
                ],
                timeout=prisma_command_timeout(),
                check=True,
                capture_output=True,
                text=True,
                env=prisma_env,
            )
    except subprocess.CalledProcessError as e:
        logger.error(
            "Failed to set REPLICA IDENTITY FULL. Logical replication "
            "consumers may reject updates to these tables. Grant table "
            "ownership to the migration user, or apply "
            "`ALTER TABLE ... REPLICA IDENTITY FULL` by hand. Error: %s",
            e.stderr,
        )
        return False
    except subprocess.TimeoutExpired:
        logger.error("Timed out setting REPLICA IDENTITY FULL on LiteLLM tables")
        return False
    except OSError as e:
        logger.error(
            "Could not run the REPLICA IDENTITY FULL statement. Logical "
            "replication consumers may reject updates to these tables. "
            "Error: %s",
            e,
        )
        return False

    logger.info("REPLICA IDENTITY FULL applied to LiteLLM tables")
    return True

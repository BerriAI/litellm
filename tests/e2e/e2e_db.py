"""Destructive DB reset for the e2e harness.

Kept at the top level next to e2e_config and lifecycle so every suite imports it
by name (`from e2e_db import ...`); no suite reaches into another's directory by
mutating sys.path.

reset_database drops the schema and replays every migration, so it removes spend
logs, keys, teams, users and orgs and leaves the DB at the migration head.
"""

import os
from pathlib import Path

from pydantic import TypeAdapter

LOCAL_DATABASE_URL = "postgresql://llmproxy:dbpassword9090@localhost:5432/litellm"

# The schema whose sibling `migrations/` holds the real history the proxy deploys
# (litellm/proxy/schema.prisma has no migrations dir, so resetting against it would
# drop the schema with nothing to replay). Absolute, because pytest's working
# directory is not the repo root.
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "litellm-proxy-extras" / "litellm_proxy_extras" / "schema.prisma"


def resolve_database_url() -> str:
    """The connection string to reset, mirroring how the proxy builds its own.

    On RDS with IAM auth there is no durable `DATABASE_URL` for the prisma CLI to
    read: the proxy assembles one at startup from `DATABASE_HOST`/`PORT`/`USER`/
    `NAME` plus a short-lived IAM token (see proxy_cli.py's "GET DB TOKEN FOR IAM
    AUTH"), and that token rotates roughly every 12 minutes. A reset that relied on
    the CLI picking `DATABASE_URL` out of the environment would abort on stage with
    "environment variable not found", or silently hit the local-docker default.

    Precedence, most specific first:
      1. An explicit `DATABASE_URL` (local docker, Neon, a tunnel).
      2. RDS parts + a freshly minted IAM token, when `IAM_TOKEN_DB_AUTH` is on.
      3. RDS parts + `DATABASE_PASSWORD`, for password auth.
      4. The local docker default.
    """
    explicit_url = os.environ.get("DATABASE_URL")
    if explicit_url:
        return explicit_url

    db_host = os.environ.get("DATABASE_HOST")
    db_user = os.environ.get("DATABASE_USER") or os.environ.get("DATABASE_USERNAME")
    db_name = os.environ.get("DATABASE_NAME")
    if not (db_host and db_user and db_name):
        return LOCAL_DATABASE_URL

    db_port = os.environ.get("DATABASE_PORT", "5432")
    secret = _resolve_database_secret(db_host=db_host, db_port=db_port, db_user=db_user)
    url = f"postgresql://{db_user}:{secret}@{db_host}:{db_port}/{db_name}"
    db_schema = os.environ.get("DATABASE_SCHEMA")
    return f"{url}?schema={db_schema}" if db_schema else url


def _resolve_database_secret(*, db_host: str, db_port: str, db_user: str) -> str:
    """An IAM auth token when IAM auth is enabled, else `DATABASE_PASSWORD`.

    The token is minted per call and URL-encoded by `generate_iam_auth_token`; it is
    valid for ~15 minutes, which is ample for one reset but is why it must never be
    cached across a session.
    """
    iam_enabled = (os.environ.get("IAM_TOKEN_DB_AUTH") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if not iam_enabled:
        return os.environ.get("DATABASE_PASSWORD", "")

    # Reuse the proxy's own minting path so the token matches what the gateway
    # presents. It is untyped (boto has no stub for generate_db_auth_token), so
    # validate the result instead of trusting it: an empty or non-string token
    # would be spliced into the URL and surface as an opaque auth failure.
    from litellm.proxy.auth.rds_iam_token import (  # noqa: PLC0415 - boto import is expensive
        generate_iam_auth_token,  # pyright: ignore[reportUnknownVariableType]  # untyped helper; result validated below
    )

    token = TypeAdapter(str).validate_python(
        generate_iam_auth_token(  # pyright: ignore[reportUnknownArgumentType]  # untyped helper
            db_host=db_host, db_port=db_port, db_user=db_user
        )
    )
    if not token:
        raise RuntimeError("RDS IAM auth token came back empty; cannot reach the database")
    return token


def reset_database() -> None:
    """`prisma migrate reset --force --skip-seed`: drop the schema, replay migrations.

    Runs through prisma's bundled Python CLI rather than `npx prisma`, so the reset
    does not depend on node being on PATH in the e2e runner image. `--skip-seed`
    keeps it from running a seed script; each suite's fixtures create what they need.

    `DATABASE_URL` is injected from resolve_database_url because the CLI cannot
    assemble an RDS/IAM connection string itself, and `--schema` is explicit and
    absolute because the repo carries three schema.prisma files and the CLI resolves
    a relative path against the working directory.
    """
    from prisma.cli import prisma as prisma_cli  # noqa: PLC0415

    if not SCHEMA_PATH.is_file():
        raise RuntimeError(f"prisma schema not found at {SCHEMA_PATH}")

    exit_code = prisma_cli.run(
        ["migrate", "reset", "--force", "--skip-seed", "--schema", str(SCHEMA_PATH)],
        check=False,
        env={"DATABASE_URL": resolve_database_url()},
    )
    if exit_code != 0:
        raise RuntimeError(f"Prisma reset failed with exit code {exit_code}")
    print(f"database reset: replayed migrations from {SCHEMA_PATH}")

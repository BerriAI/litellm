"""Assemble DATABASE_URL (+ optional DATABASE_URL_READ_REPLICA) from env.

The CLI (`proxy_cli.py`) assembles ``DATABASE_URL`` from discrete
``DATABASE_*`` env vars before Prisma initializes. The componentized
entrypoints (gateway / backend / migrations) bypass the CLI by uvicorn'ing
the app directly, so they call ``DatabaseURLSettings.from_env().apply_to_env()``
to do the same thing before importing ``proxy_server``.

The env var names this module reads are exactly the ones emitted by the
``helm/litellm`` chart's ``litellm.serverEnv`` block
(``helm/litellm/templates/_helpers.tpl``). Both auth styles and both
endpoints are covered:

  * IAM auth (``IAM_TOKEN_DB_AUTH`` truthy): mint a short-lived RDS IAM
    token and embed it as the password. The writer URL is always
    (re)written because the token is freshly minted on every startup. The
    chart omits ``DATABASE_PASSWORD`` in this mode.
  * Password auth: build a percent-encoded URL from ``DATABASE_PASSWORD``.
    The chart emits the discrete ``DATABASE_*`` fields (never a
    pre-assembled URL), so URL-reserved characters in the password survive
    instead of corrupting the URL. A pre-existing ``DATABASE_URL`` — e.g.
    one an operator pinned via ``extraEnv`` — is left untouched and wins.

The read replica is opt-in via ``DATABASE_HOST_READ_REPLICA`` and never
clobbers a pre-existing ``DATABASE_URL_READ_REPLICA``, so an IAM writer can
run alongside a password-auth reader (or a precomputed reader URL). Reader
IAM is gated on the single global ``IAM_TOKEN_DB_AUTH`` flag — the chart
only emits the reader IAM env vars when the writer also uses IAM auth.
Reader-side fields fall back to the writer's user / name / schema / port /
password when their ``*_READ_REPLICA`` counterpart is unset.
"""

import os
import urllib.parse
from typing import Optional, cast

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Imported as a module (not `from ... import generate_iam_auth_token`) so the
# AWS-touching token mint stays patchable at its canonical location in tests.
from litellm.proxy.auth import rds_iam_token

_IAM_ENV_KEY = "IAM_TOKEN_DB_AUTH"
_DEFAULT_PG_PORT = "5432"


class DatabaseURLSettings(BaseSettings):
    """Discrete ``DATABASE_*`` env vars, loaded once at process start.

    Field names are internal; ``validation_alias`` pins each one to the exact
    env var the helm chart emits. ``DATABASE_USER`` doubles as
    ``DATABASE_USERNAME`` for parity with ``construct_database_url_from_env_vars``.
    """

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    iam_token_db_auth: bool = Field(default=False, validation_alias=_IAM_ENV_KEY)

    # Writer
    database_url: Optional[str] = Field(default=None, validation_alias="DATABASE_URL")
    database_host: Optional[str] = Field(default=None, validation_alias="DATABASE_HOST")
    database_port: str = Field(
        default=_DEFAULT_PG_PORT, validation_alias="DATABASE_PORT"
    )
    database_user: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_USER", "DATABASE_USERNAME"),
    )
    database_name: Optional[str] = Field(default=None, validation_alias="DATABASE_NAME")
    database_schema: Optional[str] = Field(
        default=None, validation_alias="DATABASE_SCHEMA"
    )
    database_password: Optional[str] = Field(
        default=None, validation_alias="DATABASE_PASSWORD"
    )

    # Read replica
    database_url_read_replica: Optional[str] = Field(
        default=None, validation_alias="DATABASE_URL_READ_REPLICA"
    )
    database_host_read_replica: Optional[str] = Field(
        default=None, validation_alias="DATABASE_HOST_READ_REPLICA"
    )
    database_port_read_replica: Optional[str] = Field(
        default=None, validation_alias="DATABASE_PORT_READ_REPLICA"
    )
    database_user_read_replica: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "DATABASE_USER_READ_REPLICA", "DATABASE_USERNAME_READ_REPLICA"
        ),
    )
    database_name_read_replica: Optional[str] = Field(
        default=None, validation_alias="DATABASE_NAME_READ_REPLICA"
    )
    database_schema_read_replica: Optional[str] = Field(
        default=None, validation_alias="DATABASE_SCHEMA_READ_REPLICA"
    )
    database_password_read_replica: Optional[str] = Field(
        default=None, validation_alias="DATABASE_PASSWORD_READ_REPLICA"
    )

    @classmethod
    def from_env(cls) -> "DatabaseURLSettings":
        """Load the settings from ``os.environ`` (read at call time)."""
        return cls()

    def build_writer_url(self) -> Optional[str]:
        """Return the writer URL to set, or ``None`` to leave it as-is.

        Raises ``RuntimeError`` (naming the offending vars) when IAM auth is
        enabled but a required field is missing — the proxy cannot recover
        from this and a clear startup error beats a Prisma connect failure.
        """
        if self.iam_token_db_auth:
            missing = [
                env
                for env, val in (
                    ("DATABASE_HOST", self.database_host),
                    ("DATABASE_USER", self.database_user),
                    ("DATABASE_NAME", self.database_name),
                )
                if not val
            ]
            if missing:
                raise RuntimeError(
                    "IAM_TOKEN_DB_AUTH is enabled but required DB env var(s) "
                    f"are unset: {', '.join(missing)}. Set them so the writer "
                    "DATABASE_URL can be assembled with a minted IAM token."
                )
            host = cast(str, self.database_host)
            user = cast(str, self.database_user)
            name = cast(str, self.database_name)
            # IAM token is already URL-quoted by generate_iam_auth_token;
            # user/name embedded raw (parity with proxy_cli.py / IAMEndpoint).
            token = rds_iam_token.generate_iam_auth_token(
                db_host=host, db_port=self.database_port, db_user=user
            )
            url = f"postgresql://{user}:{token}@{host}:{self.database_port}/{name}"
            if self.database_schema:
                url += f"?schema={self.database_schema}"
            return url

        # Password auth: an operator-pinned DATABASE_URL always wins.
        if self.database_url:
            return None
        if self.database_host and self.database_user and self.database_name:
            return self._password_url(
                user=self.database_user,
                password=self.database_password,
                host=self.database_host,
                port=self.database_port,
                name=self.database_name,
                schema=self.database_schema,
            )
        return None

    def build_reader_url(self) -> Optional[str]:
        """Return the read-replica URL to set, or ``None`` to leave it as-is.

        Opt-in via ``DATABASE_HOST_READ_REPLICA``; never clobbers a
        pre-existing ``DATABASE_URL_READ_REPLICA``. Reader fields fall back
        to the writer's values.
        """
        if not self.database_host_read_replica:
            return None  # reader is opt-in
        if self.database_url_read_replica:
            return None  # never clobber an operator-supplied reader URL

        host = self.database_host_read_replica
        port = self.database_port_read_replica or self.database_port
        user = self.database_user_read_replica or self.database_user
        name = self.database_name_read_replica or self.database_name
        schema = self.database_schema_read_replica or self.database_schema
        password = self.database_password_read_replica or self.database_password

        if self.iam_token_db_auth:
            missing = [
                env
                for env, val in (
                    ("DATABASE_USER[_READ_REPLICA]", user),
                    ("DATABASE_NAME[_READ_REPLICA]", name),
                )
                if not val
            ]
            if missing:
                raise RuntimeError(
                    "IAM_TOKEN_DB_AUTH is enabled and DATABASE_HOST_READ_REPLICA "
                    "is set, but the reader could not resolve: "
                    f"{', '.join(missing)} (no *_READ_REPLICA value and no "
                    "writer fallback). Set the reader fields or the writer "
                    "defaults."
                )
            user = cast(str, user)
            name = cast(str, name)
            token = rds_iam_token.generate_iam_auth_token(
                db_host=host, db_port=port, db_user=user
            )
            url = f"postgresql://{user}:{token}@{host}:{port}/{name}"
            if schema:
                url += f"?schema={schema}"
            return url

        if user and name:
            return self._password_url(
                user=user,
                password=password,
                host=host,
                port=port,
                name=name,
                schema=schema,
            )
        return None

    @staticmethod
    def _password_url(
        *,
        user: str,
        password: Optional[str],
        host: str,
        port: str,
        name: str,
        schema: Optional[str],
    ) -> str:
        """Percent-encode credentials into a ``postgresql://`` URL.

        Parity with ``construct_database_url_from_env_vars`` in
        ``proxy/utils.py``; ``password`` may be empty for a passwordless URL.
        """
        quote = urllib.parse.quote_plus
        user_p = quote(user)
        name_p = quote(name)
        if password:
            url = f"postgresql://{user_p}:{quote(password)}@{host}:{port}/{name_p}"
        else:
            url = f"postgresql://{user_p}@{host}:{port}/{name_p}"
        if schema:
            url += f"?schema={schema}"
        return url

    def apply_to_env(self) -> bool:
        """Write the assembled URL(s) into ``os.environ``.

        Returns True iff this call set ``DATABASE_URL`` (IAM mint, or
        password auth that assembled a fresh URL). False means there was
        nothing to do — an operator-pinned URL, or no discrete fields.
        """
        wrote_writer = False
        writer_url = self.build_writer_url()
        if writer_url is not None:
            os.environ["DATABASE_URL"] = writer_url
            if self.iam_token_db_auth:
                # Normalize the toggle so downstream readers (PrismaWrapper's
                # IAM refresh) reliably see IAM on, regardless of spelling.
                os.environ[_IAM_ENV_KEY] = "True"
            wrote_writer = True

        reader_url = self.build_reader_url()
        if reader_url is not None:
            os.environ["DATABASE_URL_READ_REPLICA"] = reader_url

        return wrote_writer

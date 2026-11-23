from collections.abc import Callable
from importlib.resources import files
from typing import Final

import click
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

VERSION_TABLE: Final = "control_plane_alembic_version"
ADVISORY_LOCK: Final = 714883513942
MIGRATE_COMMAND: Final = "litellm --control-plane --skip_server_startup"


def database_url(value: SecretStr) -> str:
    raw: Final = value.get_secret_value()
    return raw.replace("postgresql://", "postgresql+psycopg://", 1) if raw.startswith("postgresql://") else raw


def engine_for(url: SecretStr) -> AsyncEngine:
    return create_async_engine(
        database_url(url),
        poolclass=NullPool,
        hide_parameters=True,
        connect_args={"prepare_threshold": None, "connect_timeout": 5},
    )


def migration_config() -> Config:
    config: Final = Config()
    config.set_main_option("script_location", str(files("litellm.proxy.control_plane.store").joinpath("migrations")))
    return config


def head_revision() -> str | None:
    return ScriptDirectory.from_config(migration_config()).get_current_head()


def migrate(url: SecretStr, upgrade: Callable[[Config, str], None] = command.upgrade) -> None:
    config: Final = migration_config()
    config.attributes["database_url"] = url
    upgrade(config, "head")


class MigrationSettings(BaseSettings):
    model_config = SettingsConfigDict(frozen=True, env_prefix="CONTROL_PLANE_", extra="ignore")

    database_url: SecretStr | None = None


def migrate_from_environment(migrate_database: Callable[[SecretStr], None] = migrate) -> None:
    url: Final = MigrationSettings().database_url
    if url is None or not url.get_secret_value().strip():
        raise click.ClickException("CONTROL_PLANE_DATABASE_URL is required for control-plane migrations.")
    try:
        migrate_database(url)
    except Exception as error:  # noqa: BLE001  # migration jobs must fail closed without exposing DSN credentials
        raise click.ClickException(
            f"Control-plane migration failed ({type(error).__name__}). Check the dedicated CONTROL_PLANE_DATABASE_URL and migration job logs."
        ) from None
    click.echo("Control-plane database migrations are current.")

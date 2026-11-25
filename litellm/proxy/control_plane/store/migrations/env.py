import asyncio
from typing import Final

from alembic import context
from pydantic import SecretStr, TypeAdapter
from sqlalchemy import Connection, text

from litellm.proxy.control_plane.store.migrate import ADVISORY_LOCK, VERSION_TABLE, engine_for
from litellm.proxy.control_plane.store.schema import metadata


def configure(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=metadata,
        version_table=VERSION_TABLE,
        transaction_per_migration=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run() -> None:
    url: Final = TypeAdapter(SecretStr).validate_python(context.config.attributes["database_url"])
    engine: Final = engine_for(url)
    try:
        # Keep the lock across per-revision commits, including through transaction pooling.
        async with engine.begin() as lock_connection:
            await lock_connection.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": ADVISORY_LOCK})
            async with engine.connect() as connection:
                await connection.run_sync(configure)
    finally:
        await engine.dispose()


asyncio.run(run())

import asyncio

from litellm.rust_bridge._native import NativeDatabase


async def _connect() -> NativeDatabase:
    return await NativeDatabase.connect()


def native_database() -> NativeDatabase:
    return asyncio.run(_connect())

import asyncio
from typing import Final

from litellm.proxy.db.native_database import native_endpoint
from litellm.rust_bridge._native import NativeDatabase

_NEVER_DIALED_URL: Final = "postgresql://litellm:secret@db.invalid:5432/proxy"


async def connect_native_database() -> NativeDatabase:
    return await NativeDatabase.connect(native_endpoint(_NEVER_DIALED_URL, tls_offered=True, max_connections=1).endpoint)


def native_database() -> NativeDatabase:
    return asyncio.run(connect_native_database())

import asyncio
from typing import Final

from litellm.proxy.db.native_database import ServiceLoggingEvents
from litellm.rust_bridge._native import DatabaseEndpoint, NativeDatabase

_NEVER_DIALED_URL: Final = "postgresql://litellm:secret@db.invalid:5432/proxy"


async def connect_native_database() -> NativeDatabase:
    return await NativeDatabase.connect(
        ServiceLoggingEvents(loop=asyncio.get_running_loop(), service_logging=lambda: None),
        await DatabaseEndpoint.resolve(_NEVER_DIALED_URL, 1),
        None,
    )


def native_database() -> NativeDatabase:
    return asyncio.run(connect_native_database())

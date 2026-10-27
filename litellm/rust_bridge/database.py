from collections.abc import Awaitable, Mapping
from typing import Final, Protocol, runtime_checkable

from pydantic import ConfigDict, TypeAdapter

from litellm.rust_bridge.loader import get_native_bridge


class NativeDatabase(Protocol):
    def find_user_banner(self) -> Awaitable[object]: ...
    def save_user_banner(self, settings: Mapping[str, object]) -> Awaitable[None]: ...
    def close(self) -> Awaitable[None]: ...


class NativeDatabaseFactory(Protocol):
    def connect(self, url: str, max_connections: int) -> Awaitable[NativeDatabase]: ...


@runtime_checkable
class NativeDatabaseBridge(Protocol):
    NativeDatabase: NativeDatabaseFactory


_NATIVE: Final[TypeAdapter[NativeDatabaseBridge]] = TypeAdapter(
    NativeDatabaseBridge, config=ConfigDict(arbitrary_types_allowed=True)
)


async def connect_native_database(url: str, max_connections: int) -> NativeDatabase:
    bridge: Final = get_native_bridge()
    if bridge is None:
        raise RuntimeError("The native database requires the Rust extension")
    return await _NATIVE.validate_python(bridge).NativeDatabase.connect(url, max_connections)

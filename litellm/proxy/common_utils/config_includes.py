from collections.abc import Awaitable, Mapping
from typing import Final, Protocol

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.rust_bridge import _native
from litellm.rust_bridge.execution import Execution, drive

_MERGED_CONFIG: Final = TypeAdapter(dict[str, object])


class ConfigReader(Protocol):
    def __call__(self, location: str, /) -> Awaitable[Mapping[str, object]]: ...


def _never_streams(_execution: Execution, _head: object) -> object:
    raise RuntimeError("resolving config includes opened a stream")


def _warn(message: str) -> None:
    verbose_proxy_logger.warning(message)


async def _merged(execution: Execution) -> dict[str, object]:
    return _MERGED_CONFIG.validate_python(await drive(execution, _never_streams))


async def resolve_file_includes(
    config: Mapping[str, object], config_file_path: str, read: ConfigReader
) -> dict[str, object]:
    """
    Merge every config file named by the `include` directive into the config that declares it.

    List values are extended and every other value is overridden. An entry names a file next to the config
    that declares it, a file already pulled in is neither read nor merged a second time, and `read` decides
    how a path is read. `config_file_path` is the absolute path of the root config.
    """
    return await _merged(_native.resolve_file_includes(dict(config), config_file_path, read, _warn))


async def resolve_object_includes(
    config: Mapping[str, object], object_key: str, read: ConfigReader
) -> dict[str, object]:
    """
    Merge every bucket object named by the `include` directive into the config that declares it.

    The merge is the one `resolve_file_includes` applies. An entry names an object under the prefix of the
    config object that declares it, and a leading "/" means the bucket root, mirroring how an absolute path
    on disk ignores the directory the including config sits in.
    """
    return await _merged(_native.resolve_object_includes(dict(config), object_key, read, _warn))

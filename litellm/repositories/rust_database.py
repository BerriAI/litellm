"""The worker's Rust database and the Rust repositories built over it, used only by the Python repositories.

``Database.open`` installs the ``NativeDatabase`` it opened and ``Database.aclose`` clears it. A Python repository
asks ``required_repository`` (or ``repository``) for its Rust repository and turns the answer into a domain model
with ``decode``; nothing outside ``litellm/repositories/`` touches a Rust repository. Each one is built once per
installed database, so its own memory tier lives as long as the pool does.
"""

from collections.abc import Callable
from typing import Final, NoReturn, TypeVar

from pydantic import BaseModel

from litellm.rust_bridge._native import NativeDatabase

RepositoryT = TypeVar("RepositoryT")
ModelT = TypeVar("ModelT", bound=BaseModel)


class _Installed:
    __slots__ = ("database", "repositories")

    def __init__(self) -> None:
        self.database: NativeDatabase | None = None
        self.repositories: Final[dict[Callable[[NativeDatabase], object], object]] = {}  # mutable-ok: one per kind


_installed: Final = _Installed()


def install(database: NativeDatabase) -> None:
    _installed.repositories.clear()
    _installed.database = database


def clear(database: NativeDatabase) -> None:
    """Forgets ``database`` and its repositories, unless another database was installed since."""
    if _installed.database is database:
        _installed.repositories.clear()
        _installed.database = None


def current() -> NativeDatabase | None:
    return _installed.database


def repository(kind: Callable[[NativeDatabase], RepositoryT]) -> RepositoryT | None:
    """The ``kind`` Rust repository over the installed database, or None when no Rust database is open."""
    database: Final = _installed.database
    if database is None:
        return None
    existing: Final = _installed.repositories.get(kind)
    if existing is not None:
        return existing  # pyright: ignore[reportReturnType]  # stored under its own kind
    built: Final = kind(database)
    _installed.repositories[kind] = built
    return built


class NoDatabaseConnected(Exception):
    """A Python repository needed the Rust database before ``Database.open`` installed one."""

    def __init__(self) -> None:
        super().__init__("No db connected")


def _raise_no_db_connected() -> NoReturn:
    raise NoDatabaseConnected()


def required_repository(kind: Callable[[NativeDatabase], RepositoryT]) -> RepositoryT:
    """The ``kind`` Rust repository, raising the proxy's "No db connected" error when no Rust database is open."""
    found: Final = repository(kind)
    if found is None:
        _raise_no_db_connected()
    return found


def decode(model_class: type[ModelT], payload: str | None) -> ModelT | None:
    """The domain model a Rust repository's JSON answer describes, or None when it found no row."""
    return None if payload is None else model_class.model_validate_json(payload)

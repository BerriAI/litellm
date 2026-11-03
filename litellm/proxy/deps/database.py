from typing import Annotated, Final, TypeAlias

from fastapi import Depends, Request

from litellm.repositories.database import Database

_MISSING_DATABASE: Final = object()


class DatabaseStateMissingError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("The proxy lifespan did not publish a Database in request state")


async def get_database(request: Request) -> Database | None:
    database: Final[object] = getattr(request.state, "database", _MISSING_DATABASE)
    if database is None or isinstance(database, Database):
        return database
    raise DatabaseStateMissingError()


DatabaseDep: TypeAlias = Annotated[Database | None, Depends(get_database)]

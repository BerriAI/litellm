from typing import Annotated, Final, TypeAlias

from fastapi import Depends, Request
from pydantic import ConfigDict, TypeAdapter

from litellm._db import Database

_DATABASE_ADAPTER: Final[TypeAdapter[Database | None]] = TypeAdapter(
    Database | None, config=ConfigDict(arbitrary_types_allowed=True)
)
_MISSING_DATABASE: Final = object()


async def get_database(request: Request) -> Database | None:
    return _DATABASE_ADAPTER.validate_python(getattr(request.state, "database", _MISSING_DATABASE))


DatabaseDependency: TypeAlias = Annotated[Database | None, Depends(get_database)]

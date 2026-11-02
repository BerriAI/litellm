from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm.caching.redis_cache import RedisCache

_KWARGS: Final = TypeAdapter(dict[str, object])


def redis_kwargs(connection: RedisCache) -> Mapping[str, object]:
    return _KWARGS.validate_python(
        connection.redis_kwargs  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped legacy dict
    )

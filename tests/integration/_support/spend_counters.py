import os
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from typing import Final

from redis import Redis

SPEND_COUNTER_TTL_SECONDS: Final = 3600
TTL_SLACK_SECONDS: Final = 600


@contextmanager
def rig_cache() -> Iterator[Redis]:
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache:
        yield cache


def key_counter(key: str) -> str:
    return f"spend:key:{sha256(key.encode()).hexdigest()}"


def counter_ttl(cache: Redis, counter_key: str) -> int:
    return int(str(cache.ttl(counter_key)))


def counter_value(cache: Redis, counter_key: str) -> float | None:
    raw: Final = cache.get(counter_key)
    return None if raw is None else float(str(raw, "utf-8") if isinstance(raw, bytes) else str(raw))


def outlives(ttl: int, expected: int = SPEND_COUNTER_TTL_SECONDS) -> bool:
    return expected - TTL_SLACK_SECONDS < ttl <= expected

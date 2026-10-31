from dataclasses import dataclass
from typing import Literal

from litellm.rust_bridge._native import NativeRedisFailure


@dataclass(frozen=True, slots=True)
class RedisFailure:
    kind: Literal["transport", "server", "decode", "breaker_open"]
    message: str
    timeout: bool = False


def from_native(failure: NativeRedisFailure) -> RedisFailure:
    return RedisFailure(kind=failure.kind, message=failure.message, timeout=failure.timeout)

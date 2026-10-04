import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

_RUN_STARTED_AT: Final = int(time.time())
_UUID: Final = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


@dataclass(frozen=True, eq=False)
class MatchesPattern:
    name: str
    pattern: str

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str) and re.fullmatch(self.pattern, other) is not None

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, eq=False)
class TimestampDuringRun:
    name: str

    def __eq__(self, other: object) -> bool:
        return type(other) is int and _RUN_STARTED_AT <= other <= int(time.time())

    def __repr__(self) -> str:
        return self.name


CHAT_COMPLETION_ID: Final = MatchesPattern("CHAT_COMPLETION_ID", rf"chatcmpl-{_UUID}")
RESPONSE_ID: Final = MatchesPattern("RESPONSE_ID", r"resp_[A-Za-z0-9_-]+={0,2}")
RESPONSE_MESSAGE_ID: Final = MatchesPattern("RESPONSE_MESSAGE_ID", rf"msg_{_UUID}")
UNIX_TIMESTAMP: Final = TimestampDuringRun("UNIX_TIMESTAMP")

Matcher: TypeAlias = MatchesPattern | TimestampDuringRun
ExpectedValue: TypeAlias = (
    Matcher | str | int | float | bool | None | Sequence["ExpectedValue"] | Mapping[str, "ExpectedValue"]
)

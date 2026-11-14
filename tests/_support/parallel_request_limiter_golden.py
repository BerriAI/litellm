import dataclasses
import json
import math
import os
import re
from collections.abc import Mapping, Set
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Final

from pydantic import BaseModel

GOLDEN_RECORD_ENV: Final = "LITELLM_PARALLEL_REQUEST_LIMITER_GOLDEN_RECORD"
GOLDEN_PATH: Final = (
    Path(__file__).resolve().parents[1] / "unit" / "proxy" / "hooks" / "parallel_request_limiter_v3_golden.json"
)
TTL_TOLERANCE_MS: Final = 10_000
_STAMP: Final = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?: UTC|[+-]\d{2}:\d{2}|Z)?")
_NUMERIC: Final = re.compile(r"-?\d+(?:\.\d+)?")
_SPAN: Final = 10**7


@dataclasses.dataclass(frozen=True, eq=False)
class Ttl:
    milliseconds: int

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Ttl) and abs(self.milliseconds - other.milliseconds) < TTL_TOLERANCE_MS


def _instant(text: str) -> datetime:
    parsed: Final = datetime.fromisoformat(text.removesuffix(" UTC").replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _relative(seconds: float, origin: float) -> str | None:
    if not origin:
        return None
    if abs(seconds - origin) < _SPAN:
        return f"<t{seconds - origin:+.3f}>"
    if abs(seconds / 1000 - origin) < _SPAN:
        return f"<tms{seconds / 1000 - origin:+.3f}>"
    return None


def _text(value: str, origin: float) -> str:
    if _NUMERIC.fullmatch(value) and (relative := _relative(float(value), origin)) is not None:
        return f"str{relative}"
    if not origin:
        return value
    return _STAMP.sub(lambda match: f"<t{_instant(match.group(0)).timestamp() - origin:+.3f}>", value)


def plain(value: object, origin: float) -> object:
    match value:
        case Ttl(milliseconds=milliseconds):
            return {"__ttl_s__": -(-milliseconds // 1000)}
        case bool() | None:
            return value
        case int() | float():
            if isinstance(value, float) and not math.isfinite(value):
                return {"__float__": repr(value)}
            relative = _relative(value, origin)
            return value if relative is None else relative
        case str():
            return _text(value, origin)
        case bytes():
            return {"__bytes__": value.hex()}
        case BaseException():
            return {"__error__": type(value).__name__, "text": _text(str(value), origin)}
        case BaseModel():
            return {"__model__": type(value).__name__, **_mapping(value.model_dump(mode="json"), origin)}
        case SimpleNamespace():
            return {"__namespace__": _mapping(vars(value), origin)}
        case Mapping():
            return _mapping(value, origin)
        case Set():
            return {"__set__": sorted((plain(item, origin) for item in value), key=_order)}
        case list() | tuple():
            return [plain(item, origin) for item in value]
        case _ if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {"__dataclass__": type(value).__name__, **_mapping(dataclasses.asdict(value), origin)}
        case _:
            raise TypeError(f"no golden form for {type(value).__qualname__}")


def _mapping(value: Mapping[object, object], origin: float) -> dict[str, object]:
    return {str(key): plain(item, origin) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}


def _order(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def matches(expected: object, actual: object) -> bool:
    match expected, actual:
        case {"__ttl_s__": int() as want}, {"__ttl_s__": int() as got}:
            return abs(want - got) * 1000 < TTL_TOLERANCE_MS
        case dict(), dict():
            return expected.keys() == actual.keys() and all(matches(expected[key], actual[key]) for key in expected)
        case list(), list():
            return len(expected) == len(actual) and all(matches(want, got) for want, got in zip(expected, actual))
        case _:
            return expected == actual


class Golden:
    def __init__(self, path: Path = GOLDEN_PATH) -> None:
        self.target: Final = os.environ.get(GOLDEN_RECORD_ENV)
        self.expected: Final[dict[str, list[object]]] = (
            {} if self.target or not path.exists() else json.loads(path.read_text())
        )
        self.recorded: Final[dict[str, list[object]]] = {}
        self.case = ""
        self.origin = 0.0
        self.position = 0
        self.ran = False

    def start(self, case: str) -> None:
        self.case = case
        self.origin = 0.0
        self.position = 0
        self.ran = False

    def anchor(self, start: datetime) -> None:
        self.origin = start.timestamp()
        self.ran = True

    def agree(self, python: object, native: object, context: object = None) -> None:
        assert python == native, (context, python, native)
        self.ran = True
        if self.target:
            self.recorded.setdefault(self.case, []).append(plain(python, self.origin))
            return
        expected: Final = self.expected.get(self.case)
        assert expected is not None, f"{self.case} has no recorded golden values"
        assert self.position < len(expected), (self.case, self.position, context)
        actual: Final = plain(native, self.origin)
        assert matches(expected[self.position], actual), (self.case, self.position, context, expected[self.position])
        self.position += 1

    def finish_case(self) -> None:
        if self.target or not self.ran:
            return
        expected: Final = self.expected.get(self.case, [])
        assert self.position == len(expected), (self.case, self.position)

    def write(self) -> None:
        if self.target:
            lines = (
                f"{json.dumps(case)}: {json.dumps(points, sort_keys=True, separators=(',', ':'))}"
                for case, points in sorted(self.recorded.items())
            )
            Path(self.target).write_text("{\n" + ",\n".join(lines) + "\n}\n")

import dataclasses
import enum
import json
import math
from collections.abc import Mapping
from contextvars import ContextVar
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final

from pydantic import BaseModel

GOLDEN_FILE: Final = Path(__file__).resolve().parent.parent / "test_model_capacity_native.golden.json"


@dataclasses.dataclass
class _Case:
    node: str
    observed: int = 0


_CASE: Final[ContextVar[_Case | None]] = ContextVar("model_capacity_golden_case", default=None)


def encode(value: object) -> object:
    match value:
        case None | bool() | str():
            return value
        case int():
            return {"int": str(value)} if abs(value) > 2**53 else value
        case float():
            return {"float": repr(value)} if math.isnan(value) or math.isinf(value) else {"float": value}
        case tuple():
            return {"tuple": [encode(item) for item in value]}
        case list():
            return [encode(item) for item in value]
        case set() | frozenset():
            return {"set": sorted((encode(item) for item in value), key=_order)}
        case enum.Enum():
            return {"enum": type(value).__name__, "value": encode(value.value)}
        case BaseModel():
            return {"model": type(value).__name__, "fields": encode(value.model_dump(mode="python"))}
        case datetime() | date():
            return {"datetime": value.isoformat()}
        case timedelta():
            return {"timedelta": value.total_seconds()}
        case BaseException():
            return {"exception": type(value).__name__, "text": str(value)}
        case Mapping():
            return {"dict": sorted(([encode(key), encode(item)] for key, item in value.items()), key=_order)}
        case _ if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                "dataclass": type(value).__name__,
                "fields": {field.name: encode(getattr(value, field.name)) for field in dataclasses.fields(value)},
            }
        case _:
            raise TypeError(f"no golden encoding for {type(value).__name__}: {value!r}")


def _order(encoded: object) -> str:
    return json.dumps(encoded, sort_keys=True)


def _load() -> Mapping[str, list[object]]:
    loaded: Final = json.loads(GOLDEN_FILE.read_text())
    assert isinstance(loaded, dict)
    return loaded


def begin(node: str) -> None:
    _CASE.set(_Case(node=node))


def finish() -> None:
    case: Final = _CASE.get()
    assert case is not None, "golden values are observed inside a test case"
    entries: Final = _load().get(case.node, [])
    assert case.observed == len(entries), f"{case.node} checked {case.observed} of {len(entries)} golden values"


def _next() -> tuple[str, int]:
    case: Final = _CASE.get()
    assert case is not None, "golden values are observed inside a test case"
    index: Final = case.observed
    case.observed += 1
    return case.node, index


def check(native: object) -> None:
    node, index = _next()
    entries: Final = _load().get(node)
    assert entries is not None and index < len(entries), f"no golden value {index} for {node}"
    assert encode(native) == entries[index], f"{node} value {index} differs from the recorded python value"

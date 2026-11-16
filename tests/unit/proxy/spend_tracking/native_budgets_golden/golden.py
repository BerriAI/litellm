from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from datetime import datetime
from functools import cache, reduce
from pathlib import Path
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, TypeAdapter

HERE: Final = Path(__file__).parent
NATIVE_BUDGETS: Final = HERE / "native_budgets.json"
NATIVE_BUDGETS_PROXY_PATH: Final = HERE / "native_budgets_proxy_path.json"

RECORDINGS: Final[dict[Path, dict[str, dict[str, object]]]] = {}  # mutable-ok: filled only by the recording script

_CASES: Final = TypeAdapter(dict[str, dict[str, object]])


def _jsonable(value: object) -> object:
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value):
        return {item.name: getattr(value, item.name) for item in dataclasses.fields(value)}
    raise TypeError(f"no golden form for {type(value).__name__}")


@cache
def _recorded(path: Path) -> Mapping[str, Mapping[str, object]]:
    return _CASES.validate_python(json.loads(path.read_text())["python"])


@dataclasses.dataclass(frozen=True, slots=True)
class Golden:
    path: Path
    test: str
    replacements: Mapping[str, str] = MappingProxyType({})

    def replacing(self, replacements: Mapping[str, str]) -> Golden:
        return dataclasses.replace(self, replacements=MappingProxyType({**self.replacements, **replacements}))

    def canonical(self, value: object) -> object:
        text: Final = reduce(
            lambda encoded, replacement: encoded.replace(*replacement),
            self.replacements.items(),
            json.dumps(value, default=_jsonable),
        )
        decoded: Final[object] = json.loads(text)
        return decoded

    def recorded(self, label: str, value: object) -> object:
        recording: Final = RECORDINGS.get(self.path)
        if recording is None:
            return _recorded(self.path)[self.test][label]
        recording.setdefault(self.test, {})[label] = self.canonical(value)
        return self.canonical(value)

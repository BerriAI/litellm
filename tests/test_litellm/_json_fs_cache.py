from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, object])


def canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True, slots=True)
class JsonFileCache:
    root: Path

    def path_for(self, key: Mapping[str, object]) -> Path:
        digest: Final = hashlib.sha256(canonical_json(key).encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def get(self, key: Mapping[str, object]) -> dict[str, object] | None:
        path: Final = self.path_for(key)
        if not path.is_file():
            return None
        return JSON_OBJECT.validate_json(path.read_text(encoding="utf-8"))

    def put(self, key: Mapping[str, object], value: Mapping[str, object]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path: Final = self.path_for(key)
        temporary_path: Final = path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary_path.replace(path)
        return path

    def values(self) -> tuple[dict[str, object], ...]:
        if not self.root.is_dir():
            return ()
        paths: Final = tuple(sorted(self.root.rglob("*.json")))
        return tuple(JSON_OBJECT.validate_json(path.read_text(encoding="utf-8")) for path in paths)

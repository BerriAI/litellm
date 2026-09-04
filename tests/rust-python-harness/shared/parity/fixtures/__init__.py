from __future__ import annotations

from typing import Final

from pydantic import TypeAdapter

JSON_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, object])

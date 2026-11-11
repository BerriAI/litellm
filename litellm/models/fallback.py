from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Final, Protocol, TypeAlias

from pydantic import Field, TypeAdapter

FallbackRule = dict[str, list[str]]
StoredFallback = Annotated[FallbackRule | dict[str, object] | str, Field(union_mode="left_to_right")]
STORED_FALLBACKS: Final = TypeAdapter(list[StoredFallback])


@dataclass(frozen=True, slots=True)
class FallbackRulesEdit:
    rules: Sequence[StoredFallback]
    replaced: bool


FallbackRulesEditor: TypeAlias = Callable[[Sequence[StoredFallback]], FallbackRulesEdit]


class RouterFallbackStore(Protocol):
    async def edit_rules(self, fallback_key: str, edit: FallbackRulesEditor) -> FallbackRulesEdit: ...

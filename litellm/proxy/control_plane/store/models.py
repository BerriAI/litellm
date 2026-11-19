from collections.abc import Mapping
from typing import Final, Literal, TypeAlias

from pydantic import ConfigDict, Field, JsonValue, TypeAdapter

from litellm.types.llms.base_model import LiteLLMBaseModel

Kind: TypeAlias = Literal["dataplane", "role"]
Entity: TypeAlias = Mapping[str, JsonValue]
ENTITY: Final = TypeAdapter(Entity)


class Change(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["upsert", "delete", "resync"]
    kind: Kind = "dataplane"
    name: str = Field(default="", max_length=2048)
    entity: Entity = Field(default_factory=dict)
    actor: str


class Entry(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    revision: str = Field(pattern=r"^\d+-\d+$")
    change: Change


def revision_key(revision: str) -> tuple[int, int]:
    milliseconds, sequence = revision.split("-")
    return int(milliseconds), int(sequence)


def maximum_revision(entries: tuple[Entry, ...]) -> str:
    return max((entry.revision for entry in entries), key=revision_key, default="0-0")

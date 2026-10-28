from importlib.resources import files
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter


class Operation(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    method: str
    path: str
    operation_id: str
    summary: str
    read_only_override: bool | None = None

    @property
    def read_only(self) -> bool:
        return self.method == "GET" if self.read_only_override is None else self.read_only_override


OPERATIONS: Final = TypeAdapter(tuple[Operation, ...]).validate_json(
    files("litellm.proxy.admin_mcp").joinpath("operations.json").read_text()
)
BY_NAME: Final = MappingProxyType({operation.name: operation for operation in OPERATIONS})

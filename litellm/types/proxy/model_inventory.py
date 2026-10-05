from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class SupplierModelInventory:
    models: Mapping[str, Mapping[str, object]]
    credential_scope: str | None = None


@dataclass(frozen=True, slots=True)
class SupplierInventoryUnavailable:
    reason: Literal["authentication", "http", "transport", "malformed"]
    credential_scope: str | None = None

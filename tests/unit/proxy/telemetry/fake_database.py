import json
from dataclasses import dataclass
from typing import Final

from typing_extensions import LiteralString


@dataclass
class SettingsDatabase:
    """In-memory ``LiteLLM_Config`` holding the telemetry instance id, hash secret and stored groups"""

    stored_groups: str | None = None
    fail_reads: bool = False
    instance_id: str = "persisted-install"
    hash_secret: str = "persisted-secret"

    async def query_raw(self, query: LiteralString, *args: object) -> object:
        if self.fail_reads:
            raise OSError("db down")
        if "AS value" in query:
            return ({"value": self.instance_id if args[0] == "telemetry_instance_id" else self.hash_secret},)
        stored: Final = self.stored_groups
        return () if stored is None else ({"param_value": json.loads(stored)},)

    async def execute_raw(self, query: LiteralString, *args: object) -> int:
        if args and args[0] == "telemetry_settings":
            self.stored_groups = str(args[1])
        return 1

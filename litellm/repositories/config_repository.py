"""Config repository for database operations on LiteLLM_Config."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, Protocol, cast

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


def _decoded_json(raw: str) -> object:
    """Decode a JSON-encoded config row value into an opaque object."""
    return cast(object, json.loads(raw))


class _ConfigRow(Protocol):
    @property
    def param_name(self) -> str: ...

    @property
    def param_value(self) -> object: ...


class _ConfigTable(Protocol):
    async def find_unique(self, *, where: Mapping[str, str]) -> _ConfigRow | None: ...

    async def find_many(self) -> Sequence[_ConfigRow]: ...

    async def upsert(self, *, where: Mapping[str, str], data: Mapping[str, Mapping[str, str]]) -> _ConfigRow: ...

    async def delete(self, *, where: Mapping[str, str]) -> _ConfigRow | None: ...


class ConfigParam:
    """Simple wrapper for config parameter from DB."""

    def __init__(self, param_name: str, param_value: object):
        self.param_name = param_name
        self.param_value = param_value


class ConfigRepository:
    """Repository for config database operations."""

    def __init__(self, prisma_client: PrismaClient | None):
        self._prisma_client: Final = prisma_client

    @property
    def prisma_client(self) -> PrismaClient:
        if self._prisma_client is None:
            raise RuntimeError("No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys")
        return self._prisma_client

    @property
    def _config_table(self) -> _ConfigTable:
        return cast(_ConfigTable, self.prisma_client.db.litellm_config)

    @property
    def table(self) -> _ConfigTable:
        return self._config_table

    async def get_param(self, param_name: str) -> ConfigParam | None:
        """Get a config parameter from the database."""
        record: Final = await self._config_table.find_unique(where={"param_name": param_name})
        if record is None:
            return None
        param_value: object = record.param_value
        if isinstance(param_value, str):
            param_value = _decoded_json(param_value)
        return ConfigParam(param_name=param_name, param_value=param_value)

    async def set_param(self, param_name: str, param_value: object) -> ConfigParam:
        """Set a config parameter in the database."""
        value_json: Final = json.dumps(param_value) if not isinstance(param_value, str) else param_value
        await self._config_table.upsert(
            where={"param_name": param_name},
            data={
                "create": {"param_name": param_name, "param_value": value_json},
                "update": {"param_value": value_json},
            },
        )
        return ConfigParam(param_name=param_name, param_value=param_value)

    async def delete_param(self, param_name: str) -> bool:
        """Delete a config parameter from the database."""
        try:
            await self._config_table.delete(where={"param_name": param_name})
            return True
        except Exception:
            return False

    async def get_all_params(self) -> dict[str, object]:
        """Get all config parameters from the database."""
        records: Final = await self._config_table.find_many()
        result: Final[dict[str, object]] = {}
        for record in records:
            param_value: object = record.param_value
            if isinstance(param_value, str):
                param_value = _decoded_json(param_value)
            result[record.param_name] = param_value
        return result

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

    @property
    def reload_revision(self) -> int | None: ...


class _ConfigTable(Protocol):
    async def find_unique(self, *, where: Mapping[str, str]) -> _ConfigRow | None: ...

    async def find_many(self) -> Sequence[_ConfigRow]: ...

    async def upsert(self, *, where: Mapping[str, str], data: Mapping[str, Mapping[str, str]]) -> _ConfigRow: ...

    async def delete(self, *, where: Mapping[str, str]) -> _ConfigRow | None: ...


class _ConfigDatabase(Protocol):
    async def query_raw(self, query: str, *args: object) -> object: ...


class ConfigParam:
    """Simple wrapper for config parameter from DB."""

    def __init__(self, param_name: str, param_value: object):
        self.param_name = param_name
        self.param_value = param_value


class ConfigRepository:
    """Repository for config database operations."""

    def __init__(self, prisma_client: PrismaClient | None, *, use_writer: bool = False):
        self._prisma_client: Final = prisma_client
        self._use_writer: Final = use_writer

    @property
    def prisma_client(self) -> PrismaClient:
        if self._prisma_client is None:
            raise RuntimeError("No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys")
        return self._prisma_client

    @property
    def _config_table(self) -> _ConfigTable:
        database: Final = self.prisma_client.writer_db if self._use_writer else self.prisma_client.db
        return cast(_ConfigTable, database.litellm_config)

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

    async def set_param_if_revision(self, param_name: str, param_value: object, revision: int) -> bool:
        delegate: Final = self.prisma_client.writer_db
        database: Final = cast(_ConfigDatabase, delegate)  # cast-ok: Prisma delegates database methods dynamically
        rows: Final = await database.query_raw(
            """INSERT INTO "LiteLLM_Config" (param_name, param_value, last_run_at)
               SELECT $1, $2::jsonb, NOW() WHERE $3::int = 0
               ON CONFLICT (param_name) DO UPDATE
               SET param_value = EXCLUDED.param_value, last_run_at = NOW()
               WHERE COALESCE(("LiteLLM_Config".param_value->>'revision')::int, 0) = $3::int
               RETURNING param_name"""
            if revision == 0
            else """UPDATE "LiteLLM_Config" SET param_value = $2::jsonb, last_run_at = NOW()
               WHERE param_name = $1 AND COALESCE((param_value->>'revision')::int, 0) = $3::int
               RETURNING param_name""",
            param_name,
            json.dumps(param_value),
            revision,
        )
        return bool(rows)

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

"""Config repository for database operations on LiteLLM_Config."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import TYPE_CHECKING, Final, Protocol, cast

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.models.fallback import STORED_FALLBACKS, FallbackRulesEdit, FallbackRulesEditor, SavedFallbackRule

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

    async def delete_many(self, *, where: Mapping[str, object]) -> int: ...


class _ConfigTransaction(Protocol):
    @property
    def litellm_config(self) -> _ConfigTable: ...

    async def query_raw(self, query: str, *args: object) -> object: ...


class _ConfigWriter(_ConfigTransaction, Protocol):
    def tx(self) -> AbstractAsyncContextManager[_ConfigTransaction]: ...


CONFIG_PARAM_LOCK_SQL: Final = "SELECT 1 AS locked FROM pg_advisory_xact_lock(hashtext($1))"
ROUTER_SETTINGS_PARAM: Final = "router_settings"
_JSON_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_JSON_RULES: Final = TypeAdapter(list[JsonValue])
_ROUTER_SETTINGS: Final = TypeAdapter(dict[str, JsonValue])


def _stored_router_settings(param_value: object) -> Mapping[str, JsonValue]:
    try:
        decoded: Final[object] = _JSON_VALUE.validate_json(param_value) if isinstance(param_value, str) else param_value
    except ValidationError:
        return {}
    if not isinstance(decoded, Mapping):
        return {}
    return _ROUTER_SETTINGS.validate_python(decoded)


def _edited_router_settings(
    param_value: object, fallback_key: str, edit: FallbackRulesEditor
) -> tuple[Mapping[str, JsonValue], FallbackRulesEdit]:
    router_settings: Final = _stored_router_settings(param_value)
    edited: Final = edit(STORED_FALLBACKS.validate_python(router_settings.get(fallback_key) or []))
    return {**router_settings, fallback_key: _JSON_RULES.validate_python(edited.rules)}, edited


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

    @property
    def _writer(self) -> _ConfigWriter:
        return cast(  # cast-ok: Prisma delegates database methods and transactions dynamically
            _ConfigWriter, self.prisma_client.writer_db
        )

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
        rows: Final = await self._writer.query_raw(
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

    async def saved_rules(self, fallback_key: str) -> Sequence[SavedFallbackRule] | None:
        row: Final = await self._config_table.find_unique(where={"param_name": ROUTER_SETTINGS_PARAM})
        if row is None:
            return None
        rules: Final = _stored_router_settings(row.param_value).get(fallback_key)
        if rules is None:
            return None
        return rules if isinstance(rules, list) else ()

    async def edit_rules(self, fallback_key: str, edit: FallbackRulesEditor) -> FallbackRulesEdit:
        param_name: Final = ROUTER_SETTINGS_PARAM
        async with self._writer.tx() as transaction:
            await transaction.query_raw(CONFIG_PARAM_LOCK_SQL, param_name)
            row: Final = await transaction.litellm_config.find_unique(where={"param_name": param_name})
            value, result = _edited_router_settings(None if row is None else row.param_value, fallback_key, edit)
            value_json: Final = json.dumps(dict(value))
            await transaction.litellm_config.upsert(
                where={"param_name": param_name},
                data={
                    "create": {"param_name": param_name, "param_value": value_json},
                    "update": {"param_value": value_json},
                },
            )
        return result

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

    async def delete_params_older_than(self, prefix: str, before: datetime) -> int:
        """Expire operational config records using the timestamp written by revision claims."""
        return await self._config_table.delete_many(
            where={"param_name": {"startswith": prefix}, "last_run_at": {"lt": before}}
        )

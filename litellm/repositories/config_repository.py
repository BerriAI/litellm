"""
Config repository for database operations on LiteLLM_Config.

This repository handles config reconciliation between database values and
YAML configmap values. DB values override configmap values except for
None values and empty lists.
"""

import asyncio
import copy
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol, cast

from pydantic import TypeAdapter
from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper


def _decoded_json(raw: str) -> object:
    """Decode a JSON-encoded config row value into an opaque object."""
    return json.loads(raw)


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


class _ConfigDb(Protocol):
    @property
    def litellm_config(self) -> _ConfigTable: ...


class _ConfigTx(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...

    @property
    def litellm_config(self) -> _ConfigTable: ...


class _ConfigTxManager(Protocol):
    async def __aenter__(self) -> _ConfigTx: ...

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> bool | None: ...


class _PrismaHandle(Protocol):
    @property
    def db(self) -> _ConfigDb: ...

    def tx(self) -> _ConfigTxManager: ...


LITELLM_SETTINGS_PARAM: Final = "litellm_settings"

_CONFIG_ADVISORY_LOCK_SQL: Final = "SELECT pg_advisory_xact_lock(hashtext($1)) IS NULL AS locked"

_SETTINGS_ADAPTER: Final = TypeAdapter(dict[str, object])

_STRING_LIST_ADAPTER: Final = TypeAdapter(tuple[str, ...])


@dataclass(frozen=True, slots=True)
class SettingsApplied:
    settings: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SettingsRejected:
    reason: str


SettingsUpdate = SettingsApplied | SettingsRejected


class SettingsTransform(Protocol):
    def __call__(self, settings: Mapping[str, object], /) -> SettingsUpdate: ...


_EMPTY_SETTINGS: Final[Mapping[str, object]] = MappingProxyType({})


def decode_settings(param_value: object) -> Mapping[str, object]:
    decoded: Final[object] = json.loads(param_value) if isinstance(param_value, str) else param_value
    if decoded is None:
        return _EMPTY_SETTINGS
    return _SETTINGS_ADAPTER.validate_python(decoded)


def encode_settings(settings: Mapping[str, object]) -> str:
    return json.dumps(dict(settings))


async def _upsert_param(table: _ConfigTable, param_name: str, value_json: str) -> None:
    await table.upsert(
        where={"param_name": param_name},
        data={
            "create": {"param_name": param_name, "param_value": value_json},
            "update": {"param_value": value_json},
        },
    )


def public_hub_list(settings: Mapping[str, object], key: str, fallback: Sequence[str]) -> tuple[str, ...]:
    stored: Final = settings.get(key)
    if stored is None:
        return tuple(fallback)
    return _STRING_LIST_ADAPTER.validate_python(stored)


class ConfigParam:
    """Simple wrapper for config parameter from DB."""

    def __init__(self, param_name: str, param_value: object):
        self.param_name = param_name
        self.param_value = param_value


class ConfigRepository:
    """Repository for config database operations with reconciliation support."""

    CONFIG_PARAMS = [
        "general_settings",
        "router_settings",
        "litellm_settings",
        "environment_variables",
    ]

    def __init__(self, prisma_client: Any):
        self._prisma_client = prisma_client

    @property
    def prisma_client(self) -> _PrismaHandle:
        if self._prisma_client is None:
            raise RuntimeError("No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys")
        return self._prisma_client

    @property
    def _config_table(self) -> _ConfigTable:
        return self.prisma_client.db.litellm_config

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
        await _upsert_param(self._config_table, param_name, value_json)
        return ConfigParam(param_name=param_name, param_value=param_value)

    async def update_litellm_settings(self, apply: SettingsTransform) -> SettingsUpdate:
        """Serialize concurrent writers with an advisory lock rather than ``SELECT ... FOR UPDATE``,
        since the row does not exist yet on a fresh database and so cannot be row-locked."""
        async with self.prisma_client.tx() as tx:
            await tx.query_raw(_CONFIG_ADVISORY_LOCK_SQL, LITELLM_SETTINGS_PARAM)

            record: Final = await tx.litellm_config.find_unique(where={"param_name": LITELLM_SETTINGS_PARAM})
            current: Final = decode_settings(record.param_value if record is not None else None)

            result: Final = apply(current)
            match result:
                case SettingsRejected():
                    return result
                case SettingsApplied(settings=settings):
                    await _upsert_param(tx.litellm_config, LITELLM_SETTINGS_PARAM, encode_settings(settings))
                    return result
        assert_never(result)

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

    def _deep_merge_dicts(self, dst: dict, src: dict) -> None:
        """Deep-merge src into dst, skipping None values and empty lists from src.

        On conflicts, src (DB) wins, but empty lists are treated as "no value"
        and don't overwrite the destination.
        """
        stack: Final = [(dst, src)]
        while stack:
            d, s = stack.pop()
            for k, v in s.items():
                if v is None:
                    continue
                if isinstance(v, list) and len(v) == 0:
                    continue
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    stack.append((d[k], v))
                else:
                    d[k] = v

    def _decrypt_env_variables(
        self, env_vars: Mapping[str, object], return_original_value: bool = True
    ) -> dict[str, str]:
        """Decrypt environment variables from database."""
        decrypted: Final[dict[str, str]] = {}
        for key, value in env_vars.items():
            if isinstance(value, str):
                decrypted_value = decrypt_value_helper(
                    value=value,
                    key=key,
                    exception_type="debug",
                    return_original_value=return_original_value,
                )
                if decrypted_value is not None:
                    decrypted[key] = decrypted_value
            else:
                decrypted[key] = str(value)
        return decrypted

    def _normalize_env_variable_keys(self, env_vars: dict[str, str]) -> dict[str, str]:
        """Normalize env variable keys to include both original and uppercase versions."""
        normalized: Final[dict[str, str]] = {}
        for key, value in env_vars.items():
            normalized[key] = value
            upper_key = key.upper()
            normalized[upper_key] = value
        return normalized

    def _update_config_fields(
        self,
        current_config: dict,
        param_name: Literal[
            "general_settings",
            "router_settings",
            "litellm_settings",
            "environment_variables",
        ],
        db_param_value: Any,
    ) -> dict:
        """Update config fields with DB values, handling the merge strategy."""
        if param_name == "environment_variables":
            decrypted_env_vars: Final = self._decrypt_env_variables(db_param_value, return_original_value=True)
            merged_env_vars: Final = self._normalize_env_variable_keys(decrypted_env_vars)
            for env_key, value in merged_env_vars.items():
                os.environ[env_key] = value

            current_config.setdefault("environment_variables", {}).update(merged_env_vars)
            return current_config

        if param_name not in current_config:
            current_config[param_name] = db_param_value
            return current_config

        if isinstance(current_config[param_name], dict) and isinstance(db_param_value, dict):
            self._deep_merge_dicts(current_config[param_name], db_param_value)
        else:
            current_config[param_name] = db_param_value

        return current_config

    async def reconcile_config(
        self,
        yaml_config: dict,
        store_model_in_db: bool | None = None,
    ) -> dict:
        """Reconcile config from YAML with database overrides.

        This is the main config reconciliation method that loads config params
        from the database and merges them with the YAML config. DB values
        override YAML values except for None values and empty lists.

        Args:
            yaml_config: The configuration loaded from YAML file
            store_model_in_db: Whether to load config from DB

        Returns:
            The merged configuration with DB overrides applied
        """
        if store_model_in_db is not True:
            verbose_proxy_logger.info("'store_model_in_db' is not True, skipping db config reconciliation")
            return yaml_config

        tasks: Final = [self.get_param(k) for k in self.CONFIG_PARAMS]
        responses: Final = await asyncio.gather(*tasks)

        config = copy.deepcopy(yaml_config)
        for response in responses:
            if response is None:
                continue

            param_name = response.param_name
            param_value = response.param_value
            verbose_proxy_logger.debug("param_name=%s, param_value=%s", param_name, param_value)

            if param_name is not None and param_value is not None:
                config = self._update_config_fields(
                    current_config=config,
                    param_name=cast(
                        Literal[
                            "general_settings",
                            "router_settings",
                            "litellm_settings",
                            "environment_variables",
                        ],
                        param_name,
                    ),
                    db_param_value=param_value,
                )

        return config

    async def prefetch_params(self, param_names: list[str]) -> None:
        """Prefetch config params to warm the cache.

        This can be called before reconcile_config to ensure all needed
        params are loaded in a single batch.
        """
        await asyncio.gather(*[self.get_param(k) for k in param_names])

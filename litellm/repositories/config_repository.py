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
from typing import Any, Dict, List, Literal, Optional, cast

from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper


class ConfigParam:
    """Simple wrapper for config parameter from DB."""

    def __init__(self, param_name: str, param_value: Any):
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
    def prisma_client(self) -> Any:
        if self._prisma_client is None:
            raise RuntimeError("No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys")
        return self._prisma_client

    @property
    def table(self) -> Any:
        return self.prisma_client.db.litellm_config

    async def get_param(self, param_name: str) -> Optional[ConfigParam]:
        """Get a config parameter from the database."""
        record = await self.table.find_unique(where={"param_name": param_name})
        if record is None:
            return None
        param_value = record.param_value
        if isinstance(param_value, str):
            param_value = json.loads(param_value)
        return ConfigParam(param_name=param_name, param_value=param_value)

    async def set_param(self, param_name: str, param_value: Any) -> ConfigParam:
        """Set a config parameter in the database."""
        value_json = json.dumps(param_value) if not isinstance(param_value, str) else param_value
        await self.table.upsert(
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
            await self.table.delete(where={"param_name": param_name})
            return True
        except Exception:
            return False

    async def get_all_params(self) -> Dict[str, Any]:
        """Get all config parameters from the database."""
        records = await self.table.find_many()
        result = {}
        for record in records:
            param_value = record.param_value
            if isinstance(param_value, str):
                param_value = json.loads(param_value)
            result[record.param_name] = param_value
        return result

    def _deep_merge_dicts(self, dst: dict, src: dict) -> None:
        """Deep-merge src into dst, skipping None values and empty lists from src.

        On conflicts, src (DB) wins, but empty lists are treated as "no value"
        and don't overwrite the destination.
        """
        stack = [(dst, src)]
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

    def _decrypt_env_variables(self, env_vars: Dict[str, Any], return_original_value: bool = True) -> Dict[str, str]:
        """Decrypt environment variables from database."""
        decrypted: Dict[str, str] = {}
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

    def _normalize_env_variable_keys(self, env_vars: Dict[str, str]) -> Dict[str, str]:
        """Normalize env variable keys to include both original and uppercase versions."""
        normalized: Dict[str, str] = {}
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
            decrypted_env_vars = self._decrypt_env_variables(db_param_value, return_original_value=True)
            merged_env_vars = self._normalize_env_variable_keys(decrypted_env_vars)
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
        store_model_in_db: Optional[bool] = None,
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

        tasks = [self.get_param(k) for k in self.CONFIG_PARAMS]
        responses = await asyncio.gather(*tasks)

        config = copy.deepcopy(yaml_config)
        for response in responses:
            if response is None:
                continue

            param_name = response.param_name
            param_value = response.param_value
            verbose_proxy_logger.debug(f"param_name={param_name}, param_value={param_value}")

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

    async def prefetch_params(self, param_names: List[str]) -> None:
        """Prefetch config params to warm the cache.

        This can be called before reconcile_config to ensure all needed
        params are loaded in a single batch.
        """
        await asyncio.gather(*[self.get_param(k) for k in param_names])

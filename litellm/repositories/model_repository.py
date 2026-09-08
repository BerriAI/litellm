"""
Model repository for database operations on LiteLLM_ProxyModelTable.
"""

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, Protocol

from litellm.models.model import LiteLLM_ProxyModelTable
from litellm.proxy.common_utils.config_sync_pubsub import wrap_table_actions_for_config_sync
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.repositories.base_repository import BaseRepository
from litellm.repositories.prisma_protocols import TableActions

if TYPE_CHECKING:
    from prisma import models as prisma_models


class _PrismaModelDb(Protocol):
    @property
    def litellm_proxymodeltable(self) -> TableActions["prisma_models.LiteLLM_ProxyModelTable"]: ...


class _PrismaClientView(Protocol):
    @property
    def db(self) -> _PrismaModelDb: ...


class ModelRepository(BaseRepository[LiteLLM_ProxyModelTable]):
    """Repository for proxy model database operations with encryption support."""

    def __init__(self, prisma_client: object, encryption_key: str | None = None) -> None:
        super().__init__(prisma_client)
        self._encryption_key = encryption_key

    @property
    def table(self) -> TableActions["prisma_models.LiteLLM_ProxyModelTable"]:
        client: Final[_PrismaClientView] = self.prisma_client
        return wrap_table_actions_for_config_sync(
            actions=client.db.litellm_proxymodeltable,
            table_name="litellm_proxymodeltable",
        )

    @property
    def model_class(self) -> type[LiteLLM_ProxyModelTable]:
        return LiteLLM_ProxyModelTable

    def _encrypt_litellm_params(self, litellm_params: Mapping[str, object]) -> Mapping[str, object]:
        """Encrypt sensitive values in litellm_params."""
        encrypted: Final = {}
        for key, value in litellm_params.items():
            if isinstance(value, str):
                encrypted[key] = encrypt_value_helper(value, new_encryption_key=self._encryption_key)
            else:
                encrypted[key] = value
        return encrypted

    def _decrypt_litellm_params(self, litellm_params: Mapping[str, object]) -> Mapping[str, object]:
        """Decrypt sensitive values in litellm_params."""
        decrypted: Final = {}
        for key, value in litellm_params.items():
            if isinstance(value, str):
                decrypted[key] = decrypt_value_helper(
                    value, key=key, exception_type="debug", return_original_value=True
                )
            else:
                decrypted[key] = value
        return decrypted

    def _to_model(self, record: Any) -> LiteLLM_ProxyModelTable | None:
        """Convert a database record to a Model with decryption."""
        if record is None:
            return None

        data: Final = record.dict() if hasattr(record, "dict") else dict(record)

        if isinstance(data.get("litellm_params"), str):
            data["litellm_params"] = json.loads(data["litellm_params"])
        if isinstance(data.get("model_info"), str):
            data["model_info"] = json.loads(data["model_info"])

        if data.get("litellm_params"):
            data["litellm_params"] = self._decrypt_litellm_params(data["litellm_params"])

        return LiteLLM_ProxyModelTable(**data)

    async def find_by_id(self, model_id: str, id_field: str = "model_id") -> LiteLLM_ProxyModelTable | None:
        return await super().find_by_id(model_id, id_field)

    async def find_by_name(self, model_name: str) -> list[LiteLLM_ProxyModelTable]:
        """Find models by name."""
        records: Final = await self.table.find_many(where={"model_name": model_name})
        return self._to_model_list(records)

    async def find_all(self) -> list[LiteLLM_ProxyModelTable]:
        """Find all models."""
        records: Final = await self.table.find_many()
        return self._to_model_list(records)

    async def find_unblocked(self) -> list[LiteLLM_ProxyModelTable]:
        """Find all models that are not blocked."""
        records: Final = await self.table.find_many(where={"blocked": False})
        return self._to_model_list(records)

    async def find_by_team_id(self, team_id: str) -> list[LiteLLM_ProxyModelTable]:
        """Find models associated with a specific team.

        Note: This filters in-memory since team_id is stored within litellm_params
        JSON. For large deployments with many models, consider adding a dedicated
        team_id column with a database index.
        """
        all_models: Final = await self.find_all()
        return [m for m in all_models if m.team_id == team_id]

    async def create_model(
        self,
        model_name: str,
        litellm_params: Mapping[str, object],
        created_by: str,
        model_id: str | None = None,
        model_info: Mapping[str, object] | None = None,
        blocked: bool = False,
    ) -> LiteLLM_ProxyModelTable:
        """Create a new model with encryption."""
        encrypted_params: Final = self._encrypt_litellm_params(litellm_params)

        data: Final[dict[str, str | bool]] = {
            "model_name": model_name,
            "litellm_params": json.dumps(encrypted_params),
            "created_by": created_by,
            "updated_by": created_by,
            "blocked": blocked,
        }
        if model_id is not None:
            data["model_id"] = model_id
        if model_info is not None:
            data["model_info"] = json.dumps(model_info)

        record: Final = await self.table.create(data=data)
        model: Final = self._to_model(record)
        assert model is not None
        return model

    async def update_model(
        self,
        model_id: str,
        updated_by: str,
        model_name: str | None = None,
        litellm_params: Mapping[str, object] | None = None,
        model_info: Mapping[str, object] | None = None,
        blocked: bool | None = None,
    ) -> LiteLLM_ProxyModelTable | None:
        """Update a model with encryption."""
        data: Final[dict[str, str | bool]] = {"updated_by": updated_by}
        if model_name is not None:
            data["model_name"] = model_name
        if litellm_params is not None:
            encrypted_params: Final = self._encrypt_litellm_params(litellm_params)
            data["litellm_params"] = json.dumps(encrypted_params)
        if model_info is not None:
            data["model_info"] = json.dumps(model_info)
        if blocked is not None:
            data["blocked"] = blocked

        record: Final = await self.table.update(where={"model_id": model_id}, data=data)
        return self._to_model(record)

    async def delete_model(self, model_id: str) -> LiteLLM_ProxyModelTable | None:
        """Delete a model."""
        return await self.delete(model_id, id_field="model_id")

    async def block_model(self, model_id: str, updated_by: str) -> LiteLLM_ProxyModelTable | None:
        """Block a model."""
        return await self.update_model(model_id, updated_by, blocked=True)

    async def unblock_model(self, model_id: str, updated_by: str) -> LiteLLM_ProxyModelTable | None:
        """Unblock a model."""
        return await self.update_model(model_id, updated_by, blocked=False)

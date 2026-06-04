"""
Credentials repository for database operations on LiteLLM_CredentialsTable.
"""

import json
from typing import Any, Dict, Optional, Type

from litellm.backend.models.credentials import Credentials
from litellm.gateway.repositories.base_repository import BaseRepository
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)


class CredentialsRepository(BaseRepository[Credentials]):
    """Repository for credentials database operations with encryption support."""

    def __init__(self, prisma_client: Any, encryption_key: Optional[str] = None):
        super().__init__(prisma_client)
        self._encryption_key = encryption_key

    @property
    def table(self) -> Any:
        return self.prisma_client.db.litellm_credentialstable

    @property
    def model_class(self) -> Type[Credentials]:
        return Credentials

    def _encrypt_credential_values(
        self, credential_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Encrypt all values in the credential_values dictionary."""
        encrypted = {}
        for key, value in credential_values.items():
            if isinstance(value, str):
                encrypted[key] = encrypt_value_helper(
                    value, new_encryption_key=self._encryption_key
                )
            else:
                encrypted[key] = value
        return encrypted

    def _decrypt_credential_values(
        self, credential_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Decrypt all values in the credential_values dictionary."""
        decrypted = {}
        for key, value in credential_values.items():
            if isinstance(value, str):
                decrypted[key] = decrypt_value_helper(
                    value, key=key, exception_type="debug", return_original_value=True
                )
            else:
                decrypted[key] = value
        return decrypted

    def _to_model(self, record: Any) -> Optional[Credentials]:
        """Convert a database record to a Credentials model with decryption."""
        if record is None:
            return None
        model = super()._to_model(record)
        if model and model.credential_values:
            model.credential_values = self._decrypt_credential_values(
                model.credential_values
            )
        return model

    async def find_by_id(
        self, credential_id: str, id_field: str = "credential_id"
    ) -> Optional[Credentials]:
        return await super().find_by_id(credential_id, id_field)

    async def find_by_name(self, credential_name: str) -> Optional[Credentials]:
        """Find credentials by name."""
        record = await self.table.find_unique(
            where={"credential_name": credential_name}
        )
        return self._to_model(record)

    async def create_credentials(
        self,
        credential_name: str,
        credential_values: Dict[str, Any],
        created_by: str,
        credential_info: Optional[Dict[str, Any]] = None,
    ) -> Credentials:
        """Create new credentials with encryption."""
        encrypted_values = self._encrypt_credential_values(credential_values)
        data: Dict[str, Any] = {
            "credential_name": credential_name,
            "credential_values": json.dumps(encrypted_values),
            "created_by": created_by,
            "updated_by": created_by,
        }
        if credential_info is not None:
            data["credential_info"] = json.dumps(credential_info)

        record = await self.table.create(data=data)
        return self._to_model(record)

    async def update_credentials(
        self,
        credential_id: str,
        updated_by: str,
        credential_values: Optional[Dict[str, Any]] = None,
        credential_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Credentials]:
        """Update credentials with encryption."""
        data: Dict[str, Any] = {"updated_by": updated_by}
        if credential_values is not None:
            encrypted_values = self._encrypt_credential_values(credential_values)
            data["credential_values"] = json.dumps(encrypted_values)
        if credential_info is not None:
            data["credential_info"] = json.dumps(credential_info)

        record = await self.table.update(
            where={"credential_id": credential_id}, data=data
        )
        return self._to_model(record)

    async def delete_credentials(self, credential_id: str) -> Optional[Credentials]:
        """Delete credentials."""
        return await self.delete(credential_id, id_field="credential_id")

from datetime import datetime
from typing import Any, Optional, Type

from litellm.models.password_reset_token import LiteLLM_PasswordResetToken
from litellm.repositories.base_repository import BaseRepository


class PasswordResetTokenRepository(BaseRepository[LiteLLM_PasswordResetToken]):
    @property
    def table(self) -> Any:
        return self.prisma_client.db.litellm_passwordresettoken

    @property
    def model_class(self) -> Type[LiteLLM_PasswordResetToken]:
        return LiteLLM_PasswordResetToken

    def _to_model(self, record: Any) -> Optional[LiteLLM_PasswordResetToken]:
        if record is None:
            return None
        data = record.dict() if hasattr(record, "dict") else dict(record)
        return LiteLLM_PasswordResetToken(**data)

    async def find_valid_by_hash(self, token_hash: str, now: datetime) -> Optional[LiteLLM_PasswordResetToken]:
        record = await self.table.find_unique(where={"token_hash": token_hash})
        model = self._to_model(record)
        if model is None:
            return None
        if model.used_at is not None:
            return None
        if model.expires_at < now:
            return None
        return model

    async def invalidate_unused_for_user(self, user_id: str, now: datetime) -> None:
        await self.table.update_many(
            where={"user_id": user_id, "used_at": None},
            data={"used_at": now},
        )

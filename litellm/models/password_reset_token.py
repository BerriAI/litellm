from datetime import datetime

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_PasswordResetToken(LiteLLMPydanticObjectBase):
    token_hash: str
    user_id: str
    requested_ip: str | None = None
    created_at: datetime
    expires_at: datetime
    used_at: datetime | None = None

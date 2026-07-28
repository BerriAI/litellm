from datetime import datetime
from typing import Optional

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_PasswordResetToken(LiteLLMPydanticObjectBase):
    token_hash: str
    user_id: str
    requested_ip: Optional[str] = None
    created_at: datetime
    expires_at: datetime
    used_at: Optional[datetime] = None

from collections.abc import Mapping
from datetime import datetime
from typing import Literal, TypeAlias

from pydantic import ConfigDict

from litellm.types.utils import LiteLLMBaseModel

AuditDecision: TypeAlias = Literal["allowed", "denied", "unreachable"]


class AuditEventResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    at: datetime
    actor: str
    actor_email: str | None
    dataplane_id: str
    method: str
    path: str
    decision: AuditDecision
    status: int | None
    credential: str | None
    detail: str | None


class AuditListResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    items: tuple[AuditEventResponse, ...]


class OcsfAuditListResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    items: tuple[Mapping[str, object], ...]
    chain_valid: bool
    sequence_gaps: int

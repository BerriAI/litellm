"""Typed views over the untyped request shapes LiteLLM hands to a guardrail."""

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError


@dataclass(frozen=True, slots=True)
class Identity:
    user: str
    session: str | None


ANONYMOUS: Final = Identity(user="anonymous", session=None)


class _FunctionName(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str


class _NamedTool(BaseModel):
    """Both tool shapes in play: {"name": ...} / {"function": {"name": ...}} dicts and pydantic tool-call objects."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str | None = None
    function: _FunctionName | None = None


class _Metadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    user_api_key_user_id: str | None = None


class _RequestIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    user: str | None = None
    litellm_session_id: str | None = None
    metadata: _Metadata | None = None


def tool_name_of(tool: object) -> str | None:
    try:
        parsed: Final = _NamedTool.model_validate(tool, from_attributes=True)
    except ValidationError:
        return None
    return parsed.name or (parsed.function.name if parsed.function else None)


def identity_of(request_data: object) -> Identity:
    try:
        parsed: Final = _RequestIdentity.model_validate(request_data, from_attributes=True)
    except ValidationError:
        return ANONYMOUS
    user: Final = parsed.user or (parsed.metadata.user_api_key_user_id if parsed.metadata else None)
    return Identity(user=user or ANONYMOUS.user, session=parsed.litellm_session_id)

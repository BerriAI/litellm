"""Types for per-entity failed request counts by HTTP status, rolled up daily."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from pydantic import ConfigDict

from litellm.types.llms.base import LiteLLMBaseModel


@dataclass(frozen=True, slots=True)
class RequestErrorKey:
    date: str
    api_key: str
    team_id: str
    user_id: str
    model_group: str
    status_code: int


RequestErrorSnapshot: TypeAlias = Mapping[RequestErrorKey, int]


class RequestErrorStatusCodeEntry(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    status_code: int
    failed_requests: int = 0


class RequestErrorDailyEntry(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    date: str
    successful_requests: int = 0
    failed_requests: int = 0
    client_errors: int = 0
    server_errors: int = 0
    by_status_code: tuple[RequestErrorStatusCodeEntry, ...] = ()


class RequestErrorEntityEntry(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    label: str | None = None
    api_requests: int = 0
    failed_requests: int = 0
    top_status_code: int | None = None
    top_status_code_requests: int = 0


class RequestErrorActivityResponse(LiteLLMBaseModel):
    """Response for GET /gateway/errors/activity."""

    model_config = ConfigDict(frozen=True)

    total_successful_requests: int = 0
    total_failed_requests: int = 0
    by_date: tuple[RequestErrorDailyEntry, ...] = ()
    by_status_code: tuple[RequestErrorStatusCodeEntry, ...] = ()
    by_key: tuple[RequestErrorEntityEntry, ...] = ()
    by_team: tuple[RequestErrorEntityEntry, ...] = ()
    by_user: tuple[RequestErrorEntityEntry, ...] = ()
    by_model: tuple[RequestErrorEntityEntry, ...] = ()

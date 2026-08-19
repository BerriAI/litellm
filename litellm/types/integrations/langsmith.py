from dataclasses import dataclass
from datetime import datetime
from typing import Any, NamedTuple

from pydantic import BaseModel
from typing_extensions import TypedDict


class LangsmithInputs(BaseModel):
    model: str | None = None
    messages: list[Any] | None = None
    stream: bool | None = None
    call_type: str | None = None
    litellm_call_id: str | None = None
    completion_start_time: datetime | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    custom_llm_provider: str | None = None
    input: list[Any] | None = None
    log_event_type: str | None = None
    original_response: Any | None = None
    response_cost: float | None = None

    # LiteLLM Virtual Key specific fields
    user_api_key: str | None = None
    user_api_key_user_id: str | None = None
    user_api_key_team_alias: str | None = None


class LangsmithCredentialsObject(TypedDict):
    LANGSMITH_API_KEY: str | None
    LANGSMITH_PROJECT: str | None
    LANGSMITH_BASE_URL: str
    LANGSMITH_TENANT_ID: str | None


class LangsmithQueueObject(TypedDict):
    """
    Langsmith Queue Object - this is what gets stored in the internal system queue before flushing to Langsmith

    We need to store:
        - data[Dict] - data that should get logged on langsmith
        - credentials[LangsmithCredentialsObject] - credentials to use for logging to langsmith
    """

    data: dict
    credentials: LangsmithCredentialsObject


class CredentialsKey(NamedTuple):
    """Immutable key for grouping credentials"""

    api_key: str
    project: str
    base_url: str
    tenant_id: str | None


@dataclass
class BatchGroup:
    """Groups credentials with their associated queue objects"""

    credentials: LangsmithCredentialsObject
    queue_objects: list[LangsmithQueueObject]

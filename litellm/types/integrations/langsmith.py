from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Optional

from pydantic import BaseModel
from typing_extensions import TypedDict


class LangsmithInputs(BaseModel):
    model: Optional[str] = None
    messages: Optional[List[Any]] = None
    stream: Optional[bool] = None
    call_type: Optional[str] = None
    litellm_call_id: Optional[str] = None
    completion_start_time: Optional[datetime] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    custom_llm_provider: Optional[str] = None
    input: Optional[List[Any]] = None
    log_event_type: Optional[str] = None
    original_response: Optional[Any] = None
    response_cost: Optional[float] = None

    # LiteLLM Virtual Key specific fields
    user_api_key: Optional[str] = None
    user_api_key_user_id: Optional[str] = None
    user_api_key_team_alias: Optional[str] = None


class LangsmithCredentialsObject(TypedDict):
    LANGSMITH_API_KEY: Optional[str]
    LANGSMITH_PROJECT: Optional[str]
    LANGSMITH_BASE_URL: str


class LangsmithQueueObject(TypedDict):
    """
    Langsmith Queue Object - this is what gets stored in the internal system queue before flushing to Langsmith

    We need to store:
        - data[Dict] - data that should get logged on langsmith
        - credentials[LangsmithCredentialsObject] - credentials to use for logging to langsmith
    """

    data: Dict
    credentials: LangsmithCredentialsObject


class CredentialsKey(NamedTuple):
    """Immutable key for grouping credentials"""

    api_key: str
    project: str
    base_url: str


@dataclass
class BatchGroup:
    """Groups credentials with their associated queue objects"""

    credentials: LangsmithCredentialsObject
    queue_objects: List[LangsmithQueueObject]

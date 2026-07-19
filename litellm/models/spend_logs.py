"""
Spend and error log table models.

Canonical definitions for ``litellm_spendlogs`` and ``litellm_errorlogs``.
Re-exported from ``litellm.proxy._types`` for backwards compatibility.
"""

from datetime import datetime
from typing import Optional, Union

from pydantic import Json

from litellm._uuid import uuid
from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_SpendLogs(LiteLLMPydanticObjectBase):
    request_id: str
    api_key: str
    model: Optional[str] = ""
    api_base: Optional[str] = ""
    call_type: str
    spend: Optional[float] = 0.0
    total_tokens: Optional[int] = 0
    prompt_tokens: Optional[int] = 0
    completion_tokens: Optional[int] = 0
    startTime: Union[str, datetime, None]
    endTime: Union[str, datetime, None]
    user: Optional[str] = ""
    metadata: Optional[Json] = {}
    cache_hit: Optional[str] = "False"
    cache_key: Optional[str] = None
    request_tags: Optional[Json] = None
    requester_ip_address: Optional[str] = None
    messages: Optional[Union[str, list, dict]]
    response: Optional[Union[str, list, dict]]


class LiteLLM_ErrorLogs(LiteLLMPydanticObjectBase):
    request_id: Optional[str] = str(uuid.uuid4())
    api_base: Optional[str] = ""
    model_group: Optional[str] = ""
    litellm_model_name: Optional[str] = ""
    model_id: Optional[str] = ""
    request_kwargs: Optional[dict] = {}
    exception_type: Optional[str] = ""
    status_code: Optional[str] = ""
    exception_string: Optional[str] = ""
    startTime: Union[str, datetime, None]
    endTime: Union[str, datetime, None]

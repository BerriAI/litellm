from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict


class EmbeddingRequest(BaseModel):
    model: str
    input: List[str] = []
    timeout: int = 600
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    api_key: Optional[str] = None
    api_type: Optional[str] = None
    caching: bool = False
    user: Optional[str] = None
    custom_llm_provider: Optional[Union[str, dict]] = None
    litellm_call_id: Optional[str] = None
    litellm_logging_obj: Optional[dict] = None
    logger_fn: Optional[str] = None

    model_config = ConfigDict(extra="allow")

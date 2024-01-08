from typing import List, Optional, Union, Dict, Tuple, Literal

from pydantic import BaseModel, validator
from .completion import CompletionRequest
from .embedding import EmbeddingRequest


class ModelConfig(BaseModel):
    model_name: str
    litellm_params: Union[CompletionRequest, EmbeddingRequest]
    tpm: int
    rpm: int


class RouterConfig(BaseModel):
    model_list: List[ModelConfig]

    redis_url: Optional[str] = None
    redis_host: Optional[str] = None
    redis_port: Optional[int] = None
    redis_password: Optional[str] = None

    cache_responses: Optional[bool] = False
    cache_kwargs: Optional[Dict] = {}
    caching_groups: Optional[List[Tuple[str, List[str]]]] = None
    client_ttl: Optional[int] = 3600
    num_retries: Optional[int] = 0
    timeout: Optional[float] = None
    default_litellm_params: Optional[Dict[str, str]] = {}
    set_verbose: Optional[bool] = False
    fallbacks: Optional[List] = []
    allowed_fails: Optional[int] = None
    context_window_fallbacks: Optional[List] = []
    model_group_alias: Optional[Dict[str, List[str]]] = {}
    retry_after: Optional[int] = 0
    routing_strategy: Literal[
        "simple-shuffle",
        "least-busy",
        "usage-based-routing",
        "latency-based-routing",
    ] = "simple-shuffle"

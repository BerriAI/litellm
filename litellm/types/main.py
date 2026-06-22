from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Optional, Union

from litellm.utils import CustomStreamWrapper, ModelResponse

if TYPE_CHECKING:
    import httpx
    from aiohttp import ClientSession

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.base_llm import BaseConfig


@dataclass(frozen=True, slots=True)
class _CompletionDispatchContext:
    _azure_detection_model: str
    acompletion: bool
    api_base: Optional[str]
    api_key: Optional[str]
    api_version: Optional[str]
    client: Any
    custom_llm_provider: str
    custom_prompt_dict: dict
    extra_headers: Optional[dict]
    headers: dict
    hf_model_name: Optional[str]
    kwargs: dict
    litellm_params: dict
    logger_fn: Optional[Callable]
    logging: LiteLLMLoggingObj
    max_retries: Optional[int]
    max_tokens: Optional[int]
    messages: list
    metadata: Optional[dict]
    model: str
    model_response: ModelResponse
    optional_params: dict
    organization: Optional[str]
    provider_config: Optional[BaseConfig]
    shared_session: Optional[ClientSession]
    stream: Optional[bool]
    temperature: Optional[float]
    text_completion: bool
    timeout: Optional[Union[float, str, httpx.Timeout]]
    top_p: Optional[float]


_CompletionDispatchResult = Union[
    Coroutine[Any, Any, Union[ModelResponse, CustomStreamWrapper]],
    ModelResponse,
    CustomStreamWrapper,
]

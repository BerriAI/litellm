"""
*New config* for using aiohttp to make the request to the custom OpenAI-like provider

This leads to 10x higher RPS than httpx
https://github.com/BerriAI/litellm/issues/6592

New config to ensure we introduce this without causing breaking changes for users
"""

from typing import TYPE_CHECKING, Any, List, Optional

from aiohttp import ClientResponse

from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class AiohttpOpenAIChatConfig(OpenAILikeChatConfig):
    def get_complete_url(
        self,
        api_base: str,
        model: str,
        optional_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Ensure - /v1/chat/completions is at the end of the url

        """

        if not api_base.endswith("/chat/completions"):
            api_base += "/chat/completions"
        return api_base

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return {"Authorization": f"Bearer {api_key}"}

    async def transform_response(  # type: ignore
        self,
        model: str,
        raw_response: ClientResponse,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        _json_response = await raw_response.json()
        return ModelResponse(**_json_response)

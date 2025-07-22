"""
Translate from OpenAI's `/v1/chat/completions` to Perplexity's `/v1/chat/completions`
"""

from typing import Any, List, Optional, Tuple

import httpx
import litellm
from litellm._logging import verbose_logger
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Usage, PromptTokensDetailsWrapper
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.utils import ModelResponse


class PerplexityChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "perplexity"

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("PERPLEXITY_API_BASE") or "https://api.perplexity.ai"  # type: ignore
        dynamic_api_key = (
            api_key
            or get_secret_str("PERPLEXITYAI_API_KEY")
            or get_secret_str("PERPLEXITY_API_KEY")
        )
        return api_base, dynamic_api_key

    def get_supported_openai_params(self, model: str) -> list:
        """
        Perplexity supports a subset of OpenAI params

        Ref: https://docs.perplexity.ai/api-reference/chat-completions

        Eg. Perplexity does not support tools, tool_choice, function_call, functions, etc.
        """
        base_openai_params = [
            "frequency_penalty",
            "max_tokens",
            "max_completion_tokens",
            "presence_penalty",
            "response_format",
            "stream",
            "temperature",
            "top_p",
            "max_retries",
            "extra_headers",
        ]

        try:
            if litellm.supports_reasoning(
                model=model, custom_llm_provider=self.custom_llm_provider
            ):
                base_openai_params.append("reasoning_effort")
        except Exception as e:
            verbose_logger.debug(f"Error checking if model supports reasoning: {e}")
        
        try:
            if litellm.supports_web_search(
                model=model, custom_llm_provider=self.custom_llm_provider
            ):
                base_openai_params.append("web_search_options")
        except Exception as e:
            verbose_logger.debug(f"Error checking if model supports web search: {e}")
        
        return base_openai_params

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
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
        # Call the parent transform_response first to handle the standard transformation
        model_response = super().transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )

        # Extract and enhance usage with Perplexity-specific fields
        try:
            raw_response_json = raw_response.json()
            self._enhance_usage_with_perplexity_fields(model_response, raw_response_json)
        except Exception as e:
            verbose_logger.debug(f"Error extracting Perplexity-specific usage fields: {e}")

        return model_response

    def _enhance_usage_with_perplexity_fields(
        self, model_response: ModelResponse, raw_response_json: dict
    ) -> None:
        """
        Extract citation tokens and search queries from Perplexity API response
        and add them to the usage object using standard LiteLLM fields.
        """
        if not hasattr(model_response, "usage") or model_response.usage is None:
            # Create a usage object if it doesn't exist (when usage was None)
            model_response.usage = Usage(  # type: ignore[attr-defined]
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0
            )

        usage = model_response.usage  # type: ignore[attr-defined]

        # Extract citation tokens count
        citations = raw_response_json.get("citations", [])
        citation_tokens = 0
        if citations:
            # Count total characters in citations as a proxy for citation tokens
            # This is an estimation - in practice, you might want to use proper tokenization
            total_citation_chars = sum(len(str(citation)) for citation in citations if citation)
            # Rough estimation: ~4 characters per token (OpenAI's general rule)
            if total_citation_chars > 0:
                citation_tokens = max(1, total_citation_chars // 4)

        # Extract search queries count from usage or response metadata
        # Perplexity might include this in the usage object or as separate metadata
        perplexity_usage = raw_response_json.get("usage", {})
        
        # Try to extract search queries from usage field first, then root level
        num_search_queries = perplexity_usage.get("num_search_queries")
        if num_search_queries is None:
            num_search_queries = raw_response_json.get("num_search_queries")
        if num_search_queries is None:
            num_search_queries = perplexity_usage.get("search_queries")
        if num_search_queries is None:
            num_search_queries = raw_response_json.get("search_queries")
        
        # Create or update prompt_tokens_details to include web search requests and citation tokens
        if citation_tokens > 0 or (num_search_queries is not None and num_search_queries > 0):
            if usage.prompt_tokens_details is None:
                usage.prompt_tokens_details = PromptTokensDetailsWrapper()
            
            # Store citation tokens count for cost calculation
            if citation_tokens > 0:
                setattr(usage, "citation_tokens", citation_tokens)
            
            # Store search queries count in the standard web_search_requests field
            if num_search_queries is not None and num_search_queries > 0:
                usage.prompt_tokens_details.web_search_requests = num_search_queries

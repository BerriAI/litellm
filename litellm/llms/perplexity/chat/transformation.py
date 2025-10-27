"""Translate from OpenAI's `/v1/chat/completions` to Perplexity's `/v1/chat/completions`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import litellm
from litellm._logging import verbose_logger
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage

if TYPE_CHECKING:
    import httpx

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import (
        AllMessageValues,
        ChatCompletionAnnotation,
        ChatCompletionAnnotationURLCitation,
    )


class PerplexityChatConfig(OpenAIGPTConfig):
    """Configuration for Perplexity chat completions."""

    @property
    def custom_llm_provider(self) -> str | None:
        """Return the custom LLM provider name."""
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

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """Validate Perplexity environment and set headers."""
        # Get API key from environment if not provided
        if api_key is None:
            _, api_key = self._get_openai_compatible_provider_info(
                api_base=api_base, api_key=api_key
            )
        
        # Validate API key is present
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key to the client or by setting the PERPLEXITY_API_KEY environment variable"
            )
        
        # Set authorization header
        headers["Authorization"] = f"Bearer {api_key}"
        
        # Ensure Content-Type is set to application/json
        if "content-type" not in headers and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        
        return headers

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


    def transform_response(  # noqa: PLR0913
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
        """Transform Perplexity response to standard format."""
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
            self.add_cost_to_usage(model_response, raw_response_json)
            self._enhance_usage_with_perplexity_fields(
                model_response, raw_response_json,
            )
            self._add_citations_as_annotations(model_response, raw_response_json)
        except (ValueError, TypeError, KeyError) as e:
            verbose_logger.debug(f"Error extracting Perplexity-specific usage fields: {e}")

        return model_response

    def _enhance_usage_with_perplexity_fields(  
        self, model_response: ModelResponse, raw_response_json: dict,
    ) -> None:
        """Extract citation tokens and search queries from Perplexity API response.

        Add them to the usage object using standard LiteLLM fields.
        """
        if not hasattr(model_response, "usage") or model_response.usage is None:
            # Create a usage object if it doesn't exist (when usage was None)
            model_response.usage = Usage(  # type: ignore[attr-defined]
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )

        usage = model_response.usage  # type: ignore[attr-defined]

        # Extract citation tokens count
        citations = raw_response_json.get("citations", [])
        citation_tokens = 0
        if citations:
            # Count total characters in citations as a proxy for citation tokens
            # This is an estimation - in practice, you might want to use proper tokenization
            total_citation_chars = sum(
                len(str(citation)) for citation in citations if citation
            )
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
        if citation_tokens > 0 or (
            num_search_queries is not None and num_search_queries > 0
        ):
            if usage.prompt_tokens_details is None:
                usage.prompt_tokens_details = PromptTokensDetailsWrapper()

            # Store citation tokens count for cost calculation
            if citation_tokens > 0:
                usage.citation_tokens = citation_tokens

            # Store search queries count in the standard web_search_requests field
            if num_search_queries is not None and num_search_queries > 0:
                usage.prompt_tokens_details.web_search_requests = num_search_queries

    def _add_citations_as_annotations(
        self, model_response: ModelResponse, raw_response_json: dict
    ) -> None:
        """
        Extract citations and search_results from Perplexity API response
        and add them as ChatCompletionAnnotation objects to the message.
        """
        if not model_response.choices:
            return

        # Get the first choice (assuming single response)
        choice = model_response.choices[0]
        if not hasattr(choice, "message") or choice.message is None:
            return

        message = choice.message
        annotations = []

        # Extract citations from the response
        citations = raw_response_json.get("citations", [])
        search_results = raw_response_json.get("search_results", [])

        # Create a mapping of URLs to search result titles
        url_to_title = {}
        for result in search_results:
            if isinstance(result, dict) and "url" in result and "title" in result:
                url_to_title[result["url"]] = result["title"]

        # Get the message content to find citation positions
        content = getattr(message, "content", "")
        if not content:
            return

        # Find all citation markers like [1], [2], [3], [4] in the text
        import re

        citation_pattern = r"\[(\d+)\]"
        citation_matches = list(re.finditer(citation_pattern, content))

        # Create a mapping of citation numbers to URLs
        citation_number_to_url = {}
        for i, citation in enumerate(citations):
            if isinstance(citation, str):
                citation_number_to_url[i + 1] = citation  # 1-indexed

        # Create annotations for each citation match found in the text
        for match in citation_matches:
            citation_number = int(match.group(1))
            if citation_number in citation_number_to_url:
                url = citation_number_to_url[citation_number]
                title = url_to_title.get(url, "")

                # Create the URL citation annotation with actual text positions
                url_citation: ChatCompletionAnnotationURLCitation = {
                    "url": url,
                    "title": title,
                    "start_index": match.start(),
                    "end_index": match.end(),
                }

                annotation: ChatCompletionAnnotation = {
                    "type": "url_citation",
                    "url_citation": url_citation,
                }

                annotations.append(annotation)

        # Add annotations to the message if we have any
        if annotations:
            if not hasattr(message, "annotations") or message.annotations is None:
                message.annotations = []
            message.annotations.extend(annotations)

        # Also add the raw citations and search_results as attributes for backward compatibility
        if citations:
            setattr(model_response, "citations", citations)
        if search_results:
            setattr(model_response, "search_results", search_results)

    def add_cost_to_usage(self, model_response: ModelResponse, raw_response_json: dict) -> None:
        """Add the cost to the usage object."""
        try:
            usage_data = raw_response_json.get("usage")
            if usage_data:
                # Try different possible cost field locations
                response_cost = None

                # Check if cost is directly in usage (flat structure)
                if "total_cost" in usage_data:
                    response_cost = usage_data["total_cost"]
                # Check if cost is nested (cost.total_cost structure)
                elif "cost" in usage_data and isinstance(usage_data["cost"], dict):
                    response_cost = usage_data["cost"].get("total_cost")
                # Check if cost is a simple value
                elif "cost" in usage_data:
                    response_cost = usage_data["cost"]

                if response_cost is not None:
                    # Store cost in hidden params for the cost calculator to use
                    if not hasattr(model_response, "_hidden_params"):
                        model_response._hidden_params = {}  
                    if "additional_headers" not in model_response._hidden_params:  
                        model_response._hidden_params["additional_headers"] = {}  
                    model_response._hidden_params["additional_headers"][  
                        "llm_provider-x-litellm-response-cost"
                    ] = float(response_cost)
        except (ValueError, TypeError, KeyError) as e:
            verbose_logger.debug(f"Error adding cost to usage: {e}")
            # If we can't extract cost, continue without it - don't fail the response

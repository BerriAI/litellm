"""
Bedrock Token Counter implementation using the CountTokens API.
"""

from typing import Any, Final

from litellm._logging import verbose_logger
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.llms.bedrock.common_utils import BedrockError, get_bedrock_base_model
from litellm.llms.bedrock.count_tokens.handler import BedrockCountTokensHandler
from litellm.types.utils import LlmProviders, TokenCountResponse


class BedrockTokenCounter(BaseTokenCounter):
    """Token counter implementation for AWS Bedrock provider using the CountTokens API."""

    def should_use_token_counting_api(
        self,
        custom_llm_provider: str | None = None,
    ) -> bool:
        """
        Returns True if we should use the Bedrock CountTokens API for token counting.
        """
        return custom_llm_provider == LlmProviders.BEDROCK.value

    async def count_tokens(
        self,
        model_to_use: str,
        messages: list[dict[str, Any]] | None,
        contents: list[dict[str, Any]] | None,
        deployment: dict[str, Any] | None = None,
        request_model: str = "",
        tools: list[dict[str, Any]] | None = None,
        system: Any | None = None,
    ) -> TokenCountResponse | None:
        """
        Count tokens using AWS Bedrock's CountTokens API.

        This method calls the existing BedrockCountTokensHandler to make an API call
        to Bedrock's token counting endpoint, bypassing the local tiktoken-based counting.

        Args:
            model_to_use: The model identifier
            messages: The messages to count tokens for
            contents: Alternative content format (not used for Bedrock)
            deployment: Deployment configuration containing litellm_params
            request_model: The original request model name

        Returns:
            TokenCountResponse with token count, or None if counting fails
        """
        if not messages:
            return None

        deployment = deployment or {}
        litellm_params: Final = deployment.get("litellm_params", {})

        # Build request data in the format expected by BedrockCountTokensHandler
        request_data: Final[dict[str, Any]] = {
            "model": model_to_use,
            "messages": messages,
        }

        if tools:
            request_data["tools"] = tools

        if system:
            request_data["system"] = system

        # Get the resolved model (strip prefixes like bedrock/, converse/, etc.)
        resolved_model: Final = get_bedrock_base_model(model_to_use)

        try:
            handler: Final = BedrockCountTokensHandler()
            result: Final = await handler.handle_count_tokens_request(
                request_data=request_data,
                litellm_params=litellm_params,
                resolved_model=resolved_model,
            )

            # Transform response to TokenCountResponse
            if result is not None:
                return TokenCountResponse(
                    total_tokens=result.get("input_tokens", 0),
                    request_model=request_model,
                    model_used=model_to_use,
                    tokenizer_type="bedrock_api",
                    original_response=result,
                )
        except BedrockError as e:
            verbose_logger.warning("Bedrock CountTokens API error: status=%s, message=%s", e.status_code, e.message)
            return TokenCountResponse(
                total_tokens=0,
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type="bedrock_api",
                error=True,
                error_message=e.message,
                status_code=e.status_code,
            )
        except Exception as e:
            verbose_logger.warning("Error calling Bedrock CountTokens API: %s", e)
            return TokenCountResponse(
                total_tokens=0,
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type="bedrock_api",
                error=True,
                error_message=str(e),
                status_code=500,
            )

        return None

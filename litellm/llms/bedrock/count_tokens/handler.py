"""
AWS Bedrock CountTokens API handler.

Simplified handler leveraging existing LiteLLM Bedrock infrastructure.
"""

from typing import Any, Final

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.bedrock.count_tokens.transformation import BedrockCountTokensConfig
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client


class BedrockCountTokensHandler(BedrockCountTokensConfig):
    """
    Simplified handler for AWS Bedrock CountTokens API requests.

    Uses existing LiteLLM infrastructure for authentication and request handling.
    """

    async def handle_count_tokens_request(
        self,
        request_data: dict[str, Any],
        litellm_params: dict[str, Any],
        resolved_model: str,
    ) -> dict[str, Any]:
        """
        Handle a CountTokens request using existing LiteLLM patterns.

        Args:
            request_data: The incoming request payload
            litellm_params: LiteLLM configuration parameters
            resolved_model: The actual model ID resolved from router

        Returns:
            Dictionary containing token count response
        """
        try:
            # Validate the request
            self.validate_count_tokens_request(request_data)

            verbose_logger.debug("Processing CountTokens request for resolved model: %s", resolved_model)

            # Get AWS region using existing LiteLLM function
            aws_region_name: Final = self._get_aws_region_name(
                optional_params=litellm_params,
                model=resolved_model,
                model_id=None,
            )

            verbose_logger.debug("Retrieved AWS region: %s", aws_region_name)

            # Transform request to Bedrock format (supports both Converse and InvokeModel)
            bedrock_request: Final = self.transform_anthropic_to_bedrock_count_tokens(request_data=request_data)

            verbose_logger.debug("Transformed request: %s", bedrock_request)

            # Get endpoint URL using simplified function
            api_base: Final = litellm_params.get("api_base", None)
            aws_bedrock_runtime_endpoint: Final = litellm_params.get("aws_bedrock_runtime_endpoint", None)
            endpoint_url: Final = self.get_bedrock_count_tokens_endpoint(
                model=resolved_model,
                aws_region_name=aws_region_name,
                api_base=api_base,
                aws_bedrock_runtime_endpoint=aws_bedrock_runtime_endpoint,
            )

            verbose_logger.debug("Making request to: %s", endpoint_url)

            # Use existing _sign_request method from BaseAWSLLM
            # Extract api_key for bearer token auth if provided
            api_key: Final = litellm_params.get("api_key", None)
            headers: Final = {"Content-Type": "application/json"}
            signed_headers, signed_body = self._sign_request(
                service_name="bedrock",
                headers=headers,
                optional_params=litellm_params,
                request_data=bedrock_request,
                api_base=endpoint_url,
                model=resolved_model,
                api_key=api_key,
            )

            async_client: Final = get_async_httpx_client(llm_provider=litellm.LlmProviders.BEDROCK)

            response: Final = await async_client.post(
                endpoint_url,
                headers=signed_headers,
                data=signed_body,
                timeout=30.0,
            )

            verbose_logger.debug("Response status: %s", response.status_code)

            if response.status_code != 200:
                error_text: Final = response.text
                verbose_logger.error("AWS Bedrock error: %s", error_text)
                raise BedrockError(
                    status_code=response.status_code,
                    message=error_text,
                )

            bedrock_response: Final = response.json()

            verbose_logger.debug("Bedrock response: %s", bedrock_response)

            # Transform response back to expected format
            final_response: Final = self.transform_bedrock_response_to_anthropic(bedrock_response)

            verbose_logger.debug("Final response: %s", final_response)

            return final_response

        except BedrockError:
            # Re-raise Bedrock exceptions as-is
            raise
        except httpx.HTTPStatusError as e:
            # HTTP errors - preserve the actual status code
            verbose_logger.error("HTTP error in CountTokens handler: %s", e)
            raise BedrockError(
                status_code=e.response.status_code,
                message=e.response.text,
            )
        except Exception as e:
            verbose_logger.error("Error in CountTokens handler: %s", e)
            raise BedrockError(
                status_code=500,
                message=f"CountTokens processing error: {e}",
            )

"""
AWS Bedrock CountTokens API handler.

Simplified handler leveraging existing LiteLLM Bedrock infrastructure.
"""

from typing import Any, Dict

from fastapi import HTTPException

import litellm
from litellm._logging import verbose_logger
from litellm.llms.bedrock.count_tokens.transformation import BedrockCountTokensConfig
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client


class BedrockCountTokensHandler(BedrockCountTokensConfig):
    """
    Simplified handler for AWS Bedrock CountTokens API requests.

    Uses existing LiteLLM infrastructure for authentication and request handling.
    """

    async def handle_count_tokens_request(
        self,
        request_data: Dict[str, Any],
        litellm_params: Dict[str, Any],
        resolved_model: str,
    ) -> Dict[str, Any]:
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

            verbose_logger.debug(
                f"Processing CountTokens request for resolved model: {resolved_model}"
            )

            # Get AWS region using existing LiteLLM function
            aws_region_name = self._get_aws_region_name(
                optional_params=litellm_params,
                model=resolved_model,
                model_id=None,
            )

            verbose_logger.debug(f"Retrieved AWS region: {aws_region_name}")

            # Transform request to Bedrock format (supports both Converse and InvokeModel)
            bedrock_request = self.transform_anthropic_to_bedrock_count_tokens(
                request_data=request_data
            )

            verbose_logger.debug(f"Transformed request: {bedrock_request}")

            # Get endpoint URL using simplified function
            endpoint_url = self.get_bedrock_count_tokens_endpoint(
                resolved_model, aws_region_name
            )

            verbose_logger.debug(f"Making request to: {endpoint_url}")

            # Use existing _sign_request method from BaseAWSLLM
            headers = {"Content-Type": "application/json"}
            signed_headers, signed_body = self._sign_request(
                service_name="bedrock",
                headers=headers,
                optional_params=litellm_params,
                request_data=bedrock_request,
                api_base=endpoint_url,
                model=resolved_model,
            )

            async_client = get_async_httpx_client(llm_provider=litellm.LlmProviders.BEDROCK)

            response = await async_client.post(
                    endpoint_url,
                    headers=signed_headers,
                    data=signed_body,
                    timeout=30.0,
                )

            verbose_logger.debug(f"Response status: {response.status_code}")

            if response.status_code != 200:
                error_text = response.text
                verbose_logger.error(f"AWS Bedrock error: {error_text}")
                raise HTTPException(
                    status_code=400,
                    detail={"error": f"AWS Bedrock error: {error_text}"},
                )

            bedrock_response = response.json()

            verbose_logger.debug(f"Bedrock response: {bedrock_response}")

            # Transform response back to expected format
            final_response = self.transform_bedrock_response_to_anthropic(
                bedrock_response
            )

            verbose_logger.debug(f"Final response: {final_response}")

            return final_response

        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            verbose_logger.error(f"Error in CountTokens handler: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={"error": f"CountTokens processing error: {str(e)}"},
            )

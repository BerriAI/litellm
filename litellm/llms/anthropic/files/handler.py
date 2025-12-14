import asyncio
import json
import time
from typing import Any, Coroutine, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
)
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.llms.openai import (
    FileContentRequest,
    HttpxBinaryResponseContent,
    OpenAIBatchResult,
    OpenAIChatCompletionResponse,
    OpenAIErrorBody,
)
from litellm.types.utils import CallTypes, LlmProviders, ModelResponse

from ..chat.transformation import AnthropicConfig
from ..common_utils import AnthropicModelInfo

# Map Anthropic error types to HTTP status codes
ANTHROPIC_ERROR_STATUS_CODE_MAP = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "permission_error": 403,
    "not_found_error": 404,
    "rate_limit_error": 429,
    "api_error": 500,
    "overloaded_error": 503,
    "timeout_error": 504,
}


class AnthropicFilesHandler:
    """
    Handles Anthropic Files API operations.
    
    Currently supports:
    - file_content() for retrieving Anthropic Message Batch results
    """

    def __init__(self):
        self.anthropic_model_info = AnthropicModelInfo()

    async def afile_content(
        self,
        file_content_request: FileContentRequest,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Union[float, httpx.Timeout] = 600.0,
        max_retries: Optional[int] = None,
    ) -> HttpxBinaryResponseContent:
        """
        Async: Retrieve file content from Anthropic.
        
        For batch results, the file_id should be the batch_id.
        This will call Anthropic's /v1/messages/batches/{batch_id}/results endpoint.
        
        Args:
            file_content_request: Contains file_id (batch_id for batch results)
            api_base: Anthropic API base URL
            api_key: Anthropic API key
            timeout: Request timeout
            max_retries: Max retry attempts (unused for now)
            
        Returns:
            HttpxBinaryResponseContent: Binary content wrapped in compatible response format
        """
        file_id = file_content_request.get("file_id")
        if not file_id:
            raise ValueError("file_id is required in file_content_request")

        # Extract batch_id from file_id
        # Handle both formats: "anthropic_batch_results:{batch_id}" or just "{batch_id}"
        if file_id.startswith("anthropic_batch_results:"):
            batch_id = file_id.replace("anthropic_batch_results:", "", 1)
        else:
            batch_id = file_id

        # Get Anthropic API credentials
        api_base = self.anthropic_model_info.get_api_base(api_base)
        api_key = api_key or self.anthropic_model_info.get_api_key()

        if not api_key:
            raise ValueError("Missing Anthropic API Key")

        # Construct the Anthropic batch results URL
        results_url = f"{api_base.rstrip('/')}/v1/messages/batches/{batch_id}/results"

        # Prepare headers
        headers = {
            "accept": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": api_key,
        }

        # Make the request to Anthropic
        async_client = get_async_httpx_client(llm_provider=LlmProviders.ANTHROPIC)
        anthropic_response = await async_client.get(
            url=results_url,
            headers=headers
        )
        anthropic_response.raise_for_status()

        # Transform Anthropic batch results to OpenAI format
        transformed_content = self._transform_anthropic_batch_results_to_openai_format(
            anthropic_response.content
        )

        # Create a new response with transformed content
        transformed_response = httpx.Response(
            status_code=anthropic_response.status_code,
            headers=anthropic_response.headers,
            content=transformed_content,
            request=anthropic_response.request,
        )

        # Return the transformed response content
        return HttpxBinaryResponseContent(response=transformed_response)


    def file_content(
        self,
        _is_async: bool,
        file_content_request: FileContentRequest,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Union[float, httpx.Timeout] = 600.0,
        max_retries: Optional[int] = None,
    ) -> Union[
        HttpxBinaryResponseContent, Coroutine[Any, Any, HttpxBinaryResponseContent]
    ]:
        """
        Retrieve file content from Anthropic.
        
        For batch results, the file_id should be the batch_id.
        This will call Anthropic's /v1/messages/batches/{batch_id}/results endpoint.
        
        Args:
            _is_async: Whether to run asynchronously
            file_content_request: Contains file_id (batch_id for batch results)
            api_base: Anthropic API base URL
            api_key: Anthropic API key
            timeout: Request timeout
            max_retries: Max retry attempts (unused for now)
            
        Returns:
            HttpxBinaryResponseContent or Coroutine: Binary content wrapped in compatible response format
        """
        if _is_async:
            return self.afile_content(
                file_content_request=file_content_request,
                api_base=api_base,
                api_key=api_key,
                max_retries=max_retries,
            )
        else:
            return asyncio.run(
                self.afile_content(
                    file_content_request=file_content_request,
                    api_base=api_base,
                    api_key=api_key,
                    timeout=timeout,
                    max_retries=max_retries,
                )
            )

    def _transform_anthropic_batch_results_to_openai_format(
        self, anthropic_content: bytes
    ) -> bytes:
        """
        Transform Anthropic batch results JSONL to OpenAI batch results JSONL format.
        
        Anthropic format:
        {
          "custom_id": "...",
          "result": {
            "type": "succeeded",
            "message": { ... }  // Anthropic message format
          }
        }
        
        OpenAI format:
        {
          "custom_id": "...",
          "response": {
            "status_code": 200,
            "request_id": "...",
            "body": { ... }  // OpenAI chat completion format
          }
        }
        """
        try:
            anthropic_config = AnthropicConfig()
            transformed_lines = []
            
            # Parse JSONL content
            content_str = anthropic_content.decode("utf-8")
            for line in content_str.strip().split("\n"):
                if not line.strip():
                    continue
                
                anthropic_result = json.loads(line)
                custom_id = anthropic_result.get("custom_id", "")
                result = anthropic_result.get("result", {})
                result_type = result.get("type", "")
                
                # Transform based on result type
                if result_type == "succeeded":
                    # Transform Anthropic message to OpenAI format
                    anthropic_message = result.get("message", {})
                    if anthropic_message:
                        openai_response_body = self._transform_anthropic_message_to_openai_format(
                            anthropic_message=anthropic_message,
                            anthropic_config=anthropic_config,
                        )
                        
                        # Create OpenAI batch result format
                        openai_result: OpenAIBatchResult = {
                            "custom_id": custom_id,
                            "response": {
                                "status_code": 200,
                                "request_id": anthropic_message.get("id", ""),
                                "body": openai_response_body,
                            },
                        }
                        transformed_lines.append(json.dumps(openai_result))
                elif result_type == "errored":
                    # Handle error case
                    error = result.get("error", {})
                    error_obj = error.get("error", {})
                    error_message = error_obj.get("message", "Unknown error")
                    error_type = error_obj.get("type", "api_error")
                    
                    status_code = ANTHROPIC_ERROR_STATUS_CODE_MAP.get(error_type, 500)
                    
                    error_body_errored: OpenAIErrorBody = {
                        "error": {
                            "message": error_message,
                            "type": error_type,
                        }
                    }
                    openai_result_errored: OpenAIBatchResult = {
                        "custom_id": custom_id,
                        "response": {
                            "status_code": status_code,
                            "request_id": error.get("request_id", ""),
                            "body": error_body_errored,
                        },
                    }
                    transformed_lines.append(json.dumps(openai_result_errored))
                elif result_type in ["canceled", "expired"]:
                    # Handle canceled/expired cases
                    error_body_canceled: OpenAIErrorBody = {
                        "error": {
                            "message": f"Batch request was {result_type}",
                            "type": "invalid_request_error",
                        }
                    }
                    openai_result_canceled: OpenAIBatchResult = {
                        "custom_id": custom_id,
                        "response": {
                            "status_code": 400,
                            "request_id": "",
                            "body": error_body_canceled,
                        },
                    }
                    transformed_lines.append(json.dumps(openai_result_canceled))
            
            # Join lines and encode back to bytes
            transformed_content = "\n".join(transformed_lines)
            if transformed_lines:
                transformed_content += "\n"  # Add trailing newline for JSONL format
            return transformed_content.encode("utf-8")
        except Exception as e:
            verbose_logger.error(
                f"Error transforming Anthropic batch results to OpenAI format: {e}"
            )
            # Return original content if transformation fails
            return anthropic_content

    def _transform_anthropic_message_to_openai_format(
        self, anthropic_message: dict, anthropic_config: AnthropicConfig
    ) -> OpenAIChatCompletionResponse:
        """
        Transform a single Anthropic message to OpenAI chat completion format.
        """
        try:
            # Create a mock httpx.Response for transformation
            mock_response = httpx.Response(
                status_code=200,
                content=json.dumps(anthropic_message).encode("utf-8"),
            )
            
            # Create a ModelResponse object
            model_response = ModelResponse()
            # Initialize with required fields - will be populated by transform_parsed_response
            model_response.choices = [
                litellm.Choices(
                    finish_reason="stop",
                    index=0,
                    message=litellm.Message(content="", role="assistant"),
                )
            ]  # type: ignore
            
            # Create a logging object for transformation
            logging_obj = Logging(
                model=anthropic_message.get("model", "claude-3-5-sonnet-20241022"),
                messages=[{"role": "user", "content": "batch_request"}],
                stream=False,
                call_type=CallTypes.aretrieve_batch,
                start_time=time.time(),
                litellm_call_id="batch_" + str(uuid.uuid4()),
                function_id="batch_processing",
                litellm_trace_id=str(uuid.uuid4()),
                kwargs={"optional_params": {}},
            )
            logging_obj.optional_params = {}
            
            # Transform using AnthropicConfig
            transformed_response = anthropic_config.transform_parsed_response(
                completion_response=anthropic_message,
                raw_response=mock_response,
                model_response=model_response,
                json_mode=False,
                prefix_prompt=None,
            )
            
            # Convert ModelResponse to OpenAI format dict - it's already in OpenAI format
            openai_body: OpenAIChatCompletionResponse = transformed_response.model_dump(exclude_none=True)
            
            # Ensure id comes from anthropic_message if not set
            if not openai_body.get("id"):
                openai_body["id"] = anthropic_message.get("id", "")
            
            return openai_body
        except Exception as e:
            verbose_logger.error(
                f"Error transforming Anthropic message to OpenAI format: {e}"
            )
            # Return a basic error response if transformation fails
            error_response: OpenAIChatCompletionResponse = {
                "id": anthropic_message.get("id", ""),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": anthropic_message.get("model", ""),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "error",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
            return error_response


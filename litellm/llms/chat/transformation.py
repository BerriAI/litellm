"""
Chat provider mapping chat completion requests to /responses API and back to chat format
"""
import httpx
import time
from typing import Any, Dict, List, Optional, Union

from openai.types.responses.response_create_params import ResponseCreateParamsBase
from openai.types.responses.response import Response as ResponsesAPIResponse
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.responses.litellm_completion_transformation.transformation import LiteLLMCompletionResponsesConfig
from litellm.types.utils import ModelResponse, Choices, Usage
from litellm.constants import OPENAI_CHAT_COMPLETION_PARAMS
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams


class ChatConfig(BaseConfig):
    """
    Provider config for chat that uses the /responses API under the hood.
    Transforms chat completion requests into Responses API requests and vice versa.
    """
    def get_supported_openai_params(self, model: str) -> List[str]:  # noqa: U100
        # Support standard OpenAI chat parameters plus responses API specific ones
        base_params = list(OPENAI_CHAT_COMPLETION_PARAMS)
        responses_specific_params = ["previous_response_id", "instructions"]
        return base_params + responses_specific_params

    def validate_environment(
        self,
        headers: Dict[str, Any],
        model: str,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Ensure proper authentication headers for OpenAI Responses API
        print(f"Chat provider validate_environment: model={model}, api_key={'***' if api_key else None}, headers_auth={'present' if headers.get('Authorization') else 'missing'}")
        
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            print(f"Chat provider: Set Authorization header from provided api_key")
        elif not headers.get("Authorization"):
            # Try to get API key from environment or other sources
            import os
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                headers["Authorization"] = f"Bearer {openai_key}"
                print(f"Chat provider: Set Authorization header from OPENAI_API_KEY environment variable")
            else:
                print("Warning: No API key found for chat provider. Set OPENAI_API_KEY or provide api_key in config.")
        else:
            print(f"Chat provider: Using existing Authorization header")
        
        # Ensure proper content type
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
            
        print(f"Chat provider final headers: {list(headers.keys())}")
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        **kwargs,
    ) -> str:
        # Default to OpenAI if no api_base provided
        if not api_base:
            api_base = "https://api.openai.com/v1"
            print(f"Warning: No api_base provided for chat provider, defaulting to: {api_base}")
        
        return api_base.rstrip("/") + "/responses"

    def transform_request(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        optional_params: Dict[str, Any],
        litellm_params: Dict[str, Any],
        headers: Dict[str, Any],  # noqa: U100
    ) -> Dict[str, Any]:
        # Convert chat completion messages back to responses API input format
        input_items: List[Any] = []
        
        # Process messages to extract system instructions and convert to responses format
        instructions = None
        converted_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")
            
            if role == "system":
                # Extract system message as instructions
                instructions = content
            elif role == "tool":
                # Convert tool message to function call output format
                input_items.append({
                    "type": "function_call_output",
                    "call_id": tool_call_id,
                    "output": content
                })
            elif role == "assistant" and tool_calls:
                # Convert assistant message with tool calls
                if content:  # If there's text content, add it first
                    input_items.append({
                        "type": "message",
                        "role": role,
                        "content": self._convert_content_to_responses_format(content)
                    })
                # Note: Tool calls will be handled by the responses API directly
                # We don't need to convert them back to input format here
            else:
                # Regular user/assistant message
                input_items.append({
                    "type": "message", 
                    "role": role,
                    "content": self._convert_content_to_responses_format(content)
                })
        
        # Build responses API request using the reverse transformation logic
        responses_api_request = ResponsesAPIOptionalRequestParams()
        
        # Set instructions if we found a system message
        if instructions:
            responses_api_request["instructions"] = instructions
            
        # Map optional parameters
        for key, value in optional_params.items():
            if value is None:
                continue
            if key in ("max_tokens", "max_completion_tokens"):
                responses_api_request["max_output_tokens"] = value
            elif key == "tools":
                # Convert chat completion tools to responses API tools format
                responses_api_request["tools"] = self._convert_tools_to_responses_format(value)
            elif key in ["temperature", "top_p", "tool_choice", "parallel_tool_calls", "user", "stream"]:
                responses_api_request[key] = value
            elif key == "metadata":
                responses_api_request["metadata"] = value
            elif key == "previous_response_id":
                # Support for responses API session management
                responses_api_request["previous_response_id"] = value
        
        # Get stream parameter from litellm_params if not in optional_params
        stream = optional_params.get("stream") or litellm_params.get("stream", False)
        print(f"Chat provider: Stream parameter: {stream}")
        
        # Ensure stream is properly set in the request
        if stream:
            responses_api_request["stream"] = True
        
        # Handle session management if previous_response_id is provided
        previous_response_id = optional_params.get("previous_response_id")
        if previous_response_id:
            # Use the existing session handler for responses API
            try:
                # This will extend the messages with previous session context
                session_request = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
                    model=model,
                    input=input_items,
                    responses_api_request=responses_api_request,
                    custom_llm_provider="openai",
                    stream=stream,
                    **litellm_params
                )
                
                # Note: Session handling for async contexts would use:
                # await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
                #     previous_response_id=previous_response_id,
                #     litellm_completion_request=session_request
                # )
                
            except Exception as e:
                # If session handling fails, continue without it but log the error
                print(f"Warning: Session handling failed for previous_response_id {previous_response_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # Convert back to responses API format for the actual request
        # Strip chat/ prefix if present for the underlying API call
        api_model = model
        if model.startswith("chat/"):
            api_model = model[5:]  # Remove "chat/" prefix
            print(f"Chat provider: Stripped chat/ prefix from model: {model} -> {api_model}")
        else:
            print(f"Chat provider: Using model as-is: {api_model}")
        
        request_data = {
            "model": api_model,
            "input": input_items,
        }
        
        print(f"Chat provider: Final request model={api_model}, input_items={len(input_items)}")
        
        # Add non-None values from responses_api_request
        for key, value in responses_api_request.items():
            if value is not None:
                if key == "instructions" and instructions:
                    request_data["instructions"] = instructions
                else:
                    request_data[key] = value
        
        print(f"Chat provider: Final request data keys: {list(request_data.keys())}")
        print(f"Chat provider: Stream in request: {request_data.get('stream', False)}")
        
        return request_data

    def transform_response(
        self,
        model: str,  # noqa: U100
        raw_response: httpx.Response,
        model_response: ModelResponse,  # noqa: U100
        logging_obj: Any,  # noqa: U100
        request_data: Dict[str, Any],
        messages: List[Any],  # noqa: U100
        optional_params: Dict[str, Any],  # noqa: U100
        litellm_params: Dict[str, Any],  # noqa: U100
        encoding: Any,  # noqa: U100
        api_key: Optional[str] = None,  # noqa: U100
        json_mode: Optional[bool] = None,  # noqa: U100
    ) -> ModelResponse:
        # Parse Responses API response and convert to chat ModelResponse
        try:
            resp_json = raw_response.json()
        except Exception as e:
            print(f"Error parsing JSON response from responses API: {e}")
            print(f"Raw response text: {raw_response.text}")
            import traceback
            traceback.print_exc()
            raise httpx.HTTPError(f"Invalid JSON from responses API: {raw_response.text}")
        
        # Use the existing transformation logic to convert responses API to chat completion
        input_param = request_data.get("input", [])
        responses_api_request = ResponsesAPIOptionalRequestParams(
            instructions=request_data.get("instructions"),
            tools=request_data.get("tools"),
            tool_choice=request_data.get("tool_choice"),
            temperature=request_data.get("temperature"),
            top_p=request_data.get("top_p"),
            text=request_data.get("text"),
            reasoning=request_data.get("reasoning"),
            store=request_data.get("store"),
            truncation=request_data.get("truncation"),
            previous_response_id=request_data.get("previous_response_id"),
            max_output_tokens=request_data.get("max_output_tokens"),
            parallel_tool_calls=request_data.get("parallel_tool_calls"),
            user=request_data.get("user"),
            metadata=request_data.get("metadata", {})
        )
        
        # Transform the responses API response to chat completion format
        print(f"Chat provider: Raw response from openai: {resp_json}")
        print(f"Chat provider: responses_api_request: {responses_api_request}")

        try:
            chat_completion_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input=input_param,
                responses_api_request=responses_api_request,
                chat_completion_response=resp_json
            )
            print(f"Chat provider: Transformation successful, output items: {len(getattr(chat_completion_response, 'output', []))}")
        except Exception as e:
            print(f"Chat provider: Transformation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Build proper ModelResponse from the responses API response
        choices = []
        
        print(f"Chat provider: Processing {len(chat_completion_response.output)} output items")
        
        # Process output items to build choices
        for i, output_item in enumerate(chat_completion_response.output):
            print(f"Chat provider: Output item {i}: type={getattr(output_item, 'type', 'unknown')}")
            
            if hasattr(output_item, 'type') and output_item.type == "message":
                # Extract content from the message
                content = ""
                if hasattr(output_item, 'content') and output_item.content:
                    print(f"Chat provider: Processing {len(output_item.content)} content items")
                    for j, content_item in enumerate(output_item.content):
                        print(f"Chat provider: Content item {j}: type={getattr(content_item, 'type', 'unknown')}, text={getattr(content_item, 'text', 'N/A')}")
                        if hasattr(content_item, 'text') and content_item.text is not None:
                            content += str(content_item.text)
                        else:
                            print(f"Chat provider: Skipping content item {j} - text is None or missing")
                else:
                    print(f"Chat provider: No content found in output item {i}")
                
                print(f"Chat provider: Final extracted content: '{content}'")
                
                message = {
                    "role": getattr(output_item, "role", "assistant"),
                    "content": content
                }
                
                # Add tool calls if present
                tool_calls = []
                for output in chat_completion_response.output:
                    if hasattr(output, 'type') and output.type == "function_call":
                        tool_calls.append({
                            "id": getattr(output, "call_id", getattr(output, "id", "")),
                            "type": "function",
                            "function": {
                                "name": getattr(output, "name", ""),
                                "arguments": getattr(output, "arguments", "")
                            }
                        })
                
                if tool_calls:
                    message["tool_calls"] = tool_calls
                
                choices.append({
                    "index": 0,
                    "message": message,
                    "finish_reason": self._map_responses_status_to_finish_reason(chat_completion_response.status)
                })
                break  # Only process the first message for now
        
        # If no message found, create a default one
        if not choices:
            choices.append({
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop"
            })
        
        # Build usage from the responses API response
        usage_obj = getattr(chat_completion_response, "usage", None)
        if usage_obj:
            usage = Usage(
                prompt_tokens=getattr(usage_obj, "input_tokens", 0),
                completion_tokens=getattr(usage_obj, "output_tokens", 0),
                total_tokens=getattr(usage_obj, "total_tokens", 0),
            )
        else:
            print("Chat provider: No usage data found, using defaults")
            usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        
        return ModelResponse(
            id=getattr(chat_completion_response, "id", "unknown"),
            choices=choices,
            created=getattr(chat_completion_response, "created_at", int(time.time())),
            model=getattr(chat_completion_response, "model", "unknown"),
            usage=usage,
        )

    def transform_streaming_response(
        self,
        model: str,  # noqa: U100
        parsed_chunk: dict,
        logging_obj: Any,  # noqa: U100
    ) -> Any:
        # Transform responses API streaming chunk to chat completion format
        print(f"Chat provider: transform_streaming_response called with chunk: {parsed_chunk}")
        
        if not parsed_chunk:
            print("Chat provider: Empty parsed_chunk, returning None")
            return None
        
        if not isinstance(parsed_chunk, dict):
            print(f"Chat provider: Invalid chunk type {type(parsed_chunk)}, returning as-is")
            return parsed_chunk
            
        # Handle different event types from responses API
        event_type = parsed_chunk.get("type")
        print(f"Chat provider: Processing event type: {event_type}")
        
        if event_type == "response.created":
            # Initial response creation event
            chunk = {
                "id": parsed_chunk.get("response", {}).get("id"),
                "object": "chat.completion.chunk",
                "created": parsed_chunk.get("response", {}).get("created_at"),
                "model": parsed_chunk.get("response", {}).get("model"),
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": None
                }]
            }
            print(f"Chat provider: response.created -> {chunk}")
            return chunk
        elif event_type == "response.output_item.added":
            # New output item added
            output_item = parsed_chunk.get("output_item", {})
            if output_item.get("type") == "message":
                return {
                    "id": parsed_chunk.get("response_id"),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": output_item.get("role", "assistant")},
                        "finish_reason": None
                    }]
                }
        elif event_type == "response.content_part.added":
            # Content part added to output
            content_part = parsed_chunk.get("part", {})
            if content_part.get("type") == "text":
                chunk = {
                    "id": parsed_chunk.get("response_id"),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": content_part.get("text", "")},
                        "finish_reason": None
                    }]
                }
                print(f"Chat provider: content_part.added -> {chunk}")
                return chunk
            else:
                print(f"Chat provider: content_part.added with non-text type {content_part.get('type')}, skipping")
                return None
        elif event_type == "response.content_part.done":
            # Content part completed
            return {
                "id": parsed_chunk.get("response_id"),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": None
                }]
            }
        elif event_type == "response.output_item.done":
            # Output item completed
            output_item = parsed_chunk.get("output_item", {})
            finish_reason = "stop"  # Default finish reason
            
            if output_item.get("type") == "message":
                status = output_item.get("status")
                finish_reason = self._map_responses_status_to_finish_reason(status)
            
            return {
                "id": parsed_chunk.get("response_id"),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason
                }]
            }
        elif event_type == "response.done":
            # Response completed - include usage if available
            response = parsed_chunk.get("response", {})
            usage_data = response.get("usage", {})
            
            chunk = {
                "id": response.get("id"),
                "object": "chat.completion.chunk",
                "created": response.get("created_at", int(time.time())),
                "model": response.get("model", model),
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": None
                }]
            }
            
            # Add usage information if available
            if usage_data:
                chunk["usage"] = {
                    "prompt_tokens": usage_data.get("input_tokens", 0),
                    "completion_tokens": usage_data.get("output_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0)
                }
            
            return chunk
        
        # For any unhandled event types, create a minimal valid chunk or skip
        print(f"Chat provider: Unhandled event type '{event_type}', creating empty chunk")
        
        # Return a minimal valid chunk for unknown events
        return {
            "id": parsed_chunk.get("response_id") or parsed_chunk.get("id") or "unknown",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": None
            }]
        }
    
    def map_openai_params(
        self,
        non_default_params: Dict[str, Any],
        optional_params: Dict[str, Any],
        model: str,
        drop_params: bool,
    ) -> Dict[str, Any]:
        # Pass through all non-default parameters into optional params
        for key, value in non_default_params.items():
            optional_params[key] = value
        return optional_params

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, "httpx.Headers"],  # noqa: F821
    ):
        # Return a BaseLLMException for chat errors
        from litellm.llms.base_llm.chat.transformation import BaseLLMException
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
    
    def _convert_content_to_responses_format(self, content: Union[str, List[Any]]) -> List[Dict[str, Any]]:
        """Convert chat completion content to responses API format"""
        print(f"Chat provider: Converting content to responses format - input type: {type(content)}")
        
        if isinstance(content, str):
            result = [{"type": "input_text", "text": content}]
            print(f"Chat provider: String content -> {result}")
            return result
        elif isinstance(content, list):
            result = []
            for i, item in enumerate(content):
                print(f"Chat provider: Processing content item {i}: {type(item)} = {item}")
                if isinstance(item, str):
                    converted = {"type": "input_text", "text": item}
                    result.append(converted)
                    print(f"Chat provider:   -> {converted}")
                elif isinstance(item, dict):
                    # Handle multimodal content
                    original_type = item.get("type")
                    if original_type == "text":
                        converted = {"type": "input_text", "text": item.get("text", "")}
                        result.append(converted)
                        print(f"Chat provider:   text -> {converted}")
                    elif original_type == "image_url":
                        # Map to responses API image format
                        converted = {"type": "input_image", "image_url": item.get("image_url", {})}
                        result.append(converted)
                        print(f"Chat provider:   image_url -> {converted}")
                    else:
                        # Try to map other types to responses API format
                        item_type = original_type or "input_text"
                        if item_type == "image":
                            converted = {"type": "input_image", **item}
                            result.append(converted)
                            print(f"Chat provider:   image -> {converted}")
                        elif item_type in ["input_text", "input_image", "output_text", "refusal", "input_file", "computer_screenshot", "summary_text"]:
                            # Already in responses API format
                            result.append(item)
                            print(f"Chat provider:   passthrough -> {item}")
                        else:
                            # Default to input_text for unknown types
                            converted = {"type": "input_text", "text": str(item.get("text", item))}
                            result.append(converted)
                            print(f"Chat provider:   unknown({original_type}) -> {converted}")
            print(f"Chat provider: Final converted content: {result}")
            return result
        else:
            result = [{"type": "input_text", "text": str(content)}]
            print(f"Chat provider: Other content type -> {result}")
            return result
    
    def _convert_tools_to_responses_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert chat completion tools to responses API tools format"""
        responses_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})
                responses_tools.append({
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {}),
                    "strict": function.get("strict", False)
                })
        return responses_tools
    
    def _map_responses_status_to_finish_reason(self, status: Optional[str]) -> str:
        """Map responses API status to chat completion finish_reason"""
        if not status:
            return "stop"
        
        status_mapping = {
            "completed": "stop",
            "incomplete": "length", 
            "failed": "stop",
            "cancelled": "stop"
        }
        
        return status_mapping.get(status, "stop")
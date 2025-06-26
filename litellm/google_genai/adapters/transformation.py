import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, cast

from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionRequest,
    ChatCompletionSystemMessage,
    ChatCompletionUserMessage,
)
from litellm.types.utils import (
    AdapterCompletionStreamWrapper,
    Choices,
    ModelResponse,
    StreamingChoices,
)


class GoogleGenAIStreamWrapper(AdapterCompletionStreamWrapper):
    """
    Wrapper for streaming Google GenAI generate_content responses.
    Transforms OpenAI streaming chunks to Google GenAI format.
    """
    
    sent_first_chunk: bool = False
    
    def __init__(self, completion_stream: Any):
        super().__init__(completion_stream)
        self.sent_first_chunk = False
    
    def __next__(self):
        try:
            for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    continue
                    
                # Transform OpenAI streaming chunk to Google GenAI format
                transformed_chunk = GoogleGenAIAdapter().translate_streaming_completion_to_generate_content(chunk)
                return transformed_chunk
                
            raise StopIteration
        except StopIteration:
            raise StopIteration
        except Exception:
            raise StopIteration
    
    async def __anext__(self):
        try:
            async for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    continue
                    
                # Transform OpenAI streaming chunk to Google GenAI format  
                transformed_chunk = GoogleGenAIAdapter().translate_streaming_completion_to_generate_content(chunk)
                return transformed_chunk
                
            raise StopAsyncIteration
        except StopAsyncIteration:
            raise StopAsyncIteration
        except Exception:
            raise StopAsyncIteration
    
    def google_genai_sse_wrapper(self) -> Iterator[bytes]:
        """
        Convert Google GenAI streaming chunks to Server-Sent Events format.
        """
        for chunk in self:
            if isinstance(chunk, dict):
                payload = f"data: {json.dumps(chunk)}\n\n"
                yield payload.encode()
            else:
                yield chunk
    
    async def async_google_genai_sse_wrapper(self) -> AsyncIterator[bytes]:
        """
        Async version of google_genai_sse_wrapper.
        """
        async for chunk in self:
            if isinstance(chunk, dict):
                payload = f"data: {json.dumps(chunk)}\n\n"
                yield payload.encode()
            else:
                yield chunk


class GoogleGenAIAdapter:
    """Adapter for transforming Google GenAI generate_content requests to/from litellm.completion format"""
    
    def __init__(self) -> None:
        pass

    def translate_generate_content_to_completion(
        self,
        model: str,
        contents: Union[List[Dict[str, Any]], Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChatCompletionRequest:
        """
        Transform generate_content request to litellm completion format
        
        Args:
            model: The model name
            contents: Generate content contents (can be list or single dict)
            config: Optional config parameters
            **kwargs: Additional parameters
            
        Returns:
            ChatCompletionRequest in OpenAI format
        """
        
        # Normalize contents to list format
        if isinstance(contents, dict):
            contents_list = [contents]
        else:
            contents_list = contents
            
        # Transform contents to OpenAI messages format
        messages = self._transform_contents_to_messages(contents_list)
        
        # Create base request
        completion_request: ChatCompletionRequest = {
            "model": model,
            "messages": messages,
        }
        
        # Add config parameters if provided
        if config:
            # Map common Google GenAI config parameters to OpenAI equivalents
            if "temperature" in config:
                completion_request["temperature"] = config["temperature"]
            if "maxOutputTokens" in config:
                completion_request["max_tokens"] = config["maxOutputTokens"]
            if "topP" in config:
                completion_request["top_p"] = config["topP"]
            if "topK" in config:
                # OpenAI doesn't have direct topK, but we can pass it as extra
                pass
            if "stopSequences" in config:
                completion_request["stop"] = config["stopSequences"]
                
        # Add any additional kwargs that are valid for completion
        valid_completion_params = [
            "temperature", "max_tokens", "top_p", "frequency_penalty", 
            "presence_penalty", "stop", "stream", "user"
        ]
        for key, value in kwargs.items():
            if key in valid_completion_params and key not in completion_request:
                completion_request[key] = value
                
        return completion_request

    def translate_completion_output_params_streaming(
        self, completion_stream: Any
    ) -> Union[AsyncIterator[bytes], None]:
        """Transform streaming completion output to Google GenAI format"""
        google_genai_wrapper = GoogleGenAIStreamWrapper(completion_stream=completion_stream)
        # Return the SSE-wrapped version for proper event formatting
        return google_genai_wrapper.async_google_genai_sse_wrapper()

    def _transform_contents_to_messages(self, contents: List[Dict[str, Any]]) -> List[AllMessageValues]:
        """Transform Google GenAI contents to OpenAI messages format"""
        messages: List[AllMessageValues] = []
        
        for content in contents:
            role = content.get("role", "user")
            parts = content.get("parts", [])
            
            if role == "user":
                # Combine all text parts into a single user message
                combined_text = ""
                for part in parts:
                    if isinstance(part, dict) and "text" in part:
                        combined_text += part["text"]
                    elif isinstance(part, str):
                        combined_text += part
                        
                if combined_text:
                    messages.append(ChatCompletionUserMessage(
                        role="user",
                        content=combined_text
                    ))
                    
            elif role == "model":
                # Transform model response to assistant message
                combined_text = ""
                for part in parts:
                    if isinstance(part, dict) and "text" in part:
                        combined_text += part["text"]
                    elif isinstance(part, str):
                        combined_text += part
                        
                if combined_text:
                    messages.append(ChatCompletionAssistantMessage(
                        role="assistant",
                        content=combined_text
                    ))
                    
        return messages

    def translate_completion_to_generate_content(
        self, response: ModelResponse
    ) -> Dict[str, Any]:
        """
        Transform litellm completion response to Google GenAI generate_content format
        
        Args:
            response: ModelResponse from litellm.completion
            
        Returns:
            Dict in Google GenAI generate_content response format
        """
        
        # Extract the main response content
        choice = response.choices[0] if response.choices else None
        if not choice:
            raise ValueError("Invalid completion response: no choices found")
            
        # Handle different choice types (Choices vs StreamingChoices)
        if isinstance(choice, Choices):
            if not choice.message:
                raise ValueError("Invalid completion response: no message found in choice")
            message_content = choice.message.content or ""
        elif isinstance(choice, StreamingChoices):
            if not choice.delta:
                raise ValueError("Invalid completion response: no delta found in streaming choice")
            message_content = choice.delta.content or ""
        else:
            # Fallback for generic choice objects
            message_content = getattr(choice, 'message', {}).get('content', '') or getattr(choice, 'delta', {}).get('content', '')
        
        # Create Google GenAI format response
        generate_content_response: Dict[str, Any] = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": message_content}],
                        "role": "model"
                    },
                    "finishReason": self._map_finish_reason(getattr(choice, 'finish_reason', None)),
                    "index": 0,
                    "safetyRatings": []
                }
            ],
            "usageMetadata": (
                self._map_usage(getattr(response, 'usage', None)) 
                if hasattr(response, 'usage') and getattr(response, 'usage', None) 
                else {
                    "promptTokenCount": 0,
                    "candidatesTokenCount": 0,
                    "totalTokenCount": 0
                }
            )
        }
        
        # Add text field for convenience (common in Google GenAI responses)
        if message_content:
            generate_content_response["text"] = message_content
            
        return generate_content_response

    def translate_streaming_completion_to_generate_content(
        self, response: ModelResponse
    ) -> Dict[str, Any]:
        """
        Transform streaming litellm completion chunk to Google GenAI generate_content format
        
        Args:
            response: Streaming ModelResponse chunk from litellm.completion
            
        Returns:
            Dict in Google GenAI streaming generate_content response format
        """
        
        # Extract the main response content from streaming chunk
        choice = response.choices[0] if response.choices else None
        if not choice:
            # Return empty chunk if no choices
            return {}
            
        # Handle streaming choice
        if isinstance(choice, StreamingChoices):
            message_content = choice.delta.content or "" if choice.delta else ""
            finish_reason = getattr(choice, 'finish_reason', None)
        else:
            # Fallback for generic choice objects
            message_content = getattr(choice, 'delta', {}).get('content', '')
            finish_reason = getattr(choice, 'finish_reason', None)
        
        # Create Google GenAI streaming format response
        streaming_chunk: Dict[str, Any] = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": message_content}] if message_content else [],
                        "role": "model"
                    },
                    "finishReason": self._map_finish_reason(finish_reason) if finish_reason else None,
                    "index": 0,
                    "safetyRatings": []
                }
            ]
        }
        
        # Add usage metadata only in the final chunk (when finish_reason is present)
        if finish_reason:
            usage_metadata = self._map_usage(getattr(response, 'usage', None)) if hasattr(response, 'usage') and getattr(response, 'usage', None) else {
                "promptTokenCount": 0,
                "candidatesTokenCount": 0,
                "totalTokenCount": 0
            }
            streaming_chunk["usageMetadata"] = usage_metadata
        
        # Add text field for convenience (common in Google GenAI responses)
        if message_content:
            streaming_chunk["text"] = message_content
            
        return streaming_chunk

    def _map_finish_reason(self, finish_reason: Optional[str]) -> str:
        """Map OpenAI finish reasons to Google GenAI finish reasons"""
        if not finish_reason:
            return "STOP"
            
        mapping = {
            "stop": "STOP",
            "length": "MAX_TOKENS", 
            "content_filter": "SAFETY",
            "tool_calls": "STOP",
            "function_call": "STOP",
        }
        
        return mapping.get(finish_reason, "STOP")

    def _map_usage(self, usage: Any) -> Dict[str, int]:
        """Map OpenAI usage to Google GenAI usage format"""
        return {
            "promptTokenCount": getattr(usage, "prompt_tokens", 0) or 0,
            "candidatesTokenCount": getattr(usage, "completion_tokens", 0) or 0,
            "totalTokenCount": getattr(usage, "total_tokens", 0) or 0,
        } 
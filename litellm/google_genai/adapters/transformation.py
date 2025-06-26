import json
from typing import Any, Dict, List, Optional, Union, cast

from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionRequest,
    ChatCompletionSystemMessage,
    ChatCompletionUserMessage,
)
from litellm.types.utils import Choices, ModelResponse, StreamingChoices


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
        generate_content_response = {
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
            "usageMetadata": self._map_usage(getattr(response, 'usage', None)) if hasattr(response, 'usage') and getattr(response, 'usage', None) else {
                "promptTokenCount": 0,
                "candidatesTokenCount": 0,
                "totalTokenCount": 0
            }
        }
        
        # Add text field for convenience (common in Google GenAI responses)
        if message_content:
            generate_content_response["text"] = message_content
            
        return generate_content_response

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
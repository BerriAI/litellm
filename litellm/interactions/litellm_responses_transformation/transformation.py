"""
Handles transforming from Interactions API -> LiteLLM Responses API

This follows the same pattern as LiteLLMCompletionResponsesConfig that transforms
from Responses API -> Chat Completion API.

Interactions API (Google) -> Responses API (OpenAI)
"""

import json
from typing import Any, Dict, List, Optional, Union

from litellm.types.interactions import (
    InteractionInput,
    InteractionsAPIOptionalRequestParams,
    InteractionsAPIResponse,
)
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIResponse,
)


class LiteLLMResponsesInteractionsConfig:
    """
    Configuration class for transforming between Interactions API and Responses API.
    
    Transforms:
    - Interactions API request -> Responses API request
    - Responses API response -> Interactions API response
    """

    @staticmethod
    def get_supported_params() -> List[str]:
        """
        Return list of supported parameters that can be passed through the bridge.
        """
        return [
            "input",
            "model",
            "agent",  # Interactions API specific
            "tools",
            "system_instruction",  # Maps to instructions
            "generation_config",  # Maps to various response params
            "stream",
            "store",
            "background",
            "response_modalities",
            "response_format",
            "response_mime_type",
            "previous_interaction_id",  # Maps to previous_response_id
        ]

    # =========================================================
    # REQUEST TRANSFORMATION: Interactions -> Responses
    # =========================================================

    @staticmethod
    def transform_interactions_api_request_to_responses_api_request(
        model: Optional[str],
        agent: Optional[str],
        input: Optional[InteractionInput],
        interactions_api_request: InteractionsAPIOptionalRequestParams,
        custom_llm_provider: Optional[str] = None,
        stream: Optional[bool] = None,
        extra_headers: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Transform an Interactions API request into a Responses API request.
        
        Interactions API -> Responses API mapping:
        - input -> input (with format transformation)
        - model -> model
        - agent -> model (use agent as model identifier if no model)
        - system_instruction -> instructions
        - tools -> tools (with format transformation)
        - generation_config.temperature -> temperature
        - generation_config.top_p -> top_p
        - generation_config.max_output_tokens -> max_output_tokens
        - generation_config.tool_choice -> tool_choice
        - previous_interaction_id -> previous_response_id
        - stream -> stream
        - store -> store
        """
        # Transform input
        transformed_input = LiteLLMResponsesInteractionsConfig._transform_interactions_input_to_responses_input(
            input=input
        )
        
        # Transform tools
        transformed_tools = LiteLLMResponsesInteractionsConfig._transform_interactions_tools_to_responses_tools(
            tools=interactions_api_request.get("tools")
        )
        
        # Extract generation config parameters
        generation_config: Optional[Dict[str, Any]] = interactions_api_request.get("generation_config")
        temperature = None
        top_p = None
        max_output_tokens = None
        tool_choice = None
        
        if generation_config:
            temperature = generation_config.get("temperature")
            top_p = generation_config.get("top_p")
            max_output_tokens = generation_config.get("max_output_tokens")
            tool_choice = LiteLLMResponsesInteractionsConfig._transform_tool_choice(
                generation_config.get("tool_choice")
            )
        
        # Build the responses API request
        responses_api_request: Dict[str, Any] = {
            "model": model or agent or "",  # Use agent as model if no model specified
            "input": transformed_input,
            "instructions": interactions_api_request.get("system_instruction"),
            "tools": transformed_tools,
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
            "tool_choice": tool_choice,
            "previous_response_id": interactions_api_request.get("previous_interaction_id"),
            "stream": stream,
            "store": interactions_api_request.get("store"),
            # LiteLLM specific params
            "custom_llm_provider": custom_llm_provider,
            "extra_headers": extra_headers,
        }
        
        # Filter out None values
        responses_api_request = {
            k: v for k, v in responses_api_request.items() if v is not None
        }
        
        return responses_api_request

    @staticmethod
    def _transform_interactions_input_to_responses_input(
        input: Optional[InteractionInput],
    ) -> Union[str, ResponseInputParam]:
        """
        Transform Interactions API input to Responses API input format.
        
        Interactions input can be:
        - str: Simple text input
        - Content: Single content object
        - List[Content]: List of content objects  
        - List[Turn]: Multi-turn conversation
        
        Responses input can be:
        - str: Simple text input
        - List of message objects with role and content
        """
        if input is None:
            return ""
        
        if isinstance(input, str):
            return input
        
        if isinstance(input, list):
            # Check if it's a list of Turn objects (multi-turn)
            if input and isinstance(input[0], dict) and "role" in input[0]:
                # List of Turn objects
                return LiteLLMResponsesInteractionsConfig._transform_turns_to_responses_input(
                    turns=input
                )
            else:
                # List of Content objects - treat as user message
                return LiteLLMResponsesInteractionsConfig._transform_content_list_to_responses_input(
                    content_list=input
                )
        
        if isinstance(input, dict):
            # Single Content object - wrap in user message
            return [
                {
                    "role": "user",
                    "content": LiteLLMResponsesInteractionsConfig._transform_content_to_responses_content(
                        input
                    ),
                }
            ]
        
        # Fallback: convert to string
        return str(input)

    @staticmethod
    def _transform_turns_to_responses_input(
        turns: List[Dict[str, Any]],
    ) -> ResponseInputParam:
        """
        Transform Interactions API Turn objects to Responses API input format.
        
        Turn: { role: str, content: str | List[Content] }
        """
        messages: List[Dict[str, Any]] = []
        
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content")
            
            # Map Interactions roles to Responses roles
            if role == "model":
                role = "assistant"
            
            if isinstance(content, str):
                messages.append({
                    "role": role,
                    "content": content,
                })
            elif isinstance(content, list):
                # List of Content objects
                transformed_content = []
                for item in content:
                    transformed_item = LiteLLMResponsesInteractionsConfig._transform_content_to_responses_content(
                        item
                    )
                    if transformed_item:
                        transformed_content.append(transformed_item)
                
                if transformed_content:
                    messages.append({
                        "role": role,
                        "content": transformed_content,
                    })
        
        return messages

    @staticmethod
    def _transform_content_list_to_responses_input(
        content_list: List[Dict[str, Any]],
    ) -> ResponseInputParam:
        """
        Transform a list of Content objects to Responses API input.
        """
        transformed_content = []
        for item in content_list:
            transformed_item = LiteLLMResponsesInteractionsConfig._transform_content_to_responses_content(
                item
            )
            if transformed_item:
                transformed_content.append(transformed_item)
        
        if transformed_content:
            return [{"role": "user", "content": transformed_content}]
        
        return ""

    @staticmethod
    def _transform_content_to_responses_content(
        content: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Transform a single Interactions API Content object to Responses API content format.
        
        Interactions Content types:
        - text, image, audio, document, video, thought
        - function_call, function_result
        - code_execution_call, code_execution_result
        - url_context_call, url_context_result
        - google_search_call, google_search_result
        - mcp_server_tool_call, mcp_server_tool_result
        """
        content_type = content.get("type")
        
        if content_type == "text":
            return {
                "type": "input_text",
                "text": content.get("text", ""),
            }
        
        elif content_type == "image":
            # Transform image content
            image_data = content.get("data")
            image_uri = content.get("uri")
            
            if image_data:
                mime_type = content.get("mime_type", "image/png")
                return {
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{image_data}",
                }
            elif image_uri:
                return {
                    "type": "input_image",
                    "image_url": image_uri,
                }
        
        elif content_type == "function_call":
            # Transform function call to Responses API format
            return {
                "type": "function_call",
                "name": content.get("name", ""),
                "arguments": json.dumps(content.get("arguments", {})),
                "call_id": content.get("id", ""),
            }
        
        elif content_type == "function_result":
            # Transform function result to Responses API format
            result = content.get("result", "")
            if isinstance(result, dict):
                result = json.dumps(result)
            
            return {
                "type": "function_call_output",
                "call_id": content.get("call_id", ""),
                "output": result,
            }
        
        elif content_type == "thought":
            # Thought content - include as text with annotation
            summary = content.get("summary", [])
            if summary:
                text_content = ""
                for item in summary:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content += item.get("text", "")
                if text_content:
                    return {
                        "type": "input_text",
                        "text": text_content,
                    }
        
        # For other types, try to extract text
        if "text" in content:
            return {
                "type": "input_text",
                "text": content.get("text", ""),
            }
        
        return None

    @staticmethod
    def _transform_interactions_tools_to_responses_tools(
        tools: Optional[List[Dict[str, Any]]],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Transform Interactions API tools to Responses API tools format.
        
        Interactions Tool types:
        - function: User-defined function
        - google_search: Google search tool
        - code_execution: Code execution tool
        - url_context: URL context tool
        - computer_use: Computer use tool
        - mcp_server: MCP server tool
        - file_search: File search tool
        
        Responses API tools:
        - function: { type: "function", name, description, parameters }
        - web_search_preview: Web search
        """
        if not tools:
            return None
        
        transformed_tools: List[Dict[str, Any]] = []
        
        for tool in tools:
            tool_type = tool.get("type")
            
            if tool_type == "function":
                # Function tool - direct mapping
                transformed_tools.append({
                    "type": "function",
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                })
            
            elif tool_type == "google_search":
                # Map to web_search_preview
                transformed_tools.append({
                    "type": "web_search_preview",
                })
            
            # Other tool types (code_execution, url_context, etc.) 
            # don't have direct mappings in Responses API
            # They could be handled by provider-specific logic
        
        return transformed_tools if transformed_tools else None

    @staticmethod
    def _transform_tool_choice(
        tool_choice: Optional[Any],
    ) -> Optional[Union[str, Dict[str, Any]]]:
        """
        Transform Interactions API tool_choice to Responses API format.
        
        Interactions tool_choice can be:
        - ToolChoiceType enum: auto, any, none, validated
        - ToolChoiceConfig: { allowed_tools: { mode, tools } }
        """
        if tool_choice is None:
            return None
        
        if isinstance(tool_choice, str):
            # Map string values
            if tool_choice in ["auto", "none"]:
                return tool_choice
            elif tool_choice in ["any", "required"]:
                return "required"
            elif tool_choice == "validated":
                return "auto"
        
        if isinstance(tool_choice, dict):
            # Handle ToolChoiceConfig
            allowed_tools = tool_choice.get("allowed_tools", {})
            mode = allowed_tools.get("mode")
            tools = allowed_tools.get("tools", [])
            
            if mode:
                return LiteLLMResponsesInteractionsConfig._transform_tool_choice(mode)
            
            # If specific tools are specified, return the first one
            if tools:
                return {
                    "type": "function",
                    "function": {"name": tools[0]},
                }
        
        return "auto"

    # =========================================================
    # RESPONSE TRANSFORMATION: Responses -> Interactions
    # =========================================================

    @staticmethod
    def transform_responses_api_response_to_interactions_api_response(
        responses_api_response: ResponsesAPIResponse,
        request_input: Optional[InteractionInput] = None,
        interactions_api_request: Optional[InteractionsAPIOptionalRequestParams] = None,
    ) -> InteractionsAPIResponse:
        """
        Transform a Responses API response into an Interactions API response.
        
        ResponsesAPIResponse:
        - id, model, status, output, usage
        
        InteractionsAPIResponse:
        - id, model, agent, status, outputs, usage
        """
        # Transform output items to Interactions format
        outputs = LiteLLMResponsesInteractionsConfig._transform_responses_output_to_interactions_outputs(
            responses_api_response.output if hasattr(responses_api_response, 'output') else []
        )
        
        # Transform usage
        usage = LiteLLMResponsesInteractionsConfig._transform_responses_usage_to_interactions_usage(
            responses_api_response.usage if hasattr(responses_api_response, 'usage') else None
        )
        
        # Map status
        status = LiteLLMResponsesInteractionsConfig._map_responses_status_to_interactions_status(
            getattr(responses_api_response, 'status', 'completed')
        )
        
        return InteractionsAPIResponse(
            id=getattr(responses_api_response, 'id', ''),
            object="interaction",
            model=getattr(responses_api_response, 'model', None),
            agent=None,  # Set from original request if needed
            status=status,
            created=str(getattr(responses_api_response, 'created_at', '')),
            role="model",
            outputs=outputs,
            usage=usage,
        )

    @staticmethod
    def _transform_responses_output_to_interactions_outputs(
        output: List[Any],
    ) -> List[Dict[str, Any]]:
        """
        Transform Responses API output to Interactions API outputs format.
        
        Responses output types:
        - message: { type: "message", content: [{ type: "output_text", text }] }
        - function_call: { type: "function_call", name, arguments, call_id }
        
        Interactions output types:
        - text: { type: "text", text }
        - function_call: { type: "function_call", name, arguments, id }
        """
        outputs: List[Dict[str, Any]] = []
        
        for item in output:
            item_dict = dict(item) if hasattr(item, '__dict__') or hasattr(item, 'model_dump') else item
            if hasattr(item, 'model_dump'):
                item_dict = item.model_dump()
            
            item_type = item_dict.get("type")
            
            if item_type == "message":
                # Extract text content from message
                content = item_dict.get("content", [])
                for content_item in content:
                    content_dict = dict(content_item) if hasattr(content_item, '__dict__') else content_item
                    if hasattr(content_item, 'model_dump'):
                        content_dict = content_item.model_dump()
                    
                    if content_dict.get("type") == "output_text":
                        outputs.append({
                            "type": "text",
                            "text": content_dict.get("text", ""),
                        })
            
            elif item_type == "function_call":
                # Transform function call
                outputs.append({
                    "type": "function_call",
                    "name": item_dict.get("name", ""),
                    "arguments": LiteLLMResponsesInteractionsConfig._parse_arguments(
                        item_dict.get("arguments", "{}")
                    ),
                    "id": item_dict.get("call_id") or item_dict.get("id", ""),
                })
            
            elif item_type == "reasoning":
                # Transform reasoning content
                content = item_dict.get("content", [])
                for content_item in content:
                    content_dict = dict(content_item) if hasattr(content_item, '__dict__') else content_item
                    if hasattr(content_item, 'model_dump'):
                        content_dict = content_item.model_dump()
                    
                    outputs.append({
                        "type": "thought",
                        "summary": [{
                            "type": "text",
                            "text": content_dict.get("text", ""),
                        }],
                    })
        
        return outputs

    @staticmethod
    def _parse_arguments(arguments: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Parse function arguments from string or dict."""
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                return {"raw": arguments}
        return {}

    @staticmethod
    def _transform_responses_usage_to_interactions_usage(
        usage: Optional[Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Transform Responses API usage to Interactions API usage format.
        """
        if usage is None:
            return None
        
        usage_dict = dict(usage) if hasattr(usage, '__dict__') else usage
        if hasattr(usage, 'model_dump'):
            usage_dict = usage.model_dump()
        
        return {
            "total_input_tokens": usage_dict.get("input_tokens", 0),
            "total_output_tokens": usage_dict.get("output_tokens", 0),
            "total_tokens": usage_dict.get("total_tokens", 0),
        }

    @staticmethod
    def _map_responses_status_to_interactions_status(
        status: str,
    ) -> str:
        """
        Map Responses API status to Interactions API status.
        
        Responses status: completed, failed, in_progress, cancelled, queued, incomplete
        Interactions status: UNSPECIFIED, IN_PROGRESS, REQUIRES_ACTION, COMPLETED, FAILED, CANCELLED
        """
        status_map = {
            "completed": "completed",
            "failed": "failed",
            "in_progress": "in_progress",
            "cancelled": "cancelled",
            "queued": "in_progress",
            "incomplete": "requires_action",
        }
        return status_map.get(status, "completed")

    # =========================================================
    # STREAMING RESPONSE TRANSFORMATION
    # =========================================================

    @staticmethod
    def transform_responses_streaming_chunk_to_interactions_chunk(
        responses_chunk: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Transform a Responses API streaming chunk to Interactions API streaming format.
        
        Responses streaming events:
        - response.created, response.in_progress, response.completed
        - output_item.added, output_item.done
        - content_part.added, content_part.done
        - output_text.delta, output_text.done
        
        Interactions streaming events:
        - interaction.start, interaction.status_update, interaction.complete
        - content.start, content.delta, content.stop
        """
        if responses_chunk is None:
            return None
        
        chunk_dict = dict(responses_chunk) if hasattr(responses_chunk, '__dict__') else responses_chunk
        if hasattr(responses_chunk, 'model_dump'):
            chunk_dict = responses_chunk.model_dump()
        
        event_type = chunk_dict.get("type")
        
        # Map event types
        if event_type == "response.created":
            return {
                "event_type": "interaction.start",
                "interaction": {
                    "id": chunk_dict.get("response", {}).get("id"),
                    "status": "in_progress",
                    "object": "interaction",
                },
            }
        
        elif event_type == "response.in_progress":
            return {
                "event_type": "interaction.status_update",
                "status": "in_progress",
            }
        
        elif event_type == "response.completed":
            response_data = chunk_dict.get("response", {})
            return {
                "event_type": "interaction.complete",
                "interaction": LiteLLMResponsesInteractionsConfig._transform_completed_response_to_interaction(
                    response_data
                ),
            }
        
        elif event_type == "response.output_text.delta":
            return {
                "event_type": "content.delta",
                "delta": {
                    "type": "text",
                    "text": chunk_dict.get("delta", ""),
                },
            }
        
        elif event_type == "response.content_part.added":
            return {
                "event_type": "content.start",
                "content": {
                    "type": "text",
                    "text": "",
                },
            }
        
        elif event_type == "response.content_part.done":
            return {
                "event_type": "content.stop",
            }
        
        elif event_type == "response.output_item.added":
            # New output item started
            return {
                "event_type": "content.start",
                "content": chunk_dict.get("item", {}),
            }
        
        elif event_type == "response.reasoning_summary_text.delta":
            return {
                "event_type": "content.delta",
                "delta": {
                    "type": "thought_summary",
                    "content": {
                        "type": "text",
                        "text": chunk_dict.get("delta", ""),
                    },
                },
            }
        
        # Unknown event type - return None to skip
        return None

    @staticmethod
    def _transform_completed_response_to_interaction(
        response_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Transform a completed response to interaction format."""
        outputs = LiteLLMResponsesInteractionsConfig._transform_responses_output_to_interactions_outputs(
            response_data.get("output", [])
        )
        
        usage = LiteLLMResponsesInteractionsConfig._transform_responses_usage_to_interactions_usage(
            response_data.get("usage")
        )
        
        return {
            "id": response_data.get("id", ""),
            "object": "interaction",
            "model": response_data.get("model"),
            "status": "completed",
            "role": "model",
            "outputs": outputs,
            "usage": usage,
        }

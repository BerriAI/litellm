"""
Support for Kimi-K2-Thinking and MiniMax M2 models in Bedrock
"""

import re
from typing import Any, List, Optional

from litellm.types.llms.openai import ChatCompletionToolCallChunk, ChatCompletionToolCallFunctionChunk


def parse_kimi_tool_calls(text_content: str) -> List[ChatCompletionToolCallChunk]:
    """
    Parse Kimi-K2-Thinking proprietary token-stream format to extract tool calls.
    
    Kimi-K2-Thinking outputs raw tokens with markers like:
    <|tool_calls_section_begin|>...<|tool_call_begin|>function_name<|tool_call_argument_begin|>{...}<|tool_call_end|>...<|tool_calls_section_end|>
    
    Args:
        text_content: The text content from the model response
        
    Returns:
        List of parsed tool call chunks
    """
    tool_calls = []
    
    # Pattern to capture individual tool calls
    # Matches: <|tool_call_begin|>function_name<|tool_call_argument_begin|>{arguments}<|tool_call_end|>
    pattern = r"<\|tool_call_begin\|>\s*([\w\.]+)\s*<\|tool_call_argument_begin\|>\s*(\{.*?\})\s*<\|tool_call_end\|>"
    
    matches = re.finditer(pattern, text_content, re.DOTALL)
    
    for i, match in enumerate(matches):
        function_name = match.group(1)
        arguments = match.group(2)
        
        # Generate a unique ID for the tool call
        tool_call_id = f"call_kimi_{i}_{hash(arguments) % 100000000}"
        
        # Create the function chunk
        function_chunk = ChatCompletionToolCallFunctionChunk(
            name=function_name,
            arguments=arguments
        )
        
        # Create the tool call chunk
        tool_call_chunk = ChatCompletionToolCallChunk(
            id=tool_call_id,
            type="function",
            function=function_chunk,
            index=i
        )
        
        tool_calls.append(tool_call_chunk)
    
    return tool_calls


def needs_kimi_parsing(text_content: str) -> bool:
    """
    Check if the text content contains Kimi-specific tool call markers.
    
    Args:
        text_content: The text content to check
        
    Returns:
        True if Kimi parsing is needed, False otherwise
    """
    return "<|tool_calls_section_begin|>" in text_content or "<|tool_call_begin|>" in text_content


def parse_kimi_streaming_content(text_content: str, remaining_buffer: str = "") -> tuple:
    """
    Parse streaming content from Kimi-K2-Thinking with buffering support.
    
    Args:
        text_content: The current chunk of text content
        remaining_buffer: Any remaining buffer from previous chunks
        
    Returns:
        tuple of (parsed_content, remaining_buffer)
    """
    # Combine buffer with current content
    full_content = remaining_buffer + text_content
    
    # Check if we have a complete tool calls section
    has_complete_section = (
        "<|tool_calls_section_begin|>" in full_content 
        and "<|tool_calls_section_end|>" in full_content
    )
    
    if has_complete_section:
        # Extract the complete section
        pattern = r"(.*<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>.*?)(.*)"
        match = re.search(pattern, full_content, re.DOTALL)
        
        if match:
            complete_section = match.group(1)
            remaining = match.group(2) or ""
            
            # Parse the tool calls from the complete section
            # The rest of the text content (complete_section) is handled by parse_kimi_tool_calls
            return complete_section, remaining
    
    # No complete section yet, keep buffering
    return full_content, ""


def map_minimax_reasoning_params(optional_params: dict, model: str) -> dict:
    """
    Map reasoning/thinking parameters for MiniMax M2 model.
    
    MiniMax M2 uses different parameter names for thinking capabilities.
    
    Args:
        optional_params: The optional parameters dict
        model: The model name
        
    Returns:
        Updated optional_params dict
    """
    # Check if this is a MiniMax M2 model
    base_model = model.lower()
    if "minimax" not in base_model:
        return optional_params
    
    # Map reasoning_effort to MiniMax-specific parameter if present
    if "reasoning_effort" in optional_params:
        reasoning_value = optional_params["reasoning_effort"]
        
        # MiniMax uses "reasoning_content" parameter
        # Map values: low -> false, medium/true -> true
        if isinstance(reasoning_value, str):
            if reasoning_value.lower() in ["low", "none"]:
                optional_params["reasoning_content"] = False
            else:
                optional_params["reasoning_content"] = True
        elif isinstance(reasoning_value, bool):
            optional_params["reasoning_content"] = reasoning_value
    
    # Ensure thinking parameter is handled correctly
    # MiniMax may require specific configuration for thinking mode
    if "thinking" in optional_params:
        optional_params["enable_thinking"] = True
    
    return optional_params


def is_kimi_model(model: str) -> bool:
    """
    Check if the model is a Kimi-K2-Thinking model.
    
    Args:
        model: The model identifier
        
    Returns:
        True if it's a Kimi model, False otherwise
    """
    return "moonshot.kimi-k2-thinking" in model.lower()


def is_minimax_m2_model(model: str) -> bool:
    """
    Check if the model is a MiniMax M2 model.
    
    Args:
        model: The model identifier
        
    Returns:
        True if it's a MiniMax M2 model, False otherwise
    """
    return "minimax.minimax-m2" in model.lower()
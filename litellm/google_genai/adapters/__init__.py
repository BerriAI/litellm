"""
Google GenAI Adapters for LiteLLM

This module provides adapters for transforming Google GenAI generate_content requests 
to/from LiteLLM completion format with full support for:
- Text content transformation
- Tool calling (function declarations, function calls, function responses)  
- Streaming (both regular and tool calling)
- Mixed content (text + tool calls)
"""

from .handler import GenerateContentToCompletionHandler
from .transformation import GoogleGenAIAdapter, GoogleGenAIStreamWrapper

__all__ = [
    "GoogleGenAIAdapter", 
    "GoogleGenAIStreamWrapper",
    "GenerateContentToCompletionHandler"
] 
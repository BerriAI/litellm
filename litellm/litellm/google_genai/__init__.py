"""
This allows using Google GenAI model in their native interface.

This module provides generate_content functionality for Google GenAI models.
"""

from .main import (
    agenerate_content,
    agenerate_content_stream,
    generate_content,
    generate_content_stream,
)

__all__ = [
    "generate_content",
    "agenerate_content", 
    "generate_content_stream",
    "agenerate_content_stream",
] 
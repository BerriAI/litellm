"""
Type definitions for litellm.compress().
"""

from typing import Dict, List, TypedDict


class CompressedResult(TypedDict):
    messages: List[dict]  # compressed messages (stubs replace low-relevance messages)
    original_tokens: int  # token count before compression
    compressed_tokens: int  # token count after compression
    compression_ratio: float  # fraction reduced, e.g. 0.6 means 60% reduction
    cache: Dict[str, str]  # key -> original content (for retrieval tool responses)
    tools: List[dict]  # [litellm_content_retrieve tool definition]

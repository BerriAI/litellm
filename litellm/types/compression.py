"""
Type definitions for litellm.compress().
"""

import sys

if sys.version_info >= (3, 11):
    from typing import Dict, List, NotRequired, TypedDict
else:
    from typing import Dict, List, TypedDict

    from typing_extensions import NotRequired


class CompressedResult(TypedDict):
    messages: List[dict]  # compressed messages (stubs replace low-relevance messages)
    original_tokens: int  # token count before compression
    compressed_tokens: int  # token count after compression
    compression_ratio: float  # fraction reduced, e.g. 0.6 means 60% reduction
    cache: Dict[str, str]  # key -> original content (for retrieval tool responses)
    tools: List[dict]  # [litellm_content_retrieve tool definition]
    compression_skipped_reason: NotRequired[str]

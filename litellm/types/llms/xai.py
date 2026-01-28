from typing import List, Literal, Optional, TypedDict


class XAIWebSearchFilters(TypedDict, total=False):
    """Filters for XAI web search tool"""
    allowed_domains: Optional[List[str]]  # Max 5 domains
    excluded_domains: Optional[List[str]]  # Max 5 domains
    
class XAIWebSearchTool(TypedDict, total=False):
    """XAI web search tool configuration"""
    type: Literal["web_search"]
    filters: Optional[XAIWebSearchFilters]
    enable_image_understanding: Optional[bool]

class XAIXSearchTool(TypedDict, total=False):
    """XAI X (Twitter) search tool configuration"""
    type: Literal["x_search"]
    allowed_x_handles: Optional[List[str]]  # Max 10 handles
    excluded_x_handles: Optional[List[str]]  # Max 10 handles
    from_date: Optional[str]  # ISO8601 format: YYYY-MM-DD
    to_date: Optional[str]  # ISO8601 format: YYYY-MM-DD
    enable_image_understanding: Optional[bool]
    enable_video_understanding: Optional[bool]
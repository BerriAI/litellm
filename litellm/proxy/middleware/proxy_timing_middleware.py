"""
Proxy Timing Middleware - Tracks total request time including auth
"""

from datetime import datetime

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class ProxyTimingMiddleware(BaseHTTPMiddleware):
    """
    Lightweight middleware to track total proxy overhead including authentication.

    Sets request.state.proxy_start_time before any processing.
    This ensures the overhead measurement includes:
    - Request parsing
    - Authentication/authorization (DB lookups, cache checks)
    - Rate limiting
    - All proxy preprocessing
    - Response postprocessing

    But excludes:
    - The actual LLM API call time (tracked separately via @track_llm_api_timing)
    """

    async def dispatch(self, request: Request, call_next):
        # Track timing for ALL requests - minimal overhead
        # Store as datetime for compatibility with rest of codebase
        request.state.proxy_start_time = datetime.now()
        response = await call_next(request)
        return response


"""
Batch Rate Limiter Hook

This hook implements rate limiting for batch API requests by:
1. Reading batch input files to count requests and estimate tokens at submission
2. Validating actual usage from output files when batches complete
3. Integrating with the existing parallel request limiter infrastructure

## Integration & Calling
This hook is automatically registered and called by the proxy system.
See BATCH_RATE_LIMITER_INTEGRATION.md for complete integration details.

Quick summary:
- Add to PROXY_HOOKS in litellm/proxy/hooks/__init__.py
- Gets auto-instantiated on proxy startup via _add_proxy_hooks()
- async_pre_call_hook() fires on POST /v1/batches (batch submission)
- async_log_success_event() fires on GET /v1/batches/{id} (batch completion)
"""

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.batches.batch_utils import (
    _get_file_content_as_dictionary,
    calculate_batch_cost_and_usage,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        _PROXY_MaxParallelRequestsHandler_v3 as _ParallelRequestLimiter,
    )
    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache
    from litellm.router import Router as _Router

    Span = Union[_Span, Any]
    InternalUsageCache = _InternalUsageCache
    Router = _Router
    ParallelRequestLimiter = _ParallelRequestLimiter
else:
    Span = Any
    InternalUsageCache = Any
    Router = Any
    ParallelRequestLimiter = Any


class _PROXY_BatchRateLimiter(CustomLogger):
    """
    Rate limiter for batch API requests.
    
    Handles rate limiting at two points:
    1. Batch submission - reads input file and reserves capacity
    2. Batch completion - reads output file and adjusts for actual usage
    """

    def __init__(
        self,
        internal_usage_cache: InternalUsageCache,
        parallel_request_limiter: ParallelRequestLimiter,
    ):
        """
        Initialize the batch rate limiter.
        
        Note: These dependencies are automatically injected by ProxyLogging._add_proxy_hooks()
        when this hook is registered in PROXY_HOOKS. See BATCH_RATE_LIMITER_INTEGRATION.md.

        Args:
            internal_usage_cache: Cache for storing rate limit data (auto-injected)
            parallel_request_limiter: Existing rate limiter to integrate with (needs custom injection)
        """
        self.internal_usage_cache = internal_usage_cache
        self.parallel_request_limiter = parallel_request_limiter

    async def count_input_file_usage(
        self,
        file_id: str,
        custom_llm_provider: Literal["openai", "azure", "vertex_ai"] = "openai",
    ) -> Tuple[int, int]:
        """
        Count number of requests and tokens in a batch input file.
        
        Args:
            file_id: The file ID to read
            custom_llm_provider: The custom LLM provider to use for token encoding
            
        Returns:
            Tuple of (total_tokens, request_count)
        """
        try:
            # Read file content
            file_content = await litellm.afile_content(
                file_id=file_id,
                custom_llm_provider=custom_llm_provider,
            )

            file_content_as_dict = _get_file_content_as_dictionary(
                file_content.content
            )
            request_count = len(file_content_as_dict)
            return (request_count, 0)
            
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error counting input file usage for {file_id}: {str(e)}"
            )
            raise
    







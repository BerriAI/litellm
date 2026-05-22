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

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from fastapi import HTTPException
from pydantic import BaseModel

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.batches.batch_utils import (
    _get_batch_job_input_file_usage,
    _get_file_content_as_dictionary,
    _get_models_from_batch_input_file_content,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        RateLimitDescriptor as _RateLimitDescriptor,
    )
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        RateLimitStatus as _RateLimitStatus,
    )
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        _PROXY_MaxParallelRequestsHandler_v3 as _ParallelRequestLimiter,
    )
    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache
    from litellm.router import Router as _Router

    Span = Union[_Span, Any]
    InternalUsageCache = _InternalUsageCache
    Router = _Router
    ParallelRequestLimiter = _ParallelRequestLimiter
    RateLimitStatus = _RateLimitStatus
    RateLimitDescriptor = _RateLimitDescriptor
else:
    Span = Any
    InternalUsageCache = Any
    Router = Any
    ParallelRequestLimiter = Any
    RateLimitStatus = Dict[str, Any]
    RateLimitDescriptor = Dict[str, Any]


class BatchFileUsage(BaseModel):
    """
    Internal model for batch file usage tracking, used for batch rate limiting
    """

    total_tokens: int
    request_count: int


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

    def _raise_rate_limit_error(
        self,
        status: "RateLimitStatus",
        descriptors: List["RateLimitDescriptor"],
        batch_usage: BatchFileUsage,
        limit_type: str,
    ) -> None:
        """Raise HTTPException for rate limit exceeded."""
        from datetime import datetime

        # Find the descriptor for this status
        descriptor_index = next(
            (
                i
                for i, d in enumerate(descriptors)
                if d.get("key") == status.get("descriptor_key")
            ),
            0,
        )
        descriptor: RateLimitDescriptor = (
            descriptors[descriptor_index]
            if descriptors
            else {"key": "", "value": "", "rate_limit": None}
        )

        now = datetime.now().timestamp()
        window_size = self.parallel_request_limiter.window_size
        reset_time = now + window_size
        reset_time_formatted = datetime.fromtimestamp(reset_time).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

        remaining_display = max(0, status["limit_remaining"])
        current_limit = status["current_limit"]

        if limit_type == "requests":
            detail = (
                f"Batch rate limit exceeded for {descriptor.get('key', 'unknown')}: {descriptor.get('value', 'unknown')}. "
                f"Batch contains {batch_usage.request_count} requests but only {remaining_display} requests remaining "
                f"out of {current_limit} RPM limit. "
                f"Limit resets at: {reset_time_formatted}"
            )
        else:  # tokens
            detail = (
                f"Batch rate limit exceeded for {descriptor.get('key', 'unknown')}: {descriptor.get('value', 'unknown')}. "
                f"Batch contains {batch_usage.total_tokens} tokens but only {remaining_display} tokens remaining "
                f"out of {current_limit} TPM limit. "
                f"Limit resets at: {reset_time_formatted}"
            )

        raise HTTPException(
            status_code=429,
            detail=detail,
            headers={
                "retry-after": str(window_size),
                "rate_limit_type": limit_type,
                "reset_at": reset_time_formatted,
            },
        )

    async def _check_and_increment_batch_counters(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: Dict,
        batch_usage: BatchFileUsage,
    ) -> None:
        """
        Atomically check + increment rate-limit counters by the batch amounts.

        Raises HTTPException if any descriptor would exceed its limit; in that
        case no counter is modified. Backed by `atomic_check_and_increment_by_n`
        which uses a Redis Lua script when available (multi-process atomic) and
        falls back to a per-process asyncio.Lock + in-memory operation.
        """
        descriptors = self.parallel_request_limiter._create_rate_limit_descriptors(
            user_api_key_dict=user_api_key_dict,
            data=data,
            rpm_limit_type=None,
            tpm_limit_type=None,
            model_has_failures=False,
        )

        increment: Dict[Literal["requests", "tokens"], int] = {
            "requests": batch_usage.request_count,
            "tokens": batch_usage.total_tokens,
        }
        increments: List[Dict[Literal["requests", "tokens"], int]] = [
            increment for _ in descriptors
        ]

        rate_limit_response = (
            await self.parallel_request_limiter.atomic_check_and_increment_by_n(
                descriptors=descriptors,
                increments=increments,
                parent_otel_span=user_api_key_dict.parent_otel_span,
            )
        )

        if rate_limit_response["overall_code"] == "OVER_LIMIT":
            for status in rate_limit_response["statuses"]:
                if status["code"] == "OVER_LIMIT":
                    self._raise_rate_limit_error(
                        status,
                        descriptors,
                        batch_usage,
                        status["rate_limit_type"],
                    )

    async def count_input_file_usage(
        self,
        file_id: str,
        custom_llm_provider: Literal["openai", "azure", "vertex_ai"] = "openai",
        user_api_key_dict: Optional[UserAPIKeyAuth] = None,
    ) -> BatchFileUsage:
        """
        Count number of requests and tokens in a batch input file.

        Args:
            file_id: The file ID to read
            custom_llm_provider: The custom LLM provider to use for token encoding
            user_api_key_dict: User authentication information for file access (required for managed files)

        Returns:
            BatchFileUsage with total_tokens and request_count
        """
        try:
            # Check if this is a managed file (base64 encoded unified file ID)
            from litellm.proxy.openai_files_endpoints.common_utils import (
                _is_base64_encoded_unified_file_id,
            )

            # Managed files require bypassing the HTTP endpoint (which runs access-check hooks)
            # and calling the managed files hook directly with the user's credentials.
            is_managed_file = _is_base64_encoded_unified_file_id(file_id)
            if is_managed_file and user_api_key_dict is not None:
                file_content = await self._fetch_managed_file_content(
                    file_id=file_id,
                    user_api_key_dict=user_api_key_dict,
                )
            else:
                # For non-managed files, use the standard litellm.afile_content
                file_content = await litellm.afile_content(
                    file_id=file_id,
                    custom_llm_provider=custom_llm_provider,
                    user_api_key_dict=user_api_key_dict,
                )

            file_content_as_dict = _get_file_content_as_dictionary(file_content.content)

            # Validate every model named in the batch JSONL against the
            # caller's per-key model allowlist. Without this, a caller
            # could smuggle restricted/expensive models inside the file
            # and the upstream provider would execute the batch under
            # the proxy's shared API key.
            if user_api_key_dict is not None:
                await self._enforce_batch_file_model_access(
                    user_api_key_dict=user_api_key_dict,
                    file_content_as_dict=file_content_as_dict,
                )

            input_file_usage = _get_batch_job_input_file_usage(
                file_content_dictionary=file_content_as_dict,
                custom_llm_provider=custom_llm_provider,
            )
            request_count = len(file_content_as_dict)
            return BatchFileUsage(
                total_tokens=input_file_usage.total_tokens,
                request_count=request_count,
            )

        except HTTPException as e:
            # Distinguish intentional 403s from `_enforce_batch_file_model_access`
            # from genuine I/O failures so security-relevant rejections show up
            # in the access log instead of getting buried in error noise.
            if e.status_code == 403:
                verbose_proxy_logger.warning(
                    f"Batch rejected: caller not authorized for a model named in {file_id}: {e.detail}"
                )
            else:
                verbose_proxy_logger.error(
                    f"Batch input file rejected for {file_id}: status={e.status_code} detail={e.detail}"
                )
            raise
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error counting input file usage for {file_id}: {str(e)}"
            )
            raise

    async def _enforce_batch_file_model_access(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        file_content_as_dict: List[dict],
    ) -> None:
        """Reject the batch if the caller is not authorized for every
        ``body.model`` named inside the JSONL.

        Reuses ``can_key_call_model`` so the same allowlist semantics
        (wildcards, access groups, ``all-proxy-models``, team aliases)
        the proxy enforces on `/chat/completions` apply here.
        """
        from litellm.proxy.auth.auth_checks import can_key_call_model
        from litellm.proxy.proxy_server import llm_router

        models = _get_models_from_batch_input_file_content(file_content_as_dict)
        if not models:
            return

        llm_model_list = llm_router.model_list if llm_router is not None else None
        for model in models:
            try:
                await can_key_call_model(
                    model=model,
                    llm_model_list=llm_model_list,
                    valid_token=user_api_key_dict,
                    llm_router=llm_router,
                )
            except HTTPException:
                raise
            except Exception as e:
                # `can_key_call_model` raises ProxyException on denial;
                # re-shape to a 403 so the batch endpoint returns a
                # consistent rejection without leaking internal types.
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": (
                            "Batch input file references a model the caller is "
                            f"not authorized to use: model={model}, reason={str(e)}"
                        )
                    },
                )

    async def _fetch_managed_file_content(
        self,
        file_id: str,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> Any:
        """
        Fetch file content from managed files hook.

        This is needed for managed files because they require proper user context
        to verify file ownership and access permissions.

        Args:
            file_id: The managed file ID (base64 encoded)
            user_api_key_dict: User authentication information

        Returns:
            HttpxBinaryResponseContent with the file content
        """
        from litellm.llms.base_llm.files.transformation import BaseFileEndpoints

        # Import proxy_server dependencies at runtime to avoid circular imports
        try:
            from litellm.proxy.proxy_server import llm_router, proxy_logging_obj
        except ImportError as e:
            raise ValueError(
                f"Cannot import proxy_server dependencies: {str(e)}. "
                "Managed files require proxy_server to be initialized."
            )

        # Get the managed files hook
        if proxy_logging_obj is None:
            raise ValueError(
                "proxy_logging_obj not available. Cannot access managed files hook."
            )

        managed_files_obj = proxy_logging_obj.get_proxy_hook("managed_files")
        if managed_files_obj is None:
            raise ValueError(
                "Managed files hook not found. Cannot access managed file."
            )

        if not isinstance(managed_files_obj, BaseFileEndpoints):
            raise ValueError("Managed files hook is not a BaseFileEndpoints instance.")

        if llm_router is None:
            raise ValueError("llm_router not available. Cannot access managed files.")

        # Use the managed files hook to get file content
        # This properly handles user permissions and file ownership
        file_content = await managed_files_obj.afile_content(
            file_id=file_id,
            litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
            llm_router=llm_router,
        )

        return file_content

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: Any,
        data: Dict,
        call_type: str,
    ) -> Union[Exception, str, Dict, None]:
        """
        Pre-call hook for batch operations.

        Only handles batch creation (acreate_batch):
        - Reads input file
        - Counts tokens and requests
        - Reserves rate limit capacity via parallel_request_limiter

        Args:
            user_api_key_dict: User authentication information
            cache: Cache instance (not used directly)
            data: Request data
            call_type: Type of call being made

        Returns:
            Modified data dict or None

        Raises:
            HTTPException: 429 if rate limit would be exceeded
        """
        # Only handle batch creation
        if call_type != "acreate_batch":
            verbose_proxy_logger.debug(
                f"Batch rate limiter: Not handling batch creation rate limiting for call type: {call_type}"
            )
            return data

        verbose_proxy_logger.debug(
            "Batch rate limiter: Handling batch creation rate limiting"
        )

        try:
            # Extract input_file_id from data
            input_file_id = data.get("input_file_id")
            if not input_file_id:
                verbose_proxy_logger.debug(
                    "No input_file_id in batch request, skipping rate limiting"
                )
                return data

            # Get custom_llm_provider for token counting
            custom_llm_provider = data.get("custom_llm_provider", "openai")

            # Count tokens and requests from input file
            verbose_proxy_logger.debug(
                f"Counting tokens from batch input file: {input_file_id}"
            )
            batch_usage = await self.count_input_file_usage(
                file_id=input_file_id,
                custom_llm_provider=custom_llm_provider,
                user_api_key_dict=user_api_key_dict,
            )

            verbose_proxy_logger.debug(
                f"Batch input file usage - Tokens: {batch_usage.total_tokens}, "
                f"Requests: {batch_usage.request_count}"
            )

            # Store batch usage in data for later reference
            data["_batch_token_count"] = batch_usage.total_tokens
            data["_batch_request_count"] = batch_usage.request_count

            # Directly increment counters by batch amounts (check happens atomically)
            # This will raise HTTPException if limits are exceeded
            await self._check_and_increment_batch_counters(
                user_api_key_dict=user_api_key_dict,
                data=data,
                batch_usage=batch_usage,
            )

            verbose_proxy_logger.debug(
                "Batch rate limit check passed, counters incremented"
            )
            return data

        except HTTPException:
            # Re-raise HTTP exceptions (rate limit exceeded)
            raise
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error in batch rate limiting: {str(e)}", exc_info=True
            )
            # Don't block the request if rate limiting fails
            return data

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

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    NoReturn,
    Optional,
    Tuple,
    Union,
)

from fastapi import HTTPException
from pydantic import BaseModel

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.batches.batch_utils import (
    _extract_file_access_credentials,
    _get_batch_job_input_file_usage,
    _get_file_content_as_dictionary,
    _get_models_from_batch_input_file_content,
)
from litellm.exceptions import RateLimitErrorCategory
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import (
    ProxyErrorTypes,
    ProxyException,
    SpecialModelNames,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.proxy_rate_limit_error import (
    ProxyRateLimitError,
    map_v3_rate_limit_type,
)
from litellm.proxy.hooks.rate_limiter_utils import resolve_llm_provider_for_rate_limit

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
        self._warned_unsupported_model_skip = False

    def _get_file_bound_batch_model(self, data: Dict) -> Optional[str]:
        """Resolve the model bound to the batch input file ID.

        ``create_batch`` routes a file-bound id (model-embedded ``file-...`` or
        unified managed file) on that bound model and ignores the top-level
        ``model``, so this is the authoritative routing model whenever the file
        binds one. The provider is then read from that deployment's trusted
        credentials for the provider-level skip decision.
        """
        input_file_id = data.get("input_file_id")
        if not isinstance(input_file_id, str) or not input_file_id:
            return None

        from litellm.proxy.openai_files_endpoints.common_utils import (
            _is_base64_encoded_unified_file_id,
            decode_model_from_file_id,
            get_models_from_unified_file_id,
        )

        model_from_file_id = decode_model_from_file_id(input_file_id)
        if model_from_file_id:
            return model_from_file_id

        unified_file_id = _is_base64_encoded_unified_file_id(input_file_id)
        if unified_file_id:
            target_model_names = get_models_from_unified_file_id(unified_file_id)
            if target_model_names:
                return target_model_names[0]

        return None

    def _get_batch_routing_model(self, data: Dict) -> Optional[str]:
        """Resolve the deployment/model used for this batch from request data.

        Mirrors ``create_batch`` routing precedence: a model bound to the input
        file id wins over the top-level ``model``, because the batch endpoint
        ignores the top-level model for file-bound ids. Resolving the provider
        skip from the top-level model first would let a caller point ``model``
        at a skip-listed provider while the file routes a rate-limited one.
        """
        file_bound_model = self._get_file_bound_batch_model(data)
        if file_bound_model:
            return file_bound_model

        model = data.get("model")
        if isinstance(model, str) and model:
            return model

        return None

    def _resolve_batch_provider(self, batch_model: Optional[str]) -> Optional[str]:
        """Resolve the provider from the deployment that serves ``batch_model``.

        The provider is read from trusted router credentials rather than the
        user-supplied ``custom_llm_provider`` request field, so a caller cannot
        spoof a skip-listed provider to bypass batch rate limiting.
        """
        if not batch_model:
            return None

        from litellm.proxy.openai_files_endpoints.common_utils import (
            get_credentials_for_model,
        )
        from litellm.proxy.proxy_server import llm_router

        if llm_router is None:
            return None

        try:
            credentials = get_credentials_for_model(
                llm_router=llm_router,
                model_id=batch_model,
                operation_context="batch input file read (rate limiting)",
            )
        except HTTPException:
            return None

        provider = credentials.get("custom_llm_provider")
        return provider if isinstance(provider, str) and provider else None

    def _create_batch_rate_limit_descriptors(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: Dict,
    ) -> List["RateLimitDescriptor"]:
        return self.parallel_request_limiter._create_rate_limit_descriptors(
            user_api_key_dict=user_api_key_dict,
            data=data,
            rpm_limit_type=None,
            tpm_limit_type=None,
            model_has_failures=False,
        )

    def _should_skip_batch_input_file_processing(
        self,
        data: Dict,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> Tuple[bool, Optional[List["RateLimitDescriptor"]]]:
        """
        Skip downloading batch input files when the operator disabled batch
        input-file rate limiting, when the batch runs entirely on a skip-listed
        provider, or when there is nothing to enforce (no applicable rate
        limits).

        A skip is only honored for keys with unrestricted model access. When
        the key has a model allowlist, the JSONL must still be downloaded so
        ``_enforce_batch_file_model_access`` can validate every ``body.model``
        entry, otherwise a restricted key could smuggle unauthorized models
        into the file via an admin-configured skip.

        The skip is never keyed on a specific model name. The models a batch
        actually runs are its JSONL ``body.model`` entries, and any model
        identifier the caller can influence (the top-level ``model`` or the
        unsigned model embedded in a ``file-...`` id) can be pointed at a
        skip-listed deployment while the file routes a different, rate-limited
        model. The provider skip is safe because the provider is read from the
        routing deployment's trusted credentials and the batch is constrained
        to run on that provider.

        Returns ``(should_skip, descriptors)`` where ``descriptors`` is the
        rate-limit descriptor list computed for the no-limits check, so the
        caller can reuse it for counter enforcement without recomputing.
        """
        from litellm.proxy.proxy_server import general_settings

        self._warn_if_unsupported_model_skip_configured(general_settings)

        if self._key_requires_batch_model_access_check(user_api_key_dict):
            return False, None

        if general_settings.get("disable_batch_input_file_rate_limiting") is True:
            return True, None

        skip_providers = (
            general_settings.get("skip_batch_input_file_rate_limiting_for_providers")
            or []
        )
        if skip_providers:
            batch_provider = self._resolve_batch_provider(
                self._get_batch_routing_model(data)
            )
            if batch_provider and batch_provider in skip_providers:
                verbose_proxy_logger.debug(
                    f"Skipping batch input file processing for provider={batch_provider}"
                )
                return True, None

        descriptors = self._create_batch_rate_limit_descriptors(
            user_api_key_dict=user_api_key_dict,
            data=data,
        )
        if not self._has_applicable_batch_rate_limits(descriptors):
            verbose_proxy_logger.debug(
                "Skipping batch input file processing: no rate limits configured"
            )
            return True, None

        return False, descriptors

    def _warn_if_unsupported_model_skip_configured(
        self, general_settings: Dict
    ) -> None:
        """Warn once that ``skip_batch_input_file_rate_limiting_for_models`` is a no-op.

        A per-model skip is intentionally not honored because the model a batch
        runs on is caller-influenced and can be pointed at a skip-listed
        deployment while the JSONL routes a different, rate-limited model.
        """
        if self._warned_unsupported_model_skip:
            return
        if general_settings.get("skip_batch_input_file_rate_limiting_for_models"):
            self._warned_unsupported_model_skip = True
            verbose_proxy_logger.warning(
                "general_settings.skip_batch_input_file_rate_limiting_for_models is not "
                "supported and has no effect. Use "
                "skip_batch_input_file_rate_limiting_for_providers or "
                "disable_batch_input_file_rate_limiting instead."
            )

    @staticmethod
    def _key_requires_batch_model_access_check(
        user_api_key_dict: UserAPIKeyAuth,
    ) -> bool:
        """True when the key may only call a subset of models (JSONL must be checked)."""
        models = user_api_key_dict.models or []
        if "*" in models:
            return False
        if SpecialModelNames.all_proxy_models.value in models:
            return False
        if user_api_key_dict.access_group_ids:
            return True
        if not models:
            return False
        return True

    @staticmethod
    def _has_applicable_batch_rate_limits(
        descriptors: List["RateLimitDescriptor"],
    ) -> bool:
        for descriptor in descriptors:
            rate_limit = descriptor.get("rate_limit") or {}
            if (
                rate_limit.get("requests_per_unit") is not None
                or rate_limit.get("tokens_per_unit") is not None
                or rate_limit.get("max_parallel_requests") is not None
            ):
                return True
        return False

    def _resolve_batch_input_file_fetch_params(
        self,
        file_id: str,
        custom_llm_provider: str,
        data: Dict,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Map proxy-facing file IDs to provider file IDs and credentials.

        Model-embedded IDs (``file-<base64>``) are not unified managed-file IDs;
        without decoding them, ``afile_content`` is called with the encoded ID
        and the upstream provider returns 404.
        """
        from litellm.proxy.openai_files_endpoints.common_utils import (
            decode_model_from_file_id,
            get_credentials_for_model,
            get_original_file_id,
        )
        from litellm.proxy.proxy_server import llm_router

        fetch_kwargs: Dict[str, Any] = {
            "custom_llm_provider": custom_llm_provider,
        }

        model_from_file_id = decode_model_from_file_id(file_id)
        if model_from_file_id:
            if llm_router is not None:
                try:
                    credentials = get_credentials_for_model(
                        llm_router=llm_router,
                        model_id=model_from_file_id,
                        operation_context="batch input file read (rate limiting)",
                    )
                    fetch_kwargs.update(_extract_file_access_credentials(credentials))
                    fetch_kwargs["model"] = model_from_file_id
                    provider = credentials.get("custom_llm_provider")
                    if provider:
                        fetch_kwargs["custom_llm_provider"] = provider
                except HTTPException:
                    pass
            return get_original_file_id(file_id), fetch_kwargs

        request_model = data.get("model")
        if isinstance(request_model, str) and request_model and llm_router is not None:
            try:
                credentials = get_credentials_for_model(
                    llm_router=llm_router,
                    model_id=request_model,
                    operation_context="batch input file read (rate limiting)",
                )
                fetch_kwargs.update(_extract_file_access_credentials(credentials))
                fetch_kwargs["model"] = request_model
                provider = credentials.get("custom_llm_provider")
                if provider:
                    fetch_kwargs["custom_llm_provider"] = provider
            except HTTPException:
                pass

        return file_id, fetch_kwargs

    def _raise_rate_limit_error(
        self,
        status: "RateLimitStatus",
        descriptors: List["RateLimitDescriptor"],
        batch_usage: BatchFileUsage,
        limit_type: str,
        requested_model: Optional[str] = None,
    ) -> NoReturn:
        """Raise :class:`ProxyRateLimitError` (a 429) for batch rate limit exceeded."""
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

        resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(
            requested_model
        )
        raise ProxyRateLimitError(
            detail=detail,
            headers={
                "retry-after": str(window_size),
                "rate_limit_type": limit_type,
                "reset_at": reset_time_formatted,
            },
            category=RateLimitErrorCategory.LITELLM_BATCH_RATE_LIMIT,
            rate_limit_type=map_v3_rate_limit_type(limit_type),
            model=resolved_model,
            llm_provider=llm_provider,
        )

    async def _check_and_increment_batch_counters(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: Dict,
        batch_usage: BatchFileUsage,
        descriptors: Optional[List["RateLimitDescriptor"]] = None,
    ) -> None:
        """
        Atomically check + increment rate-limit counters by the batch amounts.

        Raises HTTPException if any descriptor would exceed its limit; in that
        case no counter is modified. Backed by `atomic_check_and_increment_by_n`
        which uses a Redis Lua script when available (multi-process atomic) and
        falls back to a per-process asyncio.Lock + in-memory operation.

        ``descriptors`` may be passed in by the pre-call hook to reuse the list
        already computed when deciding whether to skip file processing.
        """
        if descriptors is None:
            descriptors = self._create_batch_rate_limit_descriptors(
                user_api_key_dict=user_api_key_dict,
                data=data,
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
            requested_model = data.get("model") if data else None
            for status in rate_limit_response["statuses"]:
                if status["code"] == "OVER_LIMIT":
                    self._raise_rate_limit_error(
                        status,
                        descriptors,
                        batch_usage,
                        status["rate_limit_type"],
                        requested_model=requested_model,
                    )

    async def count_input_file_usage(
        self,
        file_id: str,
        custom_llm_provider: Literal["openai", "azure", "vertex_ai"] = "openai",
        user_api_key_dict: Optional[UserAPIKeyAuth] = None,
        data: Optional[Dict] = None,
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
                get_models_from_unified_file_id,
            )

            # Managed files require bypassing the HTTP endpoint (which runs access-check hooks)
            # and calling the managed files hook directly with the user's credentials.
            is_managed_file = _is_base64_encoded_unified_file_id(file_id)
            target_model_names = (
                get_models_from_unified_file_id(is_managed_file)
                if is_managed_file
                else []
            )
            if is_managed_file and user_api_key_dict is not None:
                file_content = await self._fetch_managed_file_content(
                    file_id=file_id,
                    user_api_key_dict=user_api_key_dict,
                )
            else:
                provider_file_id, fetch_kwargs = (
                    self._resolve_batch_input_file_fetch_params(
                        file_id=file_id,
                        custom_llm_provider=custom_llm_provider,
                        data=data or {},
                    )
                )
                # For non-managed files, use the standard litellm.afile_content
                file_content = await litellm.afile_content(
                    file_id=provider_file_id,
                    user_api_key_dict=user_api_key_dict,
                    **fetch_kwargs,
                )

            file_content_bytes = getattr(file_content, "content", None)
            if not isinstance(file_content_bytes, bytes):
                raise ValueError(
                    f"Expected bytes content from file retrieval for {file_id}, "
                    f"got {type(file_content_bytes)}"
                )
            file_content_as_dict = _get_file_content_as_dictionary(file_content_bytes)

            # Validate every model named in the batch JSONL against the
            # caller's per-key model allowlist. Without this, a caller
            # could smuggle restricted/expensive models inside the file
            # and the upstream provider would execute the batch under
            # the proxy's shared API key.
            if user_api_key_dict is not None:
                await self._enforce_batch_file_model_access(
                    user_api_key_dict=user_api_key_dict,
                    file_content_as_dict=file_content_as_dict,
                    target_model_names=target_model_names or None,
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
        target_model_names: Optional[List[str]] = None,
    ) -> None:
        """Reject the batch if the caller is not authorized for the upload target.

        For managed files, ``target_model_names`` (from the unified file id) is
        the proxy alias the file was uploaded for and is used directly for auth.
        For legacy/non-managed files, falls back to ``body.model`` values in the JSONL.

        Reuses standard auth helpers so the same model access rules the proxy
        enforces on `/chat/completions` apply here.
        """
        from litellm.proxy.auth.auth_checks import (
            _check_team_member_model_access,
            _key_access_group_grants_model,
            can_key_call_model,
            can_team_access_model,
            get_team_object,
        )
        from litellm.proxy.proxy_server import llm_router
        from litellm.proxy.proxy_server import prisma_client
        from litellm.proxy.proxy_server import proxy_logging_obj
        from litellm.proxy.proxy_server import user_api_key_cache

        if target_model_names:
            models = target_model_names
        else:
            models = _get_models_from_batch_input_file_content(file_content_as_dict)
            if not models:
                return

        team_object = None
        if (
            SpecialModelNames.all_team_models.value in (user_api_key_dict.models or [])
            and user_api_key_dict.team_id is not None
            and prisma_client is not None
        ):
            try:
                team_object = await get_team_object(
                    team_id=user_api_key_dict.team_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    parent_otel_span=user_api_key_dict.parent_otel_span,
                    proxy_logging_obj=proxy_logging_obj,
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": (
                            "Batch input file model access could not be "
                            "validated against the current team."
                        )
                    },
                ) from e

        llm_model_list = llm_router.model_list if llm_router is not None else None
        for model in models:
            model_to_check = model
            try:
                if team_object is not None:
                    try:
                        await can_team_access_model(
                            model=model_to_check,
                            team_object=team_object,
                            llm_router=llm_router,
                            team_model_aliases=user_api_key_dict.team_model_aliases,
                        )
                    except ProxyException as team_denial:
                        if team_denial.type != ProxyErrorTypes.team_model_access_denied:
                            raise
                        if not await _key_access_group_grants_model(
                            model=model_to_check,
                            valid_token=user_api_key_dict,
                            team_object=team_object,
                            llm_router=llm_router,
                        ):
                            raise
                    await _check_team_member_model_access(
                        model=model_to_check,
                        team_object=team_object,
                        valid_token=user_api_key_dict,
                        llm_router=llm_router,
                        prisma_client=prisma_client,
                        user_api_key_cache=user_api_key_cache,
                        proxy_logging_obj=proxy_logging_obj,
                    )
                else:
                    await can_key_call_model(
                        model=model_to_check,
                        llm_model_list=llm_model_list,
                        valid_token=user_api_key_dict,
                        llm_router=llm_router,
                    )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": (
                            "Batch input file references a model the caller is "
                            f"not authorized to use: model={model_to_check}, reason={str(e)}"
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

            should_skip, batch_rate_limit_descriptors = (
                self._should_skip_batch_input_file_processing(
                    data=data, user_api_key_dict=user_api_key_dict
                )
            )
            if should_skip:
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
                data=data,
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
                descriptors=batch_rate_limit_descriptors,
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

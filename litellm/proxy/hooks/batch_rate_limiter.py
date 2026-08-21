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

import json
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, NoReturn, TypeAlias

from fastapi import HTTPException
from pydantic import BaseModel, Field, TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.batches.batch_utils import (
    _count_entry_tokens,
    _estimate_batch_entry_tokens,
    _extract_file_access_credentials,
    _iter_batch_input_lines,
)
from litellm.exceptions import RateLimitErrorCategory
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import (
    ProxyErrorTypes,
    ProxyException,
    SpecialModelNames,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_utils import get_model_rate_limit_from_metadata
from litellm.proxy.common_utils.proxy_rate_limit_error import (
    ProxyRateLimitError,
    map_v3_rate_limit_type,
)
from litellm.proxy.hooks.batch_enqueued_tokens import (
    BatchEnqueuedTokenOverLimit,
    BatchEnqueuedTokenReservation,
    BatchEnqueuedTokenScope,
    resolve_batch_enqueued_token_scopes,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PROJECT_ITPM_DESCRIPTOR_KEY,
    PROJECT_OTPM_DESCRIPTOR_KEY,
    get_or_create_request_stash,
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

    Span = _Span | Any
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
    RateLimitStatus = dict[str, Any]
    RateLimitDescriptor = dict[str, Any]


_BATCH_BODY_ADAPTER: Final = TypeAdapter(dict[str, object])

IncrementAmounts: TypeAlias = dict[Literal["requests", "tokens"], int]


class BatchFileUsage(BaseModel):
    """
    Internal model for batch file usage tracking, used for batch rate limiting
    """

    total_tokens: int
    request_count: int
    output_tokens: int = 0
    # Keyed by each row's own `body.model`, distinct from `total_tokens`/
    # `output_tokens` (the whole-file totals charged to the file-bound/
    # top-level routing model's key/team/model limits). A batch's rows can
    # each target a different model, so the project's per-model ITPM/OTPM
    # quota for a row's actual model must be charged with that row's own
    # tokens -- see `_create_project_io_descriptors_for_models`.
    per_model_usage: dict[str, dict[str, int]] = Field(
        default_factory=dict
    )  # mutable-ok: accumulated incrementally per row while parsing the batch file


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

    def _get_file_bound_batch_model(self, data: dict) -> str | None:
        """Resolve the model bound to the batch input file ID.

        ``create_batch`` routes a file-bound id (model-embedded ``file-...`` or
        unified managed file) on that bound model and ignores the top-level
        ``model``, so this is the authoritative routing model whenever the file
        binds one. The provider is then read from that deployment's trusted
        credentials for the provider-level skip decision.
        """
        input_file_id: Final = data.get("input_file_id")
        if not isinstance(input_file_id, str) or not input_file_id:
            return None

        from litellm.proxy.openai_files_endpoints.common_utils import (
            _is_base64_encoded_unified_file_id,
            decode_model_from_file_id,
            get_models_from_unified_file_id,
        )

        model_from_file_id: Final = decode_model_from_file_id(input_file_id)
        if model_from_file_id:
            return model_from_file_id

        unified_file_id: Final = _is_base64_encoded_unified_file_id(input_file_id)
        if unified_file_id:
            target_model_names: Final = get_models_from_unified_file_id(unified_file_id)
            if target_model_names:
                return target_model_names[0]

        return None

    def _get_batch_routing_model(self, data: dict) -> str | None:
        """Resolve the deployment/model used for this batch from request data.

        Mirrors ``create_batch`` routing precedence: a model bound to the input
        file id wins over the top-level ``model``, because the batch endpoint
        ignores the top-level model for file-bound ids. Resolving the provider
        skip from the top-level model first would let a caller point ``model``
        at a skip-listed provider while the file routes a rate-limited one.
        """
        file_bound_model: Final = self._get_file_bound_batch_model(data)
        if file_bound_model:
            return file_bound_model

        model: Final = data.get("model")
        if isinstance(model, str) and model:
            return model

        return None

    def _resolve_batch_provider(self, batch_model: str | None) -> str | None:
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
            credentials: Final = get_credentials_for_model(
                llm_router=llm_router,
                model_id=batch_model,
                operation_context="batch input file read (rate limiting)",
            )
        except HTTPException:
            return None

        provider: Final = credentials.get("custom_llm_provider")
        return provider if isinstance(provider, str) and provider else None

    def _create_batch_rate_limit_descriptors(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: dict,
    ) -> list["RateLimitDescriptor"]:
        """Build the standard key/user/team/model descriptor list a batch is charged against.

        Deliberately excludes the project-scoped ITPM/OTPM descriptors: those
        are charged per the JSONL row's own `body.model` once the file is
        parsed (`_create_project_io_descriptors_for_models`), not the
        file-bound/top-level routing model this function resolves. Charging
        project quotas here would let a caller bind the file to a model
        without a quota while rows execute against a quota-limited model.
        """
        return self.parallel_request_limiter._create_rate_limit_descriptors(
            user_api_key_dict=user_api_key_dict,
            data=data,
            rpm_limit_type=None,
            tpm_limit_type=None,
            model_has_failures=False,
        )

    @staticmethod
    def _project_has_any_io_token_limits(user_api_key_dict: UserAPIKeyAuth) -> bool:
        """True when the project has any per-model ITPM/OTPM quota configured.

        Used to stop the "skip batch input file processing" fast path from
        bypassing a project quota configured for a model other than the
        batch's file-bound/top-level routing model: the row models that
        actually drive execution and billing aren't known until the JSONL
        is parsed, so the file must be read whenever *any* model could be
        quota-limited, not only when the routing model itself is.
        """
        if user_api_key_dict.project_id is None:
            return False
        return bool(
            get_model_rate_limit_from_metadata(user_api_key_dict, "project_metadata", "model_itpm_limit")
        ) or bool(get_model_rate_limit_from_metadata(user_api_key_dict, "project_metadata", "model_otpm_limit"))

    def _create_project_io_descriptors_for_models(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        per_model_usage: Mapping[str, Mapping[str, int]],
    ) -> tuple[list["RateLimitDescriptor"], list[IncrementAmounts]]:  # mutable-ok: see below
        """Build project ITPM/OTPM descriptors charged against each row's own model.

        One descriptor pair per distinct `body.model` found in the JSONL,
        each incremented only by that model's own counted usage -- never the
        whole-batch total -- so a quota-limited model can't hide behind an
        unlimited routing model, and an unrelated model's rows can't inflate
        a different model's counter.
        """
        extra_descriptors: Final[list[RateLimitDescriptor]] = []  # mutable-ok: see above
        extra_increments: Final[list[IncrementAmounts]] = []  # mutable-ok: see above
        for model, usage in per_model_usage.items():
            model_descriptors: list[RateLimitDescriptor] = []  # mutable-ok: reset per loop iteration, not module state
            self.parallel_request_limiter.add_project_io_token_rate_limit_descriptors_from_metadata(
                user_api_key_dict=user_api_key_dict,
                requested_model=model,
                descriptors=model_descriptors,
            )
            for descriptor in model_descriptors:
                extra_descriptors.append(descriptor)
                extra_increments.append(
                    {  # mutable-ok: atomic limiter API requires mutable increment records
                        "requests": 0,
                        "tokens": usage.get("output_tokens", 0)
                        if descriptor["key"] == PROJECT_OTPM_DESCRIPTOR_KEY
                        else usage.get("total_tokens", 0),
                    }
                )
        return extra_descriptors, extra_increments

    def _should_skip_batch_input_file_processing(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        has_enqueued_scopes: bool = False,
    ) -> tuple[bool, list["RateLimitDescriptor"] | None]:
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

        The no-limits check also treats any project-configured ITPM/OTPM
        quota as an applicable limit, even when it isn't scoped to the
        routing model: a row can target a different, quota-limited model,
        and that isn't knowable without parsing the JSONL.

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

        skip_providers: Final = general_settings.get("skip_batch_input_file_rate_limiting_for_providers") or []
        if skip_providers:
            batch_provider: Final = self._resolve_batch_provider(self._get_batch_routing_model(data))
            if batch_provider and batch_provider in skip_providers:
                verbose_proxy_logger.debug("Skipping batch input file processing for provider=%s", batch_provider)
                return True, None

        descriptors: Final = self._create_batch_rate_limit_descriptors(
            user_api_key_dict=user_api_key_dict,
            data=data,
        )
        if (
            not has_enqueued_scopes
            and not self._has_applicable_batch_rate_limits(descriptors)
            and not self._project_has_any_io_token_limits(user_api_key_dict)
        ):
            verbose_proxy_logger.debug("Skipping batch input file processing: no rate limits configured")
            return True, None

        return False, descriptors

    def _warn_if_unsupported_model_skip_configured(self, general_settings: dict) -> None:
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
        models: Final = user_api_key_dict.models or []
        if "*" in models:
            return False
        if SpecialModelNames.all_proxy_models.value in models:
            return False
        if user_api_key_dict.access_group_ids:
            return True
        if not models:
            return False
        return True

    def _estimate_entry_output_tokens(
        self,
        entry: Mapping[str, object],
        min_configured_otpm_limit: int | None,
    ) -> int:
        """Conservative per-row output-token estimate for the project OTPM reservation.

        Batch completion never reconciles actual usage back into the rate
        limiter, so this pre-call estimate is the only OTPM enforcement a
        batch gets. Mirrors the real-time no-``max_tokens`` floor so a row
        that omits an output cap can't be used to bypass OTPM the way an
        unbounded streaming request could.

        Embeddings rows are identified by the row's own ``url`` (the OpenAI
        batch schema puts the target route there, e.g. ``/v1/embeddings``),
        never by body shape: a `/v1/responses` row also carries `body.input`
        with no `messages`/`prompt`, so guessing from body shape alone would
        misclassify a token-generating Responses row as a zero-output
        embeddings row and let it skip the OTPM reservation entirely.
        """
        url: Final = entry.get("url")
        if isinstance(url, str) and "embeddings" in url:
            return 0  # embeddings: no output tokens
        raw_body: Final = entry.get("body")
        body: Final[Mapping[str, object]] = (
            MappingProxyType(_BATCH_BODY_ADAPTER.validate_python(raw_body))
            if isinstance(raw_body, Mapping)
            else MappingProxyType({})  # mutable-ok: immediately frozen empty fallback
        )
        # `max_tokens`/`max_completion_tokens` cap chat completions; `/v1/responses`
        # rows cap output with `max_output_tokens` instead -- omitting it here
        # would fall through to the floor estimate for every capped Responses row.
        explicit_cap: Final = next(
            (
                v
                for v in (
                    body.get("max_tokens"),
                    body.get("max_completion_tokens"),
                    body.get("max_output_tokens"),
                )
                if v is not None
            ),
            None,
        )
        candidate_count: Final = self.parallel_request_limiter.get_output_candidate_count(body)
        if explicit_cap is not None:
            try:
                return max(0, int(explicit_cap)) * candidate_count
            except (TypeError, ValueError, OverflowError):
                pass
        return self.parallel_request_limiter.no_max_tokens_output_floor(min_configured_otpm_limit) * candidate_count

    @staticmethod
    def _has_applicable_batch_rate_limits(
        descriptors: list["RateLimitDescriptor"],
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
        data: dict,
    ) -> tuple[str, dict[str, Any]]:
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

        fetch_kwargs: Final[dict[str, Any]] = {
            "custom_llm_provider": custom_llm_provider,
        }

        model_from_file_id: Final = decode_model_from_file_id(file_id)
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

        request_model: Final = data.get("model")
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

    async def _reserve_batch_enqueued_tokens(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: Mapping[str, object],
        batch_usage: BatchFileUsage,
        scopes: tuple[BatchEnqueuedTokenScope, ...],
    ) -> None:
        """Reserve the batch's estimated tokens against the caller's enqueued-token allowance.

        Runs instead of the per-minute counter charge when the key or team
        opted in via ``batch_enqueued_token_limit`` metadata. The reservation
        is stashed on the request so the v3 limiter's post-call hooks can
        persist it (keyed by the provider batch id) and refund it when the
        batch reaches a terminal state.
        """
        outcome: Final = await self.parallel_request_limiter.batch_enqueued_token_store.reserve(
            tokens=batch_usage.total_tokens,
            scopes=scopes,
            litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
        )
        match outcome:
            case BatchEnqueuedTokenOverLimit():
                self._raise_enqueued_limit_error(over_limit=outcome, data=data, batch_usage=batch_usage)
            case BatchEnqueuedTokenReservation():
                get_or_create_request_stash().batch_enqueued_reservation = outcome

    def _raise_enqueued_limit_error(
        self,
        over_limit: BatchEnqueuedTokenOverLimit,
        data: Mapping[str, object],
        batch_usage: BatchFileUsage,
    ) -> NoReturn:
        scope: Final = over_limit.scope
        remaining: Final = max(0, scope.limit - over_limit.enqueued)
        detail: Final = (
            f"Batch enqueued token limit exceeded for {scope.key}: {scope.value}. "
            f"Batch requires {batch_usage.total_tokens} tokens but only {remaining} enqueued tokens remaining "
            f"out of {scope.limit} enqueued token limit. "
            f"Tokens free up as running batches complete or are cancelled."
        )
        raw_model: Final = data.get("model")
        resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(
            raw_model if isinstance(raw_model, str) else None
        )
        raise ProxyRateLimitError(
            detail=detail,
            headers=MappingProxyType({"rate_limit_type": "tokens"}),
            category=RateLimitErrorCategory.LITELLM_BATCH_RATE_LIMIT,
            rate_limit_type=map_v3_rate_limit_type("tokens"),
            model=resolved_model,
            llm_provider=llm_provider,
        )

    def _raise_rate_limit_error(
        self,
        status: "RateLimitStatus",
        descriptors: list["RateLimitDescriptor"],
        batch_usage: BatchFileUsage,
        limit_type: str,
        requested_model: str | None = None,
    ) -> NoReturn:
        """Raise :class:`ProxyRateLimitError` (a 429) for batch rate limit exceeded."""
        from datetime import datetime

        # Find the descriptor for this status. Matching on (key, value) is
        # required, not key alone: a batch can carry several project ITPM/OTPM
        # descriptors sharing one key (e.g. `model_per_project_otpm`) but
        # scoped to different models via `value`
        # ("{project_id}:{model}") -- key-only matching would always resolve
        # to the first same-keyed descriptor regardless of which one was
        # actually over its limit. Falls back to key-only matching for
        # statuses that predate `descriptor_value` (e.g. from should_rate_limit).
        status_descriptor_value: Final = status.get("descriptor_value")
        descriptor_index: Final = next(
            (
                i
                for i, d in enumerate(descriptors)
                if d.get("key") == status.get("descriptor_key")
                and (status_descriptor_value is None or d.get("value") == status_descriptor_value)
            ),
            0,
        )
        descriptor: Final[RateLimitDescriptor] = (
            descriptors[descriptor_index] if descriptors else {"key": "", "value": "", "rate_limit": None}
        )

        now: Final = datetime.now().timestamp()
        window_size: Final = self.parallel_request_limiter.window_size
        reset_time: Final = now + window_size
        reset_time_formatted: Final = datetime.fromtimestamp(reset_time).strftime("%Y-%m-%d %H:%M:%S UTC")

        remaining_display: Final = max(0, status["limit_remaining"])
        current_limit: Final = status["current_limit"]

        if limit_type == "requests":
            detail = (
                f"Batch rate limit exceeded for {descriptor.get('key', 'unknown')}: {descriptor.get('value', 'unknown')}. "
                f"Batch contains {batch_usage.request_count} requests but only {remaining_display} requests remaining "
                f"out of {current_limit} RPM limit. "
                f"Limit resets at: {reset_time_formatted}"
            )
        else:  # tokens
            # Project ITPM/OTPM descriptors are keyed "{project_id}:{model}" and
            # charged with that model's own rows (see
            # `_create_project_io_descriptors_for_models`), not the whole
            # batch's totals -- report the matching per-model figure when one
            # is available so the error reflects what was actually charged.
            descriptor_model: Final = (
                descriptor.get("value", "").split(":", 1)[-1]
                if descriptor.get("key") in (PROJECT_ITPM_DESCRIPTOR_KEY, PROJECT_OTPM_DESCRIPTOR_KEY)
                else None
            )
            model_usage: Final = batch_usage.per_model_usage.get(descriptor_model) if descriptor_model else None
            batch_token_count: Final = (
                (model_usage or {}).get("output_tokens", batch_usage.output_tokens)
                if descriptor.get("key") == PROJECT_OTPM_DESCRIPTOR_KEY
                else (model_usage or {}).get("total_tokens", batch_usage.total_tokens)
                if descriptor.get("key") == PROJECT_ITPM_DESCRIPTOR_KEY
                else batch_usage.total_tokens
            )
            detail = (
                f"Batch rate limit exceeded for {descriptor.get('key', 'unknown')}: {descriptor.get('value', 'unknown')}. "
                f"Batch contains {batch_token_count} tokens but only {remaining_display} tokens remaining "
                f"out of {current_limit} TPM limit. "
                f"Limit resets at: {reset_time_formatted}"
            )

        resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(requested_model)
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
        data: dict,
        batch_usage: BatchFileUsage,
        descriptors: list["RateLimitDescriptor"] | None = None,
    ) -> None:
        """
        Atomically check + increment rate-limit counters by the batch amounts.

        Raises HTTPException if any descriptor would exceed its limit; in that
        case no counter is modified. Backed by `atomic_check_and_increment_by_n`
        which uses a Redis Lua script when available (multi-process atomic) and
        falls back to a per-process asyncio.Lock + in-memory operation.

        ``descriptors`` may be passed in by the pre-call hook to reuse the list
        already computed when deciding whether to skip file processing. It
        never contains project ITPM/OTPM descriptors (those are model-specific
        and only knowable once ``batch_usage.per_model_usage`` is populated by
        parsing the JSONL), so this always builds and appends them here.
        """
        if descriptors is None:
            descriptors = self._create_batch_rate_limit_descriptors(
                user_api_key_dict=user_api_key_dict,
                data=data,
            )

        increments: list[IncrementAmounts] = [  # mutable-ok: reassigned below to append project IO increments
            {  # mutable-ok: atomic limiter API requires mutable increment records
                "requests": batch_usage.request_count,
                "tokens": batch_usage.total_tokens,
            }
            for _d in descriptors
        ]

        project_io_descriptors, project_io_increments = self._create_project_io_descriptors_for_models(
            user_api_key_dict=user_api_key_dict,
            per_model_usage=batch_usage.per_model_usage,
        )
        descriptors = [*descriptors, *project_io_descriptors]
        increments = [*increments, *project_io_increments]

        rate_limit_response: Final = await self.parallel_request_limiter.atomic_check_and_increment_by_n(
            descriptors=descriptors,
            increments=increments,
            parent_otel_span=user_api_key_dict.parent_otel_span,
        )

        if rate_limit_response["overall_code"] == "OVER_LIMIT":
            requested_model: Final = data.get("model") if data else None
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
        user_api_key_dict: UserAPIKeyAuth | None = None,
        data: dict | None = None,
        descriptors: Sequence["RateLimitDescriptor"] | None = None,
    ) -> BatchFileUsage:
        """
        Count number of requests and tokens in a batch input file.

        Args:
            file_id: The file ID to read
            custom_llm_provider: The custom LLM provider to use for token encoding
            user_api_key_dict: User authentication information for file access (required for managed files)
            descriptors: Rate limit descriptors already computed for this batch, so the
                configured project OTPM limit can scale the no-``max_tokens`` output floor

        Returns:
            BatchFileUsage with total_tokens, output_tokens, request_count, and
            per_model_usage (each row's own totals, keyed by its `body.model`)
        """
        descriptor_otpm_limits: Final = tuple(
            int(v)
            for d in (descriptors or ())
            if d.get("key") == PROJECT_OTPM_DESCRIPTOR_KEY
            for rate_limit in (d.get("rate_limit"),)
            for v in (rate_limit.get("tokens_per_unit") if rate_limit is not None else None,)
            if v is not None
        )
        # `descriptors` only ever carries the routing model's own OTPM limit
        # (see `_create_batch_rate_limit_descriptors`), but a row can target
        # any project-configured model. Folding in every configured model's
        # OTPM limit keeps the no-`max_tokens` floor from drifting wide just
        # because a row's specific model isn't known until parsed below.
        project_otpm_limits: Final = (
            tuple(int(v) for v in project_otpm_limit_map.values())
            if user_api_key_dict is not None
            and (
                project_otpm_limit_map := get_model_rate_limit_from_metadata(
                    user_api_key_dict, "project_metadata", "model_otpm_limit"
                )
            )
            else ()
        )
        min_configured_otpm_limit: Final = min((*descriptor_otpm_limits, *project_otpm_limits), default=None)
        try:
            # Check if this is a managed file (base64 encoded unified file ID)
            from litellm.proxy.openai_files_endpoints.common_utils import (
                _is_base64_encoded_unified_file_id,
                get_models_from_unified_file_id,
            )

            # Managed files require bypassing the HTTP endpoint (which runs access-check hooks)
            # and calling the managed files hook directly with the user's credentials.
            is_managed_file: Final = _is_base64_encoded_unified_file_id(file_id)
            # For managed files the unified file id encodes the proxy model
            # alias(es) the file was uploaded for; auth validates against those.
            target_model_names: Final = get_models_from_unified_file_id(is_managed_file) if is_managed_file else []
            if is_managed_file and user_api_key_dict is not None:
                file_content = await self._fetch_managed_file_content(
                    file_id=file_id,
                    user_api_key_dict=user_api_key_dict,
                )
            else:
                provider_file_id, fetch_kwargs = self._resolve_batch_input_file_fetch_params(
                    file_id=file_id,
                    custom_llm_provider=custom_llm_provider,
                    data=data or {},
                )
                # For non-managed files, use the standard litellm.afile_content
                file_content = await litellm.afile_content(
                    file_id=provider_file_id,
                    user_api_key_dict=user_api_key_dict,
                    **fetch_kwargs,
                )

            file_content_bytes: Final = getattr(file_content, "content", None)
            if not isinstance(file_content_bytes, bytes):
                raise ValueError(
                    f"Expected bytes content from file retrieval for {file_id}, got {type(file_content_bytes)}"
                )

            # Single streaming pass over the JSONL lines, accounting each row
            # independently. One bad row can never abort the pass: a malformed
            # line is skipped (its request can't run upstream anyway) and a row
            # the token counter can't measure falls back to a conservative
            # size-based estimate. This guarantees two things a restricted caller
            # must not be able to break by crafting a row that raises:
            #   1. The allowlist check below always sees every parseable
            #      ``body.model`` (the loop never stops early), so models can't be
            #      smuggled in after a bad row.
            #   2. The token total is never silently zeroed, so the TPM limit
            #      can't be evaded by sending uncountable rows.
            # Counting stays best-effort, so a legitimate (e.g. multimodal) row
            # the counter can't measure is estimated, not hard-rejected.
            models: Final[set] = set()
            # Keyed by each row's own `body.model`, so the project ITPM/OTPM
            # quota for that model is charged with only its own rows' tokens,
            # never the whole batch's -- see `_create_project_io_descriptors_for_models`.
            per_model_usage: Final[dict[str, dict[str, int]]] = {}
            total_tokens = 0
            output_tokens = 0  # rebind-ok: accumulated per JSONL row in the loop below
            request_count = 0
            for raw_line in _iter_batch_input_lines(file_content_bytes):
                request_count += 1
                try:
                    entry = json.loads(raw_line)
                except Exception:
                    entry_total_tokens = _estimate_batch_entry_tokens(raw_line)
                    entry_output_tokens = self.parallel_request_limiter.no_max_tokens_output_floor(
                        min_configured_otpm_limit
                    )
                    total_tokens += entry_total_tokens
                    output_tokens += entry_output_tokens
                    continue

                model: str | None = (entry.get("body") or {}).get("model") if isinstance(entry, dict) else None
                if model:
                    models.add(model)

                if isinstance(entry, dict):
                    entry_output_tokens = self._estimate_entry_output_tokens(entry, min_configured_otpm_limit)
                else:
                    entry_output_tokens = self.parallel_request_limiter.no_max_tokens_output_floor(
                        min_configured_otpm_limit
                    )
                output_tokens += entry_output_tokens

                try:
                    entry_total_tokens = _count_entry_tokens(entry)
                except Exception:
                    entry_total_tokens = _estimate_batch_entry_tokens(raw_line)
                total_tokens += entry_total_tokens

                if model:
                    model_usage = per_model_usage.setdefault(
                        model, {"total_tokens": 0, "output_tokens": 0, "request_count": 0}
                    )
                    model_usage["total_tokens"] += entry_total_tokens
                    model_usage["output_tokens"] += entry_output_tokens
                    model_usage["request_count"] += 1

            # Validate every model named in the batch JSONL against the
            # caller's per-key model allowlist. Without this, a caller
            # could smuggle restricted/expensive models inside the file
            # and the upstream provider would execute the batch under
            # the proxy's shared API key.
            if user_api_key_dict is not None:
                await self._enforce_batch_file_model_access(
                    user_api_key_dict=user_api_key_dict,
                    models=models,
                    target_model_names=target_model_names or None,
                )

            return BatchFileUsage(
                total_tokens=total_tokens,
                request_count=request_count,
                output_tokens=output_tokens,
                per_model_usage=per_model_usage,
            )

        except HTTPException as e:
            # Distinguish intentional 403s from `_enforce_batch_file_model_access`
            # from genuine I/O failures so security-relevant rejections show up
            # in the access log instead of getting buried in error noise.
            if e.status_code == 403:
                verbose_proxy_logger.warning(
                    "Batch rejected: caller not authorized for a model named in %s: %s", file_id, e.detail
                )
            else:
                verbose_proxy_logger.error(
                    "Batch input file rejected for %s: status=%s detail=%s", file_id, e.status_code, e.detail
                )
            raise
        except Exception as e:
            verbose_proxy_logger.error("Error counting input file usage for %s: %s", file_id, e)
            raise

    async def _enforce_batch_file_model_access(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        models: Iterable[str] | None = None,
        target_model_names: list[str] | None = None,
    ) -> None:
        """Reject the batch if the caller is not authorized for the upload target.

        For managed files, ``target_model_names`` (from the unified file id) is
        the proxy alias the file was uploaded for and is checked directly.
        Otherwise the ``body.model`` values collected from the JSONL (``models``)
        are checked.

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
        from litellm.proxy.proxy_server import llm_router, prisma_client, proxy_logging_obj, user_api_key_cache

        if target_model_names:
            models = target_model_names

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
                        "error": ("Batch input file model access could not be validated against the current team.")
                    },
                ) from e

        llm_model_list: Final = llm_router.model_list if llm_router is not None else None
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
                            f"not authorized to use: model={model_to_check}, reason={e}"
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
                f"Cannot import proxy_server dependencies: {e}. Managed files require proxy_server to be initialized."
            )

        # Get the managed files hook
        if proxy_logging_obj is None:
            raise ValueError("proxy_logging_obj not available. Cannot access managed files hook.")

        managed_files_obj: Final = proxy_logging_obj.get_proxy_hook("managed_files")
        if managed_files_obj is None:
            raise ValueError("Managed files hook not found. Cannot access managed file.")

        if not isinstance(managed_files_obj, BaseFileEndpoints):
            raise ValueError("Managed files hook is not a BaseFileEndpoints instance.")

        if llm_router is None:
            raise ValueError("llm_router not available. Cannot access managed files.")

        # Use the managed files hook to get file content
        # This properly handles user permissions and file ownership
        file_content: Final = await managed_files_obj.afile_content(
            file_id=file_id,
            litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
            llm_router=llm_router,
        )

        return file_content

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Exception | str | dict | None:
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
                "Batch rate limiter: Not handling batch creation rate limiting for call type: %s", call_type
            )
            return data

        verbose_proxy_logger.debug("Batch rate limiter: Handling batch creation rate limiting")

        try:
            # Extract input_file_id from data
            input_file_id: Final = data.get("input_file_id")
            if not input_file_id:
                verbose_proxy_logger.debug("No input_file_id in batch request, skipping rate limiting")
                return data

            enqueued_scopes: Final = resolve_batch_enqueued_token_scopes(user_api_key_dict)
            should_skip, batch_rate_limit_descriptors = self._should_skip_batch_input_file_processing(
                data=data, user_api_key_dict=user_api_key_dict, has_enqueued_scopes=bool(enqueued_scopes)
            )
            if should_skip:
                return data

            # Get custom_llm_provider for token counting
            custom_llm_provider: Final = data.get("custom_llm_provider", "openai")

            # Count tokens and requests from input file
            verbose_proxy_logger.debug("Counting tokens from batch input file: %s", input_file_id)
            batch_usage: Final = await self.count_input_file_usage(
                file_id=input_file_id,
                custom_llm_provider=custom_llm_provider,
                user_api_key_dict=user_api_key_dict,
                data=data,
                descriptors=batch_rate_limit_descriptors,
            )

            verbose_proxy_logger.debug(
                "Batch input file usage - Tokens: %s, Requests: %s", batch_usage.total_tokens, batch_usage.request_count
            )

            # Store batch usage in data for later reference
            data["_batch_token_count"] = batch_usage.total_tokens
            data["_batch_request_count"] = batch_usage.request_count

            if enqueued_scopes:
                await self._reserve_batch_enqueued_tokens(
                    user_api_key_dict=user_api_key_dict,
                    data=data,
                    batch_usage=batch_usage,
                    scopes=enqueued_scopes,
                )
                verbose_proxy_logger.debug("Batch enqueued-token reservation succeeded")
                return data

            # Directly increment counters by batch amounts (check happens atomically)
            # This will raise HTTPException if limits are exceeded
            await self._check_and_increment_batch_counters(
                user_api_key_dict=user_api_key_dict,
                data=data,
                batch_usage=batch_usage,
                descriptors=batch_rate_limit_descriptors,
            )

            verbose_proxy_logger.debug("Batch rate limit check passed, counters incremented")
            return data

        except HTTPException:
            # Re-raise HTTP exceptions (rate limit exceeded)
            raise
        except Exception as e:
            verbose_proxy_logger.error("Error in batch rate limiting: %s", e, exc_info=True)
            # Don't block the request if rate limiting fails
            return data

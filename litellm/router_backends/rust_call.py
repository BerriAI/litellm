"""One call routed by the Rust router: the Python side of its `Invoke` and `Sleep` host ops.

The native router decides which deployment each attempt goes to, when to retry, back off,
fall back and cool down. Everything that touches Python objects happens here: building an
attempt's kwargs the way `PythonRouter._acompletion` does, calling `litellm.<op>`, classifying
what it raised, and writing the router's metadata and exception updates where Python's
readers look for them.
"""

from __future__ import annotations

import asyncio
import copy
import time
from collections.abc import Awaitable, Callable, Mapping, MutableMapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.integrations.custom_guardrail import is_guardrail_intervention
from litellm.litellm_core_utils.asyncify import (
    run_async_function,  # pyright: ignore[reportUnknownVariableType]  # untyped helper, wrapped in _run_sync
)
from litellm.litellm_core_utils.exception_mapping_utils import (
    _get_response_headers,  # pyright: ignore[reportPrivateUsage]  # the header reader the cooldown callback uses
)
from litellm.litellm_core_utils.secret_redaction import redact_string
from litellm.litellm_core_utils.sensitive_data_masker import mask_sensitive_structure
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.router_backends.python_router import (
    PythonRouter,
    _anthropic_stream_pre_content_error,  # pyright: ignore[reportPrivateUsage]  # the envelope Python's Anthropic retry gives a failure
    _anthropic_stream_raised_error_status,  # pyright: ignore[reportPrivateUsage]  # the status Python's Anthropic retry reads
    _with_router_resolved_session_model,  # pyright: ignore[reportPrivateUsage]  # the session rewrite the generic helper applies
)
from litellm.router_backends.rust_streams import (
    AnthropicStreamRules,
    StreamFailed,
    anthropic_frames,
    anthropic_until_committed,
    chat_until_content,
    chat_until_content_sync,
    responses_until_output,
    unwrapped,
)
from litellm.router_strategy.complexity_router.context_compaction import compact_to_fit, surface_for_call
from litellm.router_utils.add_retry_fallback_headers import (
    add_fallback_headers_to_response,
    add_retry_headers_to_response,
)
from litellm.router_utils.common_utils import (
    format_fallback_outcome_message,
    format_no_fallback_group_message,
    provider_for_generic_call,
    truncate_fallback_error_detail,
)
from litellm.router_utils.cooldown_handlers import (
    is_advisor_orchestration_failure,
    is_background_response_cost_poll_not_found,
    is_caller_timeout_408,
)
from litellm.router_utils.fallback_event_handlers import (
    _is_fallback_target_authorized,  # pyright: ignore[reportPrivateUsage]  # run_async_fallback's access check
    _is_fallback_target_within_budget,  # pyright: ignore[reportPrivateUsage]  # run_async_fallback's budget check
)
from litellm.types.llms.openai import ResponseAPIUsage
from litellm.types.router import RouterErrors, RouterRateLimitError
from litellm.types.utils import ModelResponse
from litellm.utils import (
    _get_retry_after_from_exception_header,  # pyright: ignore[reportPrivateUsage]  # the Retry-After parser retries and cooldowns use
)

if TYPE_CHECKING:
    from litellm.router import Router

Bucket: TypeAlias = MutableMapping[str, object]
Kwargs: TypeAlias = dict[str, object]
Operation: TypeAlias = Literal["completion", "responses", "anthropic_messages"]
ResumeRoute: TypeAlias = Callable[[Mapping[str, object], Kwargs], Awaitable[object]]


class AttemptRouter(Protocol):
    """The `PythonRouter` members an attempt reuses, typed. The Rust backend's normalizer provides them."""

    cache_responses: bool | None
    total_calls: MutableMapping[str, int]  # mutable-ok: PythonRouter's per-model call counters
    success_calls: MutableMapping[str, int]  # mutable-ok: as above
    fail_calls: MutableMapping[str, int]  # mutable-ok: as above

    def get_model_info(self, id: str) -> Kwargs | None: ...

    def get_model_list(self, model_name: str | None = None) -> Sequence[Kwargs] | None: ...

    def log_retry(self, kwargs: Kwargs, e: Exception) -> object: ...

    async def set_response_headers(
        self, response: object, model_group: str | None = None, request_kwargs: Kwargs | None = None
    ) -> object: ...

    def routing_strategy_pre_call_checks(self, deployment: Kwargs) -> None: ...

    async def async_routing_strategy_pre_call_checks(
        self, deployment: Kwargs, parent_otel_span: None, logging_obj: object = None
    ) -> None: ...

    def _drop_unsupported_classifier_reasoning_effort(self, deployment: Kwargs, model: str, kwargs: Kwargs) -> None: ...

    def _deployment_params_with_request_reasoning_override(
        self, deployment_params: Mapping[str, object], kwargs: Mapping[str, object]
    ) -> Kwargs: ...

    def _update_kwargs_with_deployment(
        self, deployment: Kwargs, kwargs: Kwargs, function_name: str | None = None
    ) -> None: ...

    def _get_async_openai_model_client(self, deployment: Kwargs, kwargs: Kwargs) -> object: ...

    def _get_client(self, deployment: Kwargs, kwargs: Kwargs) -> object: ...

    def _should_raise_content_policy_error(self, model: str, response: ModelResponse, kwargs: Kwargs) -> bool: ...

    def _should_raise_anthropic_refusal_error(
        self, model: str, original_generic_function: Callable[..., object], response: object, kwargs: Kwargs
    ) -> bool: ...

    def _anthropic_messages_stream_can_retry(self, kwargs: Mapping[str, object]) -> bool: ...

    def _anthropic_messages_stream_can_fall_back(self, model_group: str, kwargs: Mapping[str, object]) -> bool: ...

    def _refusal_fallback_available(self, model_group: str, kwargs: Mapping[str, object]) -> bool: ...

    def _anthropic_messages_recoverable_frame_error(
        self,
        error_event: tuple[str, str, int] | None,
        chunk: object,
        has_generated_content: bool,
        model_group: str,
        kwargs: Mapping[str, object],
    ) -> Exception | None: ...

    def _set_deployment_num_retries_on_exception(self, exception: Exception, deployment: Kwargs) -> None: ...

    def _stamp_failed_deployment_id_with_effective_model_info(
        self, exception: Exception, deployment: Mapping[str, object], kwargs: Mapping[str, object]
    ) -> None: ...


Invoked: TypeAlias = tuple[Literal["ok"], object] | tuple[Literal["error"], BaseException, Mapping[str, object]]

_REQUEST_CONTROLS: Final = (
    "num_retries",
    "fallbacks",
    "context_window_fallbacks",
    "content_policy_fallbacks",
    "disable_fallbacks",
)

_ROUTER_ONLY_KWARGS: Final = (
    "num_retries",
    "fallbacks",
    "context_window_fallbacks",
    "content_policy_fallbacks",
    "disable_fallbacks",
    "mock_testing_fallbacks",
    "mock_testing_context_fallbacks",
    "mock_testing_content_policy_fallbacks",
    "mock_testing_rate_limit_error",
)

_UNWRAPPED_FALLBACK_TRIGGERS: Final[Mapping[Operation, tuple[type[Exception], ...]]] = {
    "completion": (),
    "responses": (litellm.ContentPolicyViolationError,),
    "anthropic_messages": (litellm.ContentPolicyViolationError, litellm.ContextWindowExceededError),
}
_GENERIC_HANDLERS: Final[Mapping[Operation, str]] = {
    "responses": "aresponses",
    "anthropic_messages": "anthropic_messages",
}

_EXCEPTION_CLASSES: Final[tuple[tuple[type[BaseException], str], ...]] = (
    (litellm.ContextWindowExceededError, "ContextWindowExceeded"),
    (litellm.ContentPolicyViolationError, "ContentPolicyViolation"),
    (litellm.BadRequestError, "BadRequest"),
    (litellm.AuthenticationError, "Authentication"),
    (litellm.PermissionDeniedError, "PermissionDenied"),
    (litellm.NotFoundError, "NotFound"),
    (litellm.Timeout, "Timeout"),
    (litellm.RateLimitError, "RateLimit"),
    (litellm.ServiceUnavailableError, "ServiceUnavailable"),
    (litellm.InternalServerError, "InternalServer"),
    (litellm.BadGatewayError, "BadGateway"),
    (litellm.APIConnectionError, "ApiConnection"),
)


class Attempt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    deployment_id: str | None = None
    mock: Literal["fallbacks", "context_window_fallbacks", "content_policy_fallbacks", "rate_limit"] | None = None
    model_group: str
    original_model_group: str
    bucket: int
    fallback_depth: int
    attempted_targets: tuple[str, ...]
    model_group_size: int
    attempted_retries: int
    max_retries: int
    stream_retry: bool = False
    ops: tuple[Mapping[str, object], ...]


class FallbackCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target: str
    model_group: str
    bucket: int
    model: str
    ops: tuple[Mapping[str, object], ...]


class Outcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_group: str
    deployment_id: str
    attempted_retries: int
    max_retries: int | None
    attempted_fallbacks: int


def exception_classes(error: BaseException) -> tuple[str, ...]:
    """The routing-relevant classes of `error`, most specific first, as `type(e).__mro__` orders them."""
    names: Final = (_class_name(cls) for cls in type(error).__mro__)
    return tuple(dict.fromkeys(name for name in names if name is not None))


def _class_name(cls: type) -> str | None:
    return next((name for known, name in _EXCEPTION_CLASSES if cls is known), None)


def _retry_after(headers: object) -> int | None:
    seconds: Final[object] = _get_retry_after_from_exception_header(headers)  # pyright: ignore[reportArgumentType]  # reads any header mapping
    return seconds if isinstance(seconds, int) else None


def _sleep_headers(error: BaseException) -> object:
    """`_time_to_sleep_before_retry`'s header precedence: `litellm_response_headers` over `response.headers`."""
    response: Final[object] = getattr(error, "response", None)
    response_headers: Final[object] = getattr(response, "headers", None) if response is not None else None
    return getattr(error, "litellm_response_headers", response_headers)


def classify(error: BaseException, kwargs: Mapping[str, object], callbacks_ran: bool) -> Mapping[str, object]:
    status: Final = getattr(error, "status_code", None)
    num_retries: Final = getattr(error, "num_retries", None)
    sleep_headers: Final = _sleep_headers(error)
    cooldown_headers: Final = _get_response_headers(original_exception=error) if isinstance(error, Exception) else None
    logging_obj: Final = kwargs.get("litellm_logging_obj")
    model_call_details: Final = getattr(logging_obj, "model_call_details", None)
    exempt: Final = (
        is_advisor_orchestration_failure(error)
        or (isinstance(error, Exception) and is_background_response_cost_poll_not_found(error, kwargs))
        or (
            isinstance(model_call_details, Mapping)
            and isinstance(status, int)
            and is_caller_timeout_408(model_call_details, status)  # pyright: ignore[reportUnknownArgumentType]  # logging state is untyped
        )
    )
    return {
        "classes": exception_classes(error),
        "type_name": type(error).__name__,
        "message": str(error),
        "status_code": status if isinstance(status, int) and not isinstance(status, bool) else None,
        "num_retries": num_retries if isinstance(num_retries, int) and num_retries >= 0 else None,
        "sleep_retry_after": (_retry_after(sleep_headers) if sleep_headers is not None else None) or -1,
        "cooldown_retry_after": _retry_after(cooldown_headers) if cooldown_headers is not None else None,
        "guardrail_intervention": isinstance(error, Exception) and is_guardrail_intervention(error),
        "callbacks_ran": callbacks_ran,
        "exempt_from_cooldown": exempt,
        "exact_litellm_type": type(error) in litellm.LITELLM_EXCEPTION_TYPES,
    }


def _not_fetched(stream: CustomStreamWrapper) -> bool:
    completion_stream: Final = cast(object, stream.completion_stream)  # cast-ok: untyped attribute
    make_call: Final = cast(object, stream.make_call)  # cast-ok: untyped attribute
    return completion_stream is None and make_call is not None


class _AttemptFailed(Exception):
    """An attempt's error that litellm did not raise, so its failure callbacks never ran for it."""

    def __init__(self, error: BaseException) -> None:
        super().__init__(str(error))
        self.error: Final = error


class RoutedCall:
    """Answers the host ops of one routed call and builds its result or error."""

    def __init__(
        self,
        normalizer: AttemptRouter,
        kwargs: Kwargs,
        operation: Operation,
        fallbacks: Callable[[str], object],
        resume: ResumeRoute | None = None,
    ) -> None:
        self._normalizer: Final = normalizer
        self._kwargs: Final = {key: value for key, value in kwargs.items() if key not in _ROUTER_ONLY_KWARGS}
        self._controls: Final = MappingProxyType({key: kwargs[key] for key in _REQUEST_CONTROLS if key in kwargs})
        self._operation: Final = operation
        metadata_key: Final = metadata_key_for(operation)
        self._metadata_key: Final = metadata_key
        self._fallbacks: Final = fallbacks
        self._resume: Final = resume
        self._partial_usage: Final[list[ResponseAPIUsage]] = []  # mutable-ok: what failed Responses streams used
        root: Final = kwargs.setdefault(metadata_key, {})
        bucket: Final[Bucket] = _bucket(root)
        self._buckets: Final[dict[int, Bucket]] = {0: bucket}  # mutable-ok: grows as fallback hops open buckets
        self._hop_kwargs: Final[dict[int, Mapping[str, object]]] = {0: {}}  # mutable-ok: grows with the buckets
        self._rejections: Final[dict[int, BaseException]] = {}  # mutable-ok: one exception per router rejection id

    def start(self, model: str) -> None:
        """`async_function_with_fallbacks`' first-hop stamps on the request's own bucket."""
        sibling: Final = "metadata" if self._metadata_key == "litellm_metadata" else "litellm_metadata"
        sibling_bucket: Final = _bucket(self._kwargs.get(sibling))
        sibling_bucket.pop("attempted_fallbacks", None)
        sibling_bucket.pop("original_model_group", None)
        self._buckets[0]["attempted_fallbacks"] = 0
        self._buckets[0]["original_model_group"] = model

    async def invoke(self, attempt: Mapping[str, object]) -> Invoked:
        parsed: Final = Attempt.model_validate(attempt)
        kwargs: Final = self._prepare(parsed)
        if parsed.mock is not None:
            return self._mock(parsed, kwargs)
        try:
            response: Final = (
                await self._acompletion(parsed, kwargs)
                if self._operation == "completion"
                else await self._ageneric(parsed, kwargs)
            )
        except StreamFailed as failed:
            return self._stream_failed(failed, kwargs)
        except _AttemptFailed as failed:
            return self._attempt_failed(parsed, failed.error, kwargs, callbacks_ran=False)
        except Exception as error:
            return self._attempt_failed(parsed, error, kwargs, callbacks_ran=True)
        return ("ok", await self._normalizer.set_response_headers(response, parsed.model_group, kwargs))

    def invoke_sync(self, attempt: Mapping[str, object]) -> Invoked:
        parsed: Final = Attempt.model_validate(attempt)
        kwargs: Final = self._prepare(parsed)
        if parsed.mock is not None:
            return self._mock(parsed, kwargs)
        try:
            response: Final = self._completion(parsed, kwargs)
        except StreamFailed as failed:
            return self._stream_failed(failed, kwargs)
        except _AttemptFailed as failed:
            return ("error", failed.error, classify(failed.error, kwargs, callbacks_ran=False))
        except Exception as error:
            return ("error", error, classify(error, kwargs, callbacks_ran=True))
        return ("ok", response)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    def sleep_sync(self, seconds: float) -> None:
        time.sleep(seconds)

    async def allow_fallback(self, check: Mapping[str, object]) -> bool:
        """`run_async_fallback`'s access and budget checks, asked of the request as the hop the
        chain falls back from left it."""
        parsed: Final = FallbackCheck.model_validate(check)
        self._apply(parsed.ops)
        kwargs: Final = {
            **self._kwargs,
            **self._hop_kwargs[parsed.bucket],
            "model": parsed.model,
            self._metadata_key: self._buckets[parsed.bucket],
        }
        router: Final = cast("Router", self._normalizer)  # cast-ok: the PythonRouter the helpers read checks off
        return await _is_fallback_target_authorized(
            router, parsed.target, parsed.model_group, kwargs
        ) and await _is_fallback_target_within_budget(router, parsed.target, parsed.model_group, kwargs)

    def allow_fallback_sync(self, check: Mapping[str, object]) -> bool:
        """Python's sync calls run the async fallback loop, checks included, through `run_async_function`."""
        return _run_sync(lambda: self.allow_fallback(check)) is True

    def success(self, response: object, outcome: Mapping[str, object], ops: Sequence[Mapping[str, object]]) -> object:
        self._apply(ops)
        parsed: Final = Outcome.model_validate(outcome)
        with_retries: Final = add_retry_headers_to_response(
            response, attempted_retries=parsed.attempted_retries, max_retries=parsed.max_retries
        )
        return add_fallback_headers_to_response(with_retries, attempted_fallbacks=parsed.attempted_fallbacks)

    def failure(self, error: object, ops: Sequence[Mapping[str, object]]) -> BaseException:
        self._apply(ops)
        return unwrapped(self._exception(error))

    def _stream_failed(self, failed: StreamFailed, kwargs: Mapping[str, object]) -> Invoked:
        if failed.partial_usage is not None:
            self._partial_usage.append(failed.partial_usage)
        classified: Final = self._enveloped(failed.error, kwargs, failed.callbacks_ran)
        return ("error", failed.error, {**classified, "stream_failure": failed.kind})

    def _attempt_failed(
        self, attempt: Attempt, error: BaseException, kwargs: Mapping[str, object], callbacks_ran: bool
    ) -> Invoked:
        """A same-group retry of a failed Anthropic stream hands a failure before its own stream opens
        to the next retry or the fallback chain in the envelope Python's retry loop gives it."""
        if not attempt.stream_retry or not isinstance(error, Exception):
            return ("error", error, classify(error, kwargs, callbacks_ran))
        envelope: Final = _anthropic_stream_pre_content_error(error, attempt.model_group)
        status: Final = _anthropic_stream_raised_error_status(error)
        retriable: Final = status is None or litellm._should_retry(status)  # pyright: ignore[reportPrivateUsage]  # the shared retry rule
        return (
            "error",
            envelope,
            {**self._enveloped(envelope, kwargs, callbacks_ran), "retries_pre_stream": retriable},
        )

    def _enveloped(
        self, error: BaseException, kwargs: Mapping[str, object], callbacks_ran: bool
    ) -> Mapping[str, object]:
        """A mid-stream fallback error is judged by its fallback trigger; Anthropic's same-group
        retry judges the provider error inside it."""
        classified: Final = classify(self._fallback_trigger(error), kwargs, callbacks_ran)
        if self._operation != "anthropic_messages":
            return classified
        return {**classified, "original": classify(unwrapped(error), kwargs, callbacks_ran)}

    def _fallback_trigger(self, error: BaseException) -> BaseException:
        """The error a stream's fallback is judged and annotated by: the mid-stream fallback error,
        or the provider error inside it for the classes this operation's wrapper unwraps."""
        if not isinstance(error, MidStreamFallbackError):
            return error
        original: Final = error.original_exception
        return original if isinstance(original, _UNWRAPPED_FALLBACK_TRIGGERS[self._operation]) else error

    def _prepare(self, attempt: Attempt) -> Kwargs:
        self._apply(attempt.ops)
        bucket: Final = self._buckets[attempt.bucket]
        if isinstance(bucket.get("model_group"), str):
            bucket["model_group_size"] = attempt.model_group_size
        bucket["attempted_retries"] = attempt.attempted_retries
        bucket["max_retries"] = attempt.max_retries
        first_hop_only: Final = () if attempt.bucket == 0 else ("mock_timeout",)
        passed_explicitly: Final = ("model", "messages") if self._operation == "completion" else ("model",)
        return {
            **{key: value for key, value in self._kwargs.items() if key not in (*passed_explicitly, *first_hop_only)},
            **self._hop_kwargs[attempt.bucket],
            self._metadata_key: bucket,
        }

    def _mock(self, attempt: Attempt, kwargs: Mapping[str, object]) -> Invoked:
        group: Final = attempt.model_group
        error: Final[BaseException] = (
            litellm.RateLimitError(
                model=group,
                llm_provider="",
                message=f"This is a mock exception for model={group}, to trigger a rate limit error.",
                num_retries=self._single_deployment_num_retries(group),
            )
            if attempt.mock == "rate_limit"
            else litellm.InternalServerError(
                model=group,
                llm_provider="",
                message=f"This is a mock exception for model={group}, to trigger a fallback. "
                f"Fallbacks={self._fallbacks('fallbacks')}",
            )
            if attempt.mock == "fallbacks"
            else litellm.ContextWindowExceededError(
                model=group,
                llm_provider="",
                message=f"This is a mock exception for model={group}, to trigger a fallback. \
                    Context_Window_Fallbacks={self._fallbacks('context_window_fallbacks')}",
            )
            if attempt.mock == "context_window_fallbacks"
            else litellm.ContentPolicyViolationError(
                model=group,
                llm_provider="",
                message=f"This is a mock exception for model={group}, to trigger a fallback. \
                    Context_Policy_Fallbacks={self._fallbacks('content_policy_fallbacks')}",
            )
        )
        return ("error", error, classify(error, kwargs, callbacks_ran=False))

    def _single_deployment_num_retries(self, group: str) -> int | None:
        """`_handle_mock_testing_rate_limit_error` reads `num_retries` off a one-deployment group."""
        deployments: Final = self._normalizer.get_model_list(model_name=group)
        if deployments is None or len(deployments) != 1:
            return None
        num_retries: Final = as_mapping(deployments[0].get("litellm_params")).get("num_retries")
        return num_retries if isinstance(num_retries, int) else None

    def _deployment(self, attempt: Attempt) -> Kwargs:
        deployment: Final = self._normalizer.get_model_info(id=attempt.deployment_id or "")
        if deployment is None:
            raise RuntimeError(f"the Rust router picked unknown deployment {attempt.deployment_id!r}")
        return deployment

    async def _acompletion(self, attempt: Attempt, kwargs: Kwargs) -> object:
        normalizer: Final = self._normalizer
        deployment: Final = self._deployment(attempt)
        messages: Final = self._kwargs.get("messages")
        model_name: str | None = None  # rebind-ok: set once the deployment's params are read
        try:
            normalizer._drop_unsupported_classifier_reasoning_effort(  # pyright: ignore[reportPrivateUsage]  # the attempt body PythonRouter runs
                deployment=deployment,  # pyright: ignore[reportArgumentType]  # router deployment dict
                model=attempt.model_group,
                kwargs=kwargs,
            )
            litellm_params: Final = normalizer._deployment_params_with_request_reasoning_override(  # pyright: ignore[reportPrivateUsage]  # as above
                as_mapping(deployment.get("litellm_params")), kwargs
            )
            kwargs.setdefault("messages", messages)
            normalizer._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)  # pyright: ignore[reportPrivateUsage]  # as above
            model_name = str(litellm_params["model"])  # rebind-ok: read by the failure counter below
            client: Final = normalizer._get_async_openai_model_client(deployment=deployment, kwargs=kwargs)  # pyright: ignore[reportPrivateUsage]  # as above
            normalizer.total_calls[model_name] += 1
            input_kwargs: Final = {
                **litellm_params,
                "messages": messages,
                "caching": normalizer.cache_responses,
                "client": client,
                **kwargs,
            }
            input_kwargs.pop("include_fallback_errors", None)
            compacted: Final = await compact_to_fit(normalizer, deployment, input_kwargs, "chat")  # pyright: ignore[reportArgumentType]  # PythonRouter and its deployment dict
            await normalizer.async_routing_strategy_pre_call_checks(
                deployment=deployment, parent_otel_span=None, logging_obj=kwargs.get("litellm_logging_obj")
            )
            response: Final = await litellm.acompletion(**compacted)
            if isinstance(response, ModelResponse) and normalizer._should_raise_content_policy_error(  # pyright: ignore[reportPrivateUsage]  # as above
                model=attempt.model_group, response=response, kwargs=kwargs
            ):
                raise _AttemptFailed(
                    litellm.ContentPolicyViolationError(
                        message="Response output was blocked.", model=attempt.model_group, llm_provider=""
                    )
                )
            if isinstance(response, CustomStreamWrapper) and _not_fetched(response):
                await response.fetch_stream()
            normalizer.success_calls[model_name] += 1
            if isinstance(response, CustomStreamWrapper):
                return await chat_until_content(response)
            return response
        except StreamFailed:
            raise
        except _AttemptFailed as failed:
            self._stamp(failed.error, deployment, kwargs, model_name)
            raise
        except Exception as error:
            if isinstance(error, litellm.Timeout) and litellm.expose_router_debug_in_errors:
                params: Final = as_mapping(deployment.get("litellm_params"))
                error.message += (
                    f"\n\nDeployment Info: request_timeout: {params.get('request_timeout')}\n"
                    f"timeout: {params.get('timeout')}"
                )
            self._stamp(error, deployment, kwargs, model_name)
            raise

    async def _ageneric(self, attempt: Attempt, kwargs: Kwargs) -> object:
        """`_ageneric_api_call_with_fallbacks_helper` after selection, with the Responses API's and
        Anthropic messages' own handling of the stream it returns."""
        normalizer: Final = self._normalizer
        deployment: Final = self._deployment(attempt)
        handler: Final = _handler(self._operation)
        hop_kwargs: Final = {**kwargs, self._metadata_key: copy.deepcopy(kwargs[self._metadata_key])}
        try:
            normalizer._update_kwargs_with_deployment(  # pyright: ignore[reportPrivateUsage]  # the attempt body PythonRouter runs
                deployment=deployment, kwargs=kwargs, function_name="_ageneric_api_call_with_fallbacks"
            )
            data: Final = dict(as_mapping(deployment.get("litellm_params")))
            model_name: Final = str(data["model"])
            normalizer.total_calls[model_name] += 1
            provider: Final = provider_for_generic_call(data)
            response_kwargs: Final = {
                **data,
                "caching": normalizer.cache_responses,
                **kwargs,
                "model": model_name,
                **_with_router_resolved_session_model(kwargs.get("session"), model_name),
                **({"custom_llm_provider": provider} if provider is not None else {}),
            }
            compacted: Final = await compact_to_fit(
                normalizer,  # pyright: ignore[reportArgumentType]  # PythonRouter
                deployment,  # pyright: ignore[reportArgumentType]  # router deployment dict
                response_kwargs,
                surface_for_call(getattr(handler, "__name__", "")),
            )
            response: Final = await handler(**compacted)
            if normalizer._should_raise_anthropic_refusal_error(  # pyright: ignore[reportPrivateUsage]  # as above
                model=attempt.model_group, original_generic_function=handler, response=response, kwargs=kwargs
            ):
                from litellm.llms.anthropic.pass_through.messages.utils import (  # noqa: PLC0415  # imports the proxy, which imports litellm.Router
                    safeguard_refusal_error,
                )

                refusal: Final = cast(Mapping[str, object], response)  # cast-ok: the refusal gate checked its shape
                stop_details: Final = as_mapping(refusal["stop_details"])
                raise _AttemptFailed(safeguard_refusal_error(model=attempt.model_group, stop_details=stop_details))
            normalizer.success_calls[model_name] += 1
        except _AttemptFailed as failed:
            self._stamp_generic(failed.error, deployment, kwargs, attempt.model_group)
            raise
        except Exception as error:
            self._stamp_generic(error, deployment, kwargs, attempt.model_group)
            raise
        if kwargs.get("stream") and isinstance(response, BaseResponsesAPIStreamingIterator):
            return await responses_until_output(
                response, self._responses_fallback(attempt, hop_kwargs), tuple(self._partial_usage)
            )
        frames: Final = anthropic_frames(response) if self._operation == "anthropic_messages" else None
        if kwargs.get("stream") and frames is not None:
            return await anthropic_until_committed(frames, self._anthropic_rules(attempt, kwargs))
        return response

    def _anthropic_rules(self, attempt: Attempt, kwargs: Mapping[str, object]) -> AnthropicStreamRules:
        """Answered by `PythonRouter`'s own rules over the request as its stream wrapper sees it:
        the hop's model group, the request's controls and the attempt's metadata bucket."""
        normalizer: Final = self._normalizer
        group: Final = attempt.model_group
        request: Final = {**kwargs, "model": group, **self._controls}
        return AnthropicStreamRules(
            model=group,
            recoverable=normalizer._anthropic_messages_stream_can_retry(request)  # pyright: ignore[reportPrivateUsage]  # Python's own gate
            or normalizer._anthropic_messages_stream_can_fall_back(group, request),  # pyright: ignore[reportPrivateUsage]  # as above
            frame_error=lambda event, chunk, committed: normalizer._anthropic_messages_recoverable_frame_error(  # pyright: ignore[reportPrivateUsage]  # as above
                event, chunk, committed, group, request
            ),
            refusal_recoverable=normalizer._refusal_fallback_available(group, request),  # pyright: ignore[reportPrivateUsage]  # as above
        )

    def _responses_fallback(
        self, attempt: Attempt, hop_kwargs: Kwargs
    ) -> Callable[[MidStreamFallbackError], Awaitable[object]]:
        """`_aresponses_fallback_attempt`: the chain of the hop `attempt` ran in, continuing from the
        text the failed stream generated."""

        async def fall_back(error: MidStreamFallbackError) -> object:
            if self._resume is None:
                raise error
            request_input: Final = cast(str, hop_kwargs.get("input"))  # cast-ok: the request's own input
            continued: Final = (
                {
                    "input": PythonRouter._build_responses_continuation_input(  # pyright: ignore[reportPrivateUsage]  # Python's continuation input
                        request_input, error.generated_content
                    )
                }
                if error.generated_content and not error.is_pre_first_chunk
                else {}
            )
            trigger: Final = self._fallback_trigger(error)
            return await self._resume(
                {
                    "model_group": attempt.model_group,
                    "fallback_depth": attempt.fallback_depth,
                    "original_model_group": attempt.original_model_group,
                    "attempted_targets": attempt.attempted_targets,
                    "error": error,
                    "classified": classify(trigger, hop_kwargs, callbacks_ran=True),
                    "deployment_id": attempt.deployment_id,
                },
                {**hop_kwargs, **continued},
            )

        return fall_back

    def _stamp_generic(
        self, error: BaseException, deployment: Mapping[str, object], kwargs: Mapping[str, object], model_group: str
    ) -> None:
        """The generic helper counts failures by model group, not by provider model."""
        self._normalizer.fail_calls[model_group] += 1
        self._stamp(error, deployment, kwargs, None)

    def _completion(self, attempt: Attempt, kwargs: Kwargs) -> object:
        normalizer: Final = self._normalizer
        deployment: Final = self._deployment(attempt)
        messages: Final = self._kwargs.get("messages")
        try:
            normalizer._drop_unsupported_classifier_reasoning_effort(  # pyright: ignore[reportPrivateUsage]  # the attempt body PythonRouter runs
                deployment=deployment,  # pyright: ignore[reportArgumentType]  # router deployment dict
                model=attempt.model_group,
                kwargs=kwargs,
            )
            litellm_params: Final = normalizer._deployment_params_with_request_reasoning_override(  # pyright: ignore[reportPrivateUsage]  # as above
                as_mapping(deployment.get("litellm_params")), kwargs
            )
            kwargs.setdefault("messages", messages)
            normalizer._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)  # pyright: ignore[reportPrivateUsage]  # as above
            potential_client: Final = normalizer._get_client(deployment=deployment, kwargs=kwargs)  # pyright: ignore[reportPrivateUsage]  # as above
            dynamic_api_key: Final = kwargs.get("api_key")
            client: Final = (
                None
                if dynamic_api_key is not None
                and potential_client is not None
                and dynamic_api_key != getattr(potential_client, "api_key", None)
                else potential_client
            )
            normalizer.routing_strategy_pre_call_checks(deployment=deployment)
            response: Final = litellm.completion(
                **{
                    **litellm_params,
                    "messages": messages,
                    "caching": normalizer.cache_responses,
                    "client": client,
                    **kwargs,
                }
            )
            if isinstance(response, ModelResponse) and normalizer._should_raise_content_policy_error(  # pyright: ignore[reportPrivateUsage]  # as above
                model=attempt.model_group, response=response, kwargs=kwargs
            ):
                raise _AttemptFailed(
                    litellm.ContentPolicyViolationError(
                        message="Response output was blocked.", model=attempt.model_group, llm_provider=""
                    )
                )
            if isinstance(response, CustomStreamWrapper) and _not_fetched(response):
                response.fetch_sync_stream()
            if isinstance(response, CustomStreamWrapper):
                return chat_until_content_sync(response)
            return response
        except StreamFailed:
            raise
        except _AttemptFailed as failed:
            self._stamp(failed.error, deployment, kwargs, None)
            raise
        except Exception as error:
            self._stamp(error, deployment, kwargs, None)
            raise

    def _stamp(
        self,
        error: BaseException,
        deployment: Mapping[str, object],
        kwargs: Mapping[str, object],
        model_name: str | None,
    ) -> None:
        if model_name is not None:
            self._normalizer.fail_calls[model_name] += 1
        if isinstance(error, Exception):
            self._normalizer._set_deployment_num_retries_on_exception(error, dict(deployment))  # pyright: ignore[reportPrivateUsage]  # as _acompletion
            self._normalizer._stamp_failed_deployment_id_with_effective_model_info(error, deployment, kwargs)  # pyright: ignore[reportPrivateUsage]  # as _acompletion

    def _exception(self, raised: object) -> BaseException:
        if isinstance(raised, BaseException):
            return raised
        rejection: Final = as_mapping(raised)
        rejection_id: Final = rejection.get("rejection_id")
        if not isinstance(rejection_id, int):
            raise TypeError(f"unexpected router error {raised!r}")
        if rejection_id not in self._rejections:
            self._rejections[rejection_id] = _rejection(rejection)
        return self._rejections[rejection_id]

    def _apply(self, ops: Sequence[Mapping[str, object]]) -> None:
        for op in ops:
            self._apply_one(op)

    def _apply_one(self, op: Mapping[str, object]) -> None:
        kind: Final = op.get("op")
        if kind == "log_retry":
            bucket_id: Final = _int(op, "bucket")
            logged: Final = self._exception(op.get("error"))
            retried: Final = unwrapped(logged) if op.get("original") is True else self._fallback_trigger(logged)
            self._normalizer.log_retry(
                kwargs={"model": op.get("model"), self._metadata_key: self._buckets[bucket_id]},
                e=retried if isinstance(retried, Exception) else Exception(str(retried)),
            )
        elif kind == "open_bucket":
            self._open_bucket(op)
        elif kind == "stamp_retries":
            error: Final = self._exception(op.get("error"))
            error.max_retries = op.get("max_retries")  # pyright: ignore[reportAttributeAccessIssue]  # as async_function_with_retries
            error.num_retries = op.get("num_retries")  # pyright: ignore[reportAttributeAccessIssue]  # as async_function_with_retries
        elif kind == "missing_typed_fallbacks":
            self._explain_missing_typed_fallbacks(op)
        elif kind == "no_fallback_group":
            self._append_message(
                op.get("error"),
                format_no_fallback_group_message(
                    tuple(str(group) for group in as_sequence(op.get("lookup_groups"))),
                    _fallback_entries(self._fallbacks("fallbacks")),
                ),
            )
        elif kind == "fallback_outcome":
            last: Final = op.get("last")
            detail: Final = (
                truncate_fallback_error_detail(redact_string(str(self._exception(last)))) if last is not None else ""
            )
            attempted: Final = op.get("attempted")
            self._append_message(
                op.get("error"),
                format_fallback_outcome_message(
                    str(op.get("model_group")),
                    tuple(as_sequence(attempted)) if attempted is not None else None,
                    detail,
                ),
            )

    def _open_bucket(self, op: Mapping[str, object]) -> None:
        bucket: Final = dict(self._buckets[_int(op, "copy_of")])
        bucket["original_model_group"] = bucket.pop("original_model_group", op.get("original_model_group"))
        bucket.pop("model_group", None)
        bucket.pop("attempted_fallbacks", None)
        bucket["model_group"] = op.get("model_group")
        bucket["attempted_fallbacks"] = op.get("attempted_fallbacks")
        self._buckets[_int(op, "id")] = bucket
        self._hop_kwargs[_int(op, "id")] = {
            "fallback_depth": op.get("attempted_fallbacks"),
            "max_fallbacks": op.get("max_fallbacks"),
        }

    def _explain_missing_typed_fallbacks(self, op: Mapping[str, object]) -> None:
        model_group: Final = op.get("model_group")
        fallbacks: Final = mask_sensitive_structure(self._fallbacks("fallbacks"))
        message: Final = (
            f"model={model_group}. context_window_fallbacks="
            f"{mask_sensitive_structure(self._fallbacks('context_window_fallbacks'))}. fallbacks={fallbacks}.\n\n"
            "Set 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks"
            if op.get("kind") == "context_window"
            else f"model={model_group}. content_policy_fallback="
            f"{mask_sensitive_structure(self._fallbacks('content_policy_fallbacks'))}. fallbacks={fallbacks}.\n\n"
            "Set 'content_policy_fallback' - https://docs.litellm.ai/docs/routing#fallbacks"
        )
        self._append_message(op.get("error"), f"\n{message}", require_message=False)

    def _append_message(self, raised: object, text: str, require_message: bool = True) -> None:
        error: Final = self._fallback_trigger(self._exception(raised))
        message: Final[object] = getattr(error, "message", None)
        if not litellm.expose_router_debug_in_errors or (require_message and not hasattr(error, "message")):
            return
        error.message = f"{message}{text}"  # pyright: ignore[reportAttributeAccessIssue]  # litellm exceptions carry a message


def _run_sync(start: Callable[[], Awaitable[object]]) -> object:
    run: Final = cast(
        Callable[[Callable[[], Awaitable[object]]], object], run_async_function
    )  # cast-ok: untyped helper
    return run(start)


def _handler(operation: Operation) -> Callable[..., Awaitable[object]]:
    """Read at call time, so a patched handler is the one called."""
    name: Final = _GENERIC_HANDLERS[operation]
    return cast(Callable[..., Awaitable[object]], getattr(litellm, name))  # cast-ok: litellm's async handler


def metadata_key_for(operation: Operation) -> Literal["metadata", "litellm_metadata"]:
    return "metadata" if operation == "completion" else "litellm_metadata"


def _rejection(raised: Mapping[str, object]) -> BaseException:
    model: Final = str(raised.get("model"))
    if raised.get("kind") == "no_healthy_deployments":
        return litellm.BadRequestError(
            message=f"You passed in model={model}. {RouterErrors.no_healthy_deployments.value}",
            model=model,
            llm_provider="",
        )
    cooldown_time: Final = raised.get("cooldown_time")
    return RouterRateLimitError(
        model=model,
        cooldown_time=cooldown_time if isinstance(cooldown_time, (int, float)) else 0.0,
        enable_pre_call_checks=raised.get("enable_pre_call_checks") is True,
        cooldown_list=list(as_sequence(raised.get("cooldown_list"))),
        model_ids=tuple(str(model_id) for model_id in as_sequence(raised.get("model_ids"))),
    )


def _int(op: Mapping[str, object], key: str) -> int:
    value: Final = op.get(key)
    if not isinstance(value, int):
        raise TypeError(f"router op field {key!r} is not an int: {value!r}")
    return value


_OBJECTS: Final = TypeAdapter(tuple[object, ...])
_MAPPING: Final = TypeAdapter(Mapping[str, object])


def as_sequence(value: object) -> Sequence[object]:
    return _OBJECTS.validate_python(value) if isinstance(value, (list, tuple)) else ()


def as_mapping(value: object) -> Mapping[str, object]:
    return _MAPPING.validate_python(value) if isinstance(value, Mapping) else {}


def _bucket(value: object) -> Bucket:
    if isinstance(value, dict):
        bucket: Final = cast(Bucket, value)  # cast-ok: the caller's dict, kept by identity for its readers
        return bucket
    return {}


def _fallback_entries(fallbacks: object) -> Sequence[Mapping[str, object]]:
    return tuple(as_mapping(entry) for entry in as_sequence(fallbacks))

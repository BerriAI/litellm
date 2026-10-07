"""One call routed by the Rust router: the Python side of its `Invoke` and `Sleep` host ops.

The native router decides which deployment each attempt goes to, when to retry, back off,
fall back and cool down. Everything that touches Python objects happens here: building an
attempt's kwargs the way `PythonRouter._acompletion` does, calling `litellm.<op>`, classifying
what it raised, and writing the router's metadata and exception updates where Python's
readers look for them.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Final, Literal, Protocol, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter

import litellm
from litellm.integrations.custom_guardrail import is_guardrail_intervention
from litellm.litellm_core_utils.exception_mapping_utils import (
    _get_response_headers,  # pyright: ignore[reportPrivateUsage]  # the header reader the cooldown callback uses
)
from litellm.litellm_core_utils.secret_redaction import redact_string
from litellm.litellm_core_utils.sensitive_data_masker import mask_sensitive_structure
from litellm.router_strategy.complexity_router.context_compaction import compact_to_fit
from litellm.router_utils.add_retry_fallback_headers import (
    add_fallback_headers_to_response,
    add_retry_headers_to_response,
)
from litellm.router_utils.common_utils import (
    format_fallback_outcome_message,
    format_no_fallback_group_message,
    truncate_fallback_error_detail,
)
from litellm.router_utils.cooldown_handlers import (
    is_advisor_orchestration_failure,
    is_background_response_cost_poll_not_found,
    is_caller_timeout_408,
)
from litellm.types.router import RouterErrors, RouterRateLimitError
from litellm.types.utils import ModelResponse
from litellm.utils import (
    _get_retry_after_from_exception_header,  # pyright: ignore[reportPrivateUsage]  # the Retry-After parser retries and cooldowns use
)

Bucket: TypeAlias = MutableMapping[str, object]
Kwargs: TypeAlias = dict[str, object]


class AttemptRouter(Protocol):
    """The `PythonRouter` members an attempt reuses, typed. The Rust backend's normalizer provides them."""

    cache_responses: bool | None
    total_calls: MutableMapping[str, int]  # mutable-ok: PythonRouter's per-model call counters
    success_calls: MutableMapping[str, int]  # mutable-ok: as above
    fail_calls: MutableMapping[str, int]  # mutable-ok: as above

    def get_model_info(self, id: str) -> Kwargs | None: ...

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

    def _update_kwargs_with_deployment(self, deployment: Kwargs, kwargs: Kwargs) -> None: ...

    def _get_async_openai_model_client(self, deployment: Kwargs, kwargs: Kwargs) -> object: ...

    def _get_client(self, deployment: Kwargs, kwargs: Kwargs) -> object: ...

    def _should_raise_content_policy_error(self, model: str, response: ModelResponse, kwargs: Kwargs) -> bool: ...

    def _set_deployment_num_retries_on_exception(self, exception: Exception, deployment: Kwargs) -> None: ...

    def _stamp_failed_deployment_id_with_effective_model_info(
        self, exception: Exception, deployment: Mapping[str, object], kwargs: Mapping[str, object]
    ) -> None: ...


Invoked: TypeAlias = tuple[Literal["ok"], object] | tuple[Literal["error"], BaseException, Mapping[str, object]]

_ROUTER_ONLY_KWARGS: Final = (
    "num_retries",
    "fallbacks",
    "context_window_fallbacks",
    "content_policy_fallbacks",
    "disable_fallbacks",
    "mock_testing_fallbacks",
    "mock_testing_context_fallbacks",
    "mock_testing_content_policy_fallbacks",
)

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
    mock: Literal["fallbacks", "context_window_fallbacks", "content_policy_fallbacks"] | None = None
    model_group: str
    bucket: int
    fallback_depth: int
    model_group_size: int
    attempted_retries: int
    max_retries: int
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
        metadata_key: Literal["metadata", "litellm_metadata"],
        fallbacks: Callable[[str], object],
    ) -> None:
        self._normalizer: Final = normalizer
        self._kwargs: Final = {key: value for key, value in kwargs.items() if key not in _ROUTER_ONLY_KWARGS}
        self._metadata_key: Final = metadata_key
        self._fallbacks: Final = fallbacks
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
            response: Final = await self._acompletion(parsed, kwargs)
        except _AttemptFailed as failed:
            return ("error", failed.error, classify(failed.error, kwargs, callbacks_ran=False))
        except Exception as error:
            return ("error", error, classify(error, kwargs, callbacks_ran=True))
        return ("ok", await self._normalizer.set_response_headers(response, parsed.model_group, kwargs))

    def invoke_sync(self, attempt: Mapping[str, object]) -> Invoked:
        parsed: Final = Attempt.model_validate(attempt)
        kwargs: Final = self._prepare(parsed)
        if parsed.mock is not None:
            return self._mock(parsed, kwargs)
        try:
            response: Final = self._completion(parsed, kwargs)
        except _AttemptFailed as failed:
            return ("error", failed.error, classify(failed.error, kwargs, callbacks_ran=False))
        except Exception as error:
            return ("error", error, classify(error, kwargs, callbacks_ran=True))
        return ("ok", response)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    def sleep_sync(self, seconds: float) -> None:
        time.sleep(seconds)

    def success(self, response: object, outcome: Mapping[str, object], ops: Sequence[Mapping[str, object]]) -> object:
        self._apply(ops)
        parsed: Final = Outcome.model_validate(outcome)
        with_retries: Final = add_retry_headers_to_response(
            response, attempted_retries=parsed.attempted_retries, max_retries=parsed.max_retries
        )
        return add_fallback_headers_to_response(with_retries, attempted_fallbacks=parsed.attempted_fallbacks)

    def failure(self, error: object, ops: Sequence[Mapping[str, object]]) -> BaseException:
        self._apply(ops)
        return self._exception(error)

    def _prepare(self, attempt: Attempt) -> Kwargs:
        self._apply(attempt.ops)
        bucket: Final = self._buckets[attempt.bucket]
        if isinstance(bucket.get("model_group"), str):
            bucket["model_group_size"] = attempt.model_group_size
        bucket["attempted_retries"] = attempt.attempted_retries
        bucket["max_retries"] = attempt.max_retries
        return {
            **{key: value for key, value in self._kwargs.items() if key not in ("model", "messages")},
            **self._hop_kwargs[attempt.bucket],
            self._metadata_key: bucket,
        }

    def _mock(self, attempt: Attempt, kwargs: Mapping[str, object]) -> Invoked:
        group: Final = attempt.model_group
        error: Final[BaseException] = (
            litellm.InternalServerError(
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
            normalizer.success_calls[model_name] += 1
            return response
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

    def _completion(self, attempt: Attempt, kwargs: Kwargs) -> object:
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
            normalizer.success_calls[model_name] += 1
            return response
        except _AttemptFailed as failed:
            self._stamp(failed.error, deployment, kwargs, model_name)
            raise
        except Exception as error:
            self._stamp(error, deployment, kwargs, model_name)
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
            self._normalizer.log_retry(
                kwargs={"model": op.get("model"), self._metadata_key: self._buckets[bucket_id]},
                e=self._as_exception(op.get("error")),
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

    def _as_exception(self, raised: object) -> Exception:
        error: Final = self._exception(raised)
        return error if isinstance(error, Exception) else Exception(str(error))

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
        error: Final = self._exception(raised)
        message: Final[object] = getattr(error, "message", None)
        if not litellm.expose_router_debug_in_errors or (require_message and not hasattr(error, "message")):
            return
        error.message = f"{message}{text}"  # pyright: ignore[reportAttributeAccessIssue]  # litellm exceptions carry a message


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

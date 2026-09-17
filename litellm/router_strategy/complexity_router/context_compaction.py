from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal, NoReturn, Protocol

import anyio
from pydantic import StrictStr, TypeAdapter

import litellm
from litellm.litellm_core_utils.get_provider_specific_headers import ProviderSpecificHeaderUtils
from litellm.litellm_core_utils.prompt_templates.compaction import (
    CompactionHistoryError,
    NativeProtocol,
    compaction_headers,
    native_compaction_payload,
    native_compaction_result,
    split_native_history,
)
from litellm.llms.anthropic.count_tokens.handler import AnthropicCountTokensHandler
from litellm.llms.openai.responses.count_tokens.handler import OpenAICountTokensHandler
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import ProviderSpecificHeader

COMPACTION_STATE_KEY: Final = "_context_window_compaction_state"
MAX_COMPACTION_SECONDS: Final = 120.0
SUMMARY_OUTPUT_TOKENS: Final = 4096
_PROTECTED_BODY_FIELDS: Final = frozenset(
    {
        "model",
        "messages",
        "input",
        "prompt",
        "instructions",
        "system",
        "tools",
        "functions",
        "tool_choice",
        "function_call",
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "thinking",
        "reasoning",
        "context_management",
        "compaction",
        "previous_response_id",
        "conversation",
    }
)
_HEADERS: Final = TypeAdapter(dict[StrictStr, StrictStr])
_COUNT_SCOPED_HEADERS: Final[TypeAdapter[ProviderSpecificHeader | tuple[ProviderSpecificHeader, ...] | None]] = (
    TypeAdapter(ProviderSpecificHeader | tuple[ProviderSpecificHeader, ...] | None)
)


@dataclass(frozen=True, slots=True)
class CompactionFailure:
    message: str


@dataclass(frozen=True, slots=True)
class ModelBudget:
    model: str
    input_limit: int
    output_limit: int
    request_defaults: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    deployment_id: str = ""

    def input_budget(self, output_tokens: int) -> int:
        return self.input_limit - output_tokens - max(64, self.input_limit // 100)


@dataclass(frozen=True, slots=True)
class CompactedRequest:
    field: str
    value: Sequence[Mapping[str, object]]
    extra_headers: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class NativeRequest:
    protocol: NativeProtocol
    payload: Mapping[str, object]
    allowed_deployment_ids: frozenset[str]
    closed: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass(frozen=True, slots=True)
class SummaryMemo:
    identity: str
    response: Mapping[str, object]


class SummaryExecutor(Protocol):
    async def __call__(self, model: str, request: NativeRequest, timeout: float) -> Mapping[str, object]: ...


class TokenCounter(Protocol):
    async def __call__(self, model: str, payload: Mapping[str, object]) -> int: ...


class NativeTokenCounter(Protocol):
    async def __call__(
        self, target: ModelBudget, payload: Mapping[str, object], protocol: NativeProtocol, timeout: float
    ) -> int: ...


_summary_executor: Final[ContextVar[SummaryExecutor | None]] = ContextVar("context_compaction_executor", default=None)
_native_request: Final[ContextVar[NativeRequest | None]] = ContextVar("native_compaction_request", default=None)


@contextmanager
def use_summary_executor(executor: SummaryExecutor) -> Generator[None]:
    token: Final = _summary_executor.set(executor)
    try:
        yield
    finally:
        _summary_executor.reset(token)


def current_summary_executor() -> SummaryExecutor | None:
    return _summary_executor.get()


@contextmanager
def native_request_scope(request: NativeRequest) -> Generator[None]:
    token: Final = _native_request.set(request)
    try:
        yield
    finally:
        request.closed.set()
        _native_request.reset(token)


def current_native_request() -> NativeRequest | None:
    request: Final = _native_request.get()
    return request if request is not None and not request.closed.is_set() else None


class CompactionState:
    def __init__(self, timeout: float = MAX_COMPACTION_SECONDS) -> None:
        self.model: str | None = None
        self.deadline: float = time.monotonic() + min(timeout, MAX_COMPACTION_SECONDS)
        self.calls: int = 0
        self.memo: SummaryMemo | None = None
        self.failure: CompactionFailure | None = None

    def arm(self, model: str) -> None:
        self.model = model

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def limit_timeout(self, timeout: float) -> None:
        self.deadline = min(self.deadline, time.monotonic() + timeout)

    def failed(self, failure: CompactionFailure) -> CompactionFailure:
        self.failure = failure
        return failure

    async def compact(self, request: NativeRequest, identity: str, executor: SummaryExecutor) -> Mapping[str, object]:
        if self.memo is not None and self.memo.identity == identity:
            return self.memo.response
        if self.model is None or self.calls or self.remaining() <= 0:
            raise ValueError("Native compaction exhausted its call or time budget")
        self.calls += 1
        with native_request_scope(request), anyio.fail_after(self.remaining()):
            response: Final = await executor(self.model, request, self.remaining())
        self.memo = SummaryMemo(identity, response)
        return response


def compaction_state(payload: Mapping[str, object] | None) -> CompactionState | None:
    candidate: Final = payload.get(COMPACTION_STATE_KEY) if payload is not None else None
    return candidate if isinstance(candidate, CompactionState) else None


def defers_context_filter(payload: Mapping[str, object] | None) -> bool:
    state: Final = compaction_state(payload)
    return state is not None and state.model is not None


def raise_compaction_failure(failure: CompactionFailure, model: str) -> NoReturn:
    raise litellm.ContextWindowExceededError(message=failure.message, model=model, llm_provider="")


def output_reservation(payload: Mapping[str, object], budget: ModelBudget) -> int | CompactionFailure:
    ceilings: Final = tuple(
        payload[key]
        for key in ("max_output_tokens", "max_completion_tokens", "max_tokens")
        if payload.get(key) is not None
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= budget.output_limit
        for value in ceilings
    ):
        return CompactionFailure("Native compaction requires valid output-token limits for the selected deployment")
    ceiling: Final = max(TypeAdapter(tuple[int, ...]).validate_python(ceilings), default=budget.output_limit)
    thinking: Final = payload.get("thinking")
    thinking_budget: Final = thinking.get("budget_tokens") if isinstance(thinking, Mapping) else None
    if thinking_budget is not None and (
        isinstance(thinking_budget, bool) or not isinstance(thinking_budget, int) or not 0 < thinking_budget < ceiling
    ):
        return CompactionFailure("The selected output ceiling must exceed the thinking budget")
    return ceiling


def validate_compaction_overrides(payload: Mapping[str, object]) -> CompactionFailure | None:
    body: Final = payload.get("extra_body")
    if body is not None and (not isinstance(body, Mapping) or not _PROTECTED_BODY_FIELDS.isdisjoint(body)):
        return CompactionFailure(
            "Native compaction does not allow extra_body to override history or token-budget fields"
        )
    return None


def native_provider(model: str) -> tuple[str, str]:
    resolved: Final = litellm.get_llm_provider(model=model)
    return resolved[0].removeprefix("responses/"), resolved[1]


def supports_native_protocol(model: str, protocol: str) -> bool:
    provider: Final = native_provider(model)[1]
    return (protocol == "responses" and provider == "openai") or (protocol == "messages" and provider == "anthropic")


def _credential(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return get_secret_str(value.removeprefix("os.environ/")) if value.startswith("os.environ/") else value


async def count_native_tokens(
    target: ModelBudget, payload: Mapping[str, object], protocol: NativeProtocol, timeout: float
) -> int:
    model, provider = native_provider(target.model)
    if (protocol, provider) not in (("responses", "openai"), ("messages", "anthropic")):
        raise ValueError("Native token counting requires a matching native provider")
    params: Final = target.request_defaults
    if any(source.get("client") is not None for source in (params, payload)) or any(
        source.get("custom_llm_provider") not in (None, provider) for source in (params, payload)
    ):
        raise ValueError("Native token counting cannot verify a custom client or provider override")
    api_key: Final = _credential(params.get("api_key"))
    api_base: Final = _credential(params.get("api_base"))
    try:
        scoped_headers: Final = _HEADERS.validate_python(
            ProviderSpecificHeaderUtils.get_provider_specific_headers(
                provider_specific_header=_COUNT_SCOPED_HEADERS.validate_python(
                    payload.get("provider_specific_header", params.get("provider_specific_header"))
                ),
                custom_llm_provider=provider,
            )
        )
        header_sources: Final = (
            *(source.get(field) for field in ("headers", "extra_headers") for source in (params, payload)),
            scoped_headers,
        )
        headers: Final = MappingProxyType(
            {
                name.lower(): value
                for source in header_sources
                for name, value in _HEADERS.validate_python(
                    source if source is not None else MappingProxyType({})
                ).items()
            }
        )
        handler: Final = OpenAICountTokensHandler() if protocol == "responses" else AnthropicCountTokensHandler()
        return await handler.count_native_tokens(
            model=model, payload=payload, api_key=api_key, api_base=api_base, headers=headers, timeout=timeout
        )
    except ValueError:
        raise ValueError("Provider-native token counting failed") from None


async def validate_native_recipient(
    request: NativeRequest,
    target: ModelBudget,
    payload: Mapping[str, object],
    timeout: float,
    counter: NativeTokenCounter = count_native_tokens,
) -> CompactionFailure | None:
    if target.deployment_id not in request.allowed_deployment_ids or not supports_native_protocol(
        target.model, request.protocol
    ):
        return CompactionFailure("Native compaction selected an incompatible deployment")
    invalid: Final = validate_compaction_overrides(payload)
    if invalid is not None:
        return invalid
    for key in ("input", "messages", "instructions", "system", "tools", "compaction"):
        if payload.get(key) != request.payload.get(key):
            return CompactionFailure("Compactor deployment defaults override the native request")
    reserve: Final = SUMMARY_OUTPUT_TOKENS if request.protocol == "messages" else target.output_limit
    if request.protocol == "messages" and payload.get("max_tokens") != SUMMARY_OUTPUT_TOKENS:
        return CompactionFailure("Compactor deployment overrides the native output allowance")
    count: Final = await counter(target, payload, request.protocol, timeout)
    if type(count) is not int or not 0 <= count <= target.input_budget(reserve):
        return CompactionFailure("The selected native compactor cannot fit the complete request")
    return None


def _has_native_state(payload: Mapping[str, object]) -> bool:
    items: Final = payload.get("input") or payload.get("messages")
    if not isinstance(items, (list, tuple)):
        return False
    return any(
        isinstance(item, Mapping)
        and (
            item.get("type") in ("compaction", "reasoning", "item_reference")
            or isinstance(item.get("content"), (list, tuple))
            and any(isinstance(block, Mapping) and block.get("type") == "compaction" for block in item["content"])
        )
        for item in items
    )


async def prepare_compaction(
    payload: Mapping[str, object],
    target: ModelBudget,
    state: CompactionState,
    summary_budgets: tuple[ModelBudget, ...],
    executor: SummaryExecutor,
    counter: TokenCounter,
    protocol: Literal["chat", "responses", "messages"] = "chat",
    native_counter: NativeTokenCounter = count_native_tokens,
) -> CompactedRequest | CompactionFailure | None:
    invalid: Final = validate_compaction_overrides(payload)
    if invalid is not None:
        return invalid
    reserve: Final = output_reservation(payload, target)
    if isinstance(reserve, CompactionFailure):
        return reserve
    input_budget: Final = target.input_budget(reserve)
    try:
        if not _has_native_state(payload) and await counter(target.model, payload) <= input_budget:
            return None
        if protocol == "chat" or not supports_native_protocol(target.model, protocol):
            return CompactionFailure(
                "Native compaction requires Responses/OpenAI or Messages/Anthropic; this route is unsupported"
            )
        if payload.get("previous_response_id") is not None or payload.get("conversation") is not None:
            return CompactionFailure("Native overflow compaction requires the complete conversation history")
        with anyio.fail_after(state.remaining()):
            input_tokens: Final = await native_counter(target, payload, protocol, state.remaining())
            if type(input_tokens) is not int or input_tokens < 0:
                return CompactionFailure("Native token counting returned an invalid input count")
            if input_tokens <= input_budget:
                return None
            history: Final = split_native_history(payload, protocol)
            if isinstance(history, CompactionHistoryError):
                return CompactionFailure(history.message)
            eligible: Final = tuple(
                budget
                for budget in summary_budgets
                if budget.deployment_id
                and supports_native_protocol(budget.model, protocol)
                and budget.output_limit >= SUMMARY_OUTPUT_TOKENS
                and input_tokens
                <= budget.input_budget(SUMMARY_OUTPUT_TOKENS if protocol == "messages" else budget.output_limit)
            )
            if not eligible:
                return CompactionFailure("No compatible configured native compactor can fit the complete input")
            body: Final = native_compaction_payload(history, payload, protocol)
            request: Final = NativeRequest(protocol, body, frozenset(budget.deployment_id for budget in eligible))
            identity: Final = hashlib.sha256(
                json.dumps(
                    MappingProxyType(
                        {"model": state.model, "protocol": protocol, "payload": body, "tools": payload.get("tools")}
                    ),
                    sort_keys=True,
                    default=dict,
                ).encode()
            ).hexdigest()
            response: Final = await state.compact(request, identity, executor)
            result: Final = native_compaction_result(response, history, protocol)
            if isinstance(result, CompactionHistoryError):
                return CompactionFailure(result.message)
            headers: Final = (
                compaction_headers(payload.get("extra_headers")) if protocol == "messages" else MappingProxyType({})
            )
            prepared: Final = MappingProxyType(
                {
                    **payload,
                    history.field: result,
                    "extra_headers": headers or payload.get("extra_headers") or MappingProxyType({}),
                }
            )
            final_count: Final = await native_counter(target, prepared, protocol, state.remaining())
            if type(final_count) is not int or not 0 <= final_count <= input_budget:
                return CompactionFailure("Native compaction output does not fit the pinned target")
            return CompactedRequest(history.field, result, headers)
    except Exception as error:  # noqa: BLE001  # provider failures become sanitized context errors at this boundary
        return state.failed(
            CompactionFailure(f"Native compaction failed ({type(error).__name__}); original history retained")
        )

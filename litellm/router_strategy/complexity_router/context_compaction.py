from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from itertools import takewhile
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NoReturn, TypeAlias
from typing import Protocol as CallableProtocol

from pydantic import TypeAdapter

from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
    inherit_message_logging_privacy,
    initialize_standard_callback_dynamic_params,
)
from litellm.litellm_core_utils.internal_call_metadata import (
    parent_session_kwargs,
    sanitized_forwardable_call_metadata,
)
from litellm.litellm_core_utils.redact_messages import (
    should_redact_message_logging,  # pyright: ignore[reportUnknownVariableType]  # legacy owner receives validated call details
)
from litellm.llms.anthropic import compaction as anthropic_compaction
from litellm.router_strategy.complexity_router.config import ContextCompactionConfig

if TYPE_CHECKING:
    from litellm.router import Router

Protocol: TypeAlias = Literal["chat", "messages"]


class CompactionExecutor(CallableProtocol):
    async def __call__(self, protocol: Protocol, payload: Mapping[str, object]) -> Mapping[str, object]: ...


compaction_executor: Final[ContextVar[CompactionExecutor | None]] = ContextVar("compaction_executor", default=None)
_native_child: Final[ContextVar[bool]] = ContextVar("native_compaction_child", default=False)
_STATE_KEY: Final = "_context_compaction_state"
_MAPPING: Final = TypeAdapter(Mapping[str, object])
_MESSAGES: Final = TypeAdapter(list[dict[str, object]])
_DICT: Final = TypeAdapter(dict[str, object])
_INPUT: Final[TypeAdapter[str | list[object] | None]] = TypeAdapter(
    str | list[object] | None
)  # mutable-ok: Responses normalization requires a JSON list
_SEQUENCE: Final = TypeAdapter(tuple[object, ...])
_EMPTY: Final[Mapping[str, object]] = MappingProxyType({})


@dataclass(slots=True, repr=False)
class CompactionState:
    config: ContextCompactionConfig | None = None
    candidates: tuple[str, ...] = ()
    summary: tuple[str, asyncio.Task[str]] | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class InputBudget:
    window: int | None
    available: int | None


@contextmanager
def native_compaction_call() -> Generator[None]:
    token: Final = _native_child.set(True)
    try:
        yield
    finally:
        _native_child.reset(token)


def initialize_compaction_state(kwargs: Mapping[str, object]) -> CompactionState:
    existing: Final = kwargs.get(_STATE_KEY)
    return existing if isinstance(existing, CompactionState) else CompactionState()


def arm_compaction(
    kwargs: Mapping[str, object],
    config: ContextCompactionConfig | Literal[False] | None,
    candidates: tuple[str, ...] = (),
) -> None:
    state: Final = kwargs.get(_STATE_KEY)
    if isinstance(state, CompactionState):
        state.config = config if isinstance(config, ContextCompactionConfig) else None
        state.candidates = candidates


def reject_recursive_compactor(model: str) -> None:
    if _native_child.get():
        _reject(model, "The compactor must be a regular model group, not an auto-router")


def is_native_compaction_call() -> bool:
    return _native_child.get()


def compaction_pending(kwargs: Mapping[str, object] | None) -> bool:
    state: Final = kwargs.get(_STATE_KEY) if kwargs is not None else None
    return isinstance(state, CompactionState) and state.config is not None


def _reject(model: str, reason: str) -> NoReturn:
    from litellm.exceptions import BadRequestError

    raise BadRequestError(message=f"Context compaction: {reason}", model=model, llm_provider="")


def _positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _objects(value: object) -> tuple[Mapping[str, object], ...]:
    return (
        tuple(
            _MAPPING.validate_python(block) for block in _SEQUENCE.validate_python(value) if isinstance(block, Mapping)
        )
        if isinstance(value, (list, tuple))
        else ()
    )


def _blocks(message: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return _objects(message.get("content"))


def _tool_ids(messages: Sequence[Mapping[str, object]], *, results: bool) -> frozenset[str]:
    return frozenset(
        str(identifier)
        for message in messages
        for identifier in (
            *((message.get("tool_call_id"),) if results and message.get("role") == "tool" else ()),
            *(
                block.get("tool_use_id" if results else "id")
                for block in _blocks(message)
                if block.get("type") == ("tool_result" if results else "tool_use")
            ),
            *(call.get("id") for call in _objects(message.get("tool_calls")) if not results),
        )
        if identifier is not None
    )


def _history(
    messages: Sequence[Mapping[str, object]], model: str
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    instructions: Final = tuple(takewhile(lambda message: message.get("role") in ("system", "developer"), messages))
    conversation: Final = tuple(messages[len(instructions) :])
    if any(message.get("role") in ("system", "developer") for message in conversation):
        _reject(model, "Mid-conversation instructions cannot be compacted")
    split: Final = next(
        (
            index
            for index in range(len(conversation) - 1, -1, -1)
            if conversation[index].get("role") == "user"
            and not any(block.get("type") == "tool_result" for block in _blocks(conversation[index]))
        ),
        0,
    )
    prefix: Final = conversation[:split]
    if not prefix or _tool_ids(prefix, results=False) != _tool_ids(prefix, results=True):
        _reject(model, "No closed older conversation is available to compact without changing the latest request")
    return instructions, prefix, conversation[split:]


async def _count(router: Router, payload: Mapping[str, object]) -> int:
    return await asyncio.to_thread(
        router._count_pre_call_check_tokens,  # pyright: ignore[reportPrivateUsage]  # Reuse Router's shared admission counter
        messages=_MESSAGES.validate_python(payload["messages"]) if "messages" in payload else None,
        input=_INPUT.validate_python(payload.get("input")),
        request_kwargs=payload,
    )


def _budget(
    router: Router, deployment: Mapping[str, object], payload: Mapping[str, object], ratio: float
) -> InputBudget:
    model: Final = str(payload.get("model", ""))
    info: Final = _MAPPING.validate_python(
        router.get_router_model_info(deployment=_DICT.validate_python(deployment), received_model_name=model)
    )
    window_value: Final = info.get("max_input_tokens")
    window: Final = window_value if isinstance(window_value, int) else None
    output_value: Final = next(
        (
            payload[key]
            for key in ("max_completion_tokens", "max_output_tokens", "max_tokens")
            if payload.get(key) is not None
        ),
        info.get("max_output_tokens"),
    )
    output: Final = _positive_int(output_value)
    if output_value is not None and output is None:
        _reject(model, "The output allowance must be a positive integer")
    return InputBudget(window, int(window * ratio) - output if window is not None and output is not None else None)


async def _compactor_model(router: Router, state: CompactionState, payload: Mapping[str, object]) -> str:
    config: Final = state.config
    if config is not None and config.model is not None:
        return config.model
    needed: Final = await _count(router, payload)
    for candidate in state.candidates:
        deployments: Final = tuple(router.get_model_list(model_name=candidate) or ())
        if deployments and all(
            anthropic_compaction.supports_native_compaction(_MAPPING.validate_python(deployment["litellm_params"]))
            and (budget := _budget(router, deployment, payload, 0.9)).available is not None
            and needed <= budget.available
            for deployment in deployments
        ):
            return candidate
    return _reject(
        str(payload.get("model", "")),
        "No configured tier model supports native compaction with enough context; set context_compaction.model",
    )


async def _generate_summary(
    router: Router, protocol: Protocol, payload: Mapping[str, object], timeout_seconds: float
) -> str:
    executor: Final = compaction_executor.get()
    with native_compaction_call():
        response: Final = await asyncio.wait_for(
            executor(protocol, payload)
            if executor is not None
            else anthropic_compaction.dispatch(router, protocol, payload),
            timeout=timeout_seconds,
        )
    return anthropic_compaction.extract_summary(protocol, response, str(payload["model"]))


async def compact_to_fit(
    router: Router,
    deployment: Mapping[str, object],
    payload: Mapping[str, object],
    protocol: Protocol | None,
) -> Mapping[str, object]:
    state: Final = payload.get(_STATE_KEY)
    config: Final = state.config if isinstance(state, CompactionState) else None
    if config is None and not _native_child.get():
        return payload
    model: Final = str(payload.get("model", ""))
    limits: Final = _budget(router, deployment, payload, config.trigger_ratio if config is not None else 0.9)
    budget: Final = limits.available
    if budget is None or budget <= 0:
        if (
            config is not None
            and config.model is None
            and (limits.window is None or await _count(router, payload) <= limits.window)
        ):
            return payload
        _reject(model, "A known input window and a smaller output allowance are required")
    if _native_child.get():
        anthropic_compaction.validate_compactor_defaults(payload)
        if await _count(router, payload) > budget:
            _reject(model, "The selected compactor cannot fit the history and output allowance")
        return payload
    if await _count(router, payload) <= budget:
        return payload
    if protocol is None or config is None or not isinstance(state, CompactionState):
        _reject(model, "Only Chat Completions and Anthropic Messages support compact-to-fit")
    if any(payload.get(key) is not None for key in anthropic_compaction.CLIENT_COMPACTION_PARAMS):
        _reject(model, "Do not combine compact-to-fit with client-managed compaction")
    messages: Final = _MESSAGES.validate_python(payload["messages"])
    instructions, prefix, tail = _history(messages, model)
    if any(
        message.get("thinking_blocks")
        or any(block.get("type") in ("thinking", "redacted_thinking") for block in _blocks(message))
        for message in tail
    ):
        _reject(model, "Retained native reasoning cannot be replayed after text compaction")
    retained: Final = MappingProxyType({**payload, "messages": _MESSAGES.validate_python((*instructions, *tail))})
    if await _count(router, retained) >= budget:
        _reject(model, "The retained instructions, tools and latest turn leave no room for a summary")
    parent_metadata: Final = _MAPPING.validate_python(
        payload.get("litellm_metadata") or payload.get("metadata") or _EMPTY
    )
    forwarded_metadata: Final = sanitized_forwardable_call_metadata(parent_metadata, "autorouter_compaction")
    logging_disabled: Final = should_redact_message_logging(
        _DICT.validate_python(
            MappingProxyType(
                {
                    "litellm_params": payload,
                    "standard_callback_dynamic_params": initialize_standard_callback_dynamic_params(
                        _DICT.validate_python(payload)
                    ),
                }
            )
        )
    )
    metadata: Final = _DICT.validate_python(
        MappingProxyType({key: value for key, value in forwarded_metadata.items() if key != "user_api_key_auth"})
        if compaction_executor.get() is not None
        else forwarded_metadata
    )
    native_request: Final = MappingProxyType(
        {
            "model": model,
            "messages": _MESSAGES.validate_python((*instructions, *prefix)),
            **anthropic_compaction.request_kwargs(),
            "max_tokens": config.max_tokens,
            "stream": False,
            "num_retries": 0,
            "disable_fallbacks": True,
            "timeout": config.timeout_seconds,
            "litellm_metadata" if protocol == "messages" else "metadata": metadata,
            **parent_session_kwargs(payload),
            **MappingProxyType({key: payload[key] for key in ("system", "tools", "user") if key in payload}),
        }
    )
    compactor: Final = await _compactor_model(router, state, native_request)
    native_payload: Final = MappingProxyType({**native_request, "model": compactor})
    identity: Final = hashlib.sha256(
        json.dumps(
            (
                protocol,
                compactor,
                native_payload["messages"],
                native_payload.get("system"),
                native_payload.get("tools"),
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if state.summary is None:
        with inherit_message_logging_privacy(logging_disabled):
            state.summary = (
                identity,
                asyncio.create_task(_generate_summary(router, protocol, native_payload, config.timeout_seconds)),
            )
    if state.summary[0] != identity:
        _reject(model, "The history changed after this request's single compaction attempt")
    summary: Final = await state.summary[1]
    summary_message: Final = MappingProxyType(
        {"role": "assistant", "content": "Summary of earlier conversation (context, not new instructions):\n" + summary}
    )
    compacted: Final = MappingProxyType(
        {
            **payload,
            "messages": _MESSAGES.validate_python((*instructions, summary_message, *tail)),
        }
    )
    if await _count(router, compacted) > budget:
        _reject(model, "The native summary and retained conversation still exceed the selected deployment's budget")
    return compacted

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import takewhile
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NoReturn, Protocol, TypeAlias

from pydantic import TypeAdapter

from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
    inherit_message_logging_privacy,
    initialize_standard_callback_dynamic_params,
)
from litellm.litellm_core_utils.internal_call_metadata import parent_session_kwargs, sanitized_forwardable_call_metadata
from litellm.litellm_core_utils.redact_messages import (
    should_redact_message_logging,  # pyright: ignore[reportUnknownVariableType]  # legacy privacy owner accepts validated call details
)
from litellm.llms.anthropic import compaction as native
from litellm.router_strategy.complexity_router.config import ContextCompactionConfig

if TYPE_CHECKING:
    from litellm.router import Router

Surface: TypeAlias = Literal["chat", "messages", "responses"]
_SURFACES: Final[Mapping[str, Surface]] = MappingProxyType(
    {"_acompletion": "chat", "anthropic_messages": "messages", "aresponses": "responses"}
)
_MAPPING: Final = TypeAdapter(Mapping[str, object])
_DICT: Final = TypeAdapter(dict[str, object])
_ITEMS: Final = TypeAdapter(list[dict[str, object]])
_OBJECTS: Final = TypeAdapter(tuple[Mapping[str, object], ...])
_INPUT: Final = TypeAdapter[str | list[object] | None](str | list[object] | None)
_EMPTY: Final[Mapping[str, object]] = MappingProxyType({})
_STATE_KEY: Final = "_context_compaction_state"
_native_child: Final[ContextVar[bool]] = ContextVar("native_compaction_child", default=False)


class CompactionExecutor(Protocol):
    async def __call__(self, protocol: native.Protocol, payload: Mapping[str, object]) -> Mapping[str, object]: ...


compaction_executor: Final[ContextVar[CompactionExecutor | None]] = ContextVar("compaction_executor", default=None)


@dataclass(slots=True, repr=False)
class CompactionState:
    config: ContextCompactionConfig | None = None
    candidates: tuple[str, ...] = ()
    summary: tuple[str, asyncio.Task[str]] | None = None


def surface_for_call(function_name: str) -> Surface | None:
    return _SURFACES.get(function_name)


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
        state.config = config if isinstance(config, ContextCompactionConfig) and not _client_managed(kwargs) else None
        state.candidates = candidates


def _client_managed(payload: Mapping[str, object]) -> bool:
    return any(
        payload.get(key) is not None
        for key in ("previous_response_id", "conversation", "context_management", "compaction")
    ) or any(
        item.get("type") in ("reasoning", "compaction", "item_reference")
        or item.get("encrypted_content") is not None
        or any(block.get("type") == "encrypted_content" for block in _blocks(item))
        for item in _blocks(payload, "input")
    )


def is_native_compaction_call() -> bool:
    return _native_child.get()


def reject_recursive_compactor(model: str) -> None:
    if _native_child.get():
        _reject(model, "The compactor must be a regular model group, not an auto-router")


def compaction_pending(kwargs: Mapping[str, object] | None) -> bool:
    state: Final = kwargs.get(_STATE_KEY) if kwargs is not None else None
    return isinstance(state, CompactionState) and state.config is not None


def _reject(model: str, reason: str) -> NoReturn:
    from litellm.exceptions import BadRequestError

    raise BadRequestError(message=f"Context compaction: {reason}", model=model, llm_provider="")


def _blocks(item: Mapping[str, object], key: str = "content") -> tuple[Mapping[str, object], ...]:
    value: Final = item.get(key)
    return _OBJECTS.validate_python(value) if isinstance(value, (list, tuple)) else ()


def _tool_ids(items: Sequence[Mapping[str, object]], *, results: bool) -> tuple[str, ...]:
    return tuple(
        identifier if isinstance(identifier, str) else ""
        for item in items
        for identifier in (
            *((item.get("tool_call_id"),) if results and item.get("role") == "tool" else ()),
            *(
                (item.get("call_id"),)
                if item.get("type") == ("function_call_output" if results else "function_call")
                else ()
            ),
            *(
                block.get("tool_use_id" if results else "id")
                for block in _blocks(item)
                if block.get("type") == ("tool_result" if results else "tool_use")
            ),
            *(call.get("id") for call in _blocks(item, "tool_calls") if not results),
        )
    )


def _history(
    items: Sequence[Mapping[str, object]], model: str
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    instructions: Final = tuple(takewhile(lambda item: item.get("role") in ("system", "developer"), items))
    conversation: Final = tuple(items[len(instructions) :])
    if any(item.get("role") in ("system", "developer") for item in conversation):
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
    calls: Final = _tool_ids(prefix, results=False)
    results: Final = _tool_ids(prefix, results=True)
    if (
        not prefix
        or "" in calls
        or "" in results
        or len(calls) != len(frozenset(calls))
        or sorted(calls) != sorted(results)
    ):
        _reject(model, "No closed older conversation is available without changing the latest request")
    return instructions, prefix, conversation[split:]


async def _count(router: Router, payload: Mapping[str, object]) -> int:
    return await asyncio.to_thread(
        router._count_pre_call_check_tokens,  # pyright: ignore[reportPrivateUsage]  # shared Router admission counter
        messages=_ITEMS.validate_python(payload["messages"]) if "messages" in payload else None,
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
    raw_window: Final = info.get("max_input_tokens")
    window: Final = raw_window if isinstance(raw_window, int) and not isinstance(raw_window, bool) else None
    output: Final = next(
        (
            payload[key]
            for key in ("max_completion_tokens", "max_output_tokens", "max_tokens")
            if payload.get(key) is not None
        ),
        info.get("max_output_tokens"),
    )
    if output is not None and (not isinstance(output, int) or isinstance(output, bool) or output <= 0):
        _reject(model, "The output allowance must be a positive integer")
    return InputBudget(window, int(window * ratio) - output if window is not None and isinstance(output, int) else None)


async def _compactor_model(router: Router, state: CompactionState, payload: Mapping[str, object]) -> str:
    needed: Final = await _count(router, payload)
    candidates: Final = (
        (state.config.model,) if state.config is not None and state.config.model is not None else state.candidates
    )
    selected: Final = next(
        (
            candidate
            for candidate in candidates
            if (deployments := tuple(router.get_model_list(model_name=candidate) or ()))
            and all(
                native.supports_native_compaction(params := _MAPPING.validate_python(deployment["litellm_params"]))
                and native.compatible_defaults(params)
                and (budget := _budget(router, deployment, payload, 0.9)).available is not None
                and needed <= budget.available
                for deployment in deployments
            )
        ),
        None,
    )
    return (
        selected
        if selected is not None
        else _reject(
            str(payload["model"]),
            "No configured compactor supports native compaction with enough context and compatible defaults",
        )
    )


def _native_prefix(payload: Mapping[str, object], surface: Surface) -> Mapping[str, object]:
    if surface != "responses":
        return payload
    from openai.types.responses.response_create_params import ResponseInputParam

    from litellm.responses.litellm_completion_transformation.transformation import LiteLLMCompletionResponsesConfig

    messages: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=TypeAdapter(ResponseInputParam).validate_python(payload["input"]),
        responses_api_request=_DICT.validate_python(payload),
    )
    tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
        _OBJECTS.validate_python(payload.get("tools") or ())
    )
    return MappingProxyType({**payload, "messages": _ITEMS.validate_python(messages), "tools": tools})


async def _generate_summary(
    router: Router, protocol: native.Protocol, payload: Mapping[str, object], timeout: float
) -> str:
    executor: Final = compaction_executor.get()
    with native_compaction_call():
        response: Final = await asyncio.wait_for(
            executor(protocol, payload) if executor is not None else native.dispatch(router, protocol, payload),
            timeout=timeout,
        )
    summary: Final = native.extract_summary(protocol, response)
    return (
        summary
        if summary is not None
        else _reject(str(payload["model"]), "The provider did not return one complete native compaction block")
    )


async def compact_to_fit(
    router: Router, deployment: Mapping[str, object], payload: Mapping[str, object], surface: Surface | None
) -> Mapping[str, object]:
    state: Final = payload.get(_STATE_KEY)
    config: Final = state.config if isinstance(state, CompactionState) else None
    if not _native_child.get() and (config is None or _client_managed(payload)):
        return payload
    model: Final = str(payload["model"])
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
        if (
            not native.compatible_defaults(payload)
            or not native.supports_native_compaction(payload)
            or await _count(router, payload) > budget
        ):
            _reject(model, "The selected compactor's effective request is incompatible or exceeds its input budget")
        return payload
    if await _count(router, payload) <= budget:
        return payload
    if surface is None or config is None or not isinstance(state, CompactionState):
        _reject(model, "This request surface cannot be compacted")
    key: Final = "input" if surface == "responses" else "messages"
    raw_items: Final = payload[key]
    if isinstance(raw_items, str):
        _reject(model, "A single user input cannot be compacted without changing the latest request")
    items: Final = _ITEMS.validate_python(raw_items)
    if surface == "responses" and any(
        item.get("type", "message") not in ("message", "function_call", "function_call_output") for item in items
    ):
        _reject(model, "Opaque or provider-managed Responses items require client-managed native compaction")
    if surface == "responses" and (
        any(block.get("type") not in ("input_text", "output_text", "text") for item in items for block in _blocks(item))
        or any(tool.get("type") != "function" for tool in _blocks(payload, "tools"))
    ):
        _reject(model, "Only text history and ordinary function tools support portable Responses compaction")
    instructions, prefix, tail = _history(items, model)
    if any(
        item.get("thinking_blocks")
        or any(block.get("type") in ("thinking", "redacted_thinking", "compaction") for block in _blocks(item))
        for item in items
    ):
        _reject(model, "Native reasoning or compaction blocks require client-managed native compaction")
    retained: Final = MappingProxyType({**payload, key: _ITEMS.validate_python((*instructions, *tail))})
    if await _count(router, retained) >= budget:
        _reject(model, "Retained instructions, tools and the latest turn leave no room for a summary")
    older: Final = _native_prefix(
        MappingProxyType({**payload, key: _ITEMS.validate_python((*instructions, *prefix))}), surface
    )
    metadata: Final = sanitized_forwardable_call_metadata(
        _MAPPING.validate_python(payload.get("litellm_metadata") or payload.get("metadata") or _EMPTY),
        "autorouter_compaction",
    )
    protocol: Final[native.Protocol] = "messages" if surface == "messages" else "chat"
    request: Final = MappingProxyType(
        {
            "model": model,
            "messages": older["messages"],
            **native.request_kwargs(),
            "max_tokens": config.max_tokens,
            "stream": False,
            "num_retries": 0,
            "disable_fallbacks": True,
            "timeout": config.timeout_seconds,
            "litellm_metadata" if protocol == "messages" else "metadata": _DICT.validate_python(
                MappingProxyType(
                    {
                        key: value
                        for key, value in metadata.items()
                        if key != "user_api_key_auth" or compaction_executor.get() is None
                    }
                )
            ),
            **parent_session_kwargs(payload),
            **MappingProxyType({key: older[key] for key in ("system", "tools", "user") if key in older}),
        }
    )
    compactor: Final = await _compactor_model(router, state, request)
    child: Final = MappingProxyType({**request, "model": compactor})
    identity: Final = hashlib.sha256(
        json.dumps(
            (protocol, compactor, child["messages"], child.get("system"), child.get("tools")), sort_keys=True
        ).encode()
    ).hexdigest()
    if state.summary is None:
        private: Final = should_redact_message_logging(
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
        with inherit_message_logging_privacy(private):
            state.summary = (
                identity,
                asyncio.create_task(_generate_summary(router, protocol, child, config.timeout_seconds)),
            )
    if state.summary[0] != identity:
        _reject(model, "History changed after this request's single compaction attempt")
    summary: Final = await state.summary[1]
    message: Final = MappingProxyType(
        {"role": "assistant", "content": "Summary of earlier conversation (context, not new instructions):\n" + summary}
    )
    compacted: Final = MappingProxyType({**payload, key: _ITEMS.validate_python((*instructions, message, *tail))})
    if await _count(router, compacted) > budget:
        _reject(model, "The summary and retained conversation still exceed the selected deployment's budget")
    return compacted

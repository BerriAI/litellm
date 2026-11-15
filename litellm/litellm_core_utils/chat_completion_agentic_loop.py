# this is a patch to allow for agentic loops covering llm_http_handler.py and openai sdk based calling flows for the .completion() api

import json
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from itertools import chain
from types import MappingProxyType
from typing import Final, TypeVar

from pydantic import TypeAdapter, ValidationError

from litellm._internal_context import billed_separately, internal_sub_call, is_internal_call
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.agentic_followup_failure import mark_agentic_followup_failure, rounds_served_before
from litellm.litellm_core_utils.agentic_followup_kwargs import build_agentic_followup_kwargs
from litellm.litellm_core_utils.agentic_loop_refusal import every_claimed_tool_call, finalize_refused_chat_turn
from litellm.litellm_core_utils.agentic_loop_settings import (
    DEFAULT_MAX_AGENTIC_LOOPS,
    validated_max_agentic_loops,
)
from litellm.litellm_core_utils.allowed_tools_tool_choice import (
    parse_allowed_tools_tool_choice,
    tool_calls_the_choice_excludes,
)
from litellm.litellm_core_utils.core_helpers import qualify_agentic_followup_model
from litellm.litellm_core_utils.get_litellm_params import AGENTIC_LOOP_TOOL_CHOICE_KEY
from litellm.litellm_core_utils.hidden_params import get_hidden_params, served_from_cache, set_hidden_params
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObject
from litellm.litellm_core_utils.litellm_logging import UncachedPrice
from litellm.litellm_core_utils.llm_cost_calc.guardrail_cost import is_str_keyed_dict, object_items
from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
from litellm.responses.mcp.parked_stream_dispatch import SERVED_FROM_CACHE
from litellm.responses.mcp.round_billing import billed_for_every_round, summed_cost_breakdown
from litellm.types.integrations.custom_logger import (
    CHAT_COMPLETION_AGENTIC_SURFACE,
    NON_CODE_INTERPRETER_INTERCEPTION_INTERNAL_PREFIXES,
    AgenticLoopLimitError,
    AgenticLoopPlan,
    AgenticLoopRequestPatch,
    AgenticLoopSafetyError,
    converted_stream_requested,
    is_interception_internal_key,
    stashed_stream_options,
)
from litellm.types.utils import (
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageToolCall,
    ModelResponse,
)
from litellm.utils import CustomStreamWrapper

_DispatchResult: Final = TypeVar("_DispatchResult")

_FOLLOWUP_INTERNAL_PARAMS: Final = frozenset(
    (
        "acompletion",
        "litellm_logging_obj",
        "custom_llm_provider",
        "model_alias_map",
        "stream_response",
        "custom_prompt_dict",
        "_agentic_loop_api_surface",
    )
)


def _first_round_bills_itself(logging_obj: object) -> bool:
    return (
        isinstance(logging_obj, LiteLLMLoggingObject)
        and logging_obj.bridged_chat_request is not None
        and logging_obj.stream is not True
        and not is_internal_call.get()
        and logging_obj.async_success_log_enqueued
    )


def _follow_ups_bill_themselves(logging_obj: object, request_kwargs: Mapping[str, object]) -> bool:
    return _first_round_bills_itself(logging_obj) and not converted_stream_requested(request_kwargs)


async def bill_first_round_on_failure(
    dispatch: Awaitable[_DispatchResult], *, logging_obj: object, response: object, stream: bool
) -> _DispatchResult:
    try:
        return await dispatch
    except Exception as error:
        if stream or not isinstance(logging_obj, LiteLLMLoggingObject):
            mark_agentic_followup_failure(error, rounds_served_before(error))
            raise
        this_round: Final = None if _first_round_bills_itself(logging_obj) else logging_obj.served_round(response)
        served_rounds: Final = (*rounds_served_before(error), *(() if this_round is None else (this_round,)))
        mark_agentic_followup_failure(error, served_rounds)
        logging_obj.record_served_rounds_for_failure(served_rounds)
        raise


def _gate_overridden(callback: CustomLogger) -> bool:
    return type(callback).async_should_run_agentic_loop is not CustomLogger.async_should_run_agentic_loop


def _build_plan_overridden(callback: CustomLogger) -> bool:
    return type(callback).async_build_agentic_loop_plan is not CustomLogger.async_build_agentic_loop_plan


def _chat_build_plan_overridden(callback: CustomLogger) -> bool:
    return (
        type(callback).async_build_chat_completion_agentic_loop_plan
        is not CustomLogger.async_build_chat_completion_agentic_loop_plan
    )


def runs_its_own_chat_completion_loop(callback: CustomLogger) -> bool:
    return callback.wants_chat_completion_hooks_on_bridged_chat


def _post_hook_overridden(callback: CustomLogger) -> bool:
    return (
        type(callback).async_post_agentic_loop_response_hook is not CustomLogger.async_post_agentic_loop_response_hook
    )


_STREAM_OPTIONS_ADAPTER: Final[TypeAdapter[Mapping[str, object] | None]] = TypeAdapter(Mapping[str, object] | None)
_HOOK_RESPONSE: Final = TypeAdapter(object)
_PRICE_OF_EVERY_ROUND: Final = "agentic_loop_price_of_every_round"


def _plan_response_override(plan: AgenticLoopPlan) -> object | None:
    return plan.response_override


def _stashed_stream_options(kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
    try:
        return _STREAM_OPTIONS_ADAPTER.validate_python(stashed_stream_options(kwargs))
    except ValidationError:
        return None


def _coerce_int(value: object, default: int) -> int:
    return int(value) if isinstance(value, (int, str)) else default


def _agentic_loop_settings(kwargs: Mapping[str, object]) -> tuple[int, int, tuple[str, ...]]:
    depth: Final = _coerce_int(kwargs.get("_agentic_loop_depth"), 0)
    configured: Final = validated_max_agentic_loops(
        kwargs.get("max_agentic_loops"), field="litellm_params.max_agentic_loops"
    )
    max_loops: Final = DEFAULT_MAX_AGENTIC_LOOPS if configured is None else configured
    raw_fingerprints: Final = kwargs.get("_agentic_loop_fingerprints")
    fingerprints: Final = tuple(str(fp) for fp in raw_fingerprints) if isinstance(raw_fingerprints, list) else ()
    return depth, max_loops, fingerprints


def _fingerprint_tools(tool_calls: object) -> str:
    try:
        return json.dumps(tool_calls, sort_keys=True, default=str)
    except Exception:
        return str(tool_calls)


def _check_agentic_loop_safety(
    tool_calls: object,
    fingerprints: Sequence[str],
    depth: int,
    max_loops: int,
    model: str,
) -> str:
    fingerprint: Final = _fingerprint_tools(tool_calls)
    if fingerprint in fingerprints:
        raise AgenticLoopSafetyError("Agentic loop detected repeated tool-call fingerprint; aborting rerun")
    if depth >= max_loops:
        raise AgenticLoopLimitError(f"Exceeded max_agentic_loops={max_loops} for model={model}")
    return fingerprint


def _converted_stream_replay(
    response: object,
    *,
    kwargs: Mapping[str, object],
    depth: int,
    model: str,
    custom_llm_provider: str,
    logging_obj: object,
) -> CustomStreamWrapper | None:
    if depth or not converted_stream_requested(kwargs):
        return None
    if isinstance(response, CustomStreamWrapper):
        return response
    if not isinstance(response, ModelResponse) or not isinstance(logging_obj, LiteLLMLoggingObject):
        return None
    every_round: Final = _price_of_every_round(response)
    if every_round is not None:
        logging_obj.record_every_round_breakdown(every_round.cost_breakdown)

    return CustomStreamWrapper(
        completion_stream=MockResponseIterator(model_response=response),
        model=model,
        custom_llm_provider=custom_llm_provider,
        logging_obj=logging_obj,
        stream_options=_stashed_stream_options(kwargs),
        response_cost=None if every_round is None else every_round.cost,
    )


def _end_refused_turn(
    finalized: ModelResponse,
    *,
    kwargs: Mapping[str, object],
    depth: int,
    model: str,
    custom_llm_provider: str,
    logging_obj: object,
) -> ModelResponse | CustomStreamWrapper:
    replay: Final = _converted_stream_replay(
        finalized,
        kwargs=kwargs,
        depth=depth,
        model=model,
        custom_llm_provider=custom_llm_provider,
        logging_obj=logging_obj,
    )
    return finalized if replay is None else replay


def _tool_call_name(call: ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall) -> str:
    match call:
        case ChatCompletionMessageCustomToolCall():
            return call.custom.name
        case ChatCompletionMessageToolCall():
            return call.function.name or ""


def _called_tool_names(response: ModelResponse) -> tuple[str, ...]:
    calls: Final = chain.from_iterable(choice.message.tool_calls or () for choice in response.choices)
    return tuple(map(_tool_call_name, calls))


def _server_fulfilled_tool_names(callbacks: Iterable[CustomLogger]) -> frozenset[str]:
    return frozenset(chain.from_iterable(callback.server_fulfilled_tool_names for callback in callbacks))


def holds_a_call_the_choice_excludes(
    tool_names: Iterable[str], kwargs: Mapping[str, object], callbacks: Iterable[CustomLogger]
) -> bool:
    server_fulfilled: Final = _server_fulfilled_tool_names(callbacks)
    client_tool_names: Final = (name for name in tool_names if name not in server_fulfilled)
    return bool(tool_calls_the_choice_excludes(kwargs.get(AGENTIC_LOOP_TOOL_CHOICE_KEY), client_tool_names))


def with_caller_tool_choice(
    kwargs: Mapping[str, object], optional_params: Mapping[str, object]
) -> Mapping[str, object]:
    tool_choice: Final = optional_params.get("tool_choice")
    if AGENTIC_LOOP_TOOL_CHOICE_KEY in kwargs or parse_allowed_tools_tool_choice(tool_choice) is None:
        return kwargs
    return {**kwargs, AGENTIC_LOOP_TOOL_CHOICE_KEY: tool_choice}


def _request_tools(optional_params: Mapping[str, object]) -> list[dict[str, object]]:
    return [tool for tool in object_items(optional_params.get("tools")) if is_str_keyed_dict(tool)]


def _custom_loggers(logging_obj: object) -> tuple[CustomLogger, ...]:
    import litellm

    dynamic: Final = getattr(logging_obj, "dynamic_success_callbacks", None) or []
    return tuple(callback for callback in (*litellm.callbacks, *dynamic) if isinstance(callback, CustomLogger))


async def _refused_chat_turn(
    callbacks: Sequence[CustomLogger],
    refusing: CustomLogger,
    refused_tool_calls: object,
    *,
    response: ModelResponse,
    model: str,
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None,
    custom_llm_provider: str,
    kwargs: Mapping[str, object],
) -> ModelResponse:
    surface_kwargs: Final = {
        **kwargs,
        "_agentic_loop_api_surface": CHAT_COMPLETION_AGENTIC_SURFACE,
        "custom_llm_provider": custom_llm_provider,
    }

    def other_gates() -> Iterator[Awaitable[tuple[bool, object]]]:
        for callback in callbacks:
            if callback is refusing:
                continue
            yield callback.async_should_run_chat_completion_agentic_loop(
                response=response,
                model=model,
                messages=messages,
                tools=tools,
                stream=False,
                custom_llm_provider=custom_llm_provider,
                kwargs=dict(kwargs),
            )
            if _gate_overridden(callback):
                yield callback.async_should_run_agentic_loop(
                    response=response,
                    model=model,
                    messages=messages,
                    tools=tools,
                    stream=False,
                    custom_llm_provider=custom_llm_provider,
                    kwargs=dict(surface_kwargs),
                )

    claimed: Final = await every_claimed_tool_call(refused_tool_calls, other_gates())
    return finalize_refused_chat_turn(response, claimed)


def _fingerprint_of_a_turn_to_run(
    *,
    response: ModelResponse,
    request_kwargs: Mapping[str, object],
    callbacks: Sequence[CustomLogger],
    tool_calls: object,
    fingerprints: Sequence[str],
    depth: int,
    max_loops: int,
    model: str,
    stream: bool,
) -> str | None:
    if holds_a_call_the_choice_excludes(_called_tool_names(response), request_kwargs, callbacks):
        return None
    try:
        return _check_agentic_loop_safety(
            tool_calls=tool_calls,
            fingerprints=fingerprints,
            depth=depth,
            max_loops=max_loops,
            model=model,
        )
    except AgenticLoopSafetyError as e:
        if stream:
            raise
        verbose_logger.warning("LiteLLM.AgenticLoopRefused: ending turn [rail=%s]", type(e).__name__)
        return None


def _with_agentic_loop_metadata(kwargs_for_followup: Mapping[str, object]) -> Mapping[str, object]:
    metadata: Final = kwargs_for_followup.get("litellm_metadata")
    return MappingProxyType(
        {
            **kwargs_for_followup,
            "litellm_metadata": dict(
                chain(
                    metadata.items() if isinstance(metadata, dict) else (),
                    (
                        (key, value)
                        for key, value in kwargs_for_followup.items()
                        if key.startswith("_agentic_loop")
                        or key == "max_agentic_loops"
                        or is_interception_internal_key(key)
                    ),
                )
            ),
        }
    )


def _filter_followup_kwargs(source: dict[str, object]) -> dict[str, object]:
    return {
        k: v
        for k, v in source.items()
        if not is_interception_internal_key(k, prefixes=NON_CODE_INTERPRETER_INTERCEPTION_INTERNAL_PREFIXES)
        and k not in _FOLLOWUP_INTERNAL_PARAMS
    }


def _followup_billing_scope(logging_obj: object, request_kwargs: Mapping[str, object]) -> AbstractContextManager[None]:
    if _follow_ups_bill_themselves(logging_obj, request_kwargs):
        return billed_separately()
    return internal_sub_call(logging_obj.litellm_call_id if isinstance(logging_obj, LiteLLMLoggingObject) else None)


def _billed_with_the_round_it_follows(round_served: object, follow_up: object, logging_obj: object) -> object:
    if not (
        isinstance(round_served, ModelResponse)
        and isinstance(follow_up, ModelResponse)
        and isinstance(logging_obj, LiteLLMLoggingObject)
        and logging_obj.bridged_chat_request is not None
    ):
        return follow_up
    if not (is_internal_call.get() or logging_obj.unpark_success_log()):
        return follow_up
    first_price: Final = _price_of_round(round_served, logging_obj)
    final_price: Final = _price_of_round(follow_up, logging_obj)
    first: Final = _priced(round_served, first_price)
    final: Final = _priced(follow_up, final_price)
    breakdown: Final = summed_cost_breakdown(first, first_price.cost_breakdown, final, final_price.cost_breakdown)
    logging_obj.record_every_round_breakdown(breakdown)
    billed: Final = billed_for_every_round(first, first_price.cost_breakdown, final, final_price.cost_breakdown)
    billed_hidden_params: Final = get_hidden_params(billed) or {}
    total_cost: Final = billed_hidden_params.get("response_cost")
    set_hidden_params(
        billed,
        {
            **billed_hidden_params,
            _PRICE_OF_EVERY_ROUND: UncachedPrice(
                cost=float(total_cost) if isinstance(total_cost, (int, float)) else None, cost_breakdown=breakdown
            ),
        },
    )
    return billed


def _price_of_every_round(response: ModelResponse) -> UncachedPrice | None:
    every_round: Final = (get_hidden_params(response) or {}).get(_PRICE_OF_EVERY_ROUND)
    return every_round if isinstance(every_round, UncachedPrice) else None


def _price_of_round(round_response: ModelResponse, logging_obj: LiteLLMLoggingObject) -> UncachedPrice:
    every_round: Final = _price_of_every_round(round_response)
    if every_round is not None:
        return every_round
    return SERVED_FROM_CACHE if served_from_cache(round_response) else logging_obj.uncached_price(round_response)


def _priced(round_response: ModelResponse, price: UncachedPrice) -> ModelResponse:
    priced: Final = round_response.model_copy()
    set_hidden_params(priced, {**(get_hidden_params(round_response) or {}), "response_cost": price.cost})
    return priced


def _leave_pricing_to_the_request(response: object) -> None:
    hidden_params: Final = get_hidden_params(response)
    if hidden_params is not None and not served_from_cache(response):
        hidden_params.pop("response_cost", None)


async def _execute_chat_completion_agentic_plan(
    *,
    response: ModelResponse,
    plan: AgenticLoopPlan,
    callback: CustomLogger,
    model: str,
    optional_params: dict[str, object],
    kwargs: dict[str, object],
    logging_obj: object,
    custom_llm_provider: str,
    depth: int,
    max_loops: int,
    fingerprints: Sequence[str],
    fingerprint: str,
) -> object:
    import litellm

    patch: Final = plan.request_patch or AgenticLoopRequestPatch()
    if patch.messages is None:
        raise ValueError("Agentic loop plan missing patched messages")

    full_model_name: Final = qualify_agentic_followup_model(patch.model, model, custom_llm_provider)

    optional_params_for_followup: Final = {**optional_params, **patch.optional_params}
    if patch.tools is not None:
        optional_params_for_followup["tools"] = patch.tools
    if "tool_choice" not in patch.optional_params:
        optional_params_for_followup.pop("tool_choice", None)

    kwargs_for_followup: Final = _with_agentic_loop_metadata(
        build_agentic_followup_kwargs(
            request_kwargs=_filter_followup_kwargs(kwargs),
            patch_kwargs=_filter_followup_kwargs(patch.kwargs),
            request_params=frozenset((*optional_params_for_followup, "model", "messages")),
            depth=depth,
            max_loops=max_loops,
            fingerprints=fingerprints,
            fingerprint=fingerprint,
        )
    )

    try:
        with _followup_billing_scope(logging_obj, kwargs):
            response_followup = await litellm.acompletion(
                model=full_model_name,
                messages=patch.messages,
                **optional_params_for_followup,
                **kwargs_for_followup,
            )
        if _post_hook_overridden(callback):
            try:
                response_followup = _HOOK_RESPONSE.validate_python(
                    await callback.async_post_agentic_loop_response_hook(
                        response=response_followup, plan=plan, kwargs=kwargs
                    )
                )
            except Exception as e:
                _call_id = getattr(logging_obj, "litellm_call_id", "unknown")
                verbose_logger.exception(
                    "LiteLLM.AgenticHookError: Exception in "
                    "async_post_agentic_loop_response_hook [call_id=%s model=%s]: %s",
                    _call_id,
                    model,
                    str(e),
                )
        _leave_pricing_to_the_request(response_followup)
        billed: Final = _billed_with_the_round_it_follows(response, response_followup, logging_obj)
        replay: Final = _converted_stream_replay(
            billed,
            kwargs=kwargs,
            depth=depth,
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
        )
        return billed if replay is None else replay
    finally:
        try:
            await callback.async_agentic_loop_cleanup_hook(plan=plan, kwargs=kwargs)
        except Exception as e:
            _call_id = getattr(logging_obj, "litellm_call_id", "unknown")
            verbose_logger.exception(
                "LiteLLM.AgenticHookError: Exception in async_agentic_loop_cleanup_hook [call_id=%s model=%s]: %s",
                _call_id,
                model,
                str(e),
            )


async def maybe_run_chat_completion_agentic_loop(
    *,
    response: ModelResponse,
    model: str,
    messages: list[dict[str, object]],
    optional_params: dict[str, object],
    kwargs: dict[str, object],
    logging_obj: object,
    custom_llm_provider: str,
    stream: bool,
) -> ModelResponse | CustomStreamWrapper | None:
    callbacks: Final = _custom_loggers(logging_obj)
    request_kwargs: Final = with_caller_tool_choice(kwargs, optional_params)
    depth, max_loops, fingerprints = _agentic_loop_settings(request_kwargs)
    tools: Final = _request_tools(optional_params)

    for callback in callbacks:
        if not _gate_overridden(callback):
            continue

        hook_kwargs = {
            **request_kwargs,
            "_agentic_loop_api_surface": CHAT_COMPLETION_AGENTIC_SURFACE,
            "custom_llm_provider": custom_llm_provider,
        }
        try:
            should_run, tool_calls = await callback.async_should_run_agentic_loop(
                response=response,
                model=model,
                messages=messages,
                tools=tools,
                stream=stream,
                custom_llm_provider=custom_llm_provider,
                kwargs=hook_kwargs,
            )
        except Exception as e:
            verbose_logger.exception(
                "LiteLLM.AgenticHookError: Exception in chat completion agentic gate: %s",
                str(e),
            )
            continue

        if not should_run:
            continue

        fingerprint = _fingerprint_of_a_turn_to_run(
            response=response,
            request_kwargs=request_kwargs,
            callbacks=callbacks,
            tool_calls=tool_calls,
            fingerprints=fingerprints,
            depth=depth,
            max_loops=max_loops,
            model=model,
            stream=stream,
        )
        if fingerprint is None:
            return _end_refused_turn(
                await _refused_chat_turn(
                    callbacks,
                    callback,
                    tool_calls,
                    response=response,
                    model=model,
                    messages=messages,
                    tools=tools,
                    custom_llm_provider=custom_llm_provider,
                    kwargs=request_kwargs,
                ),
                kwargs=request_kwargs,
                depth=depth,
                model=model,
                custom_llm_provider=custom_llm_provider,
                logging_obj=logging_obj,
            )

        if not _build_plan_overridden(callback):
            return await callback.async_run_agentic_loop(
                tools=tool_calls,
                model=model,
                messages=messages,
                response=response,
                anthropic_messages_provider_config=None,
                anthropic_messages_optional_request_params=optional_params,
                logging_obj=logging_obj,
                stream=stream,
                kwargs=hook_kwargs,
            )

        plan = await callback.async_build_agentic_loop_plan(
            tools=tool_calls,
            model=model,
            messages=messages,
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params=optional_params,
            logging_obj=logging_obj,
            stream=stream,
            kwargs=hook_kwargs,
        )

        if plan.response_override is not None:
            return plan.response_override
        if plan.terminate:
            return response
        if not plan.run_agentic_loop:
            continue

        return await _execute_chat_completion_agentic_plan(
            response=response,
            plan=plan,
            callback=callback,
            model=model,
            optional_params=optional_params,
            kwargs=dict(request_kwargs),
            logging_obj=logging_obj,
            custom_llm_provider=custom_llm_provider,
            depth=depth,
            max_loops=max_loops,
            fingerprints=fingerprints,
            fingerprint=fingerprint,
        )

    bridged_chat_turn: Final = (
        await run_chat_completion_agentic_hooks(
            response=response,
            model=model,
            messages=messages,
            optional_params=optional_params,
            kwargs=dict(request_kwargs),
            logging_obj=logging_obj,
            custom_llm_provider=custom_llm_provider,
            stream=stream,
            asking_only=runs_its_own_chat_completion_loop,
        )
        if isinstance(logging_obj, LiteLLMLoggingObject) and logging_obj.bridged_chat_request is not None
        else None
    )
    match bridged_chat_turn:
        case CustomStreamWrapper():
            return bridged_chat_turn
        case ModelResponse():
            return _end_refused_turn(
                bridged_chat_turn,
                kwargs=request_kwargs,
                depth=depth,
                model=model,
                custom_llm_provider=custom_llm_provider,
                logging_obj=logging_obj,
            )
        case _:
            return _converted_stream_replay(
                response,
                kwargs=request_kwargs,
                depth=depth,
                model=model,
                custom_llm_provider=custom_llm_provider,
                logging_obj=logging_obj,
            )


async def run_chat_completion_agentic_hooks(
    *,
    response: ModelResponse,
    model: str,
    messages: list[dict[str, object]],
    optional_params: dict[str, object],
    kwargs: dict[str, object],
    logging_obj: LiteLLMLoggingObject,
    custom_llm_provider: str,
    stream: bool,
    asking_only: Callable[[CustomLogger], bool] | None = None,
) -> object | None:
    callbacks: Final = _custom_loggers(logging_obj)
    depth, max_loops, fingerprints = _agentic_loop_settings(kwargs)
    tools: Final = _request_tools(optional_params)
    hook_kwargs: Final = {**kwargs, "custom_llm_provider": custom_llm_provider}

    for callback in callbacks if asking_only is None else tuple(filter(asking_only, callbacks)):
        try:
            should_run, tool_calls = await callback.async_should_run_chat_completion_agentic_loop(
                response=response,
                model=model,
                messages=messages,
                tools=tools,
                stream=stream,
                custom_llm_provider=custom_llm_provider,
                kwargs=kwargs,
            )
        except Exception:
            verbose_logger.exception(
                "LiteLLM.AgenticHookError: Exception in async_should_run_chat_completion_agentic_loop"
            )
            continue

        if not should_run:
            continue

        fingerprint = _fingerprint_of_a_turn_to_run(
            response=response,
            request_kwargs=kwargs,
            callbacks=callbacks,
            tool_calls=tool_calls,
            fingerprints=fingerprints,
            depth=depth,
            max_loops=max_loops,
            model=model,
            stream=stream,
        )
        if fingerprint is None:
            return await _refused_chat_turn(
                callbacks,
                callback,
                tool_calls,
                response=response,
                model=model,
                messages=messages,
                tools=tools,
                custom_llm_provider=custom_llm_provider,
                kwargs=kwargs,
            )

        if not _chat_build_plan_overridden(callback):
            return await callback.async_run_chat_completion_agentic_loop(
                tools=tool_calls,
                model=model,
                messages=messages,
                response=response,
                optional_params=optional_params,
                logging_obj=logging_obj,
                stream=stream,
                kwargs=hook_kwargs,
            )

        plan = await callback.async_build_chat_completion_agentic_loop_plan(
            tools=tool_calls,
            model=model,
            messages=messages,
            response=response,
            optional_params=optional_params,
            logging_obj=logging_obj,
            stream=stream,
            kwargs=hook_kwargs,
        )

        if (response_override := _plan_response_override(plan)) is not None:
            return response_override
        if plan.terminate:
            verbose_logger.debug(
                "Agentic chat loop terminated by callback=%s reason=%s", type(callback).__name__, plan.stop_reason
            )
            return response
        if not plan.run_agentic_loop:
            continue

        return await _execute_chat_completion_agentic_plan(
            response=response,
            plan=plan,
            callback=callback,
            model=model,
            optional_params=optional_params,
            kwargs=hook_kwargs,
            logging_obj=logging_obj,
            custom_llm_provider=custom_llm_provider,
            depth=depth,
            max_loops=max_loops,
            fingerprints=fingerprints,
            fingerprint=fingerprint,
        )

    return None

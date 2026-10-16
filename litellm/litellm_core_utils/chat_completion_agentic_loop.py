# this is a patch to allow for agentic loops covering llm_http_handler.py and openai sdk based calling flows for the .completion() api

import json
from collections.abc import Mapping
from itertools import chain
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.agentic_followup_kwargs import build_agentic_followup_kwargs
from litellm.litellm_core_utils.agentic_loop_settings import (
    DEFAULT_MAX_AGENTIC_LOOPS,
    validated_max_agentic_loops,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObject
from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
from litellm.types.integrations.custom_logger import (
    CHAT_COMPLETION_AGENTIC_SURFACE,
    NON_CODE_INTERPRETER_INTERCEPTION_INTERNAL_PREFIXES,
    AgenticLoopPlan,
    AgenticLoopRequestPatch,
    converted_stream_requested,
    is_interception_internal_key,
    stashed_stream_options,
)
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper

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


def _gate_overridden(callback: CustomLogger) -> bool:
    base: Final = CustomLogger.async_should_run_agentic_loop
    func: Final = type(callback).async_should_run_agentic_loop
    return getattr(func, "__func__", func) is not getattr(base, "__func__", base)


def _build_plan_overridden(callback: CustomLogger) -> bool:
    base: Final = CustomLogger.async_build_agentic_loop_plan
    func: Final = type(callback).async_build_agentic_loop_plan
    return getattr(func, "__func__", func) is not getattr(base, "__func__", base)


def _post_hook_overridden(callback: CustomLogger) -> bool:
    base: Final = CustomLogger.async_post_agentic_loop_response_hook
    func: Final = type(callback).async_post_agentic_loop_response_hook
    return getattr(func, "__func__", func) is not getattr(base, "__func__", base)


_STREAM_OPTIONS_ADAPTER: Final[TypeAdapter[Mapping[str, object] | None]] = TypeAdapter(Mapping[str, object] | None)


def _stashed_stream_options(kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
    try:
        return _STREAM_OPTIONS_ADAPTER.validate_python(stashed_stream_options(kwargs))
    except ValidationError:
        return None


def _coerce_int(value: object, default: int) -> int:
    return int(value) if isinstance(value, (int, str)) else default


def _agentic_loop_settings(kwargs: dict[str, object]) -> tuple[int, int, list[str]]:
    depth: Final = _coerce_int(kwargs.get("_agentic_loop_depth"), 0)
    configured: Final = validated_max_agentic_loops(
        kwargs.get("max_agentic_loops"), field="litellm_params.max_agentic_loops"
    )
    max_loops: Final = DEFAULT_MAX_AGENTIC_LOOPS if configured is None else configured
    raw_fingerprints: Final = kwargs.get("_agentic_loop_fingerprints")
    fingerprints: Final = [str(fp) for fp in raw_fingerprints] if isinstance(raw_fingerprints, list) else []
    return depth, max_loops, fingerprints


def _fingerprint_tools(tool_calls: object) -> str:
    try:
        return json.dumps(tool_calls, sort_keys=True, default=str)
    except Exception:
        return str(tool_calls)


def _check_agentic_loop_safety(
    tool_calls: object,
    fingerprints: list[str],
    depth: int,
    max_loops: int,
    model: str,
) -> str:
    fingerprint: Final = _fingerprint_tools(tool_calls)
    if fingerprint in fingerprints:
        raise ValueError("Agentic loop detected repeated tool-call fingerprint; aborting rerun")
    if depth >= max_loops:
        raise ValueError(f"Exceeded max_agentic_loops={max_loops} for model={model}")
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

    return CustomStreamWrapper(
        completion_stream=MockResponseIterator(model_response=response),
        model=model,
        custom_llm_provider=custom_llm_provider,
        logging_obj=logging_obj,
        stream_options=_stashed_stream_options(kwargs),
    )


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


async def _execute_chat_completion_agentic_plan(
    *,
    plan: AgenticLoopPlan,
    callback: CustomLogger,
    model: str,
    optional_params: dict[str, object],
    kwargs: dict[str, object],
    logging_obj: object,
    custom_llm_provider: str,
    depth: int,
    max_loops: int,
    fingerprints: list[str],
    fingerprint: str,
) -> object:
    import litellm

    patch: Final = plan.request_patch or AgenticLoopRequestPatch()
    if patch.messages is None:
        raise ValueError("Agentic loop plan missing patched messages")

    full_model_name = patch.model or model
    if "/" not in full_model_name:
        full_model_name = f"{custom_llm_provider}/{full_model_name}"

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
        response_followup = await litellm.acompletion(
            model=full_model_name,
            messages=patch.messages,
            **optional_params_for_followup,
            **kwargs_for_followup,
        )
        if _post_hook_overridden(callback):
            try:
                response_followup = await callback.async_post_agentic_loop_response_hook(
                    response=response_followup, plan=plan, kwargs=kwargs
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
        replay: Final = _converted_stream_replay(
            response_followup,
            kwargs=kwargs,
            depth=depth,
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
        )
        return response_followup if replay is None else replay
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
    messages: list,
    optional_params: dict,
    kwargs: dict[str, object],
    logging_obj: object,
    custom_llm_provider: str,
    stream: bool,
) -> ModelResponse | CustomStreamWrapper | None:
    import litellm

    callbacks: Final = litellm.callbacks + (getattr(logging_obj, "dynamic_success_callbacks", None) or [])
    depth, max_loops, fingerprints = _agentic_loop_settings(kwargs)
    tools: Final = optional_params.get("tools", [])

    for callback in callbacks:
        if not isinstance(callback, CustomLogger):
            continue

        if not _gate_overridden(callback):
            continue

        hook_kwargs = {
            **kwargs,
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

        fingerprint = _check_agentic_loop_safety(
            tool_calls=tool_calls,
            fingerprints=fingerprints,
            depth=depth,
            max_loops=max_loops,
            model=model,
        )

        try:
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
                plan=plan,
                callback=callback,
                model=model,
                optional_params=optional_params,
                kwargs=kwargs,
                logging_obj=logging_obj,
                custom_llm_provider=custom_llm_provider,
                depth=depth,
                max_loops=max_loops,
                fingerprints=fingerprints,
                fingerprint=fingerprint,
            )
        except Exception as e:
            verbose_logger.exception(
                "LiteLLM.AgenticHookError: Exception in chat completion agentic hooks: %s",
                str(e),
            )

    return _converted_stream_replay(
        response,
        kwargs=kwargs,
        depth=depth,
        model=model,
        custom_llm_provider=custom_llm_provider,
        logging_obj=logging_obj,
    )

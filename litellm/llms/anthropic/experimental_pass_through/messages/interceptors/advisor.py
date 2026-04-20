"""
Advisor Orchestration Handler

Implements the advisor tool loop for providers that don't support
advisor_20260301 natively (i.e. everything except Anthropic direct for now).

How it works:
1. Detects advisor_20260301 in tools + non-native provider → intercepts.
2. Translates the advisor tool to a regular function tool the provider understands.
3. Calls the executor model (non-streaming).
4. If the executor makes a tool_use call named "advisor", runs the advisor model
   and injects the result as a tool_result before re-calling the executor.
5. Repeats until the executor produces a final text response or max_uses is hit.
6. Wraps in FakeAnthropicMessagesStreamIterator if the caller requested streaming.
"""

import asyncio
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import litellm
import litellm.constants as _c
from litellm._internal_context import is_internal_call
from litellm._logging import verbose_logger
from litellm.llms.anthropic.common_utils import strip_advisor_blocks_from_messages
from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
    AnthropicUsageIteration,
)
from litellm.types.llms.anthropic import ANTHROPIC_ADVISOR_TOOL_TYPE
from litellm.utils import (
    resolve_proxy_model_alias_to_litellm_model,
    supports_native_advisor_tool,
)

ADVISOR_MAX_USES: int = _c.ADVISOR_MAX_USES
ADVISOR_TOOL_DESCRIPTION: str = _c.ADVISOR_TOOL_DESCRIPTION

from .base import MessagesInterceptor


class AdvisorMaxIterationsError(Exception):
    """Raised when the advisor loop exceeds max_uses."""


class AdvisorOrchestrationHandler(MessagesInterceptor):
    """Orchestrates the advisor tool loop for /v1/messages requests."""

    def can_handle(
        self,
        tools: Optional[List[Dict]],
        custom_llm_provider: Optional[str],
    ) -> bool:
        if not tools:
            return False
        advisor_tools = [
            t for t in tools if t.get("type") == ANTHROPIC_ADVISOR_TOOL_TYPE
        ]
        if not advisor_tools:
            return False
        # Direct Anthropic /messages: the API handles advisor_20260301 natively
        # *only* when the tool's model resolves to a native Anthropic advisor
        # model. When an operator remaps the tool's model via model_group_alias
        # to a non-native model (e.g. claude-opus-4-7 -> o3) we must take over
        # the loop here so the sub-call is routed through litellm.
        if custom_llm_provider == "anthropic":
            for advisor_tool in advisor_tools:
                if not _advisor_tool_uses_native_anthropic_model(advisor_tool):
                    return True
            return False
        return True

    async def handle(
        self,
        *,
        model: str,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        stream: Optional[bool],
        max_tokens: int,
        custom_llm_provider: Optional[str],
        **kwargs,
    ) -> Union[AnthropicMessagesResponse, AsyncIterator]:
        from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
            FakeAnthropicMessagesStreamIterator,
        )

        # Extract advisor tool config.
        advisor_tool = next(
            (t for t in (tools or []) if t.get("type") == ANTHROPIC_ADVISOR_TOOL_TYPE),
            None,
        )
        if advisor_tool is None:
            raise ValueError(
                f"handle() called but no {ANTHROPIC_ADVISOR_TOOL_TYPE} tool found in tools list"
            )
        advisor_model_alias: str = advisor_tool.get("model") or ""
        if not advisor_model_alias:
            advisor_model_alias = _resolve_default_advisor_model()
        if not advisor_model_alias:
            raise ValueError(
                "No advisor model specified. Either:\n"
                "  1. Set 'default_advisor_model' in advisor_interception_params in your proxy config YAML, or\n"
                "  2. Include a 'model' field in the advisor tool definition."
            )
        # Resolve the tool's ``model`` (which may be a proxy model_group_alias
        # like ``claude-opus-4-7`` pointing at ``o3``) to the actual underlying
        # litellm model for the sub-call. Keep the alias separate so every
        # client-visible surface (iterations[].model) continues to show the
        # original name the caller sent — the remap is opaque to the caller.
        resolved_advisor_model: str = (
            resolve_proxy_model_alias_to_litellm_model(advisor_model_alias)
            or advisor_model_alias
        )
        _raw_max_uses = advisor_tool.get("max_uses")
        max_uses: int = ADVISOR_MAX_USES if _raw_max_uses is None else int(_raw_max_uses)
        # Optional routing overrides for the advisor sub-call (e.g. proxy routing).
        # If not set in the tool definition, litellm resolves from env vars.
        advisor_api_key: Optional[str] = advisor_tool.get("api_key")
        advisor_api_base: Optional[str] = advisor_tool.get("api_base")

        # Build the synthetic tool definition the provider will receive.
        synthetic_advisor_tool = _make_synthetic_advisor_tool()

        # Executor tools = all original tools with advisor replaced by the synthetic one.
        executor_tools: List[Dict] = [
            (
                synthetic_advisor_tool
                if t.get("type") == ANTHROPIC_ADVISOR_TOOL_TYPE
                else t
            )
            for t in (tools or [])
        ]

        # Strip prior advisor blocks from history, preserving advice text as context.
        current_messages: List[Dict] = strip_advisor_blocks_from_messages(
            [dict(m) for m in messages], replace_with_text=True
        )

        parent_request_id: str = str(
            kwargs.pop("litellm_call_id", None) or uuid.uuid4()
        )
        metadata_base: Dict = dict(kwargs.pop("metadata", None) or {})
        # Hold the outer ``anthropic_messages`` logging obj — we attach the
        # aggregated cost/breakdown to it in _finalize_orchestrated_response.
        #
        # We *keep* this object in kwargs so inner executor sub-calls share it:
        # their inner @client wrappers (aresponses/acompletion) still populate
        # ``custom_llm_provider``, ``api_base``, ``model_id`` on this shared
        # model_call_details via ``update_environment_variables`` — fields the
        # outer anthropic_messages path never sets on its own.
        #
        # NOTE: ``_is_litellm_internal_call`` in kwargs is not sufficient to
        # suppress @client logging; wrapper_async checks the ContextVar
        # ``is_internal_call``. Keep the kwarg for compatibility, but also set
        # the ContextVar around the orchestration loop so nested sub-calls do
        # not emit separate proxy billing rows.
        litellm_logging_obj = kwargs.get("litellm_logging_obj", None)
        kwargs["_is_litellm_internal_call"] = True
        iteration = 0
        advisor_interactions: List[Dict] = []
        iterations: List[AnthropicUsageIteration] = []
        advisor_first_call_cost: float = 0.0
        advisor_subcall_cost: float = 0.0

        _prev_internal = is_internal_call.get()
        is_internal_call.set(True)
        try:
            while True:
                # --- Executor call (always non-streaming) ---
                executor_response: AnthropicMessagesResponse = await _call_messages_handler(
                    model=model,
                    messages=current_messages,
                    tools=executor_tools,
                    stream=False,
                    max_tokens=max_tokens,
                    custom_llm_provider=custom_llm_provider,
                    metadata={
                        **metadata_base,
                        "advisor_sub_call": False,
                        "parent_request_id": parent_request_id,
                    },
                    **kwargs,
                )

                executor_cost = _get_response_cost(executor_response, model=model)
                iterations.append(
                    _build_iteration_entry(
                        response=executor_response, iteration_type="message"
                    )
                )

                advisor_use_block = _find_advisor_tool_use(executor_response)

                if advisor_use_block is None:
                    # No more advisor calls — this is the final response.
                    # Inject advisor_tool_result blocks to match Anthropic native format.
                    _inject_advisor_blocks_into_response(
                        executor_response, advisor_interactions
                    )
                    total_cost = (
                        advisor_first_call_cost + advisor_subcall_cost + executor_cost
                    )
                    _finalize_orchestrated_response(
                        response=executor_response,
                        iterations=iterations,
                        total_cost=total_cost,
                        final_executor_cost=executor_cost,
                        advisor_first_call_cost=advisor_first_call_cost,
                        advisor_subcall_cost=advisor_subcall_cost,
                        litellm_logging_obj=litellm_logging_obj,
                    )
                    if stream:
                        # The outer ``@client`` async wrapper skips
                        # ``_client_async_logging_helper`` for streaming
                        # requests — it assumes a ``CustomStreamWrapper`` will
                        # fire logging on iteration. ``FakeAnthropicMessagesStreamIterator``
                        # is a plain iterator over pre-built SSE bytes and
                        # does not know about the logging obj, so nothing fires
                        # the proxy log row. We've already aggregated the full
                        # response into ``executor_response`` (same dict shape
                        # the non-streaming path uses for logging), so fire
                        # the success handler ourselves with that dict before
                        # wrapping — this mirrors the non-streaming flow and
                        # avoids double-logging (the @client path is skipped).
                        _fire_async_success_logging(
                            litellm_logging_obj=litellm_logging_obj,
                            result=executor_response,
                        )
                        return FakeAnthropicMessagesStreamIterator(executor_response)
                    return executor_response

                # Executor response triggered another advisor call → count it as a
                # "first/intermediate" executor turn. Only the terminating turn is
                # treated as the base response.
                advisor_first_call_cost += executor_cost

                iteration += 1
                if iteration > max_uses:
                    raise AdvisorMaxIterationsError(
                        f"Advisor orchestration loop exceeded max_uses={max_uses}. "
                        "Increase max_uses in the advisor tool definition or cap the request."
                    )

                # --- Build advisor context ---
                advisor_messages = _build_advisor_context(
                    current_messages, executor_response, advisor_use_block
                )

                # --- Advisor sub-call (always non-streaming, no tools) ---
                # Use the resolved model so router routing / cost lookup hit the
                # real underlying deployment; the alias is kept only for the
                # client-visible iteration entry below.
                advisor_response: AnthropicMessagesResponse = await _call_advisor_with_router(
                    model=resolved_advisor_model,
                    messages=advisor_messages,
                    max_tokens=max_tokens,
                    metadata={
                        **metadata_base,
                        "advisor_sub_call": True,
                        "parent_request_id": parent_request_id,
                    },
                    api_key=advisor_api_key,
                    api_base=advisor_api_base,
                )

                advisor_call_cost = _get_response_cost(
                    advisor_response, model=resolved_advisor_model
                )
                advisor_subcall_cost += advisor_call_cost
                iterations.append(
                    _build_iteration_entry(
                        response=advisor_response,
                        iteration_type="advisor_message",
                        model=advisor_model_alias,
                    )
                )

                advisor_text = _extract_response_text(advisor_response)

                # Record the interaction for later injection into the final response.
                advisor_interactions.append({
                    "tool_use_id": advisor_use_block.get(
                        "id", f"srvtoolu_{uuid.uuid4().hex[:24]}"
                    ),
                    "advisor_text": advisor_text,
                })

                # --- Inject advisor result and continue loop ---
                current_messages = _inject_advisor_turn(
                    current_messages,
                    executor_response,
                    advisor_use_block,
                    advisor_text,
                )
        finally:
            is_internal_call.set(_prev_internal)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_default_advisor_model() -> str:
    """Resolve the default advisor model from proxy config / litellm settings."""
    import litellm

    params = getattr(litellm, "advisor_interception_params", None) or {}
    return params.get("default_advisor_model", "") or ""


def _advisor_tool_uses_native_anthropic_model(advisor_tool: Dict) -> bool:
    """
    Return True iff the advisor tool's ``model`` (after proxy alias resolution)
    is a native Anthropic advisor model.

    Used by :class:`AdvisorOrchestrationHandler.can_handle` to decide whether
    to let Anthropic's server-side advisor handle the tool or to intercept it
    and route the sub-call through LiteLLM.
    """
    advisor_model = advisor_tool.get("model") or _resolve_default_advisor_model()
    if not advisor_model:
        # No model specified — let native Anthropic handle it (or fail there
        # with its own error). This path should not be hit in practice because
        # advisor_interception_params enforces a default upstream.
        return True
    resolved_model = (
        resolve_proxy_model_alias_to_litellm_model(advisor_model) or advisor_model
    )
    if resolved_model.startswith("anthropic/"):
        resolved_model = resolved_model.split("/", 1)[1]
    return supports_native_advisor_tool(
        model=resolved_model, custom_llm_provider="anthropic"
    )


_SYNTHETIC_ADVISOR_TOOL_NAME = "consult_advisor"


def _make_synthetic_advisor_tool() -> Dict:
    """Build a regular tool definition the executor provider can understand.

    Uses a name that does NOT collide with ``_ADVISOR_TOOL_NAMES`` in the
    chat-completions interception handler so pre-request hooks won't
    double-convert it.
    """
    return {
        "name": _SYNTHETIC_ADVISOR_TOOL_NAME,
        "description": ADVISOR_TOOL_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question or challenge you want guidance on.",
                }
            },
            "required": ["question"],
        },
    }


def _get_response_cost(response: Any, model: Optional[str] = None) -> float:
    """
    Extract the cost of a single sub-call response.

    Prefers a pre-set ``_hidden_params["response_cost"]`` (produced by the
    ``@client`` wrapper on BaseModel responses / preserved in
    :func:`_openai_response_to_anthropic_dict`). Falls back to
    ``litellm.completion_cost`` when the hidden param is missing — this covers
    executor responses from ``anthropic_messages`` which return a plain dict.
    """
    if response is None:
        return 0.0

    hidden_params: Any = None
    if isinstance(response, dict):
        hidden_params = response.get("_hidden_params")
    else:
        hidden_params = getattr(response, "_hidden_params", None)

    if isinstance(hidden_params, dict):
        cost = hidden_params.get("response_cost")
        if isinstance(cost, (int, float)):
            return float(cost)

    try:
        cost = litellm.completion_cost(completion_response=response, model=model)
        if isinstance(cost, (int, float)):
            return float(cost)
    except Exception as cost_error:
        verbose_logger.debug(
            "AdvisorOrchestration: completion_cost fallback failed for model '%s': %s",
            model,
            str(cost_error),
        )
    return 0.0


def _build_iteration_entry(
    response: Any,
    iteration_type: str,
    model: Optional[str] = None,
) -> AnthropicUsageIteration:
    """
    Build one entry for ``usage.iterations[]`` on advisor-orchestrated
    ``/v1/messages`` responses. Reads tokens from Anthropic-shaped usage on
    the dict response.
    """
    usage: Dict[str, Any] = {}
    if isinstance(response, dict):
        maybe_usage = response.get("usage")
        if isinstance(maybe_usage, dict):
            usage = maybe_usage

    entry: AnthropicUsageIteration = {
        "type": iteration_type,  # type: ignore[typeddict-item]
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "cache_read_input_tokens": int(usage.get("cache_read_input_tokens", 0) or 0),
        "cache_creation_input_tokens": int(
            usage.get("cache_creation_input_tokens", 0) or 0
        ),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
    }
    if iteration_type == "advisor_message" and model is not None:
        entry["model"] = model
    return entry


def _finalize_orchestrated_response(
    response: Any,
    iterations: List[AnthropicUsageIteration],
    total_cost: float,
    final_executor_cost: float,
    advisor_first_call_cost: float,
    advisor_subcall_cost: float,
    litellm_logging_obj: Any,
) -> None:
    """
    Attach aggregated usage, per-iteration breakdown and total cost to the
    terminating executor response, and publish the advisor cost split to the
    parent ``litellm_logging_obj`` so the proxy UI can render it (same
    plumbing Azure Router uses).
    """
    if not isinstance(response, dict):
        return

    aggregated_usage: Dict[str, Any] = {
        "input_tokens": sum(it.get("input_tokens", 0) for it in iterations),
        "output_tokens": sum(it.get("output_tokens", 0) for it in iterations),
        "cache_read_input_tokens": sum(
            it.get("cache_read_input_tokens", 0) for it in iterations
        ),
        "cache_creation_input_tokens": sum(
            it.get("cache_creation_input_tokens", 0) for it in iterations
        ),
        "iterations": list(iterations),
    }
    response["usage"] = aggregated_usage

    # Give the orchestrated response its own Anthropic-style id so the outer
    # ``anthropic_messages`` log row is distinct from any inner sub-call log
    # row in the proxy UI (the inner final executor sub-call shares this
    # ``litellm_logging_obj`` and would otherwise overwrite its request_id).
    response["id"] = f"msg_{uuid.uuid4().hex}"

    hidden_params = response.get("_hidden_params")
    if not isinstance(hidden_params, dict):
        hidden_params = {}
    hidden_params["response_cost"] = total_cost
    response["_hidden_params"] = hidden_params

    if litellm_logging_obj is not None:
        try:
            # Ensure downstream logging emits the aggregated cost even when the
            # transformed ModelResponse path (which drops our dict hidden_params)
            # is taken.
            litellm_logging_obj.model_call_details["response_cost"] = total_cost
        except Exception:
            pass

        # Restore call_type to ``anthropic_messages`` so the outer success
        # handler runs ``_handle_anthropic_messages_response_logging`` and
        # converts the final Anthropic dict to a ``ModelResponse`` — without
        # this, the proxy UI's log row has no renderable output (the raw
        # Anthropic ``content`` blocks are not recognised by the OpenAI-shaped
        # renderer). Inner executor sub-calls (e.g. aresponses for OpenAI
        # models) flip call_type to ``acompletion`` to dodge a separate bug in
        # their own success path; that leaves us with the wrong call_type on
        # the shared logging obj by the time orchestration finishes.
        try:
            from litellm.types.utils import CallTypes

            litellm_logging_obj.call_type = CallTypes.anthropic_messages.value
            litellm_logging_obj.model_call_details[
                "call_type"
            ] = CallTypes.anthropic_messages.value
        except Exception:
            pass

        if hasattr(litellm_logging_obj, "set_cost_breakdown"):
            additional_costs: Dict[str, float] = {}
            if advisor_first_call_cost > 0:
                additional_costs["Main Model (initial)"] = advisor_first_call_cost
            if advisor_subcall_cost > 0:
                additional_costs["Advisor Model"] = advisor_subcall_cost
            try:
                litellm_logging_obj.set_cost_breakdown(
                    input_cost=final_executor_cost,
                    output_cost=0.0,
                    total_cost=total_cost,
                    cost_for_built_in_tools_cost_usd_dollar=0.0,
                    additional_costs=additional_costs or None,
                )
            except Exception as breakdown_error:
                verbose_logger.debug(
                    "AdvisorOrchestration: failed to store cost breakdown: %s",
                    str(breakdown_error),
                )


def _fire_async_success_logging(
    litellm_logging_obj: Any,
    result: Any,
) -> None:
    """
    Manually enqueue the ``async_success_handler`` for a streaming advisor
    response.

    The outer ``@client`` async wrapper only calls
    ``_client_async_logging_helper`` for non-streaming results; streaming
    results are expected to log from inside a ``CustomStreamWrapper``. Our
    synthetic :class:`FakeAnthropicMessagesStreamIterator` has no logging
    hook, so without this helper the proxy UI would never get a row for
    streaming advisor calls. We already built the aggregated response dict
    (same shape the non-streaming path logs from), so we can fire logging
    exactly once here with the same arguments the non-streaming path uses.
    """
    if litellm_logging_obj is None:
        return
    try:
        import datetime as _dt

        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        start_time = getattr(litellm_logging_obj, "start_time", None) or _dt.datetime.now()
        end_time = _dt.datetime.now()
        GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(
            async_coroutine=litellm_logging_obj.async_success_handler(
                result=result, start_time=start_time, end_time=end_time
            )
        )
        try:
            litellm_logging_obj.handle_sync_success_callbacks_for_async_calls(
                result=result,
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as sync_cb_error:
            verbose_logger.debug(
                "AdvisorOrchestration: sync success callbacks failed: %s",
                str(sync_cb_error),
            )
    except Exception as logging_error:
        verbose_logger.debug(
            "AdvisorOrchestration: failed to fire async success logging: %s",
            str(logging_error),
        )


def _find_advisor_tool_use(response: Any) -> Optional[Dict]:
    """Return the first tool_use block whose name matches our synthetic advisor."""
    content = response.get("content") if isinstance(response, dict) else []
    if not isinstance(content, list):
        return None
    for block in content:
        if (
            isinstance(block, dict)
            and block.get("type") == "tool_use"
            and block.get("name") == _SYNTHETIC_ADVISOR_TOOL_NAME
        ):
            return block
    return None


def _openai_response_to_anthropic_dict(response: Any) -> Dict:
    """Convert an OpenAI ChatCompletion response to a minimal Anthropic Messages dict.

    Preserves ``usage`` (mapped to Anthropic's shape) and ``_hidden_params`` so
    the orchestration loop can aggregate cost and per-iteration token usage.
    """
    try:
        choices = response.choices if hasattr(response, "choices") else []
        content_blocks: List[Dict] = []
        for choice in choices:
            msg = choice.message if hasattr(choice, "message") else choice.get("message", {})
            text = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if text:
                content_blocks.append({"type": "text", "text": text})
        anthropic_dict: Dict[str, Any] = {
            "id": getattr(response, "id", ""),
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "stop_reason": "end_turn",
            "usage": _openai_usage_to_anthropic_usage(response),
        }
        hidden_params = getattr(response, "_hidden_params", None)
        if hidden_params is not None:
            anthropic_dict["_hidden_params"] = (
                dict(hidden_params) if isinstance(hidden_params, dict) else hidden_params
            )
        return anthropic_dict
    except Exception:
        return {"content": [], "stop_reason": "end_turn"}


def _openai_usage_to_anthropic_usage(response: Any) -> Dict[str, int]:
    """Translate an OpenAI usage block (if any) to Anthropic token shape."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

    def _get(obj: Any, key: str, default: int = 0) -> int:
        if isinstance(obj, dict):
            value = obj.get(key, default)
        else:
            value = getattr(obj, key, default)
        return int(value or 0)

    cache_read = _get(usage, "cache_read_input_tokens")
    if not cache_read:
        prompt_details = (
            usage.get("prompt_tokens_details")
            if isinstance(usage, dict)
            else getattr(usage, "prompt_tokens_details", None)
        )
        if prompt_details is not None:
            cache_read = _get(prompt_details, "cached_tokens")

    return {
        "input_tokens": _get(usage, "prompt_tokens"),
        "output_tokens": _get(usage, "completion_tokens"),
        "cache_creation_input_tokens": _get(usage, "cache_creation_input_tokens"),
        "cache_read_input_tokens": cache_read,
    }


def _inject_advisor_blocks_into_response(
    response: Any, advisor_interactions: List[Dict]
) -> None:
    """
    Mutate *response* in place so its ``content`` array includes
    ``advisor_tool_result`` blocks that mirror Anthropic's native advisor
    response format.

    Each advisor interaction produces two blocks appended after existing
    content:

    * ``server_tool_use``  – records the executor's call to the advisor
    * ``advisor_tool_result`` – carries the advisor's answer

    This ensures callers see an identical structure regardless of whether the
    advisor ran natively or via LiteLLM interception.
    """
    if not advisor_interactions:
        return

    content = response.get("content") if isinstance(response, dict) else None
    if not isinstance(content, list):
        return

    for interaction in advisor_interactions:
        tool_use_id = interaction["tool_use_id"]
        advisor_text = interaction["advisor_text"]

        # ``input`` is required by
        # ``AnthropicConfig.convert_tool_use_to_openai_format`` (it does
        # ``json.dumps(block["input"])`` unconditionally when logging converts
        # the Anthropic response to OpenAI format). We did not observe a real
        # tool-call payload here — the advisor was invoked via a synthetic
        # tool — so an empty object is the correct stub.
        content.append({
            "type": "server_tool_use",
            "id": tool_use_id,
            "name": "advisor",
            "input": {},
        })
        content.append({
            "type": "advisor_tool_result",
            "tool_use_id": tool_use_id,
            "content": {
                "type": "advisor_result",
                "text": advisor_text,
            },
        })


def _extract_response_text(response: Any) -> str:
    """Extract concatenated text from all text blocks in a response."""
    content = response.get("content") if isinstance(response, dict) else []
    if not isinstance(content, list):
        return ""
    parts = [
        b.get("text", "")
        for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    return "\n".join(parts).strip()


_PROVIDER_SPECIFIC_KEYS = frozenset({"provider_specific_fields"})


def _build_advisor_context(
    messages: List[Dict],
    executor_response: Any,
    advisor_use_block: Dict,
) -> List[Dict]:
    """
    Build the message list for the advisor sub-call.

    Passes the full conversation + any text the executor produced so far, then
    poses the advisor question as the last user turn.

    tool_use blocks are excluded because Anthropic requires tool_use to be
    immediately followed by tool_result — not the advisor question.

    Messages stay in Anthropic ``/v1/messages`` shape here; ``_call_advisor_with_router``
    runs the same ``LiteLLMMessagesToCompletionTransformationHandler`` path used when
    a client calls the messages endpoint with a non-Anthropic model, so provider
    translation (including interleaved ``thinking`` blocks) matches the rest of
    the stack.
    """
    question = (advisor_use_block.get("input") or {}).get("question") or (
        "Please provide guidance on the current task."
    )
    raw_content = (
        executor_response.get("content") if isinstance(executor_response, dict) else []
    ) or []
    # Keep only text blocks — strip tool_use and provider-specific fields.
    executor_text_blocks = [
        {k: v for k, v in block.items() if k not in _PROVIDER_SPECIFIC_KEYS}
        for block in raw_content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    result = list(messages)
    if executor_text_blocks:
        result.append({"role": "assistant", "content": executor_text_blocks})
    result.append({"role": "user", "content": question})
    return result


def _inject_advisor_turn(
    messages: List[Dict],
    executor_response: Any,
    advisor_use_block: Dict,
    advisor_text: str,
) -> List[Dict]:
    """
    Append the executor's response (as an assistant turn) and the advisor
    result (as a user tool_result turn) so the executor can continue.
    """
    executor_content = (
        executor_response.get("content") if isinstance(executor_response, dict) else []
    ) or []
    tool_use_id = advisor_use_block.get("id", "")
    return [
        *messages,
        {"role": "assistant", "content": executor_content},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": advisor_text,
                }
            ],
        },
    ]


def _inject_max_uses_error(
    messages: List[Dict],
    executor_response: Any,
    advisor_use_block: Dict,
) -> List[Dict]:
    """
    Inject a max_uses_exceeded error tool_result so the executor continues
    without further advisor calls (mirrors Anthropic's server-side behaviour).
    """
    executor_content = (
        executor_response.get("content") if isinstance(executor_response, dict) else []
    ) or []
    tool_use_id = advisor_use_block.get("id", "")
    return [
        *messages,
        {"role": "assistant", "content": executor_content},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": "Advisor unavailable: max_uses limit reached. Continue without advisor guidance.",
                }
            ],
        },
    ]


async def _call_messages_handler(
    model: str,
    messages: List[Dict],
    tools: Optional[List[Dict]],
    stream: bool,
    max_tokens: int,
    custom_llm_provider: Optional[str],
    **kwargs,
) -> Any:
    """
    Dispatch an orchestration sub-call (executor turn) by invoking the
    inner ``anthropic_messages_handler`` directly, **bypassing** the @client
    wrapper around ``anthropic_messages``.

    Why bypass @client here:
        - ``anthropic_messages`` is @client-decorated, which ``kwargs.pop``s
          ``_is_litellm_internal_call`` before the body runs. Once popped, the
          flag is gone from the kwargs forwarded to the inner
          ``litellm.aresponses()`` / ``litellm.acompletion()`` call, so *those*
          inner @client wrappers emit their own log rows.
        - For advisor orchestration we want exactly ONE log row — the outer
          ``anthropic_messages`` call the proxy received. Calling the inner
          handler directly keeps ``_is_litellm_internal_call=True`` in kwargs
          all the way down, so the inner aresponses/acompletion skip logging
          and the outer call's @client emits the single aggregated entry.

    The shared ``litellm_logging_obj`` is intentionally passed through in
    kwargs so the inner aresponses/acompletion @client populates provider
    metadata (``custom_llm_provider``, ``api_base``, ``model_id``) on the
    outer log row.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages_handler,
    )

    kwargs["is_async"] = True
    result = anthropic_messages_handler(
        model=model,
        messages=messages,
        tools=tools,
        stream=stream,
        max_tokens=max_tokens,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )
    if asyncio.iscoroutine(result):
        return await result
    return result


async def _call_advisor_with_router(
    model: str,
    messages: List[Dict],
    max_tokens: int,
    metadata: Optional[Dict] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Any:
    """
    Call the advisor model via ``llm_router.acompletion()`` (proxy) or
    ``litellm.acompletion()`` (SDK-only).

    Returns a dict in Anthropic Messages format so the orchestration loop
    can process it uniformly.
    """
    import litellm as _litellm

    llm_router = None
    try:
        from litellm.proxy.proxy_server import llm_router as _router

        llm_router = _router
    except ImportError:
        pass

    # Mark as internal so the @client decorator on acompletion skips
    # emitting a log row — only the outer anthropic_messages call should
    # produce a single aggregated log entry for the advisor request.
    kwargs: Dict[str, Any] = {"_is_litellm_internal_call": True}
    if metadata is not None:
        kwargs["metadata"] = metadata
    if api_key is not None:
        kwargs["api_key"] = api_key
    if api_base is not None:
        kwargs["api_base"] = api_base

    # Same translation path as ``/v1/messages`` → non-Anthropic model: Anthropic
    # request shape → Chat Completions kwargs for the target provider.
    (
        completion_kwargs,
        _tool_name_mapping,
    ) = LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
        max_tokens=max_tokens,
        messages=messages,
        model=model,
        metadata=metadata,
        stream=False,
        extra_kwargs=kwargs,
    )
    # Inner advisor call is always a plain completion (no tools).
    completion_kwargs["tools"] = None

    openai_response = None
    if llm_router is not None:
        try:
            openai_response = await llm_router.acompletion(**completion_kwargs)
        except Exception:
            verbose_logger.debug(
                "AdvisorOrchestration: Router call for advisor model '%s' failed, "
                "falling back to direct litellm.acompletion()",
                model,
            )

    if openai_response is None:
        openai_response = await _litellm.acompletion(**completion_kwargs)

    return _openai_response_to_anthropic_dict(openai_response)

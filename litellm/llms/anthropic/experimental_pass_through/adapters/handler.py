from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Coroutine,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.asyncify import run_async_function
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    AnthropicAdapter,
)
from litellm.llms.anthropic.experimental_pass_through.context_management import (
    AnthropicContextManagementError,
    PolyfillResult,
    apply_context_management,
)
from litellm.llms.anthropic.experimental_pass_through.utils import (
    is_reasoning_auto_summary_enabled,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.utils import ModelResponse
from litellm.utils import get_model_info

if TYPE_CHECKING:
    pass


# Anthropic-only keys already mapped by the translator; strip on extra_kwargs re-merge.
ANTHROPIC_ONLY_REQUEST_KEYS: frozenset[str] = frozenset({"output_config"})


def _messages_have_compaction_block(messages: List[Dict]) -> bool:
    """Return True when any message carries a ``compaction`` content block."""
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "compaction":
                return True
    return False


def _extract_proxy_litellm_metadata(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return ``kwargs["litellm_metadata"]`` when it's a dict; ``None`` otherwise.

    The proxy attaches its auth/spend-attribution fields (``user_api_key``,
    ``user_api_key_team_id``, ``litellm_call_id``, the full ``UserAPIKeyAuth``
    object under ``user_api_key_auth``, ...) to ``data["litellm_metadata"]``
    for ``/v1/messages`` (see
    ``LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata`` and
    ``LITELLM_METADATA_ROUTES``). The Anthropic-shape ``metadata`` arg only
    carries ``user_id`` and must not be conflated. Returns ``None`` for SDK
    callers that bypass the proxy entirely.
    """
    litellm_metadata = kwargs.get("litellm_metadata")
    if not isinstance(litellm_metadata, dict):
        return None
    return litellm_metadata


async def _prepare_context_managed_request(
    *,
    model: str,
    messages: List[Dict],
    tools: Optional[List[Dict]],
    system: Optional[Any],
    context_management_spec: Any,
    litellm_metadata: Optional[Dict],
    drop_params: Optional[bool],
    llm_router: Any,
    user_api_key_auth: Any = None,
) -> Optional[PolyfillResult]:
    """Apply client compaction history, then optional context_management polyfill."""
    from litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact import (
        apply_client_compaction_block_history,
    )

    # Skip the client-history pre-processing when a ``compact_20260112``
    # polyfill spec will run: that editor already slices around any client-sent
    # compaction block in its Phase A (and uses the full post-compaction tail
    # for its token-threshold check). Pre-collapsing to just the latest user
    # question here would starve the polyfill of conversation context and
    # silently drop intermediate turns.
    polyfill_will_run = _polyfill_will_run(
        context_management_spec=context_management_spec,
        drop_params=drop_params,
    )

    if polyfill_will_run:
        history_result: Optional[PolyfillResult] = None
        working_messages: List[Dict] = messages
        working_system: Optional[Any] = system
    else:
        history_result = apply_client_compaction_block_history(
            messages=cast(List[Dict[str, Any]], messages),
            system=system,
        )
        working_messages = history_result.messages if history_result is not None else messages
        working_system = history_result.system if history_result is not None else system

    polyfill_result = await _run_polyfill_if_enabled(
        model=model,
        messages=working_messages,
        tools=tools,
        system=working_system,
        context_management_spec=context_management_spec,
        litellm_metadata=litellm_metadata,
        drop_params=drop_params,
        llm_router=llm_router,
        user_api_key_auth=user_api_key_auth,
    )

    if polyfill_result is not None:
        return polyfill_result

    # Safety net: if we skipped client-history pre-processing because a
    # ``compact_20260112`` polyfill was expected to handle the compaction
    # block itself but the polyfill ultimately did not produce a result
    # (e.g. it crashed and was best-effort swallowed in
    # ``_run_polyfill_if_enabled``), apply the slice-only fallback now so
    # Anthropic-specific ``compaction`` content blocks don't leak through
    # to non-Anthropic backends that would reject them.
    if polyfill_will_run and history_result is None:
        history_result = apply_client_compaction_block_history(
            messages=cast(List[Dict[str, Any]], messages),
            system=system,
        )
    return history_result


def _polyfill_will_run(
    *,
    context_management_spec: Any,
    drop_params: Optional[bool],
) -> bool:
    """Return True when ``compact_20260112`` will run via the polyfill dispatcher.

    Mirrors the gating in ``_run_polyfill_if_enabled``: an empty spec or
    effective ``drop_params`` short-circuits the polyfill. The pre-processing
    skip only applies when the dispatcher will actually invoke
    ``apply_compact_20260112`` (which has its own compaction-block slicing).
    """
    edits = _normalize_spec_edits(
        context_management_spec=context_management_spec,
        drop_params=drop_params,
    )
    if edits is None:
        return False

    from litellm.llms.anthropic.experimental_pass_through.context_management.constants import (
        COMPACT_EDIT_TYPE,
    )

    return any(isinstance(edit, dict) and edit.get("type") == COMPACT_EDIT_TYPE for edit in edits)


def _spec_has_non_compact_edits(
    *,
    context_management_spec: Any,
    drop_params: Optional[bool],
) -> bool:
    """Return True when the spec includes edits other than ``compact_20260112``.

    Used to decide whether a polyfill failure can be silently swallowed
    (compact-only specs have a safe compaction-block slicing fallback) or
    must be surfaced (other editors like ``clear_tool_uses_20250919`` have
    no slice-only fallback and would otherwise be dropped without notice).
    """
    edits = _normalize_spec_edits(
        context_management_spec=context_management_spec,
        drop_params=drop_params,
    )
    if edits is None:
        return False

    from litellm.llms.anthropic.experimental_pass_through.context_management.constants import (
        COMPACT_EDIT_TYPE,
    )

    return any(
        isinstance(edit, dict) and isinstance(edit.get("type"), str) and edit.get("type") != COMPACT_EDIT_TYPE
        for edit in edits
    )


def _normalize_spec_edits(
    *,
    context_management_spec: Any,
    drop_params: Optional[bool],
) -> Optional[List[Dict[str, Any]]]:
    """Return the normalized ``edits`` list, or ``None`` if the polyfill won't run.

    Delegates spec-shape normalization to the dispatcher's ``_normalize_spec``
    so the prediction here can't drift from what the dispatcher actually does.
    """
    if not context_management_spec:
        return None

    effective_drop_params = drop_params if drop_params is not None else litellm.drop_params
    if effective_drop_params:
        return None

    from litellm.llms.anthropic.experimental_pass_through.context_management.dispatcher import (
        _normalize_spec,
    )

    try:
        return _normalize_spec(context_management_spec)
    except Exception:
        return None


async def _run_polyfill_if_enabled(
    *,
    model: str,
    messages: List[Dict],
    tools: Optional[List[Dict]],
    system: Optional[Any],
    context_management_spec: Any,
    litellm_metadata: Optional[Dict],
    drop_params: Optional[bool],
    llm_router: Any,
    user_api_key_auth: Any = None,
) -> Optional[PolyfillResult]:
    """Run the async context_management polyfill if a spec is present.

    Returns ``None`` when the spec is empty or drop_params is on. Raises
    ``AnthropicContextManagementError`` so the /v1/messages endpoint can
    emit an Anthropic-format 400. All other exceptions are best-effort
    swallowed (matches v0 behavior).
    """
    if not context_management_spec:
        return None

    effective_drop_params = drop_params if drop_params is not None else litellm.drop_params
    if effective_drop_params:
        return None

    try:
        return await apply_context_management(
            model=model,
            messages=messages,
            tools=tools,
            system=system,
            context_management_spec=context_management_spec,
            litellm_metadata=litellm_metadata,
            llm_router=llm_router,
            user_api_key_auth=user_api_key_auth,
        )
    except AnthropicContextManagementError:
        # Surface validation errors so the endpoint can emit an Anthropic-format
        # 400. Other exception types fall into the best-effort branch below.
        raise
    except Exception as e:
        verbose_logger.exception("context_management polyfill: skipping edits due to error: %s", e)
        # Best-effort swallow is only safe for compact-only specs, where the
        # caller's compaction-block-slicing safety net produces a correct
        # (if degraded) result. When the spec also requested non-compact
        # edits (e.g. ``clear_tool_uses_20250919``), the safety net does
        # NOT re-run those editors, so silently returning ``None`` would
        # drop them with no error surface. Raise instead so the endpoint
        # emits an Anthropic-format error.
        if _spec_has_non_compact_edits(
            context_management_spec=context_management_spec,
            drop_params=drop_params,
        ):
            raise AnthropicContextManagementError(
                status_code=500,
                message=f"context_management polyfill failed: {e}",
            ) from e
        return None


########################################################
# init adapter
ANTHROPIC_ADAPTER = AnthropicAdapter()
########################################################


class LiteLLMMessagesToCompletionTransformationHandler:
    @staticmethod
    def _route_openai_thinking_to_responses_api_if_needed(
        completion_kwargs: Dict[str, Any],
        *,
        thinking: Optional[Dict[str, Any]],
    ) -> None:
        """
        When users call `litellm.anthropic.messages.*` with a non-Anthropic model and
        `thinking={"type": "enabled", ...}`, LiteLLM converts this into OpenAI
        `reasoning_effort`.

        For OpenAI models, Chat Completions typically does not return reasoning text
        (only token accounting). To return a thinking-like content block in the
        Anthropic response format, we route the request through OpenAI's Responses API.
        If the user provides a `summary` field in the thinking dict, it is passed
        through to the OpenAI reasoning params (opt-in per OpenAI spec).
        """
        custom_llm_provider = completion_kwargs.get("custom_llm_provider")
        if custom_llm_provider is None:
            try:
                _, inferred_provider, _, _ = litellm.utils.get_llm_provider(
                    model=cast(str, completion_kwargs.get("model"))
                )
                custom_llm_provider = inferred_provider
            except Exception:
                custom_llm_provider = None

        if custom_llm_provider != "openai":
            return

        if not isinstance(thinking, dict) or thinking.get("type") != "enabled":
            return

        model = completion_kwargs.get("model")
        try:
            model_info = get_model_info(model=cast(str, model), custom_llm_provider=custom_llm_provider)
            if model_info and model_info.get("supports_reasoning") is False:
                # Model doesn't support reasoning/responses API, don't route
                return
        except Exception:
            pass

        if isinstance(model, str) and model and not model.startswith("responses/"):
            # Prefix model with "responses/" to route to OpenAI Responses API
            completion_kwargs["model"] = f"responses/{model}"

        auto_summary = is_reasoning_auto_summary_enabled()

        reasoning_effort = completion_kwargs.get("reasoning_effort")
        summary = thinking.get("summary")
        if isinstance(reasoning_effort, str) and reasoning_effort:
            reasoning_dict: Dict[str, Any] = {"effort": reasoning_effort}
            if summary:
                reasoning_dict["summary"] = summary
            elif auto_summary:
                reasoning_dict["summary"] = "detailed"
            completion_kwargs["reasoning_effort"] = reasoning_dict
        elif isinstance(reasoning_effort, dict):
            if "summary" not in reasoning_effort and "generate_summary" not in reasoning_effort:
                effective_summary = summary if summary else ("detailed" if auto_summary else None)
                if effective_summary:
                    updated_reasoning_effort = dict(reasoning_effort)
                    updated_reasoning_effort["summary"] = effective_summary
                    completion_kwargs["reasoning_effort"] = updated_reasoning_effort

    @staticmethod
    def _normalize_reasoning_effort(
        completion_kwargs: Dict[str, Any],
    ) -> None:
        """
        Normalize reasoning_effort values based on target model capabilities.

        Handles both string ("max") and dict ({"effort": "max", "summary": ...})
        formats. Uses model registry to check supports_xhigh/supports_minimal.
        """
        from litellm.llms.anthropic.experimental_pass_through.utils import (
            normalize_reasoning_effort_value,
        )

        reasoning_effort = completion_kwargs.get("reasoning_effort")
        if reasoning_effort is None:
            return

        model = cast(str, completion_kwargs.get("model", ""))
        custom_llm_provider = completion_kwargs.get("custom_llm_provider")

        if isinstance(reasoning_effort, str):
            normalized = normalize_reasoning_effort_value(
                reasoning_effort, model=model, custom_llm_provider=custom_llm_provider
            )
            if normalized != reasoning_effort:
                completion_kwargs["reasoning_effort"] = normalized
        elif isinstance(reasoning_effort, dict) and "effort" in reasoning_effort:
            effort = reasoning_effort["effort"]
            normalized = normalize_reasoning_effort_value(effort, model=model, custom_llm_provider=custom_llm_provider)
            if normalized != effort:
                completion_kwargs["reasoning_effort"] = {
                    **reasoning_effort,
                    "effort": normalized,
                }

    @staticmethod
    def _prepare_completion_kwargs(
        *,
        max_tokens: int,
        messages: List[Dict],
        model: str,
        metadata: Optional[Dict] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: Optional[bool] = False,
        system: Optional[Union[str, List[Dict[str, Any]]]] = None,
        temperature: Optional[float] = None,
        thinking: Optional[Dict] = None,
        tool_choice: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        output_format: Optional[Dict] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Prepare kwargs for litellm.completion/acompletion.

        Returns:
            Tuple of (completion_kwargs, tool_name_mapping)
            - tool_name_mapping maps truncated tool names back to original names
              for tools that exceeded OpenAI's 64-char limit
        """
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObject,
        )

        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        if metadata:
            request_data["metadata"] = metadata
        if stop_sequences:
            request_data["stop_sequences"] = stop_sequences
        if system:
            request_data["system"] = system
        if temperature is not None:
            request_data["temperature"] = temperature
        if thinking:
            request_data["thinking"] = thinking
        if tool_choice:
            request_data["tool_choice"] = tool_choice
        if tools:
            request_data["tools"] = tools
        if top_k is not None:
            request_data["top_k"] = top_k
        if top_p is not None:
            request_data["top_p"] = top_p
        if output_format:
            request_data["output_format"] = output_format

        # Extract output_config from extra_kwargs so the translator can use it
        # (e.g. output_config.effort for adaptive thinking → reasoning_effort,
        # output_config.format → response_format for structured outputs).
        # Use explicit None check rather than `or {}` so an explicit empty dict
        # caller-passed argument is preserved (matters for tests that drive
        # the fallback inference path).
        extra_kwargs = extra_kwargs if extra_kwargs is not None else {}
        if "output_config" in extra_kwargs:
            request_data["output_config"] = extra_kwargs["output_config"]

        (
            openai_request,
            tool_name_mapping,
        ) = ANTHROPIC_ADAPTER.translate_completion_input_params_with_tool_mapping(request_data)

        if openai_request is None:
            raise ValueError("Failed to translate request to OpenAI format")

        completion_kwargs: Dict[str, Any] = dict(openai_request)

        if stream:
            completion_kwargs["stream"] = stream
            completion_kwargs["stream_options"] = {
                "include_usage": True,
            }

        # Keys that must NOT be forwarded as raw extras into the OpenAI-format
        # ``completion_kwargs`` after translation. The translator above has
        # already consumed the meaningful parts of these inputs (e.g.
        # ``output_config.format`` → ``response_format``, ``output_config.effort``
        # → ``reasoning_effort`` for non-Claude targets). Re-adding the raw
        # Anthropic-shaped key here causes 400 "Extra inputs are not permitted"
        # on non-Anthropic backends (Azure OpenAI, Fireworks, Bedrock Nova,
        # etc.) and is silently lossy on Anthropic-family targets, which would
        # see the translated key ``response_format`` AND a duplicate, conflicting
        # ``output_config``.
        #
        # Maintainability: when adding a new Anthropic-only request param to
        # ``AnthropicMessagesRequestOptionalParams``, also extend
        # ``ANTHROPIC_ONLY_REQUEST_KEYS`` here so it doesn't silently leak.
        excluded_keys = ANTHROPIC_ONLY_REQUEST_KEYS | {"anthropic_messages"}
        # NOTE: extra_kwargs was already coerced from None to {} at the top of
        # this method (line ~220). It is guaranteed to be a dict here.
        for key, value in extra_kwargs.items():
            if key == "litellm_logging_obj" and value is not None and isinstance(value, LiteLLMLoggingObject):
                from litellm.types.utils import CallTypes

                setattr(value, "call_type", CallTypes.anthropic_messages.value)
                setattr(value, "stream_options", completion_kwargs.get("stream_options"))
            if key not in excluded_keys and key not in completion_kwargs and value is not None:
                completion_kwargs[key] = value

        # Normalize reasoning_effort based on model capabilities
        # (e.g. "max" → "xhigh"/"high", "minimal" → "low" if unsupported)
        # Must run BEFORE _route_openai_thinking, which prepends "responses/"
        # to the model name and would break get_model_info() lookups.
        LiteLLMMessagesToCompletionTransformationHandler._normalize_reasoning_effort(completion_kwargs)

        LiteLLMMessagesToCompletionTransformationHandler._route_openai_thinking_to_responses_api_if_needed(
            completion_kwargs,
            thinking=thinking,
        )

        return completion_kwargs, tool_name_mapping

    @staticmethod
    async def async_anthropic_messages_handler(
        max_tokens: int,
        messages: List[Dict],
        model: str,
        metadata: Optional[Dict] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: Optional[bool] = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking: Optional[Dict] = None,
        tool_choice: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        output_format: Optional[Dict] = None,
        **kwargs,
    ) -> Union[AnthropicMessagesResponse, AsyncIterator[Any], Iterator[bytes]]:
        """Handle non-Anthropic models asynchronously using the adapter"""
        context_management = kwargs.pop("context_management", None)
        drop_params: Optional[bool] = kwargs.get("drop_params", None)
        litellm_router = kwargs.pop("litellm_router", None)
        if litellm_router is None:
            try:
                from litellm.proxy.proxy_server import llm_router as _proxy_router

                litellm_router = _proxy_router
            except Exception:
                pass

        proxy_litellm_metadata = _extract_proxy_litellm_metadata(kwargs)
        user_api_key_auth = (
            proxy_litellm_metadata.get("user_api_key_auth") if proxy_litellm_metadata is not None else None
        )

        polyfill_result = await _prepare_context_managed_request(
            model=model,
            messages=messages,
            tools=tools,
            system=system,
            context_management_spec=context_management,
            litellm_metadata=proxy_litellm_metadata,
            drop_params=drop_params,
            llm_router=litellm_router,
            user_api_key_auth=user_api_key_auth,
        )

        effective_messages = polyfill_result.messages if polyfill_result is not None else messages
        effective_system = polyfill_result.system if polyfill_result is not None else system

        (
            completion_kwargs,
            tool_name_mapping,
        ) = LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
            max_tokens=max_tokens,
            messages=effective_messages,
            model=model,
            metadata=metadata,
            stop_sequences=stop_sequences,
            stream=stream,
            system=effective_system,
            temperature=temperature,
            thinking=thinking,
            tool_choice=tool_choice,
            tools=tools,
            top_k=top_k,
            top_p=top_p,
            output_format=output_format,
            extra_kwargs=kwargs,
        )

        completion_response = await litellm.acompletion(**completion_kwargs)

        if stream:
            transformed_stream = ANTHROPIC_ADAPTER.translate_completion_output_params_streaming(
                completion_response,
                model=model,
                tool_name_mapping=tool_name_mapping,
                polyfill_result=polyfill_result,
                is_async=True,
            )
            if transformed_stream is not None:
                return transformed_stream
            raise ValueError("Failed to transform streaming response")
        else:
            anthropic_response = ANTHROPIC_ADAPTER.translate_completion_output_params(
                cast(ModelResponse, completion_response),
                tool_name_mapping=tool_name_mapping,
                polyfill_result=polyfill_result,
            )
            if anthropic_response is not None:
                return anthropic_response
            raise ValueError("Failed to transform response to Anthropic format")

    @staticmethod
    def anthropic_messages_handler(
        max_tokens: int,
        messages: List[Dict],
        model: str,
        metadata: Optional[Dict] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: Optional[bool] = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking: Optional[Dict] = None,
        tool_choice: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        output_format: Optional[Dict] = None,
        _is_async: bool = False,
        **kwargs,
    ) -> Union[
        AnthropicMessagesResponse,
        Iterator[bytes],
        AsyncIterator[Any],
        Coroutine[
            Any,
            Any,
            Union[AnthropicMessagesResponse, AsyncIterator[Any], Iterator[bytes]],
        ],
    ]:
        """Handle non-Anthropic models using the adapter."""
        if _is_async is True:
            return LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
                max_tokens=max_tokens,
                messages=messages,
                model=model,
                metadata=metadata,
                stop_sequences=stop_sequences,
                stream=stream,
                system=system,
                temperature=temperature,
                thinking=thinking,
                tool_choice=tool_choice,
                tools=tools,
                top_k=top_k,
                top_p=top_p,
                output_format=output_format,
                **kwargs,
            )

        # Run the context_management polyfill on the sync path too so that
        # ``litellm.messages.create()`` callers don't silently lose edits like
        # ``clear_tool_uses_20250919``. The dispatcher is async (so the
        # ``compact_20260112`` editor can ``await`` the summarization model);
        # bridge to it via ``run_async_function``.
        context_management = kwargs.pop("context_management", None)
        drop_params: Optional[bool] = kwargs.get("drop_params", None)
        # Deliberately do NOT auto-attach the proxy ``llm_router`` here:
        # ``run_async_function`` spawns a new event loop in a worker thread
        # to bridge to the async dispatcher, but the proxy router's httpx
        # ``AsyncClient`` instances are bound to the proxy's main event loop.
        # Reusing them from the new thread's loop violates httpx's single-loop
        # invariant and can raise ``RuntimeError: Event loop is closed`` or
        # produce stalled connections. The summary editor falls back to
        # ``litellm.acompletion`` (which creates a fresh client per call) when
        # ``llm_router`` is ``None``, which is safe to call from the bridged
        # loop. The async ``async_anthropic_messages_handler`` path is
        # unaffected because it ``await``s within the original event loop.
        litellm_router = kwargs.pop("litellm_router", None)

        # Skip the async bridge entirely when there is nothing for either the
        # polyfill or the client-history slice-only fallback to do. The vast
        # majority of sync ``litellm.messages.create()`` requests carry no
        # ``context_management`` spec and no client-sent ``compaction`` block,
        # and bridging through a worker-thread event loop just to discover
        # there is no work is pure overhead.
        if context_management is None and not _messages_have_compaction_block(messages):
            polyfill_result: Optional[PolyfillResult] = None
        else:
            proxy_litellm_metadata = _extract_proxy_litellm_metadata(kwargs)
            user_api_key_auth = (
                proxy_litellm_metadata.get("user_api_key_auth") if proxy_litellm_metadata is not None else None
            )
            polyfill_result = run_async_function(
                _prepare_context_managed_request,
                model=model,
                messages=messages,
                tools=tools,
                system=system,
                context_management_spec=context_management,
                litellm_metadata=proxy_litellm_metadata,
                drop_params=drop_params,
                llm_router=litellm_router,
                user_api_key_auth=user_api_key_auth,
            )

        effective_messages = polyfill_result.messages if polyfill_result is not None else messages
        effective_system = polyfill_result.system if polyfill_result is not None else system

        (
            completion_kwargs,
            tool_name_mapping,
        ) = LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
            max_tokens=max_tokens,
            messages=effective_messages,
            model=model,
            metadata=metadata,
            stop_sequences=stop_sequences,
            stream=stream,
            system=effective_system,
            temperature=temperature,
            thinking=thinking,
            tool_choice=tool_choice,
            tools=tools,
            top_k=top_k,
            top_p=top_p,
            output_format=output_format,
            extra_kwargs=kwargs,
        )

        completion_response = litellm.completion(**completion_kwargs)

        if stream:
            transformed_stream = ANTHROPIC_ADAPTER.translate_completion_output_params_streaming(
                completion_response,
                model=model,
                tool_name_mapping=tool_name_mapping,
                polyfill_result=polyfill_result,
                is_async=False,
            )
            if transformed_stream is not None:
                return transformed_stream
            raise ValueError("Failed to transform streaming response")
        else:
            anthropic_response = ANTHROPIC_ADAPTER.translate_completion_output_params(
                cast(ModelResponse, completion_response),
                tool_name_mapping=tool_name_mapping,
                polyfill_result=polyfill_result,
            )
            if anthropic_response is not None:
                return anthropic_response
            raise ValueError("Failed to transform response to Anthropic format")

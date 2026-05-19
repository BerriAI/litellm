from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import litellm
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    AnthropicAdapter,
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


# Anthropic-only fields that the translator above already maps into the
# OpenAI-format completion_kwargs (output_config → reasoning_effort /
# response_format, etc.). They must be filtered out of the raw
# extra_kwargs re-merge below or non-Anthropic backends reject the call
# with 400 "Extra inputs are not permitted". Add new entries here when
# extending AnthropicMessagesRequestOptionalParams with another Anthropic-
# specific key.
ANTHROPIC_ONLY_REQUEST_KEYS: frozenset[str] = frozenset({"output_config"})

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
            model_info = get_model_info(
                model=cast(str, model), custom_llm_provider=custom_llm_provider
            )
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
            if (
                "summary" not in reasoning_effort
                and "generate_summary" not in reasoning_effort
            ):
                effective_summary = (
                    summary if summary else ("detailed" if auto_summary else None)
                )
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
            normalized = normalize_reasoning_effort_value(
                effort, model=model, custom_llm_provider=custom_llm_provider
            )
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
        system: Optional[str] = None,
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
        ) = ANTHROPIC_ADAPTER.translate_completion_input_params_with_tool_mapping(
            request_data
        )

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
            if (
                key == "litellm_logging_obj"
                and value is not None
                and isinstance(value, LiteLLMLoggingObject)
            ):
                from litellm.types.utils import CallTypes

                setattr(value, "call_type", CallTypes.anthropic_messages.value)
                setattr(
                    value, "stream_options", completion_kwargs.get("stream_options")
                )
            if (
                key not in excluded_keys
                and key not in completion_kwargs
                and value is not None
            ):
                completion_kwargs[key] = value

        # Normalize reasoning_effort based on model capabilities
        # (e.g. "max" → "xhigh"/"high", "minimal" → "low" if unsupported)
        # Must run BEFORE _route_openai_thinking, which prepends "responses/"
        # to the model name and would break get_model_info() lookups.
        LiteLLMMessagesToCompletionTransformationHandler._normalize_reasoning_effort(
            completion_kwargs
        )

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
    ) -> Union[AnthropicMessagesResponse, AsyncIterator]:
        """Handle non-Anthropic models asynchronously using the adapter"""
        (
            completion_kwargs,
            tool_name_mapping,
        ) = LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
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
            extra_kwargs=kwargs,
        )

        completion_response = await litellm.acompletion(**completion_kwargs)

        if stream:
            transformed_stream = (
                ANTHROPIC_ADAPTER.translate_completion_output_params_streaming(
                    completion_response,
                    model=model,
                    tool_name_mapping=tool_name_mapping,
                )
            )
            if transformed_stream is not None:
                return transformed_stream
            raise ValueError("Failed to transform streaming response")
        else:
            anthropic_response = ANTHROPIC_ADAPTER.translate_completion_output_params(
                cast(ModelResponse, completion_response),
                tool_name_mapping=tool_name_mapping,
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
        AsyncIterator[Any],
        Coroutine[Any, Any, Union[AnthropicMessagesResponse, AsyncIterator[Any]]],
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

        (
            completion_kwargs,
            tool_name_mapping,
        ) = LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
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
            extra_kwargs=kwargs,
        )

        completion_response = litellm.completion(**completion_kwargs)

        if stream:
            transformed_stream = (
                ANTHROPIC_ADAPTER.translate_completion_output_params_streaming(
                    completion_response,
                    model=model,
                    tool_name_mapping=tool_name_mapping,
                )
            )
            if transformed_stream is not None:
                return transformed_stream
            raise ValueError("Failed to transform streaming response")
        else:
            anthropic_response = ANTHROPIC_ADAPTER.translate_completion_output_params(
                cast(ModelResponse, completion_response),
                tool_name_mapping=tool_name_mapping,
            )
            if anthropic_response is not None:
                return anthropic_response
            raise ValueError("Failed to transform response to Anthropic format")

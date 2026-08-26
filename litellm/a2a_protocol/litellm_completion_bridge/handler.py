"""
Handler for A2A to LiteLLM completion bridge.

Routes A2A requests through litellm.acompletion based on custom_llm_provider.

A2A Streaming Events (in order):
1. Task event (kind: "task") - Initial task creation with status "submitted"
2. Status update (kind: "status-update") - Status change to "working"
3. Artifact update (kind: "artifact-update") - Content/artifact delivery
4. Status update (kind: "status-update") - Final status "completed" with final=true
"""

from collections.abc import AsyncIterator, Callable, Coroutine, Mapping
from typing import Any, Final

import litellm
from litellm._logging import verbose_logger
from litellm.a2a_protocol.litellm_completion_bridge.transformation import (
    A2ACompletionBridgeTransformation,
    A2AStreamingContext,
)
from litellm.a2a_protocol.providers.config_manager import A2AProviderConfigManager
from litellm.interactions.agents.utils import merge_agent_headers
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponse

# litellm_params key carrying the authenticated principal (hashed virtual key) so
# A2A provider configs can scope provider-side state (e.g. LangFlow session memory)
# per key instead of trusting the client-supplied A2A contextId.
A2A_USER_API_KEY_HASH_PARAM: Final = "litellm_a2a_user_api_key_hash"

# Agent metadata fields stored in litellm_params that are not valid litellm.acompletion() kwargs
_AGENT_ONLY_PARAMS: Final = frozenset(
    {
        "is_public",
        "agent_name",
        "agent_id",
        "agent_card_params",
        A2A_USER_API_KEY_HASH_PARAM,
    }
)


class A2ACompletionBridgeHandler:
    """
    Static methods for handling A2A requests via LiteLLM completion.
    """

    @staticmethod
    def _merge_stream_values(previous: object, current: object) -> object:
        if isinstance(previous, Mapping) and isinstance(current, Mapping):
            merged = dict(previous)
            for key, value in current.items():
                merged[key] = (
                    A2ACompletionBridgeHandler._merge_stream_values(merged[key], value) if key in merged else value
                )
            return merged
        if isinstance(previous, list) and isinstance(current, list):
            return [*previous, *current]
        return current

    @staticmethod
    def _build_completion_params(
        params: dict[str, Any],
        litellm_params: Mapping[str, Any],
        api_base: str | None,
        agent_extra_headers: Mapping[str, str] | None,
        *,
        stream: bool,
    ) -> Mapping[str, object]:
        # Extract message from params
        message: Final = params.get("message", {})

        # Transform A2A message to OpenAI format
        supplied_messages: Final = params.get("messages")
        openai_messages: Final = (
            supplied_messages
            if isinstance(supplied_messages, list)
            else A2ACompletionBridgeTransformation.a2a_message_to_openai_messages(message)
        )

        # Get completion params
        custom_llm_provider: Final = litellm_params.get("custom_llm_provider")
        model: Final[str] = litellm_params.get("model", "agent")

        # Build full model string if provider specified
        # Skip prepending if model already starts with the provider prefix
        if custom_llm_provider and not model.startswith(f"{custom_llm_provider}/"):
            full_model = f"{custom_llm_provider}/{model}"
        else:
            full_model = model

        if stream:
            verbose_logger.info("A2A completion bridge streaming: model=%s, api_base=%s", full_model, api_base)
        else:
            verbose_logger.info("A2A completion bridge: model=%s, api_base=%s", full_model, api_base)

        # Build completion params dict
        completion_params: Final[dict[str, Any]] = {
            "model": full_model,
            "messages": openai_messages,
            "api_base": api_base,
            "stream": stream,
        }
        configured_headers: Final[object] = litellm_params.get("extra_headers") or litellm_params.get("headers")
        # Add litellm_params (contains api_key, client_id, client_secret, tenant_id, etc.)
        litellm_params_to_add: Final = {
            k: v
            for k, v in litellm_params.items()
            if k not in ("model", "custom_llm_provider", "extra_headers", "headers", "api_base", "stream")
            and k not in _AGENT_ONLY_PARAMS
        }
        completion_params.update(litellm_params_to_add)
        # Apply forward metadata AFTER the litellm_params merge so the helper
        # sees any agent-owner-configured ``extra_body.metadata`` and can keep
        # those keys authoritative over the client-supplied A2A metadata.
        A2ACompletionBridgeTransformation.apply_forward_metadata_to_completion_params(
            completion_params=completion_params,
            a2a_message=message,
            params=params,
        )

        if agent_extra_headers or configured_headers:
            completion_params["extra_headers"] = merge_agent_headers(
                dynamic_headers=agent_extra_headers,
                static_headers=configured_headers if isinstance(configured_headers, Mapping) else None,
            )

        return completion_params

    @staticmethod
    async def _acompletion(completion_params: Mapping[str, object]) -> ModelResponse | CustomStreamWrapper:
        acompletion_fn: Final[Callable[..., Coroutine[object, object, ModelResponse | CustomStreamWrapper]]] = vars(
            litellm
        )["acompletion"]
        return await acompletion_fn(**completion_params)

    @staticmethod
    async def handle_non_streaming(
        request_id: str,
        params: dict[str, object],
        litellm_params: dict[str, Any],
        api_base: str | None = None,
        agent_extra_headers: dict[str, str] | None = None,
        agent_static_headers: Mapping[str, str] | None = None,
        *,
        _skip_a2a_provider_routing: bool = False,
    ) -> dict[str, object]:
        """
        Handle non-streaming A2A request via litellm.acompletion.

        Args:
            request_id: A2A JSON-RPC request ID
            params: A2A MessageSendParams containing the message
            litellm_params: Agent's litellm_params (custom_llm_provider, model, etc.)
            api_base: API base URL from agent_card_params
            agent_extra_headers: Per-request headers (from x-a2a-{agent}-* rewrite and
                admin extra_headers) to forward on the upstream HTTP call.
            agent_static_headers: Configured headers for provider-specific routing.

        Returns:
            A2A SendMessageResponse dict
        """
        custom_llm_provider: Final = litellm_params.get("custom_llm_provider")
        if not _skip_a2a_provider_routing:
            a2a_provider_config: Final = A2AProviderConfigManager.get_provider_config(
                custom_llm_provider=custom_llm_provider,
                model=litellm_params.get("model"),
            )

            if a2a_provider_config is not None:
                verbose_logger.info("A2A: Using provider config for %s", custom_llm_provider)

                provider_params: Final = dict(params)
                if custom_llm_provider == "pydantic_ai_agents" or (
                    custom_llm_provider == "bedrock"
                    and isinstance(litellm_params.get("model"), str)
                    and "agentcore" in litellm_params["model"]
                ):
                    provider_params.pop("messages", None)
                provider_kwargs: Final[dict[str, Any]] = {
                    "request_id": request_id,
                    "params": provider_params,
                    "api_base": api_base,
                    "litellm_params": litellm_params,
                    "agent_extra_headers": agent_extra_headers,
                    "agent_static_headers": agent_static_headers,
                }
                if litellm_params.get("timeout") is not None:
                    provider_kwargs["timeout"] = litellm_params["timeout"]
                return await a2a_provider_config.handle_non_streaming(**provider_kwargs)

        completion_params: Final = A2ACompletionBridgeHandler._build_completion_params(
            params=params,
            litellm_params=litellm_params,
            api_base=api_base,
            agent_extra_headers=agent_extra_headers,
            stream=False,
        )

        # Call litellm.acompletion
        response: Final = await A2ACompletionBridgeHandler._acompletion(completion_params)

        # Transform response to A2A format
        a2a_response: Final = A2ACompletionBridgeTransformation.openai_response_to_a2a_response(
            response=response,
            request_id=request_id,
        )

        verbose_logger.info("A2A completion bridge completed: request_id=%s", request_id)

        return a2a_response

    @staticmethod
    async def handle_streaming(
        request_id: str,
        params: dict[str, Any],
        litellm_params: dict[str, Any],
        api_base: str | None = None,
        agent_extra_headers: dict[str, str] | None = None,
        agent_static_headers: Mapping[str, str] | None = None,
        *,
        _skip_a2a_provider_routing: bool = False,
    ) -> AsyncIterator[dict[str, object]]:
        """
        Handle streaming A2A request via litellm.acompletion with stream=True.

        Emits proper A2A streaming events:
        1. Task event (kind: "task") - Initial task with status "submitted"
        2. Status update (kind: "status-update") - Status "working"
        3. Artifact update (kind: "artifact-update") - Content delivery
        4. Status update (kind: "status-update") - Final "completed" status

        Args:
            request_id: A2A JSON-RPC request ID
            params: A2A MessageSendParams containing the message
            litellm_params: Agent's litellm_params (custom_llm_provider, model, etc.)
            api_base: API base URL from agent_card_params
            agent_extra_headers: Per-request headers (from x-a2a-{agent}-* rewrite and
                admin extra_headers) to forward on the upstream HTTP call.
            agent_static_headers: Configured headers for provider-specific routing.

        Yields:
            A2A streaming response events
        """
        custom_llm_provider: Final = litellm_params.get("custom_llm_provider")
        if not _skip_a2a_provider_routing:
            a2a_provider_config: Final = A2AProviderConfigManager.get_provider_config(
                custom_llm_provider=custom_llm_provider,
                model=litellm_params.get("model"),
            )

            if a2a_provider_config is not None:
                verbose_logger.info("A2A: Using provider config for %s (streaming)", custom_llm_provider)

                provider_params: Final = dict(params)
                if custom_llm_provider == "pydantic_ai_agents" or (
                    custom_llm_provider == "bedrock"
                    and isinstance(litellm_params.get("model"), str)
                    and "agentcore" in litellm_params["model"]
                ):
                    provider_params.pop("messages", None)
                provider_kwargs: Final[dict[str, Any]] = {
                    "request_id": request_id,
                    "params": provider_params,
                    "api_base": api_base,
                    "litellm_params": litellm_params,
                    "agent_extra_headers": agent_extra_headers,
                    "agent_static_headers": agent_static_headers,
                }
                if litellm_params.get("timeout") is not None:
                    provider_kwargs["timeout"] = litellm_params["timeout"]
                provider_stream: Final = a2a_provider_config.handle_streaming(**provider_kwargs)
                try:
                    async for chunk in provider_stream:
                        yield chunk
                finally:
                    close_provider_stream = getattr(provider_stream, "aclose", None)
                    if close_provider_stream is not None:
                        await close_provider_stream()

                return

        # Create streaming context
        ctx: Final = A2AStreamingContext(
            request_id=request_id,
            input_message=params.get("message", {}),
        )

        completion_params: Final = A2ACompletionBridgeHandler._build_completion_params(
            params=params,
            litellm_params=litellm_params,
            api_base=api_base,
            agent_extra_headers=agent_extra_headers,
            stream=True,
        )

        # 1. Emit initial task event (kind: "task", status: "submitted")
        task_event: Final = A2ACompletionBridgeTransformation.create_task_event(ctx)
        yield task_event

        # 2. Emit status update (kind: "status-update", status: "working")
        working_event: Final = A2ACompletionBridgeTransformation.create_status_update_event(
            ctx=ctx,
            state="working",
            final=False,
            message_text="Processing request...",
        )
        yield working_event

        # Call litellm.acompletion with streaming
        response: Final = await A2ACompletionBridgeHandler._acompletion(completion_params)

        # 3. Forward content as artifact updates
        accumulated_tool_calls: Final[list[object]] = []  # mutable-ok: collect streaming tool-call deltas
        choice_texts: dict[int, str] = {}
        choice_tool_calls: dict[int, list[object]] = {}
        choice_delta_fields: dict[int, dict[str, object]] = {}
        choice_logprobs: dict[int, dict[str, object]] = {}
        choice_finish_reasons: dict[int, str] = {}
        stream_metadata: dict[str, str] = {}
        stream_usage: object | None = None
        stream_finish_reason: str | None = None
        chunk_count = 0
        try:
            async for chunk in response:
                chunk_count += 1

                raw_usage = getattr(chunk, "usage", None)
                if isinstance(raw_usage, Mapping):
                    stream_usage = raw_usage
                else:
                    dump_usage = getattr(raw_usage, "model_dump", None)
                    if callable(dump_usage):
                        dumped_usage = dump_usage(exclude_none=True)
                        if isinstance(dumped_usage, Mapping):
                            stream_usage = dumped_usage
                    else:
                        dict_usage = getattr(raw_usage, "dict", None)
                        if callable(dict_usage):
                            dumped_usage = dict_usage(exclude_none=True)
                            if isinstance(dumped_usage, Mapping):
                                stream_usage = dumped_usage

                for metadata_name in ("system_fingerprint", "service_tier"):
                    metadata_value = getattr(chunk, metadata_name, None)
                    if not isinstance(metadata_value, str):
                        chunk_fields = A2ACompletionBridgeTransformation._model_dump(chunk)
                        metadata_value = chunk_fields.get(metadata_name)
                    if isinstance(metadata_value, str) and metadata_value:
                        stream_metadata[metadata_name] = metadata_value

                # Extract delta content
                choices = getattr(chunk, "choices", None) if chunk is not None else None
                if isinstance(choices, (list, tuple)):
                    for choice_position, choice in enumerate(choices):
                        raw_index = getattr(choice, "index", choice_position)
                        choice_index = raw_index if isinstance(raw_index, int) else choice_position
                        choice_texts.setdefault(choice_index, "")
                        raw_finish_reason = getattr(choice, "finish_reason", None)
                        if isinstance(raw_finish_reason, str) and raw_finish_reason:
                            choice_finish_reasons[choice_index] = raw_finish_reason
                            if choice_index == 0 or stream_finish_reason is None:
                                stream_finish_reason = raw_finish_reason
                        content = ""
                        delta = getattr(choice, "delta", None)
                        if delta:
                            raw_content = getattr(delta, "content", None)
                            content = raw_content if isinstance(raw_content, str) else ""
                            choice_texts[choice_index] += content
                            tool_calls = getattr(delta, "tool_calls", None)
                            if isinstance(tool_calls, (list, tuple)):
                                accumulated_tool_calls.extend(tool_calls)
                                choice_tool_calls.setdefault(choice_index, []).extend(tool_calls)
                            delta_fields = A2ACompletionBridgeTransformation._model_dump(delta)
                            if delta_fields:
                                choice_fields = choice_delta_fields.setdefault(choice_index, {})
                                for field, value in delta_fields.items():
                                    if field in {"content", "role", "tool_calls"} or value is None:
                                        continue
                                    previous = choice_fields.get(field)
                                    if (isinstance(previous, str) and isinstance(value, str)) or (
                                        isinstance(previous, list) and isinstance(value, list)
                                    ):
                                        choice_fields[field] = previous + value
                                    elif isinstance(previous, Mapping) and isinstance(value, Mapping):
                                        choice_fields[field] = {**previous, **value}
                                    else:
                                        choice_fields[field] = value

                        raw_logprobs = getattr(choice, "logprobs", None)
                        serialized_logprobs = A2ACompletionBridgeTransformation._model_dump(raw_logprobs)
                        if serialized_logprobs:
                            previous_logprobs = choice_logprobs.get(choice_index, {})
                            merged_logprobs = A2ACompletionBridgeHandler._merge_stream_values(
                                previous_logprobs, serialized_logprobs
                            )
                            if isinstance(merged_logprobs, dict):
                                choice_logprobs[choice_index] = merged_logprobs

                        if content:
                            artifact_event: Final = A2ACompletionBridgeTransformation.create_artifact_update_event(
                                ctx=ctx,
                                text=content,
                                index=choice_index,
                            )
                            yield artifact_event
        finally:
            close_response = getattr(response, "aclose", None)
            if close_response is not None:
                await close_response()

        # 4. Emit final status update (kind: "status-update", status: "completed", final: true)
        completed_event: Final = A2ACompletionBridgeTransformation.create_status_update_event(
            ctx=ctx,
            state="completed",
            final=True,
        )
        if accumulated_tool_calls:
            completed_event["result"]["tool_calls"] = accumulated_tool_calls
        if stream_finish_reason:
            completed_event["result"]["finish_reason"] = stream_finish_reason
        if stream_usage is not None:
            completed_event["usage"] = stream_usage
        for metadata_name, metadata_value in stream_metadata.items():
            completed_event[metadata_name] = metadata_value
        choice_indices = sorted(
            set(choice_texts)
            | set(choice_tool_calls)
            | set(choice_delta_fields)
            | set(choice_logprobs)
            | set(choice_finish_reasons)
        )
        if choice_indices:
            choice_payloads: list[dict[str, object]] = []
            for choice_index in choice_indices:
                choice_payload: dict[str, object] = {
                    "index": choice_index,
                    "message": {
                        "kind": "message",
                        "role": "agent",
                        "parts": [{"kind": "text", "text": ""}],
                        **(
                            {"tool_calls": choice_tool_calls[choice_index]}
                            if choice_tool_calls.get(choice_index)
                            else {}
                        ),
                    },
                    **(
                        {"finish_reason": choice_finish_reasons[choice_index]}
                        if choice_index in choice_finish_reasons
                        else {}
                    ),
                    **({"logprobs": choice_logprobs[choice_index]} if choice_index in choice_logprobs else {}),
                }
                if choice_delta_fields.get(choice_index):
                    choice_payload["delta"] = choice_delta_fields[choice_index]
                choice_payloads.append(choice_payload)
            completed_event["result"]["choices"] = choice_payloads
        yield completed_event

        verbose_logger.info(
            "A2A completion bridge streaming completed: request_id=%s, chunks=%s", request_id, chunk_count
        )


# Convenience functions that delegate to the class methods
async def handle_a2a_completion(
    request_id: str,
    params: dict[str, object],
    litellm_params: dict[str, object],
    api_base: str | None = None,
    agent_extra_headers: dict[str, str] | None = None,
) -> dict[str, object]:
    """Convenience function for non-streaming A2A completion."""
    return await A2ACompletionBridgeHandler.handle_non_streaming(
        request_id=request_id,
        params=params,
        litellm_params=litellm_params,
        api_base=api_base,
        agent_extra_headers=agent_extra_headers,
    )


async def handle_a2a_completion_streaming(
    request_id: str,
    params: dict[str, object],
    litellm_params: dict[str, object],
    api_base: str | None = None,
    agent_extra_headers: dict[str, str] | None = None,
) -> AsyncIterator[dict[str, object]]:
    """Convenience function for streaming A2A completion."""
    async for chunk in A2ACompletionBridgeHandler.handle_streaming(
        request_id=request_id,
        params=params,
        litellm_params=litellm_params,
        api_base=api_base,
        agent_extra_headers=agent_extra_headers,
    ):
        yield chunk

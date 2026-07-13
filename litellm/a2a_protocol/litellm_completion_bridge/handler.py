"""
Handler for A2A to LiteLLM completion bridge.

Routes A2A requests through litellm.acompletion based on custom_llm_provider.

A2A Streaming Events (in order):
1. Task event (kind: "task") - Initial task creation with status "submitted"
2. Status update (kind: "status-update") - Status change to "working"
3. Artifact update (kind: "artifact-update") - Content/artifact delivery
4. Status update (kind: "status-update") - Final status "completed" with final=true
"""

from typing import Any, AsyncIterator, Dict, Optional

import litellm
from litellm._logging import verbose_logger
from litellm.a2a_protocol.litellm_completion_bridge.transformation import (
    A2ACompletionBridgeTransformation,
    A2AStreamingContext,
)
from litellm.a2a_protocol.providers.config_manager import A2AProviderConfigManager
from litellm.interactions.agents.utils import merge_agent_headers

# litellm_params key carrying the authenticated principal (hashed virtual key) so
# A2A provider configs can scope provider-side state (e.g. LangFlow session memory)
# per key instead of trusting the client-supplied A2A contextId.
A2A_USER_API_KEY_HASH_PARAM = "litellm_a2a_user_api_key_hash"

# Agent metadata fields stored in litellm_params that are not valid litellm.acompletion() kwargs
_AGENT_ONLY_PARAMS = frozenset(
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
    async def handle_non_streaming(
        request_id: str,
        params: Dict[str, Any],
        litellm_params: Dict[str, Any],
        api_base: Optional[str] = None,
        agent_extra_headers: Optional[Dict[str, str]] = None,
        *,
        _skip_a2a_provider_routing: bool = False,
    ) -> Dict[str, Any]:
        """
        Handle non-streaming A2A request via litellm.acompletion.

        Args:
            request_id: A2A JSON-RPC request ID
            params: A2A MessageSendParams containing the message
            litellm_params: Agent's litellm_params (custom_llm_provider, model, etc.)
            api_base: API base URL from agent_card_params
            agent_extra_headers: Per-request headers (from x-a2a-{agent}-* rewrite and
                admin extra_headers) to forward on the upstream HTTP call.

        Returns:
            A2A SendMessageResponse dict
        """
        custom_llm_provider = litellm_params.get("custom_llm_provider")
        if not _skip_a2a_provider_routing:
            a2a_provider_config = A2AProviderConfigManager.get_provider_config(
                custom_llm_provider=custom_llm_provider,
                model=litellm_params.get("model"),
            )

            if a2a_provider_config is not None:
                verbose_logger.info(f"A2A: Using provider config for {custom_llm_provider}")

                return await a2a_provider_config.handle_non_streaming(
                    request_id=request_id,
                    params=params,
                    api_base=api_base,
                    litellm_params=litellm_params,
                    agent_extra_headers=agent_extra_headers,
                )

        # Extract message from params
        message = params.get("message", {})

        # Transform A2A message to OpenAI format
        openai_messages = A2ACompletionBridgeTransformation.a2a_message_to_openai_messages(message)

        # Get completion params
        custom_llm_provider = litellm_params.get("custom_llm_provider")
        model = litellm_params.get("model", "agent")

        # Build full model string if provider specified
        # Skip prepending if model already starts with the provider prefix
        if custom_llm_provider and not model.startswith(f"{custom_llm_provider}/"):
            full_model = f"{custom_llm_provider}/{model}"
        else:
            full_model = model

        verbose_logger.info(f"A2A completion bridge: model={full_model}, api_base={api_base}")

        # Build completion params dict
        completion_params: Dict[str, Any] = {
            "model": full_model,
            "messages": openai_messages,
            "api_base": api_base,
            "stream": False,
        }
        # Add litellm_params (contains api_key, client_id, client_secret, tenant_id, etc.)
        litellm_params_to_add = {
            k: v
            for k, v in litellm_params.items()
            if k not in ("model", "custom_llm_provider") and k not in _AGENT_ONLY_PARAMS
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

        if agent_extra_headers:
            completion_params["extra_headers"] = merge_agent_headers(
                dynamic_headers=agent_extra_headers,
                static_headers=completion_params.get("extra_headers"),
            )

        # Call litellm.acompletion
        response = await litellm.acompletion(**completion_params)

        # Transform response to A2A format
        a2a_response = A2ACompletionBridgeTransformation.openai_response_to_a2a_response(
            response=response,
            request_id=request_id,
        )

        verbose_logger.info(f"A2A completion bridge completed: request_id={request_id}")

        return a2a_response

    @staticmethod
    async def handle_streaming(
        request_id: str,
        params: Dict[str, Any],
        litellm_params: Dict[str, Any],
        api_base: Optional[str] = None,
        agent_extra_headers: Optional[Dict[str, str]] = None,
        *,
        _skip_a2a_provider_routing: bool = False,
    ) -> AsyncIterator[Dict[str, Any]]:
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

        Yields:
            A2A streaming response events
        """
        custom_llm_provider = litellm_params.get("custom_llm_provider")
        if not _skip_a2a_provider_routing:
            a2a_provider_config = A2AProviderConfigManager.get_provider_config(
                custom_llm_provider=custom_llm_provider,
                model=litellm_params.get("model"),
            )

            if a2a_provider_config is not None:
                verbose_logger.info(f"A2A: Using provider config for {custom_llm_provider} (streaming)")

                async for chunk in a2a_provider_config.handle_streaming(
                    request_id=request_id,
                    params=params,
                    api_base=api_base,
                    litellm_params=litellm_params,
                    agent_extra_headers=agent_extra_headers,
                ):
                    yield chunk

                return

        # Extract message from params
        message = params.get("message", {})

        # Create streaming context
        ctx = A2AStreamingContext(
            request_id=request_id,
            input_message=message,
        )

        # Transform A2A message to OpenAI format
        openai_messages = A2ACompletionBridgeTransformation.a2a_message_to_openai_messages(message)

        # Get completion params
        custom_llm_provider = litellm_params.get("custom_llm_provider")
        model = litellm_params.get("model", "agent")

        # Build full model string if provider specified
        # Skip prepending if model already starts with the provider prefix
        if custom_llm_provider and not model.startswith(f"{custom_llm_provider}/"):
            full_model = f"{custom_llm_provider}/{model}"
        else:
            full_model = model

        verbose_logger.info(f"A2A completion bridge streaming: model={full_model}, api_base={api_base}")

        # Build completion params dict
        completion_params: Dict[str, Any] = {
            "model": full_model,
            "messages": openai_messages,
            "api_base": api_base,
            "stream": True,
        }
        # Add litellm_params (contains api_key, client_id, client_secret, tenant_id, etc.)
        litellm_params_to_add = {
            k: v
            for k, v in litellm_params.items()
            if k not in ("model", "custom_llm_provider") and k not in _AGENT_ONLY_PARAMS
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

        if agent_extra_headers:
            completion_params["extra_headers"] = merge_agent_headers(
                dynamic_headers=agent_extra_headers,
                static_headers=completion_params.get("extra_headers"),
            )

        # 1. Emit initial task event (kind: "task", status: "submitted")
        task_event = A2ACompletionBridgeTransformation.create_task_event(ctx)
        yield task_event

        # 2. Emit status update (kind: "status-update", status: "working")
        working_event = A2ACompletionBridgeTransformation.create_status_update_event(
            ctx=ctx,
            state="working",
            final=False,
            message_text="Processing request...",
        )
        yield working_event

        # Call litellm.acompletion with streaming
        response = await litellm.acompletion(**completion_params)

        # 3. Accumulate content and emit artifact update
        accumulated_text = ""
        chunk_count = 0
        async for chunk in response:  # type: ignore[union-attr]
            chunk_count += 1

            # Extract delta content
            content = ""
            if chunk is not None and hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, "delta") and choice.delta:
                    content = choice.delta.content or ""

            if content:
                accumulated_text += content

        # Emit artifact update with accumulated content
        if accumulated_text:
            artifact_event = A2ACompletionBridgeTransformation.create_artifact_update_event(
                ctx=ctx,
                text=accumulated_text,
            )
            yield artifact_event

        # 4. Emit final status update (kind: "status-update", status: "completed", final: true)
        completed_event = A2ACompletionBridgeTransformation.create_status_update_event(
            ctx=ctx,
            state="completed",
            final=True,
        )
        yield completed_event

        verbose_logger.info(f"A2A completion bridge streaming completed: request_id={request_id}, chunks={chunk_count}")


# Convenience functions that delegate to the class methods
async def handle_a2a_completion(
    request_id: str,
    params: Dict[str, Any],
    litellm_params: Dict[str, Any],
    api_base: Optional[str] = None,
    agent_extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
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
    params: Dict[str, Any],
    litellm_params: Dict[str, Any],
    api_base: Optional[str] = None,
    agent_extra_headers: Optional[Dict[str, str]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Convenience function for streaming A2A completion."""
    async for chunk in A2ACompletionBridgeHandler.handle_streaming(
        request_id=request_id,
        params=params,
        litellm_params=litellm_params,
        api_base=api_base,
        agent_extra_headers=agent_extra_headers,
    ):
        yield chunk

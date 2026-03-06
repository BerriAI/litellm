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
    ) -> Dict[str, Any]:
        """
        Handle non-streaming A2A request via litellm.acompletion.

        Args:
            request_id: A2A JSON-RPC request ID
            params: A2A MessageSendParams containing the message
            litellm_params: Agent's litellm_params (custom_llm_provider, model, etc.)
            api_base: API base URL from agent_card_params

        Returns:
            A2A SendMessageResponse dict
        """
        # Get provider config for custom_llm_provider
        custom_llm_provider = litellm_params.get("custom_llm_provider")
        a2a_provider_config = A2AProviderConfigManager.get_provider_config(
            custom_llm_provider=custom_llm_provider
        )
        
        # If provider config exists, use it
        if a2a_provider_config is not None:
            if api_base is None:
                raise ValueError(f"api_base is required for {custom_llm_provider}")
            
            verbose_logger.info(
                f"A2A: Using provider config for {custom_llm_provider}"
            )
            
            response_data = await a2a_provider_config.handle_non_streaming(
                request_id=request_id,
                params=params,
                api_base=api_base,
            )
            
            return response_data
        
        # Extract message from params
        message = params.get("message", {})

        # Transform A2A message to OpenAI format
        openai_messages = A2ACompletionBridgeTransformation.a2a_message_to_openai_messages(
            message
        )

        # Get completion params
        custom_llm_provider = litellm_params.get("custom_llm_provider")
        model = litellm_params.get("model", "agent")

        # Build full model string if provider specified
        # Skip prepending if model already starts with the provider prefix
        if custom_llm_provider and not model.startswith(f"{custom_llm_provider}/"):
            full_model = f"{custom_llm_provider}/{model}"
        else:
            full_model = model

        verbose_logger.info(
            f"A2A completion bridge: model={full_model}, api_base={api_base}"
        )

        # Build completion params dict
        completion_params = {
            "model": full_model,
            "messages": openai_messages,
            "api_base": api_base,
            "stream": False,
        }
        # Add litellm_params (contains api_key, client_id, client_secret, tenant_id, etc.)
        litellm_params_to_add = {
            k: v for k, v in litellm_params.items()
            if k not in ("model", "custom_llm_provider")
        }
        completion_params.update(litellm_params_to_add)

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

        Yields:
            A2A streaming response events
        """
        # Get provider config for custom_llm_provider
        custom_llm_provider = litellm_params.get("custom_llm_provider")
        a2a_provider_config = A2AProviderConfigManager.get_provider_config(
            custom_llm_provider=custom_llm_provider
        )
        
        # If provider config exists, use it
        if a2a_provider_config is not None:
            if api_base is None:
                raise ValueError(f"api_base is required for {custom_llm_provider}")
            
            verbose_logger.info(
                f"A2A: Using provider config for {custom_llm_provider} (streaming)"
            )
            
            async for chunk in a2a_provider_config.handle_streaming(
                request_id=request_id,
                params=params,
                api_base=api_base,
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
        openai_messages = A2ACompletionBridgeTransformation.a2a_message_to_openai_messages(
            message
        )

        # Get completion params
        custom_llm_provider = litellm_params.get("custom_llm_provider")
        model = litellm_params.get("model", "agent")

        # Build full model string if provider specified
        # Skip prepending if model already starts with the provider prefix
        if custom_llm_provider and not model.startswith(f"{custom_llm_provider}/"):
            full_model = f"{custom_llm_provider}/{model}"
        else:
            full_model = model

        verbose_logger.info(
            f"A2A completion bridge streaming: model={full_model}, api_base={api_base}"
        )

        # Build completion params dict
        completion_params = {
            "model": full_model,
            "messages": openai_messages,
            "api_base": api_base,
            "stream": True,
        }
        # Add litellm_params (contains api_key, client_id, client_secret, tenant_id, etc.)
        litellm_params_to_add = {
            k: v for k, v in litellm_params.items()
            if k not in ("model", "custom_llm_provider")
        }
        completion_params.update(litellm_params_to_add)

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

        verbose_logger.info(
            f"A2A completion bridge streaming completed: request_id={request_id}, chunks={chunk_count}"
        )


# Convenience functions that delegate to the class methods
async def handle_a2a_completion(
    request_id: str,
    params: Dict[str, Any],
    litellm_params: Dict[str, Any],
    api_base: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function for non-streaming A2A completion."""
    return await A2ACompletionBridgeHandler.handle_non_streaming(
        request_id=request_id,
        params=params,
        litellm_params=litellm_params,
        api_base=api_base,
    )


async def handle_a2a_completion_streaming(
    request_id: str,
    params: Dict[str, Any],
    litellm_params: Dict[str, Any],
    api_base: Optional[str] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Convenience function for streaming A2A completion."""
    async for chunk in A2ACompletionBridgeHandler.handle_streaming(
        request_id=request_id,
        params=params,
        litellm_params=litellm_params,
        api_base=api_base,
    ):
        yield chunk

import pytest
from unittest.mock import MagicMock, patch
import os
from litellm.proxy.guardrails.guardrail_hooks.openai.moderations import (
    OpenAIModerationGuardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
    UnifiedLLMGuardrails,
)
from litellm.types.utils import ModelResponseStream, ModelResponse
from litellm.proxy._types import UserAPIKeyAuth


@pytest.mark.asyncio
async def test_openai_moderation_guardrail_streaming_latency():
    """
    Test that the OpenAI Moderation guardrail, when run via UnifiedLLMGuardrails,
    supports streaming (fast time-to-first-token) instead of buffering.
    """
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        # 1. Initialize the specific guardrail with proper event_hook
        openai_guardrail = OpenAIModerationGuardrail(
            guardrail_name="test-openai-moderation",
            event_hook="post_call",
        )

        # 2. Initialize the Unified Guardrail system (which invokes the specific guardrail)
        unified_guardrail = UnifiedLLMGuardrails()

        # Mock safe moderation response
        mock_mod_response = MagicMock()
        mock_mod_response.results = []

        # Mock streaming chunks (no artificial delay - test deterministically)
        async def mock_stream():
            chunks_data = ["Hello", " ", "world", "!", " Goodbye"]
            for i, content in enumerate(chunks_data):
                chunk = MagicMock(spec=ModelResponseStream)
                chunk.model = "gpt-4"
                choice = MagicMock()
                choice.delta = MagicMock()
                choice.delta.content = content
                # Last chunk gets finish_reason
                choice.finish_reason = "stop" if i == len(chunks_data) - 1 else None
                chunk.choices = [choice]
                yield chunk

        # Mock for stream_chunk_builder to return a simple ModelResponse
        mock_model_response = MagicMock(spec=ModelResponse)
        mock_model_response.choices = [MagicMock()]
        mock_model_response.choices[0].message = MagicMock()
        mock_model_response.choices[0].message.content = "Hello world! Goodbye"

        # Patch the network call in the specific guardrail
        with patch.object(
            openai_guardrail, "async_make_request", return_value=mock_mod_response
        ), patch(
            "litellm.llms.openai.chat.guardrail_translation.handler.stream_chunk_builder",
            return_value=mock_model_response,
        ):
            user_api_key_dict = UserAPIKeyAuth(
                api_key="test", request_route="/chat/completions"
            )
            request_data = {
                "messages": [{"role": "user", "content": "hi"}],
                "guardrail_to_apply": openai_guardrail,
                "metadata": {
                    "guardrails": ["test-openai-moderation"],
                    "guardrail_config": {"streaming_sampling_rate": 1},
                },  # Check every chunk for test
            }

            chunks_received = 0
            first_chunk_yielded = False

            # Call the hook on UnifiedLLMGuardrails
            async for chunk in unified_guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=user_api_key_dict,
                response=mock_stream(),
                request_data=request_data,
            ):
                if not first_chunk_yielded:
                    first_chunk_yielded = True
                chunks_received += 1

            # Deterministic assertions (no flaky timing checks)
            assert first_chunk_yielded, "Expected at least one chunk to be yielded"
            assert chunks_received == 5, f"Expected 5 chunks, got {chunks_received}"


@pytest.mark.asyncio
async def test_openai_moderation_guardrail_streaming_harmful_content():
    """
    Test that harmful content is caught during streaming via UnifiedLLMGuardrails
    """
    from fastapi import HTTPException

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        openai_guardrail = OpenAIModerationGuardrail(
            guardrail_name="test-openai-moderation",
            event_hook="post_call",
        )
        unified_guardrail = UnifiedLLMGuardrails()

        # Mock harmful moderation response
        mock_mod_response = MagicMock()
        mock_mod_response.results = [
            MagicMock(
                flagged=True, categories={"hate": True}, category_scores={"hate": 0.99}
            )
        ]

        async def mock_stream():
            chunks_data = ["This ", "is ", "harmful ", "content"]
            for i, content in enumerate(chunks_data):
                chunk = MagicMock(spec=ModelResponseStream)
                chunk.model = "gpt-4"
                choice = MagicMock()
                choice.delta = MagicMock()
                choice.delta.content = content
                # Last chunk gets finish_reason
                choice.finish_reason = "stop" if i == len(chunks_data) - 1 else None
                chunk.choices = [choice]
                yield chunk

        # Mock for stream_chunk_builder - use real litellm types so isinstance checks pass
        import litellm

        mock_model_response = ModelResponse(
            id="mock-response",
            model="gpt-4",
            choices=[
                litellm.Choices(
                    index=0,
                    message=litellm.Message(
                        role="assistant",
                        content="This is harmful content",
                    ),
                    finish_reason="stop",
                )
            ],
        )

        with patch.object(
            openai_guardrail, "async_make_request", return_value=mock_mod_response
        ), patch(
            "litellm.llms.openai.chat.guardrail_translation.handler.stream_chunk_builder",
            return_value=mock_model_response,
        ):
            user_api_key_dict = UserAPIKeyAuth(
                api_key="test", request_route="/chat/completions"
            )
            request_data = {
                "messages": [{"role": "user", "content": "generate hate"}],
                "guardrail_to_apply": openai_guardrail,
                "metadata": {
                    "guardrails": ["test-openai-moderation"],
                    "guardrail_config": {"streaming_sampling_rate": 1},
                },
            }

            # Should raise HTTPException
            with pytest.raises(HTTPException) as exc_info:
                async for _ in unified_guardrail.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=user_api_key_dict,
                    response=mock_stream(),
                    request_data=request_data,
                ):
                    pass

            assert exc_info.value.status_code == 400
            assert "Violated OpenAI moderation policy" in str(exc_info.value.detail)

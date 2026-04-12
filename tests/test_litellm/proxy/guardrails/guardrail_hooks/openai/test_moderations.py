#!/usr/bin/env python3
"""
Test OpenAI Moderation Guardrail
"""
import os
import sys

sys.path.insert(0, os.path.abspath("../../../../../.."))

from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.openai.moderations import (
    OpenAIModerationGuardrail,
)
from litellm.types.llms.openai import OpenAIModerationResponse, OpenAIModerationResult


@pytest.mark.asyncio
async def test_openai_moderation_guardrail_init():
    """Test OpenAI moderation guardrail initialization"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        guardrail = OpenAIModerationGuardrail(
            guardrail_name="test-openai-moderation",
        )

        assert guardrail.guardrail_name == "test-openai-moderation"
        assert guardrail.api_key == "test-key"
        assert guardrail.model == "omni-moderation-latest"
        assert guardrail.api_base == "https://api.openai.com/v1"


@pytest.mark.asyncio
async def test_openai_moderation_guardrail_adds_to_litellm_callbacks():
    """Test that OpenAI moderation guardrail adds itself to litellm callbacks during initialization"""
    import litellm
    from litellm.proxy.guardrails.guardrail_hooks.openai import (
        initialize_guardrail as openai_initialize_guardrail,
    )
    from litellm.types.guardrails import (
        Guardrail,
        LitellmParams,
        SupportedGuardrailIntegrations,
    )

    # Clear existing callbacks for clean test
    original_callbacks = litellm.callbacks.copy()
    litellm.logging_callback_manager._reset_all_callbacks()

    try:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            guardrail_litellm_params = LitellmParams(
                guardrail=SupportedGuardrailIntegrations.OPENAI_MODERATION,
                api_key="test-key",
                model="omni-moderation-latest",
                mode="pre_call",
            )
            guardrail = openai_initialize_guardrail(
                litellm_params=guardrail_litellm_params,
                guardrail=Guardrail(
                    guardrail_name="test-openai-moderation",
                    litellm_params=guardrail_litellm_params,
                ),
            )

            # Check that the guardrail was added to litellm callbacks
            assert guardrail in litellm.callbacks
            assert len(litellm.callbacks) == 1

            # Verify it's the correct guardrail
            callback = litellm.callbacks[0]
            assert isinstance(callback, OpenAIModerationGuardrail)
            assert callback.guardrail_name == "test-openai-moderation"
    finally:
        # Restore original callbacks
        litellm.logging_callback_manager._reset_all_callbacks()
        for callback in original_callbacks:
            litellm.logging_callback_manager.add_litellm_callback(callback)


@pytest.mark.asyncio
async def test_openai_moderation_guardrail_safe_content():
    """Test OpenAI moderation guardrail with safe content via apply_guardrail"""
    from litellm.types.utils import GenericGuardrailAPIInputs

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        guardrail = OpenAIModerationGuardrail(
            guardrail_name="test-openai-moderation",
        )

        # Mock safe moderation response
        mock_response = OpenAIModerationResponse(
            id="modr-123",
            model="omni-moderation-latest",
            results=[
                OpenAIModerationResult(
                    flagged=False,
                    categories={
                        "sexual": False,
                        "hate": False,
                        "harassment": False,
                        "self-harm": False,
                        "violence": False,
                    },
                    category_scores={
                        "sexual": 0.001,
                        "hate": 0.001,
                        "harassment": 0.001,
                        "self-harm": 0.001,
                        "violence": 0.001,
                    },
                    category_applied_input_types={
                        "sexual": [],
                        "hate": [],
                        "harassment": [],
                        "self-harm": [],
                        "violence": [],
                    },
                )
            ],
        )

        with patch.object(guardrail, "async_make_request", return_value=mock_response):
            # Test apply_guardrail with safe content using structured_messages
            inputs = GenericGuardrailAPIInputs(
                structured_messages=[
                    {"role": "user", "content": "Hello, how are you today?"}
                ]
            )

            result = await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={
                    "messages": [
                        {"role": "user", "content": "Hello, how are you today?"}
                    ]
                },
                input_type="request",
            )

            # Should return the original inputs unchanged
            assert result == inputs


@pytest.mark.asyncio
async def test_openai_moderation_guardrail_apply_guardrail():
    """Test OpenAI moderation guardrail apply_guardrail method (unified guardrail interface)"""
    from litellm.types.utils import GenericGuardrailAPIInputs

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        guardrail = OpenAIModerationGuardrail(
            guardrail_name="test-openai-moderation",
        )

        # Mock safe moderation response
        mock_response = OpenAIModerationResponse(
            id="modr-123",
            model="omni-moderation-latest",
            results=[
                OpenAIModerationResult(
                    flagged=False,
                    categories={
                        "sexual": False,
                        "hate": False,
                        "harassment": False,
                        "self-harm": False,
                        "violence": False,
                    },
                    category_scores={
                        "sexual": 0.001,
                        "hate": 0.001,
                        "harassment": 0.001,
                        "self-harm": 0.001,
                        "violence": 0.001,
                    },
                    category_applied_input_types={
                        "sexual": [],
                        "hate": [],
                        "harassment": [],
                        "self-harm": [],
                        "violence": [],
                    },
                )
            ],
        )

        with patch.object(guardrail, "async_make_request", return_value=mock_response):
            # Test apply_guardrail with texts (embeddings-style input)
            inputs = GenericGuardrailAPIInputs(
                texts=["Hello, how are you?", "What is the weather?"]
            )

            result = await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="request",
            )

            # Should return inputs unchanged (moderation doesn't modify, only blocks)
            assert result == inputs


@pytest.mark.asyncio
async def test_openai_moderation_guardrail_harmful_content():
    """Test OpenAI moderation guardrail with harmful content via apply_guardrail"""
    from litellm.types.utils import GenericGuardrailAPIInputs

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        guardrail = OpenAIModerationGuardrail(
            guardrail_name="test-openai-moderation",
        )

        # Mock harmful moderation response
        mock_response = OpenAIModerationResponse(
            id="modr-123",
            model="omni-moderation-latest",
            results=[
                OpenAIModerationResult(
                    flagged=True,
                    categories={
                        "sexual": False,
                        "hate": True,
                        "harassment": False,
                        "self-harm": False,
                        "violence": False,
                    },
                    category_scores={
                        "sexual": 0.001,
                        "hate": 0.95,
                        "harassment": 0.001,
                        "self-harm": 0.001,
                        "violence": 0.001,
                    },
                    category_applied_input_types={
                        "sexual": [],
                        "hate": ["text"],
                        "harassment": [],
                        "self-harm": [],
                        "violence": [],
                    },
                )
            ],
        )

        with patch.object(guardrail, "async_make_request", return_value=mock_response):
            # Test apply_guardrail with harmful content using structured_messages
            inputs = GenericGuardrailAPIInputs(
                structured_messages=[
                    {"role": "user", "content": "This is hateful content"}
                ]
            )

            # Should raise HTTPException
            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc_info:
                await guardrail.apply_guardrail(
                    inputs=inputs,
                    request_data={
                        "messages": [
                            {"role": "user", "content": "This is hateful content"}
                        ]
                    },
                    input_type="request",
                )

            assert exc_info.value.status_code == 400
            assert "Violated OpenAI moderation policy" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_openai_moderation_guardrail_streaming_safe_content():
    """Test OpenAI moderation guardrail with streaming safe content via UnifiedLLMGuardrails"""
    from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
        UnifiedLLMGuardrails,
    )

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        guardrail = OpenAIModerationGuardrail(
            guardrail_name="test-openai-moderation",
            event_hook="post_call",
        )
        unified_guardrail = UnifiedLLMGuardrails()

        # Mock safe moderation response
        mock_response = OpenAIModerationResponse(
            id="modr-123",
            model="omni-moderation-latest",
            results=[
                OpenAIModerationResult(
                    flagged=False,
                    categories={
                        "sexual": False,
                        "hate": False,
                        "harassment": False,
                        "self-harm": False,
                        "violence": False,
                    },
                    category_scores={
                        "sexual": 0.001,
                        "hate": 0.001,
                        "harassment": 0.001,
                        "self-harm": 0.001,
                        "violence": 0.001,
                    },
                    category_applied_input_types={
                        "sexual": [],
                        "hate": [],
                        "harassment": [],
                        "self-harm": [],
                        "violence": [],
                    },
                )
            ],
        )

        # Mock streaming chunks
        async def mock_stream():
            # Simulate streaming chunks with safe content
            chunk1 = MagicMock()
            chunk1.model = "gpt-4"
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta = MagicMock()
            chunk1.choices[0].delta.content = "Hello "
            chunk1.choices[0].finish_reason = None
            
            chunk2 = MagicMock()
            chunk2.model = "gpt-4"
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta = MagicMock()
            chunk2.choices[0].delta.content = "world"
            chunk2.choices[0].finish_reason = None
            
            # Last chunk with finish_reason
            chunk3 = MagicMock()
            chunk3.model = "gpt-4"
            chunk3.choices = [MagicMock()]
            chunk3.choices[0].delta = MagicMock()
            chunk3.choices[0].delta.content = "!"
            chunk3.choices[0].finish_reason = "stop"
            
            for chunk in [chunk1, chunk2, chunk3]:
                yield chunk

        # Mock for stream_chunk_builder
        mock_model_response = MagicMock()
        mock_model_response.choices = [MagicMock()]
        mock_model_response.choices[0].message = MagicMock()
        mock_model_response.choices[0].message.content = "Hello world!"

        with patch.object(guardrail, "async_make_request", return_value=mock_response), patch(
            "litellm.llms.openai.chat.guardrail_translation.handler.stream_chunk_builder",
            return_value=mock_model_response,
        ):
            user_api_key_dict = UserAPIKeyAuth(
                api_key="test", request_route="/chat/completions"
            )
            request_data = {
                "messages": [{"role": "user", "content": "Hello, how are you today?"}],
                "guardrail_to_apply": guardrail,
                "metadata": {"guardrails": ["test-openai-moderation"]},
            }

            # Test streaming hook with safe content via UnifiedLLMGuardrails
            result_chunks = []
            async for chunk in unified_guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=user_api_key_dict,
                response=mock_stream(),
                request_data=request_data,
            ):
                result_chunks.append(chunk)

            # Should return all chunks without blocking
            assert len(result_chunks) == 3


@pytest.mark.asyncio
async def test_openai_moderation_guardrail_streaming_harmful_content():
    """Test OpenAI moderation guardrail with streaming harmful content via UnifiedLLMGuardrails"""
    from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
        UnifiedLLMGuardrails,
    )

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        guardrail = OpenAIModerationGuardrail(
            guardrail_name="test-openai-moderation",
            event_hook="post_call",
        )
        unified_guardrail = UnifiedLLMGuardrails()

        # Mock harmful moderation response
        mock_response = OpenAIModerationResponse(
            id="modr-123",
            model="omni-moderation-latest",
            results=[
                OpenAIModerationResult(
                    flagged=True,
                    categories={
                        "sexual": False,
                        "hate": True,
                        "harassment": False,
                        "self-harm": False,
                        "violence": False,
                    },
                    category_scores={
                        "sexual": 0.001,
                        "hate": 0.95,
                        "harassment": 0.001,
                        "self-harm": 0.001,
                        "violence": 0.001,
                    },
                    category_applied_input_types={
                        "sexual": [],
                        "hate": ["text"],
                        "harassment": [],
                        "self-harm": [],
                        "violence": [],
                    },
                )
            ],
        )

        # Mock streaming chunks with harmful content
        async def mock_stream():
            # First chunk - no finish_reason
            chunk1 = MagicMock()
            chunk1.model = "gpt-4"
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta = MagicMock()
            chunk1.choices[0].delta.content = "This is "
            chunk1.choices[0].finish_reason = None
            
            # Last chunk - with finish_reason to signal end of stream
            chunk2 = MagicMock()
            chunk2.model = "gpt-4"
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta = MagicMock()
            chunk2.choices[0].delta.content = "harmful content"
            chunk2.choices[0].finish_reason = "stop"
            
            for chunk in [chunk1, chunk2]:
                yield chunk

        # Mock for stream_chunk_builder - use real litellm types so isinstance checks pass
        from litellm.types.utils import ModelResponse
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

        with patch.object(guardrail, "async_make_request", return_value=mock_response), patch(
            "litellm.llms.openai.chat.guardrail_translation.handler.stream_chunk_builder",
            return_value=mock_model_response,
        ):
            user_api_key_dict = UserAPIKeyAuth(
                api_key="test", request_route="/chat/completions"
            )
            request_data = {
                "messages": [{"role": "user", "content": "Generate harmful content"}],
                "guardrail_to_apply": guardrail,
                "metadata": {"guardrails": ["test-openai-moderation"]},
            }

            # Should raise HTTPException when processing streaming harmful content
            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc_info:
                result_chunks = []
                async for chunk in unified_guardrail.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=user_api_key_dict,
                    response=mock_stream(),
                    request_data=request_data,
                ):
                    result_chunks.append(chunk)

            assert exc_info.value.status_code == 400
            assert "Violated OpenAI moderation policy" in str(exc_info.value.detail)

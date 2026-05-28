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
        with (
            patch.object(
                openai_guardrail, "async_make_request", return_value=mock_mod_response
            ),
            patch(
                "litellm.llms.openai.chat.guardrail_translation.handler.stream_chunk_builder",
                return_value=mock_model_response,
            ),
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
            async for (
                chunk
            ) in unified_guardrail.async_post_call_streaming_iterator_hook(
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

        with (
            patch.object(
                openai_guardrail, "async_make_request", return_value=mock_mod_response
            ),
            patch(
                "litellm.llms.openai.chat.guardrail_translation.handler.stream_chunk_builder",
                return_value=mock_model_response,
            ),
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
                async for (
                    _
                ) in unified_guardrail.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=user_api_key_dict,
                    response=mock_stream(),
                    request_data=request_data,
                ):
                    pass

            assert exc_info.value.status_code == 400
            assert "Violated OpenAI moderation policy" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_openai_moderation_streaming_end_of_stream_request_data_passthrough():
    """Test that streaming end-of-stream guardrail info flows through to the
    real request_data (Bug 1 fix for streaming path)."""
    from litellm.types.llms.openai import (
        OpenAIModerationResponse,
        OpenAIModerationResult,
    )

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        openai_guardrail = OpenAIModerationGuardrail(
            guardrail_name="test-openai-moderation",
            event_hook="post_call",
        )
        unified_guardrail = UnifiedLLMGuardrails()

        mock_mod_response = OpenAIModerationResponse(
            id="modr-stream-test",
            model="omni-moderation-latest",
            results=[
                OpenAIModerationResult(
                    flagged=False,
                    categories={"hate": False, "violence": False},
                    category_scores={"hate": 0.001, "violence": 0.002},
                    category_applied_input_types={"hate": [], "violence": []},
                )
            ],
        )

        async def mock_stream():
            import litellm

            chunks_data = ["Hello", " world"]
            for i, content in enumerate(chunks_data):
                chunk = MagicMock(spec=ModelResponseStream)
                chunk.model = "gpt-4"
                choice = MagicMock()
                choice.delta = MagicMock()
                choice.delta.content = content
                choice.finish_reason = "stop" if i == len(chunks_data) - 1 else None
                chunk.choices = [choice]
                yield chunk

        import litellm

        mock_model_response = ModelResponse(
            id="mock-stream-response",
            model="gpt-4",
            choices=[
                litellm.Choices(
                    index=0,
                    message=litellm.Message(role="assistant", content="Hello world"),
                    finish_reason="stop",
                )
            ],
        )

        request_data = {
            "messages": [{"role": "user", "content": "hi"}],
            "guardrail_to_apply": openai_guardrail,
            "metadata": {
                "guardrails": ["test-openai-moderation"],
                "guardrail_config": {"streaming_sampling_rate": 1},
            },
        }

        with (
            patch.object(
                openai_guardrail, "async_make_request", return_value=mock_mod_response
            ),
            patch(
                "litellm.llms.openai.chat.guardrail_translation.handler.stream_chunk_builder",
                return_value=mock_model_response,
            ),
        ):
            user_api_key_dict = UserAPIKeyAuth(
                api_key="test", request_route="/chat/completions"
            )

            async for _ in unified_guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=user_api_key_dict,
                response=mock_stream(),
                request_data=request_data,
            ):
                pass

        # Verify guardrail info reached the REAL request_data (not a throwaway)
        guardrail_info_list = request_data["metadata"].get(
            "standard_logging_guardrail_information"
        )
        assert (
            guardrail_info_list is not None
        ), "Guardrail info should be in request_data after streaming"
        info = guardrail_info_list[0]
        assert info["guardrail_status"] == "success"

        # Full moderation response dict, NOT the simplified "allow" string
        guardrail_resp = info["guardrail_response"]
        assert isinstance(
            guardrail_resp, dict
        ), f"Expected full moderation response dict, got {type(guardrail_resp)}: {guardrail_resp}"
        assert "results" in guardrail_resp


@pytest.mark.asyncio
async def test_openai_moderation_defaults_to_end_of_stream_only():
    """
    Regression for LIT-3320 / "Post guardrail with OpenAI moderation is too slow".

     should default to 
    so it issues exactly one  call per streamed completion. Mid-stream
    sampling produces no safety benefit for moderation (it cannot redact, only block)
    and previously inflated post-guardrail latency by ~4x at sampling_rate=5.

    Users can still set  in the guardrail
    config to restore mid-stream sampling — verified below as well.
    """
    from litellm.types.utils import ModelResponseStream, StreamingChoices, Delta
    from litellm.types.llms.openai import (
        OpenAIModerationResponse,
        OpenAIModerationResult,
    )

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        # 1. Default — no kwargs about streaming
        default_guardrail = OpenAIModerationGuardrail(
            guardrail_name="test-openai-moderation",
            event_hook="post_call",
        )
        assert default_guardrail.streaming_end_of_stream_only is True, (
            "Default streaming_end_of_stream_only must be True so OpenAI moderation "
            "only runs once at end-of-stream (LIT-3320)."
        )

        # 2. Explicit opt-out still honored
        opted_out = OpenAIModerationGuardrail(
            guardrail_name="test-openai-moderation",
            event_hook="post_call",
            streaming_end_of_stream_only=False,
        )
        assert opted_out.streaming_end_of_stream_only is False, (
            "User opt-out via streaming_end_of_stream_only=False must be preserved."
        )

        # 3. End-to-end behavior: drive the real UnifiedLLMGuardrails streaming
        #    hook over a 20-chunk stream, mock the /moderations call, and assert
        #    it is invoked exactly once.
        unified = UnifiedLLMGuardrails()

        chunks_data = [
            "The", " capital", " of", " France", " is", " Paris", ".",
            " It", " is", " also", " the", " largest", " city", " in",
            " the", " country", " and", " a", " global", " hub",
        ]

        async def real_stream():
            for i, content in enumerate(chunks_data):
                yield ModelResponseStream(
                    id=f"chunk-{i}",
                    model="gpt-4",
                    choices=[
                        StreamingChoices(
                            index=0,
                            delta=Delta(content=content, role="assistant"),
                            finish_reason="stop"
                            if i == len(chunks_data) - 1
                            else None,
                        )
                    ],
                )

        call_counter = {"n": 0}

        async def fake_moderation(self, input_text):
            call_counter["n"] += 1
            return OpenAIModerationResponse(
                id="mod",
                model="omni-moderation-latest",
                results=[
                    OpenAIModerationResult(
                        categories={},
                        category_applied_input_types={},
                        category_scores={},
                        flagged=False,
                    )
                ],
            )

        request_data = {
            "messages": [{"role": "user", "content": "hi"}],
            "guardrail_to_apply": default_guardrail,
            "metadata": {"guardrails": ["test-openai-moderation"]},
            "model": "gpt-4",
        }
        user_api_key_dict = UserAPIKeyAuth(
            api_key="test", request_route="/chat/completions"
        )

        with patch.object(
            OpenAIModerationGuardrail, "async_make_request", fake_moderation
        ):
            chunks_received = 0
            async for _ in unified.async_post_call_streaming_iterator_hook(
                user_api_key_dict=user_api_key_dict,
                response=real_stream(),
                request_data=request_data,
            ):
                chunks_received += 1

        assert chunks_received == len(chunks_data), (
            f"Expected all {len(chunks_data)} chunks to be yielded, got {chunks_received}"
        )
        moderation_calls = call_counter["n"]
        assert moderation_calls == 1, (
            f"Expected exactly 1 /moderations call at end-of-stream with the new "
            f"default; got {moderation_calls}. "
            f"This is the LIT-3320 regression."
        )

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
async def test_lit3320_streaming_end_of_stream_only_default_and_opt_in():
    """
    Regression for LIT-3320 / "Post guardrail with OpenAI moderation is too slow".

    Both reviewer bots (Greptile P2 + Veria Medium) flagged the previous PR for
    defaulting ``streaming_end_of_stream_only=True``: that change silently
    shifted every existing OpenAI Moderation streaming deployment to deliver
    the full response to the client before the post-call guardrail ever ran,
    which is a moderation bypass on upgrade.

    Correct contract verified here:

      1. Default ``streaming_end_of_stream_only`` is ``False`` — same as every
         other streaming-aware post-call guardrail in ``unified_guardrail.py``.
         The unified dispatcher samples mid-stream so a flagged response is
         interrupted at the next ``streaming_sampling_rate`` tick.
      2. Operators that prioritise latency over mid-stream interruption can
         still opt in via ``streaming_end_of_stream_only=True``, which
         collapses post-call moderation to exactly one ``/moderations``
         round-trip per streamed completion. This is the LIT-3320 perf knob.
      3. In the safe default with ``streaming_sampling_rate=5`` over a 20-chunk
         stream, the LIT-3320 dispatcher dedup (in
         ``unified_guardrail.async_post_call_streaming_iterator_hook``) skips
         the redundant end-of-stream pass that would otherwise duplicate the
         chunk-20 sample, so we see exactly 4 calls (chunks 5/10/15/20)
         instead of 5.
    """
    from litellm.types.utils import ModelResponseStream, StreamingChoices, Delta
    from litellm.types.llms.openai import (
        OpenAIModerationResponse,
        OpenAIModerationResult,
    )

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        # 1. Default — no kwargs about streaming. Must be False so behaviour
        #    on upgrade matches all other streaming-aware post-call guardrails.
        default_guardrail = OpenAIModerationGuardrail(
            guardrail_name="test-openai-moderation",
            event_hook="post_call",
        )
        assert default_guardrail.streaming_end_of_stream_only is False, (
            "Default streaming_end_of_stream_only must be False so OpenAI "
            "Moderation preserves mid-stream blocking on upgrade (LIT-3320 "
            "safety contract)."
        )

        # 2. Explicit opt-in still honoured — this is the LIT-3320 perf knob.
        opted_in = OpenAIModerationGuardrail(
            guardrail_name="test-openai-moderation",
            event_hook="post_call",
            streaming_end_of_stream_only=True,
        )
        assert opted_in.streaming_end_of_stream_only is True, (
            "User opt-in via streaming_end_of_stream_only=True must be "
            "preserved — this is the documented low-latency mode."
        )

        # Explicit False still works
        explicit_false = OpenAIModerationGuardrail(
            guardrail_name="test-openai-moderation",
            event_hook="post_call",
            streaming_end_of_stream_only=False,
        )
        assert explicit_false.streaming_end_of_stream_only is False

        # 3-4. End-to-end driving the real UnifiedLLMGuardrails streaming hook
        #     over a 20-chunk stream, in both modes.
        unified = UnifiedLLMGuardrails()

        chunks_data = [
            "The", " capital", " of", " France", " is", " Paris", ".",
            " It", " is", " also", " the", " largest", " city", " in",
            " the", " country", " and", " a", " global", " hub",
        ]
        assert len(chunks_data) == 20

        def make_stream():
            async def _gen():
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
            return _gen()

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

        user_api_key_dict = UserAPIKeyAuth(
            api_key="test", request_route="/chat/completions"
        )

        # --- 3. End-of-stream-only opt-in => exactly 1 /moderations call ---
        call_counter = {"n": 0}
        request_data = {
            "messages": [{"role": "user", "content": "hi"}],
            "guardrail_to_apply": opted_in,
            "metadata": {"guardrails": ["test-openai-moderation"]},
            "model": "gpt-4",
        }
        with patch.object(
            OpenAIModerationGuardrail, "async_make_request", fake_moderation
        ):
            chunks_received = 0
            async for _ in unified.async_post_call_streaming_iterator_hook(
                user_api_key_dict=user_api_key_dict,
                response=make_stream(),
                request_data=request_data,
            ):
                chunks_received += 1

        assert chunks_received == len(chunks_data), (
            f"Expected all {len(chunks_data)} chunks to be yielded; got "
            f"{chunks_received}"
        )
        assert call_counter["n"] == 1, (
            "Opt-in streaming_end_of_stream_only=True must issue exactly "
            f"1 /moderations call per streamed completion; got {call_counter[chr(34)+n+chr(34)]}."
        )

        # --- 4. Safe default => mid-stream sampling, EOS pass deduped ---
        #
        # 20 chunks at sampling_rate=5 fires the mid-stream guardrail pass at
        # chunks 5, 10, 15, 20. The final end-of-stream pass used to also run
        # on the same 20 chunks, giving 5 outbound /moderations calls. The
        # LIT-3320 dispatcher dedup skips the redundant EOS pass when the
        # last chunk already ran a mid-stream pass on the same
        # responses_so_far - so we expect exactly 4 calls.
        call_counter = {"n": 0}
        request_data = {
            "messages": [{"role": "user", "content": "hi"}],
            "guardrail_to_apply": default_guardrail,
            "metadata": {"guardrails": ["test-openai-moderation"]},
            "model": "gpt-4",
        }
        with patch.object(
            OpenAIModerationGuardrail, "async_make_request", fake_moderation
        ):
            chunks_received = 0
            async for _ in unified.async_post_call_streaming_iterator_hook(
                user_api_key_dict=user_api_key_dict,
                response=make_stream(),
                request_data=request_data,
            ):
                chunks_received += 1

        assert chunks_received == len(chunks_data)
        assert call_counter["n"] == 4, (
            "Safe default (sampling_rate=5 over 20 chunks) should issue 4 "
            "mid-stream /moderations calls (chunks 5/10/15/20). The final "
            "end-of-stream pass is deduped by the LIT-3320 dispatcher fix "
            f"because chunk 20 already moderated the same content; got {call_counter[chr(34)+n+chr(34)]}."
        )


@pytest.mark.asyncio
async def test_lit3320_dispatcher_dedup_skips_redundant_eos_pass():
    """
    Regression for LIT-3320 dispatcher dedup: when the final chunk index is an
    exact multiple of ``streaming_sampling_rate``, the unified streaming hook
    used to run the guardrail twice on the same ``responses_so_far`` (once
    mid-stream at the last chunk, once again in the post-loop end-of-stream
    block). The dedup tracks ``last_chunk_processed_mid_stream`` and skips
    the EOS pass in that case.

    Drives the unified dispatcher with OpenAI Moderation as the guardrail
    and asserts:

      * 20 chunks at sampling_rate=5  =>  4 passes (chunks 5/10/15/20, EOS skipped)
      * 17 chunks at sampling_rate=5  =>  4 passes (chunks 5/10/15 mid-stream + EOS)
    """
    from litellm.types.utils import ModelResponseStream, StreamingChoices, Delta

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        guardrail = OpenAIModerationGuardrail(
            guardrail_name="test-openai-moderation",
            event_hook="post_call",
        )
        # Force a known sampling rate independent of any future default change.
        guardrail.streaming_sampling_rate = 5
        # Mid-stream sampling must be on for this test.
        guardrail.streaming_end_of_stream_only = False

        unified = UnifiedLLMGuardrails()
        user_api_key_dict = UserAPIKeyAuth(
            api_key="test", request_route="/chat/completions"
        )

        def make_stream(n_chunks):
            async def _gen():
                for i in range(n_chunks):
                    yield ModelResponseStream(
                        id=f"chunk-{i}",
                        model="gpt-4",
                        choices=[
                            StreamingChoices(
                                index=0,
                                delta=Delta(content=f"tok-{i} ", role="assistant"),
                                finish_reason="stop" if i == n_chunks - 1 else None,
                            )
                        ],
                    )
            return _gen()

        # Patch the chat-completions endpoint translations

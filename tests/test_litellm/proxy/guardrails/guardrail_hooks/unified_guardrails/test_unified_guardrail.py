"""Tests for unified guardrail."""

import pytest

import litellm
from litellm.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
    openai_messages_without_system,
    openai_messages_without_tool,
)
from litellm.llms.openai.chat.guardrail_translation.handler import (
    OpenAIChatCompletionsHandler,
)
from litellm.llms.base_llm.ocr.transformation import OCRPage, OCRResponse
from litellm.llms.mistral.ocr.guardrail_translation.handler import OCRHandler
from litellm.proxy._experimental.mcp_server.guardrail_translation.handler import (
    MCPGuardrailTranslationHandler,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail import (
    unified_guardrail as unified_module,
)
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
    UnifiedLLMGuardrails,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypes, Delta, ModelResponseStream, StreamingChoices


class RecordingGuardrail(CustomGuardrail):
    """Records the event types it is asked to run for."""

    def __init__(self):
        super().__init__(guardrail_name="recording-guardrail")
        self.event_history = []
        self.apply_calls = []

    def should_run_guardrail(self, data, event_type):  # type: ignore[override]
        self.event_history.append(event_type)
        return True

    async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
        self.apply_calls.append({"inputs": inputs, "input_type": input_type})
        return {"texts": inputs.get("texts", [])}


class _NoopTranslation(BaseTranslation):
    """Test translation handler that simply echoes input/output."""

    async def process_input_messages(self, data, guardrail_to_apply, litellm_logging_obj=None):  # type: ignore[override]
        return data

    async def process_output_response(  # type: ignore[override]
        self,
        response,
        guardrail_to_apply,
        litellm_logging_obj=None,
        user_api_key_dict=None,
    ):
        return response


@pytest.fixture(autouse=True)
def _inject_mcp_handler_mapping():
    """Inject MCP handler mapping so the unified guardrail can run inside tests."""
    unified_module.endpoint_guardrail_translation_mappings = {
        CallTypes.call_mcp_tool: MCPGuardrailTranslationHandler,
        CallTypes.anthropic_messages: _NoopTranslation,
        CallTypes.ocr: OCRHandler,
        CallTypes.aocr: OCRHandler,
    }
    yield
    unified_module.endpoint_guardrail_translation_mappings = None


class TestUnifiedLLMGuardrails:
    class TestSkipSystemMessageForChatCompletions:
        def test_openai_messages_without_system(self):
            msgs = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ]
            out = openai_messages_without_system(msgs)
            assert len(out) == 1
            assert out[0]["role"] == "user"
            assert msgs[0]["content"] == "sys"

        def test_effective_skip_respects_per_guardrail_over_global(self, monkeypatch):
            monkeypatch.setattr(
                litellm, "skip_system_message_in_guardrail", True, raising=False
            )

            class G:
                skip_system_message_in_guardrail = False

            assert effective_skip_system_message_for_guardrail(G()) is False

            class G2:
                skip_system_message_in_guardrail = None

            assert effective_skip_system_message_for_guardrail(G2()) is True

        @pytest.mark.asyncio
        async def test_openai_handler_skips_system_in_guardrail_inputs(
            self, monkeypatch
        ):
            monkeypatch.setattr(
                litellm, "skip_system_message_in_guardrail", True, raising=False
            )

            captured = {}

            class MockGuardrail:
                skip_system_message_in_guardrail = None

                async def apply_guardrail(
                    self, inputs, request_data, input_type, logging_obj=None
                ):
                    captured["inputs"] = inputs
                    return inputs

            data = {
                "messages": [
                    {"role": "system", "content": "secret system"},
                    {"role": "user", "content": "hello"},
                ],
                "model": "gpt-4o",
            }

            handler = OpenAIChatCompletionsHandler()
            await handler.process_input_messages(
                data=data,
                guardrail_to_apply=MockGuardrail(),
                litellm_logging_obj=None,
            )

            assert captured["inputs"]["texts"] == ["hello"]
            sm = captured["inputs"].get("structured_messages") or []
            assert all(m.get("role") != "system" for m in sm)
            assert data["messages"][0]["content"] == "secret system"

        @pytest.mark.asyncio
        async def test_openai_handler_per_guardrail_skip_false_overrides_global(
            self, monkeypatch
        ):
            monkeypatch.setattr(
                litellm, "skip_system_message_in_guardrail", True, raising=False
            )

            captured = {}

            class MockGuardrail:
                skip_system_message_in_guardrail = False

                async def apply_guardrail(
                    self, inputs, request_data, input_type, logging_obj=None
                ):
                    captured["inputs"] = inputs
                    return inputs

            data = {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "u"},
                ],
            }

            await OpenAIChatCompletionsHandler().process_input_messages(
                data=data,
                guardrail_to_apply=MockGuardrail(),
                litellm_logging_obj=None,
            )

            assert "sys" in captured["inputs"]["texts"]
            roles = {
                m.get("role")
                for m in (captured["inputs"].get("structured_messages") or [])
            }
            assert "system" in roles

    class TestSkipToolMessageForChatCompletions:
        def test_openai_messages_without_tool(self):
            msgs = [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "content": "tool result", "tool_call_id": "call_1"},
            ]
            out = openai_messages_without_tool(msgs)
            assert len(out) == 2
            assert all(m["role"] != "tool" for m in out)
            assert msgs[2]["content"] == "tool result"

        def test_effective_skip_tool_respects_per_guardrail_over_global(
            self, monkeypatch
        ):
            monkeypatch.setattr(
                litellm, "skip_tool_message_in_guardrail", True, raising=False
            )

            class G:
                skip_tool_message_in_guardrail = False

            assert effective_skip_tool_message_for_guardrail(G()) is False

            class G2:
                skip_tool_message_in_guardrail = None

            assert effective_skip_tool_message_for_guardrail(G2()) is True

        @pytest.mark.asyncio
        async def test_openai_handler_skips_tool_in_guardrail_inputs(self, monkeypatch):
            monkeypatch.setattr(
                litellm, "skip_tool_message_in_guardrail", True, raising=False
            )

            captured = {}

            class MockGuardrail:
                skip_tool_message_in_guardrail = None

                async def apply_guardrail(
                    self, inputs, request_data, input_type, logging_obj=None
                ):
                    captured["inputs"] = inputs
                    return inputs

            data = {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": "secret tool result",
                        "tool_call_id": "call_1",
                    },
                ],
                "model": "gpt-4o",
            }

            handler = OpenAIChatCompletionsHandler()
            await handler.process_input_messages(
                data=data,
                guardrail_to_apply=MockGuardrail(),
                litellm_logging_obj=None,
            )

            assert "secret tool result" not in captured["inputs"]["texts"]
            sm = captured["inputs"].get("structured_messages") or []
            assert all(m.get("role") != "tool" for m in sm)
            assert data["messages"][2]["content"] == "secret tool result"

        @pytest.mark.asyncio
        async def test_openai_handler_per_guardrail_skip_tool_false_overrides_global(
            self, monkeypatch
        ):
            monkeypatch.setattr(
                litellm, "skip_tool_message_in_guardrail", True, raising=False
            )

            captured = {}

            class MockGuardrail:
                skip_tool_message_in_guardrail = False

                async def apply_guardrail(
                    self, inputs, request_data, input_type, logging_obj=None
                ):
                    captured["inputs"] = inputs
                    return inputs

            data = {
                "messages": [
                    {"role": "user", "content": "u"},
                    {"role": "tool", "content": "tr", "tool_call_id": "call_1"},
                ],
            }

            await OpenAIChatCompletionsHandler().process_input_messages(
                data=data,
                guardrail_to_apply=MockGuardrail(),
                litellm_logging_obj=None,
            )

            assert "tr" in captured["inputs"]["texts"]
            roles = {
                m.get("role")
                for m in (captured["inputs"].get("structured_messages") or [])
            }
            assert "tool" in roles

    class TestAsyncPreCallHook:
        @pytest.mark.asyncio
        async def test_uses_mcp_event_type(self):
            """pre_call hook should swap to GuardrailEventHooks.pre_mcp_call for MCP calls."""
            handler = UnifiedLLMGuardrails()
            guardrail = RecordingGuardrail()
            cache = DualCache()

            data = {
                "guardrail_to_apply": guardrail,
                "messages": [{"role": "user", "content": "Tool: test\nArguments: {}"}],
                "model": "mcp-tool-call",
            }

            await handler.async_pre_call_hook(
                user_api_key_dict=None,
                cache=cache,
                data=data,
                call_type=CallTypes.call_mcp_tool.value,
            )

            assert guardrail.event_history == [GuardrailEventHooks.pre_mcp_call]

    class TestAsyncModerationHook:
        @pytest.mark.asyncio
        async def test_uses_mcp_event_type(self):
            """moderation hook should request GuardrailEventHooks.during_mcp_call for MCP calls."""
            handler = UnifiedLLMGuardrails()
            guardrail = RecordingGuardrail()

            data = {
                "guardrail_to_apply": guardrail,
                "messages": [{"role": "user", "content": "Tool: test\nArguments: {}"}],
                "model": "mcp-tool-call",
            }

            await handler.async_moderation_hook(
                data=data,
                user_api_key_dict=None,
                call_type=CallTypes.call_mcp_tool.value,
            )

            assert guardrail.event_history == [GuardrailEventHooks.during_mcp_call]

        @pytest.mark.asyncio
        async def test_runs_for_anthropic_messages(self):
            """Ensure anthropic_messages requests still trigger guardrail moderation."""
            handler = UnifiedLLMGuardrails()
            guardrail = RecordingGuardrail()

            data = {
                "guardrail_to_apply": guardrail,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello Anthropics",
                    }
                ],
                "model": "anthropic.claude-3",
            }

            await handler.async_moderation_hook(
                data=data,
                user_api_key_dict=None,
                call_type=CallTypes.anthropic_messages.value,
            )

            assert guardrail.event_history == [GuardrailEventHooks.during_call]

    class TestAsyncPostCallStreamingIteratorHook:
        @pytest.mark.asyncio
        async def test_streaming_content_not_lost_on_sampled_chunks(self):
            """
            Verify that every chunk's content is preserved in the output stream.

            The bug: process_output_streaming_response puts the combined
            guardrailed text in the first chunk and clears all subsequent
            chunks to "". The hook then yielded processed_items[-1] (the
            cleared last item), permanently losing every Nth chunk's content.
            """

            class _ContentClearingTranslation(BaseTranslation):
                """Simulates the real OpenAI handler behavior that triggers the bug."""

                async def process_input_messages(self, data, guardrail_to_apply, litellm_logging_obj=None):  # type: ignore[override]
                    return data

                async def process_output_response(self, response, guardrail_to_apply, litellm_logging_obj=None, user_api_key_dict=None):  # type: ignore[override]
                    return response

                async def process_output_streaming_response(
                    self,
                    responses_so_far,
                    guardrail_to_apply,
                    litellm_logging_obj=None,
                    user_api_key_dict=None,
                    request_data=None,
                ):
                    # Simulate what the real handler does:
                    # put combined text in first chunk, clear the rest
                    combined = ""
                    for resp in responses_so_far:
                        for choice in resp.choices:
                            if choice.delta and choice.delta.content:
                                combined += choice.delta.content

                    first_set = False
                    for resp in responses_so_far:
                        for choice in resp.choices:
                            if not first_set:
                                choice.delta.content = combined
                                first_set = True
                            else:
                                choice.delta.content = ""

                    return responses_so_far

            # Override the mapping to use our content-clearing translation
            unified_module.endpoint_guardrail_translation_mappings = {
                CallTypes.acompletion: _ContentClearingTranslation,
            }

            handler = UnifiedLLMGuardrails()
            guardrail = RecordingGuardrail()

            # Create 10 streaming chunks with distinct content
            chunks = []
            for i in range(10):
                chunk = ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(content=f"word{i} ", role="assistant"),
                            finish_reason=None,
                        )
                    ],
                )
                chunks.append(chunk)

            async def mock_stream():
                for chunk in chunks:
                    yield chunk

            user_api_key_dict = UserAPIKeyAuth(
                api_key="test-key",
                request_route="/v1/chat/completions",
            )

            request_data = {
                "guardrail_to_apply": guardrail,
                "model": "gpt-4",
            }

            # Collect all yielded chunks
            yielded_contents = []
            async for item in handler.async_post_call_streaming_iterator_hook(
                user_api_key_dict=user_api_key_dict,
                response=mock_stream(),
                request_data=request_data,
            ):
                content = (
                    item.choices[0].delta.content if item.choices[0].delta else None
                )
                yielded_contents.append(content)

            # Every chunk should have non-empty content
            for i, content in enumerate(yielded_contents):
                assert content is not None and content != "", (
                    f"Chunk {i} lost its content (got {content!r}). "
                    f"Expected non-empty content for every streamed chunk."
                )

    class TestOCRGuardrailE2E:
        """End-to-end tests: UnifiedLLMGuardrails -> OCRHandler."""

        @pytest.mark.asyncio
        async def test_pre_call_hook_invokes_ocr_handler_for_input(self):
            """
            Verify that async_pre_call_hook with call_type=aocr routes through
            the OCR handler and calls apply_guardrail with the document URL.
            """
            handler = UnifiedLLMGuardrails()
            guardrail = RecordingGuardrail()
            cache = DualCache()

            data = {
                "guardrail_to_apply": guardrail,
                "model": "mistral/mistral-ocr-latest",
                "document": {
                    "type": "document_url",
                    "document_url": "https://arxiv.org/pdf/2201.04234",
                },
            }

            result = await handler.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
                cache=cache,
                data=data,
                call_type=CallTypes.aocr.value,
            )

            # Guardrail should have been checked and invoked
            assert guardrail.event_history == [GuardrailEventHooks.pre_call]
            assert len(guardrail.apply_calls) == 1
            assert guardrail.apply_calls[0]["input_type"] == "request"
            assert (
                "https://arxiv.org/pdf/2201.04234"
                in guardrail.apply_calls[0]["inputs"]["texts"]
            )

            # Data should be returned with document intact
            assert (
                result["document"]["document_url"] == "https://arxiv.org/pdf/2201.04234"
            )

        @pytest.mark.asyncio
        async def test_moderation_hook_invokes_ocr_handler(self):
            """
            Verify that async_moderation_hook with call_type=aocr routes through
            the OCR handler correctly.
            """
            handler = UnifiedLLMGuardrails()
            guardrail = RecordingGuardrail()

            data = {
                "guardrail_to_apply": guardrail,
                "model": "mistral/mistral-ocr-latest",
                "document": {
                    "type": "image_url",
                    "image_url": "https://example.com/scan.png",
                },
            }

            await handler.async_moderation_hook(
                data=data,
                user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
                call_type=CallTypes.aocr.value,
            )

            assert guardrail.event_history == [GuardrailEventHooks.during_call]
            assert len(guardrail.apply_calls) == 1
            assert (
                "https://example.com/scan.png"
                in guardrail.apply_calls[0]["inputs"]["texts"]
            )

        @pytest.mark.asyncio
        async def test_post_call_success_hook_guardrails_ocr_output(self):
            """
            Verify that async_post_call_success_hook resolves the OCR route
            to the OCR handler and applies guardrails to page markdown.
            """

            class TextModifyingGuardrail(CustomGuardrail):
                def __init__(self):
                    super().__init__(guardrail_name="text-modifier")

                def should_run_guardrail(self, data, event_type):  # type: ignore[override]
                    return True

                async def apply_guardrail(
                    self, inputs, request_data, input_type, **kwargs
                ):
                    texts = inputs.get("texts", [])
                    return {"texts": [t.replace("SECRET", "[REDACTED]") for t in texts]}

            handler = UnifiedLLMGuardrails()
            guardrail = TextModifyingGuardrail()

            ocr_response = OCRResponse(
                pages=[
                    OCRPage(index=0, markdown="Page 1 has a SECRET value"),
                    OCRPage(index=1, markdown="Page 2 is clean"),
                    OCRPage(index=2, markdown="Page 3 also has SECRET data"),
                ],
                model="mistral/mistral-ocr-latest",
            )

            user_api_key_dict = UserAPIKeyAuth(
                api_key="test-key",
                request_route="/v1/ocr",
            )

            data = {
                "guardrail_to_apply": guardrail,
                "model": "mistral/mistral-ocr-latest",
            }

            result = await handler.async_post_call_success_hook(
                data=data,
                user_api_key_dict=user_api_key_dict,
                response=ocr_response,
            )

            # Verify the SECRET text was redacted across pages
            assert result.pages[0].markdown == "Page 1 has a [REDACTED] value"
            assert result.pages[1].markdown == "Page 2 is clean"
            assert result.pages[2].markdown == "Page 3 also has [REDACTED] data"

        @pytest.mark.asyncio
        async def test_post_call_success_hook_ocr_route_resolves_call_type(self):
            """
            Verify that request_route=/v1/ocr correctly resolves to the OCR
            call type and the handler is invoked (not skipped).
            """
            handler = UnifiedLLMGuardrails()
            guardrail = RecordingGuardrail()

            ocr_response = OCRResponse(
                pages=[OCRPage(index=0, markdown="Some text")],
                model="mistral/mistral-ocr-latest",
            )

            user_api_key_dict = UserAPIKeyAuth(
                api_key="test-key",
                request_route="/v1/ocr",
            )

            data = {
                "guardrail_to_apply": guardrail,
                "model": "mistral/mistral-ocr-latest",
            }

            result = await handler.async_post_call_success_hook(
                data=data,
                user_api_key_dict=user_api_key_dict,
                response=ocr_response,
            )

            # Guardrail was invoked
            assert guardrail.event_history == [GuardrailEventHooks.post_call]
            assert len(guardrail.apply_calls) == 1
            assert guardrail.apply_calls[0]["input_type"] == "response"
            assert guardrail.apply_calls[0]["inputs"]["texts"] == ["Some text"]

            # Response returned with pages intact
            assert result.pages[0].markdown == "Some text"


class _StreamingTextGuardrail(CustomGuardrail):
    """Guardrail whose apply_guardrail rewrites (uppercases) response text.

    Optionally schedules a per-response-call ``stream_holdback_chars`` (indexed
    like ``texts``) and can force the mutated text shorter than the input to
    exercise the streaming underflow guard.
    """

    def __init__(self, *, holdback_schedule=None, shrink_to=None, shrink_after=0, sampling_rate=1):
        super().__init__(guardrail_name="streaming-text-guardrail")
        self.streaming_transform_mode = "incremental_diff"
        self.streaming_sampling_rate = sampling_rate
        self.streaming_end_of_stream_only = False
        self.guardrail_config = {}
        self._holdback_schedule = list(holdback_schedule or [])
        self._shrink_to = shrink_to
        self._shrink_after = shrink_after
        self.response_calls = 0
        self.received_texts = []
        self.received_tool_calls = []

    def should_run_guardrail(self, data, event_type):  # type: ignore[override]
        return True

    async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
        texts = inputs.get("texts", [])
        if input_type != "response":
            return {"texts": [t.upper() for t in texts]}

        if inputs.get("tool_calls"):
            self.received_tool_calls.append(inputs.get("tool_calls"))
        self.received_texts.append(list(texts))
        idx = self.response_calls
        self.response_calls += 1
        if self._shrink_to is not None and idx >= self._shrink_after:
            transformed = [self._shrink_to for _ in texts]
        else:
            transformed = [t.upper() for t in texts]
        result = {"texts": transformed}
        if idx < len(self._holdback_schedule):
            result["stream_holdback_chars"] = [self._holdback_schedule[idx]] * len(texts)
        return result


def _stream_chunk(content, finish_reason=None, index=0):
    return ModelResponseStream(
        choices=[
            StreamingChoices(
                index=index,
                delta=Delta(content=content, role="assistant"),
                finish_reason=finish_reason,
            )
        ],
    )


async def _drive_stream(handler, guardrail, chunks, request_route="/v1/chat/completions"):
    async def _mock_stream():
        for chunk in chunks:
            yield chunk

    user_api_key_dict = UserAPIKeyAuth(api_key="test-key", request_route=request_route)
    request_data = {"guardrail_to_apply": guardrail, "model": "gpt-4"}
    out = []
    async for item in handler.async_post_call_streaming_iterator_hook(
        user_api_key_dict=user_api_key_dict,
        response=_mock_stream(),
        request_data=request_data,
    ):
        out.append(item)
    return out


def _delta_text(item):
    if not getattr(item, "choices", None):
        return ""
    return item.choices[0].delta.content or ""


class TestStreamingTransform:
    """Streaming text-transformation (incremental_diff) path on the OpenAI chat
    completions streaming surface."""

    @pytest.fixture(autouse=True)
    def _use_openai_handler_mapping(self):
        unified_module.endpoint_guardrail_translation_mappings = {
            CallTypes.acompletion: OpenAIChatCompletionsHandler,
        }
        yield
        unified_module.endpoint_guardrail_translation_mappings = None

    @pytest.mark.asyncio
    async def test_block_only_drops_text_rewrites(self):
        """Default block_only: the guardrail's uppercasing never reaches the
        client; the original lowercase chunks are streamed verbatim."""
        guardrail = _StreamingTextGuardrail()
        guardrail.streaming_transform_mode = "block_only"

        chunks = [
            _stream_chunk("hello "),
            _stream_chunk("world"),
            _stream_chunk("", finish_reason="stop"),
        ]

        out = await _drive_stream(UnifiedLLMGuardrails(), guardrail, chunks)
        streamed = "".join(_delta_text(i) for i in out)

        assert streamed == "hello world"
        assert streamed != streamed.upper()

    @pytest.mark.asyncio
    async def test_incremental_diff_emits_uppercased_deltas(self):
        """incremental_diff: the client receives uppercased deltas whose
        concatenation equals uppercase(full)."""
        guardrail = _StreamingTextGuardrail()

        full = "hello world this is streaming"
        words = ["hello ", "world ", "this ", "is ", "streaming"]
        chunks = [_stream_chunk(w) for w in words]
        chunks.append(_stream_chunk("", finish_reason="stop"))

        out = await _drive_stream(UnifiedLLMGuardrails(), guardrail, chunks)
        streamed = "".join(_delta_text(i) for i in out)

        assert streamed == full.upper()
        # No raw lowercase content leaked onto the wire.
        assert "hello" not in streamed

    @pytest.mark.asyncio
    async def test_incremental_diff_holdback_boundary(self):
        """Holdback=5 on the first sample withholds the trailing chars until the
        next round; the final concatenation matches with no loss or duplication."""
        guardrail = _StreamingTextGuardrail(holdback_schedule=[5, 0])

        # No finish_reason: every sample uses the combined-text branch so the
        # scheduled holdback is applied on the first round.
        chunks = [_stream_chunk("abcdef"), _stream_chunk("ghij")]

        out = await _drive_stream(UnifiedLLMGuardrails(), guardrail, chunks)
        deltas = [_delta_text(i) for i in out]
        streamed = "".join(deltas)

        # First sample: "ABCDEF" with holdback 5 -> only "A" is emitted.
        assert deltas[0] == "A"
        assert streamed == "ABCDEFGHIJ"

    @pytest.mark.asyncio
    async def test_incremental_diff_underflow_raises(self):
        """A transform shorter than what was already streamed cannot retract
        bytes: it raises HTTPException(stream_transform_underflow)."""
        # First sample emits "ABCDEF" (6 chars); second sample shrinks to 3.
        guardrail = _StreamingTextGuardrail(shrink_to="ABC", shrink_after=1)

        chunks = [_stream_chunk("abcdef"), _stream_chunk("ghij")]

        with pytest.raises(unified_module.HTTPException) as exc_info:
            await _drive_stream(UnifiedLLMGuardrails(), guardrail, chunks)

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "stream_transform_underflow"

    @pytest.mark.asyncio
    async def test_incremental_diff_final_chunk_preserves_finish_reason(self):
        """The final synthetic chunk carries the finish_reason of the last raw
        chunk."""
        guardrail = _StreamingTextGuardrail()

        chunks = [
            _stream_chunk("hello "),
            _stream_chunk("world"),
            _stream_chunk("", finish_reason="stop"),
        ]

        out = await _drive_stream(UnifiedLLMGuardrails(), guardrail, chunks)

        assert out, "expected at least one synthetic chunk"
        assert out[-1].choices[0].finish_reason == "stop"
        assert "".join(_delta_text(i) for i in out) == "HELLO WORLD"

    @pytest.mark.asyncio
    async def test_end_of_stream_only_emits_single_final_chunk(self):
        """incremental_diff + streaming_end_of_stream_only: a single post-stream
        synthetic chunk carries the whole guardrailed text and the finish_reason."""
        guardrail = _StreamingTextGuardrail()
        guardrail.streaming_end_of_stream_only = True

        chunks = [
            _stream_chunk("hello "),
            _stream_chunk("world"),
            _stream_chunk("", finish_reason="stop"),
        ]

        out = await _drive_stream(UnifiedLLMGuardrails(), guardrail, chunks)

        non_empty = [i for i in out if _delta_text(i)]
        assert len(non_empty) == 1
        assert _delta_text(non_empty[0]) == "HELLO WORLD"
        assert out[-1].choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_unsupported_route_falls_back_to_block_only(self):
        """A route that does not resolve to the OpenAI chat handler falls back to
        block_only rather than transforming."""
        guardrail = _StreamingTextGuardrail()

        chunks = [_stream_chunk("hello "), _stream_chunk("world", finish_reason="stop")]

        # request_route=None => no resolvable call type => block_only fallback.
        out = await _drive_stream(UnifiedLLMGuardrails(), guardrail, chunks, request_route=None)
        streamed = "".join(_delta_text(i) for i in out)

        assert streamed == "hello world"

    @pytest.mark.asyncio
    async def test_emit_streaming_http_error_a2a_yields_jsonrpc_chunk(self):
        """The shared streaming error helper emits an in-stream JSON-RPC error for
        A2A call types instead of raising."""
        import json

        handler = UnifiedLLMGuardrails()
        exc = unified_module.HTTPException(
            status_code=400,
            detail={"error": "stream_transform_underflow", "message": "boom"},
        )

        emitted = []
        async for item in handler._emit_streaming_http_error(
            exc,
            call_type=CallTypes.asend_message.value,
            responses_so_far=[{"id": "req-1"}],
            request_data={},
        ):
            emitted.append(item)

        assert len(emitted) == 1
        payload = json.loads(emitted[0])
        assert payload["error"]["message"] == "stream_transform_underflow"
        assert payload["id"] == "req-1"

    def test_final_chunk_preserves_per_choice_finish_reason(self):
        """The final flush must carry each choice's own finish_reason, not
        choices[0]'s, for n > 1 (e.g. "stop" vs "length")."""
        reference_chunk = ModelResponseStream(choices=[StreamingChoices(index=0, delta=Delta(content="a"))])

        synthetic = UnifiedLLMGuardrails()._build_transform_chunk(
            reference_chunk=reference_chunk,
            mutated_text_per_choice={0: "A", 1: "B"},
            emitted_text_per_choice={},
            holdback_per_choice={},
            finish_reason_per_choice={0: "stop", 1: "length"},
            is_final=True,
        )

        by_index = {c.index: c for c in synthetic.choices}
        assert by_index[0].finish_reason == "stop"
        assert by_index[1].finish_reason == "length"
        assert by_index[0].delta.content == "A"
        assert by_index[1].delta.content == "B"

    def test_synthetic_chunk_drops_raw_tool_calls(self):
        """v1 does not transform streamed tool calls; the synthetic chunk must not
        pass raw upstream tool_calls through (they would bypass the guardrail)."""
        reference_chunk = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content="hi",
                        tool_calls=[
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "leak", "arguments": '{"ssn": "123-45-6789"}'},
                            }
                        ],
                    ),
                    finish_reason=None,
                ),
            ],
        )

        synthetic = UnifiedLLMGuardrails()._build_transform_chunk(
            reference_chunk=reference_chunk,
            mutated_text_per_choice={0: "HI"},
            emitted_text_per_choice={},
            holdback_per_choice={},
            finish_reason_per_choice={},
            is_final=False,
        )

        assert synthetic.choices[0].delta.tool_calls is None
        assert synthetic.choices[0].delta.content == "HI"

    def test_rewriting_already_emitted_prefix_raises(self):
        """If a later transform rewrites bytes already streamed (not a forward
        extension), the framework fails closed rather than leaking the original."""
        reference_chunk = ModelResponseStream(
            choices=[StreamingChoices(index=0, delta=Delta(content="x"), finish_reason=None)],
        )

        with pytest.raises(unified_module.HTTPException) as exc_info:
            UnifiedLLMGuardrails()._build_transform_chunk(
                reference_chunk=reference_chunk,
                # already streamed "My SSN is 123"; the guardrail now wants to
                # redact those already-sent chars -> not a forward extension.
                mutated_text_per_choice={0: "My SSN is [REDACTED]"},
                emitted_text_per_choice={0: "My SSN is 123"},
                holdback_per_choice={},
                finish_reason_per_choice={},
                is_final=False,
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "stream_transform_underflow"

    @pytest.mark.asyncio
    async def test_tool_call_chunks_pass_through_and_not_dropped(self):
        """A tool-call chunk is passed through raw under incremental_diff (v1 does
        not transform tool calls) rather than being withheld and dropped, and no
        bogus empty-choices chunk is emitted for a tool-call-only turn."""
        guardrail = _StreamingTextGuardrail()

        tool_chunk = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=None,
                        tool_calls=[
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "{}"},
                            }
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )

        out = await _drive_stream(UnifiedLLMGuardrails(), guardrail, [tool_chunk])

        assert len(out) == 1
        assert out[0].choices[0].delta.tool_calls
        assert out[0].choices[0].finish_reason == "tool_calls"
        # The tool call is delivered raw but still inspected by the guardrail at
        # end of stream (it can block), matching block_only.
        assert guardrail.received_tool_calls

    @pytest.mark.asyncio
    async def test_per_choice_finish_reason_when_choices_finish_in_different_chunks(self):
        """n>1: a choice finishing before the stream's last chunk keeps its own
        finish_reason (it must not be lost because it is not on last_chunk)."""
        guardrail = _StreamingTextGuardrail()
        guardrail.streaming_end_of_stream_only = True  # only the flush emits

        chunks = [
            _stream_chunk("aa", index=0),
            _stream_chunk("bb", index=1),
            _stream_chunk("", finish_reason="stop", index=0),
            _stream_chunk("", finish_reason="length", index=1),
        ]

        out = await _drive_stream(UnifiedLLMGuardrails(), guardrail, chunks)

        by_index = {}
        for item in out:
            for choice in item.choices:
                if choice.finish_reason is not None:
                    by_index[choice.index] = choice.finish_reason
        assert by_index == {0: "stop", 1: "length"}

    @pytest.mark.asyncio
    async def test_short_guardrail_texts_withheld_not_leaked(self):
        """If the guardrail returns fewer texts than sent (contract violation),
        the unmatched choice is withheld (fail closed), not emitted raw."""

        class _DropsSecondChoice(_StreamingTextGuardrail):
            async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
                texts = inputs.get("texts", [])
                if input_type != "response":
                    return {"texts": [t.upper() for t in texts]}
                # Return only the first choice's transformed text.
                return {"texts": [texts[0].upper()] if texts else []}

        guardrail = _DropsSecondChoice()
        guardrail.streaming_end_of_stream_only = True

        chunks = [
            _stream_chunk("secret-a", index=0),
            _stream_chunk("secret-b", index=1),
            _stream_chunk("", finish_reason="stop", index=0),
            _stream_chunk("", finish_reason="stop", index=1),
        ]

        out = await _drive_stream(UnifiedLLMGuardrails(), guardrail, chunks)

        streamed = "".join(_delta_text(i) for i in out)
        assert "SECRET-A" in streamed
        # Choice 1 had no guardrailed text returned: withheld, never leaked raw.
        assert "secret-b" not in streamed
        assert "SECRET-B" not in streamed

    @pytest.mark.asyncio
    async def test_no_spurious_chunk_after_text_then_tool_call_finish(self):
        """When a choice streams text and then finishes via a tool-call chunk, the
        raw tool-call chunk carries the finish_reason and no spurious empty chunk
        for that choice is emitted afterwards (protocol: no delta after finish)."""
        guardrail = _StreamingTextGuardrail()

        tool_chunk = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=None,
                        tool_calls=[
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "{}"},
                            }
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )
        chunks = [_stream_chunk("let me check "), tool_chunk]

        out = await _drive_stream(UnifiedLLMGuardrails(), guardrail, chunks)

        # Exactly: one synthetic text delta, then the raw tool-call chunk. No
        # trailing empty chunk for choice 0 after it already finished.
        assert len(out) == 2
        assert _delta_text(out[0]) == "LET ME CHECK "
        assert out[1].choices[0].delta.tool_calls
        assert out[1].choices[0].finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_tool_call_blocking_guardrail_is_enforced(self):
        """A guardrail that blocks on tool calls must terminate the incremental_diff
        stream: tool calls go through the block decision, not bypass it."""
        from litellm.exceptions import GuardrailRaisedException

        class _ToolCallBlocker(_StreamingTextGuardrail):
            async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
                if input_type == "response" and inputs.get("tool_calls"):
                    raise GuardrailRaisedException(
                        guardrail_name="tc-block",
                        message="blocked tool call",
                        should_wrap_with_default_message=False,
                    )
                return await super().apply_guardrail(inputs, request_data, input_type, **kwargs)

        tool_chunk = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=None,
                        tool_calls=[
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "exfiltrate", "arguments": "{}"},
                            }
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )

        with pytest.raises(GuardrailRaisedException):
            await _drive_stream(UnifiedLLMGuardrails(), _ToolCallBlocker(), [tool_chunk])

    @pytest.mark.asyncio
    async def test_mixed_content_and_tool_call_chunk_does_not_leak_text(self):
        """A chunk carrying BOTH delta.content and a tool call must not be yielded
        raw: the text has to go through the transform, only tool-call fields pass
        through raw (content stripped)."""
        guardrail = _StreamingTextGuardrail()

        mixed = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content="secret",
                        role="assistant",
                        tool_calls=[
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )

        out = await _drive_stream(UnifiedLLMGuardrails(), guardrail, [mixed])

        streamed = "".join(_delta_text(i) for i in out)
        # Raw text never reaches the client; only the transformed text does.
        assert "secret" not in streamed
        assert "SECRET" in streamed
        # The tool call is delivered, but its chunk carries no text.
        tool_chunks = [i for i in out if i.choices[0].delta.tool_calls]
        assert tool_chunks
        assert all(not (c.choices[0].delta.content or "") for c in tool_chunks)

    @pytest.mark.asyncio
    async def test_n_gt_1_text_and_tool_call_in_same_chunk_no_text_leak(self):
        """n>1 chunk where one choice streams text and another a tool call: the
        text choice must be transformed, not emitted raw alongside the tool call."""
        guardrail = _StreamingTextGuardrail()

        chunk = ModelResponseStream(
            choices=[
                StreamingChoices(index=0, delta=Delta(content="secret", role="assistant"), finish_reason=None),
                StreamingChoices(
                    index=1,
                    delta=Delta(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    ),
                    finish_reason="tool_calls",
                ),
            ],
        )

        out = await _drive_stream(UnifiedLLMGuardrails(), guardrail, [chunk])

        # choice 0's text is transformed, never delivered raw on the tool chunk.
        for item in out:
            for c in item.choices:
                if c.delta.tool_calls:
                    assert not (c.delta.content or "")
        all_text = "".join(c.delta.content or "" for i in out for c in i.choices)
        assert "secret" not in all_text
        assert "SECRET" in all_text

    @pytest.mark.asyncio
    async def test_mixed_chunk_finish_reason_arrives_after_transformed_text(self):
        """Fix #1: when a single chunk carries both delta.content and tool_calls
        with finish_reason set, the passthrough must NOT emit finish_reason
        before the transformed text — SSE clients that stop reading at
        finish_reason would silently drop the guardrailed text. finish_reason
        must ride on a terminator after the transformed text."""
        guardrail = _StreamingTextGuardrail()
        mixed = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content="secret",
                        role="assistant",
                        tool_calls=[
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )

        out = await _drive_stream(UnifiedLLMGuardrails(), guardrail, [mixed])

        passthrough_idx = next(i for i, item in enumerate(out) if item.choices and item.choices[0].delta.tool_calls)
        # Passthrough MUST NOT carry finish_reason for a mixed chunk (deferred).
        assert out[passthrough_idx].choices[0].finish_reason is None, (
            f"passthrough of mixed chunk leaked finish_reason: {out[passthrough_idx].choices[0].finish_reason}"
        )
        # finish_reason arrives via a terminator that comes AFTER the passthrough,
        # so an SSE client reading top-down sees the transformed text before it
        # sees the terminator.
        finish_carriers = [
            i for i, item in enumerate(out) if item.choices and item.choices[0].finish_reason == "tool_calls"
        ]
        assert finish_carriers, "finish_reason=tool_calls never delivered"
        assert min(finish_carriers) > passthrough_idx
        # And the redacted text ("SECRET") reached the wire on some non-tool
        # chunk (i.e. the text terminator).
        transformed = "".join(
            item.choices[0].delta.content or ""
            for item in out
            if item.choices and not item.choices[0].delta.tool_calls
        )
        assert "SECRET" in transformed
        assert "secret" not in transformed

    @pytest.mark.asyncio
    async def test_text_flush_precedes_tool_call_passthrough(self):
        """Fix #3: text chunks followed by a pure tool-call chunk carrying
        finish_reason="tool_calls" must emit transformed text BEFORE the
        passthrough, otherwise SSE-compliant clients stop reading at
        finish_reason and drop the transformed text."""
        # sampling_rate 5: no mid-stream round would fire on 2 text chunks
        # without the pre-tool-call flush.
        guardrail = _StreamingTextGuardrail(sampling_rate=5)
        chunks = [
            _stream_chunk("hello "),
            _stream_chunk("world"),
            ModelResponseStream(
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(
                            content=None,
                            role="assistant",
                            tool_calls=[
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "f", "arguments": "{}"},
                                }
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
            ),
        ]

        out = await _drive_stream(UnifiedLLMGuardrails(), guardrail, chunks)

        text_indices = [
            i
            for i, item in enumerate(out)
            if item.choices and (item.choices[0].delta.content or "") and not item.choices[0].delta.tool_calls
        ]
        tool_indices = [i for i, item in enumerate(out) if item.choices and item.choices[0].delta.tool_calls]
        assert text_indices, "transformed text was never emitted"
        assert tool_indices, "tool-call passthrough missing"
        assert max(text_indices) < min(tool_indices)
        transformed = "".join(out[i].choices[0].delta.content or "" for i in text_indices)
        assert "HELLO WORLD" in transformed

    @pytest.mark.asyncio
    async def test_final_finish_reason_flushed_when_guardrail_suppresses_text(self):
        """Fix #4: when the guardrail returns texts=[] (full suppression) and a
        mixed content+tool_call chunk had deferred its finish_reason to the
        text flush, the final flush must still emit a terminator chunk carrying
        finish_reason. Otherwise the SSE stream ends without finish_reason."""

        class _SuppressAll(_StreamingTextGuardrail):
            async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
                self.received_texts.append(list(inputs.get("texts") or []))
                return {"texts": []}

        mixed = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content="secret",
                        role="assistant",
                        tool_calls=[
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )

        out = await _drive_stream(UnifiedLLMGuardrails(), _SuppressAll(), [mixed])
        finishes = [c.finish_reason for item in out for c in item.choices if c.index == 0]
        assert "tool_calls" in finishes

    @pytest.mark.asyncio
    async def test_transform_sends_texts_sorted_by_choice_index(self):
        """Fix #2: for n>1 streams where choice 1 emits before choice 0, the
        transform must send texts to the guardrail in ascending choice-index
        order so its returned texts realign to the correct choice indices."""

        class _RecordingGuardrail(_StreamingTextGuardrail):
            async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
                self.received_texts.append(list(inputs.get("texts") or []))
                return {"texts": list(inputs.get("texts") or [])}

        guardrail = _RecordingGuardrail()
        chunks = [
            _stream_chunk("beta", index=1),
            _stream_chunk("alpha", index=0),
            _stream_chunk("", index=0, finish_reason="stop"),
        ]
        await _drive_stream(UnifiedLLMGuardrails(), guardrail, chunks)
        last = guardrail.received_texts[-1]
        # Sorted ascending: alpha (index 0) before beta (index 1).
        assert last[0].startswith("alpha")
        assert last[1].startswith("beta")

    @pytest.mark.asyncio
    async def test_guardrail_always_sees_raw_accumulated_text(self):
        """The guardrail must receive the raw accumulated output each round, not a
        transformed-prefix + raw-suffix mix (responses_so_far stays untouched)."""
        guardrail = _StreamingTextGuardrail()

        chunks = [_stream_chunk("aa "), _stream_chunk("bb "), _stream_chunk("cc", finish_reason="stop")]

        await _drive_stream(UnifiedLLMGuardrails(), guardrail, chunks)

        # Every recorded input is the cumulative RAW (lowercase) text; if the
        # accumulator were corrupted by write-back, later rounds would contain
        # uppercased prefixes like "AA bb ".
        for received in guardrail.received_texts:
            assert received[0] == received[0].lower()
        assert guardrail.received_texts[-1] == ["aa bb cc"]

    def test_accumulate_keys_by_choice_index_not_position(self):
        """Single-choice chunks carrying a non-zero .index (n>1 streaming) must be
        keyed by index, not enumerate position (which would collapse to 0)."""
        handler = OpenAIChatCompletionsHandler()
        chunks = [
            _stream_chunk("hello", index=1),
            _stream_chunk(" world", index=1),
        ]

        accumulated = handler._accumulate_string_content_by_choice_index(chunks)

        assert accumulated == {1: "hello world"}

    @pytest.mark.asyncio
    async def test_terminal_chunk_not_guardrailed_twice(self):
        """A terminal (finish_reason) chunk that is also a sampling boundary must
        be processed once by the end-of-stream flush, not by a sampled round too."""
        guardrail = _StreamingTextGuardrail()  # sampling_rate=1

        chunks = [_stream_chunk("aa "), _stream_chunk("bb", finish_reason="stop")]

        await _drive_stream(UnifiedLLMGuardrails(), guardrail, chunks)

        # Round 1 (chunk 1) + end-of-stream flush = 2 calls. Without the terminal
        # skip, chunk 2 would be guardrailed by a sampled round AND the flush (3).
        assert guardrail.response_calls == 2

    @pytest.mark.asyncio
    async def test_malformed_holdback_from_in_process_guardrail_degrades(self):
        """An in-process guardrail (bypassing from_dict) returning a null holdback
        must degrade to 0 in the handler, not raise and abort the stream."""
        guardrail = _StreamingTextGuardrail(holdback_schedule=[None])

        chunks = [_stream_chunk("abc"), _stream_chunk("def", finish_reason="stop")]

        out = await _drive_stream(UnifiedLLMGuardrails(), guardrail, chunks)

        # None holdback treated as 0: full text emitted, no crash.
        assert "".join(_delta_text(i) for i in out) == "ABCDEF"

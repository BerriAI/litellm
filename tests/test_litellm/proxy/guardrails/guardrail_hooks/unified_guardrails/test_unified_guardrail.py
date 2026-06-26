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

    class TestActionMode:
        """Streaming action protocol (auto-enabled when the guardrail
        implements apply_guardrail_action)."""

        @staticmethod
        def _make_chunk(content=None, tool_calls=None, finish_reason=None):
            return ModelResponseStream(
                id="cmpl-test",
                created=0,
                model="test-model",
                object="chat.completion.chunk",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(
                            content=content,
                            role="assistant",
                            tool_calls=tool_calls,
                        ),
                        finish_reason=finish_reason,
                    )
                ],
            )

        @staticmethod
        def _content_chunks(parts, finish_reason="stop"):
            n = len(parts)

            async def gen():
                for i, p in enumerate(parts):
                    yield TestUnifiedLLMGuardrails.TestActionMode._make_chunk(
                        content=p,
                        finish_reason=finish_reason if i == n - 1 else None,
                    )

            return gen()

        @staticmethod
        async def _collect(generator):
            out = []
            async for item in generator:
                out.append(item)
            return out

        class _ScriptedActionGuardrail(CustomGuardrail):
            """Drives apply_guardrail_action with a scripted sequence of decisions."""

            def __init__(self, scripted_decisions, sampling_rate=2):
                super().__init__(guardrail_name="scripted-action-guardrail")
                # Each decision: (action, texts, blocked_reason).
                # apply_guardrail_action's presence on this class is what the
                # iterator hook auto-detects to drive action mode.
                self.script = list(scripted_decisions)
                self.calls = []
                self.streaming_sampling_rate = sampling_rate
                self.streaming_end_of_stream_only = False
                self.unreachable_fallback = "fail_closed"

            def should_run_guardrail(self, data, event_type):  # type: ignore[override]
                return True

            async def apply_guardrail_action(
                self, *, inputs, request_data, input_type, logging_obj=None
            ):
                from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import (
                    GenericGuardrailAPIResponse,
                )

                if not self.script:
                    raise AssertionError(
                        "scripted guardrail ran out of decisions"
                    )
                action, texts, blocked_reason = self.script.pop(0)
                self.calls.append(
                    {
                        "text": (inputs.get("texts") or [""])[0],
                        "is_final": inputs.get("is_final"),
                        "tool_calls": inputs.get("tool_calls"),
                        "action": action,
                    }
                )
                return GenericGuardrailAPIResponse(
                    action=action,
                    texts=[texts] if texts is not None else None,
                    blocked_reason=blocked_reason,
                )

        @pytest.mark.asyncio
        async def test_auto_detect_falls_back_to_moderation(self):
            """A guardrail without apply_guardrail_action gets moderation mode.

            The iterator hook dispatches on the presence of a callable
            apply_guardrail_action method — RecordingGuardrail doesn't have
            one, so the historical observe-only iterator hook should run and
            yield the original chunks unmodified.
            """
            handler = UnifiedLLMGuardrails()
            guardrail = RecordingGuardrail()
            assert not callable(
                getattr(guardrail, "apply_guardrail_action", None)
            ), "this test relies on RecordingGuardrail not implementing the action protocol"

            chunks = [
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(content=f"w{i}", role="assistant"),
                            finish_reason=None,
                        )
                    ],
                )
                for i in range(3)
            ]

            async def upstream():
                for c in chunks:
                    yield c

            user = UserAPIKeyAuth(
                api_key="k", request_route="/v1/chat/completions"
            )
            request_data = {"guardrail_to_apply": guardrail, "model": "gpt-4"}

            out = await TestUnifiedLLMGuardrails.TestActionMode._collect(
                handler.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=user,
                    response=upstream(),
                    request_data=request_data,
                )
            )
            # Moderation passes through original chunks unchanged.
            assert len(out) == 3
            assert "".join(
                c.choices[0].delta.content for c in out if c.choices[0].delta.content
            ) == "w0w1w2"

        @pytest.mark.asyncio
        async def test_action_mode_emits_modified_text(self):
            """GUARDRAIL_INTERVENED: client receives modified text only."""
            handler = UnifiedLLMGuardrails()
            guardrail = TestUnifiedLLMGuardrails.TestActionMode._ScriptedActionGuardrail(
                [
                    ("GUARDRAIL_INTERVENED", "ABCD", None),
                    ("GUARDRAIL_INTERVENED", "ABCDEFGH", None),
                    ("GUARDRAIL_INTERVENED", "ABCDEFGH", None),
                ],
                sampling_rate=2,
            )
            user = UserAPIKeyAuth(
                api_key="k", request_route="/v1/chat/completions"
            )
            request_data = {"guardrail_to_apply": guardrail}

            out = await TestUnifiedLLMGuardrails.TestActionMode._collect(
                handler.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=user,
                    response=TestUnifiedLLMGuardrails.TestActionMode._content_chunks(
                        ["ab", "cd", "ef", "gh"]
                    ),
                    request_data=request_data,
                )
            )
            text = "".join(
                c.choices[0].delta.content or "" for c in out
            )
            assert text == "ABCDEFGH"
            assert out[-1].choices[0].finish_reason == "stop"
            assert [c["is_final"] for c in guardrail.calls] == [False, False, True]

        @pytest.mark.asyncio
        async def test_action_mode_buffers_during_wait(self):
            """WAIT yields nothing; on resume, the full delta reaches the client."""
            handler = UnifiedLLMGuardrails()
            guardrail = TestUnifiedLLMGuardrails.TestActionMode._ScriptedActionGuardrail(
                [
                    ("WAIT", None, None),
                    ("WAIT", None, None),
                    ("WAIT", None, None),
                    ("GUARDRAIL_INTERVENED", "FINAL", None),
                ],
                sampling_rate=2,
            )
            user = UserAPIKeyAuth(
                api_key="k", request_route="/v1/chat/completions"
            )
            out = await TestUnifiedLLMGuardrails.TestActionMode._collect(
                handler.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=user,
                    response=TestUnifiedLLMGuardrails.TestActionMode._content_chunks(
                        ["ab", "cd", "ef", "gh"]
                    ),
                    request_data={"guardrail_to_apply": guardrail},
                )
            )
            text = "".join(c.choices[0].delta.content or "" for c in out)
            assert text == "FINAL"
            # WAIT collapses sample-rate to 1: every subsequent chunk triggers a call.
            assert [c["is_final"] for c in guardrail.calls] == [
                False,
                False,
                False,
                True,
            ]

        @pytest.mark.asyncio
        async def test_action_mode_blocks_mid_stream(self):
            """BLOCKED raises GuardrailRaisedException with the blocked_reason."""
            from litellm.exceptions import GuardrailRaisedException

            handler = UnifiedLLMGuardrails()
            guardrail = TestUnifiedLLMGuardrails.TestActionMode._ScriptedActionGuardrail(
                [("BLOCKED", None, "policy denied")],
                sampling_rate=2,
            )
            user = UserAPIKeyAuth(
                api_key="k", request_route="/v1/chat/completions"
            )
            with pytest.raises(GuardrailRaisedException, match="policy denied"):
                await TestUnifiedLLMGuardrails.TestActionMode._collect(
                    handler.async_post_call_streaming_iterator_hook(
                        user_api_key_dict=user,
                        response=TestUnifiedLLMGuardrails.TestActionMode._content_chunks(
                            ["ab", "cd"]
                        ),
                        request_data={"guardrail_to_apply": guardrail},
                    )
                )

        @pytest.mark.asyncio
        async def test_action_mode_wait_at_eos_is_violation(self):
            """WAIT at is_final=True is a protocol violation; fail_closed raises."""
            from litellm.exceptions import GuardrailRaisedException

            handler = UnifiedLLMGuardrails()
            guardrail = TestUnifiedLLMGuardrails.TestActionMode._ScriptedActionGuardrail(
                # sampling_rate=100 ensures only the EOS call is made
                [("WAIT", None, None)],
                sampling_rate=100,
            )
            user = UserAPIKeyAuth(
                api_key="k", request_route="/v1/chat/completions"
            )
            with pytest.raises(
                GuardrailRaisedException, match="end of stream"
            ):
                await TestUnifiedLLMGuardrails.TestActionMode._collect(
                    handler.async_post_call_streaming_iterator_hook(
                        user_api_key_dict=user,
                        response=TestUnifiedLLMGuardrails.TestActionMode._content_chunks(
                            ["ab"]
                        ),
                        request_data={"guardrail_to_apply": guardrail},
                    )
                )

        @pytest.mark.asyncio
        async def test_action_mode_modify_shrink_is_violation(self):
            """GUARDRAIL_INTERVENED with text shorter than cursor raises."""
            from litellm.exceptions import GuardrailRaisedException

            handler = UnifiedLLMGuardrails()
            guardrail = TestUnifiedLLMGuardrails.TestActionMode._ScriptedActionGuardrail(
                [
                    ("GUARDRAIL_INTERVENED", "abcd", None),  # cursor=4
                    ("GUARDRAIL_INTERVENED", "ab", None),  # shrink → violation
                ],
                sampling_rate=2,
            )
            user = UserAPIKeyAuth(
                api_key="k", request_route="/v1/chat/completions"
            )
            with pytest.raises(
                GuardrailRaisedException, match="retract already-emitted"
            ):
                await TestUnifiedLLMGuardrails.TestActionMode._collect(
                    handler.async_post_call_streaming_iterator_hook(
                        user_api_key_dict=user,
                        response=TestUnifiedLLMGuardrails.TestActionMode._content_chunks(
                            ["ab", "cd", "ef", "gh"]
                        ),
                        request_data={"guardrail_to_apply": guardrail},
                    )
                )

        @pytest.mark.asyncio
        async def test_action_mode_end_of_stream_only_blocks_before_emit(
            self,
        ):
            """
            With `streaming_end_of_stream_only=True`, action-mode guardrails
            must not emit any content past cursor before the EOS decision.
            A BLOCKED at is_final=true terminates the stream cleanly without
            anything having reached the client mid-stream.
            """
            from litellm.exceptions import GuardrailRaisedException

            handler = UnifiedLLMGuardrails()
            guardrail = TestUnifiedLLMGuardrails.TestActionMode._ScriptedActionGuardrail(
                # Only one decision — should only be called once at EOS.
                [("BLOCKED", None, "policy violation at EOS")],
                sampling_rate=1,  # would normally fire every chunk
            )
            guardrail.streaming_end_of_stream_only = True

            user = UserAPIKeyAuth(
                api_key="k", request_route="/v1/chat/completions"
            )
            with pytest.raises(
                GuardrailRaisedException, match="policy violation at EOS"
            ):
                await TestUnifiedLLMGuardrails.TestActionMode._collect(
                    handler.async_post_call_streaming_iterator_hook(
                        user_api_key_dict=user,
                        response=TestUnifiedLLMGuardrails.TestActionMode._content_chunks(
                            ["ab", "cd", "ef", "gh"]
                        ),
                        request_data={"guardrail_to_apply": guardrail},
                    )
                )
            # Single guardrail call, at is_final=True. Mid-stream calls
            # were suppressed by end_of_stream_only.
            assert len(guardrail.calls) == 1, (
                f"expected single EOS call, got {len(guardrail.calls)}: "
                f"{guardrail.calls}"
            )
            assert guardrail.calls[0]["is_final"] is True

        @pytest.mark.asyncio
        async def test_action_mode_end_of_stream_only_emits_modified_at_eos(
            self,
        ):
            """
            With `streaming_end_of_stream_only=True`, mid-stream samples are
            suppressed. A successful GUARDRAIL_INTERVENED at is_final=true
            emits the modified text in one delta and terminates.
            """
            handler = UnifiedLLMGuardrails()
            guardrail = TestUnifiedLLMGuardrails.TestActionMode._ScriptedActionGuardrail(
                [("GUARDRAIL_INTERVENED", "REWRITTEN", None)],
                sampling_rate=1,
            )
            guardrail.streaming_end_of_stream_only = True

            user = UserAPIKeyAuth(
                api_key="k", request_route="/v1/chat/completions"
            )
            out = await TestUnifiedLLMGuardrails.TestActionMode._collect(
                handler.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=user,
                    response=TestUnifiedLLMGuardrails.TestActionMode._content_chunks(
                        ["ab", "cd", "ef", "gh"]
                    ),
                    request_data={"guardrail_to_apply": guardrail},
                )
            )
            text = "".join(c.choices[0].delta.content or "" for c in out)
            assert text == "REWRITTEN", text
            assert out[-1].choices[0].finish_reason == "stop"
            # Single guardrail call, at is_final=True.
            assert len(guardrail.calls) == 1
            assert guardrail.calls[0]["is_final"] is True

        @pytest.mark.asyncio
        async def test_action_mode_none_after_expansion_keeps_stream_alive(
            self,
        ):
            """
            A guardrail that previously returned an expanded
            GUARDRAIL_INTERVENED (modified text longer than raw accumulated
            text) and then returns NONE must not terminate the stream.

            The protocol must defer the NONE-with-shrink call (no emission,
            cursor stays put) and let a later GUARDRAIL_INTERVENED refine
            cleanly. Raising on this edge would terminate an otherwise
            healthy stream when a guardrail momentarily lacks the context
            needed to reproduce its earlier expansion.
            """
            handler = UnifiedLLMGuardrails()
            guardrail = TestUnifiedLLMGuardrails.TestActionMode._ScriptedActionGuardrail(
                [
                    # Sample 1 (after chunks 1+2, accumulated="aabb", 4 chars):
                    # guardrail expands to 18 chars. Cursor 0 → 18.
                    ("GUARDRAIL_INTERVENED", "AABB-EXPANDED-MORE", None),
                    # Sample 2 (after chunks 3+4, accumulated="aabbccdd",
                    # 8 chars): NONE on raw text shorter than cursor.
                    # Must defer, not raise.
                    ("NONE", None, None),
                    # EOS (is_final=True): guardrail returns expanded text
                    # again. 24 >= 18 → emit "-FINAL".
                    (
                        "GUARDRAIL_INTERVENED",
                        "AABB-EXPANDED-MORE-FINAL",
                        None,
                    ),
                ],
                sampling_rate=2,
            )
            user = UserAPIKeyAuth(
                api_key="k", request_route="/v1/chat/completions"
            )
            out = await TestUnifiedLLMGuardrails.TestActionMode._collect(
                handler.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=user,
                    response=TestUnifiedLLMGuardrails.TestActionMode._content_chunks(
                        ["aa", "bb", "cc", "dd"]
                    ),
                    request_data={"guardrail_to_apply": guardrail},
                )
            )
            text = "".join(c.choices[0].delta.content or "" for c in out)
            # Stream did not terminate; final delta extends the established
            # expanded-text trajectory.
            assert text == "AABB-EXPANDED-MORE-FINAL", text
            assert out[-1].choices[0].finish_reason == "stop"
            # All three scripted decisions consumed (sample 1, sample 2 NONE
            # which deferred, EOS).
            assert [c["is_final"] for c in guardrail.calls] == [
                False,
                False,
                True,
            ]

        @pytest.mark.asyncio
        async def test_action_mode_eos_shrink_does_not_leak_raw_tail(self):
            """At EOS, a guardrail returning text shorter than the cursor
            must not cause the raw, unmodified tail to leak.

            Regression scenario: a redaction guardrail substitutes cleanly
            on every mid-stream sample, then at is_final=True returns short
            text (e.g. due to a bug or backend timeout). Falling back to
            `accumulated_text` would emit `accumulated_text[cursor:]` —
            the raw, unredacted tail of the upstream stream. The correct
            behavior is to emit nothing further; the bytes through cursor
            were already sent (with substitutions) and we cannot extend
            without potentially leaking source content past cursor.
            """
            handler = UnifiedLLMGuardrails()
            # Mid-stream sample at cursor=2 (after chunk 2): substitute
            # "ab" with "AB". Cursor advances to 2.
            # Mid-stream sample at cursor=4: substitute with "ABCD". Cursor=4.
            # EOS sample: guardrail returns short text "AB" (len 2 < cursor 4)
            # — should not leak the raw tail "efgh".
            guardrail = TestUnifiedLLMGuardrails.TestActionMode._ScriptedActionGuardrail(
                [
                    ("GUARDRAIL_INTERVENED", "AB", None),
                    ("GUARDRAIL_INTERVENED", "ABCD", None),
                    ("GUARDRAIL_INTERVENED", "AB", None),  # EOS shrink
                ],
                sampling_rate=2,
            )
            user = UserAPIKeyAuth(
                api_key="k", request_route="/v1/chat/completions"
            )
            out = await TestUnifiedLLMGuardrails.TestActionMode._collect(
                handler.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=user,
                    response=TestUnifiedLLMGuardrails.TestActionMode._content_chunks(
                        ["ab", "cd", "ef", "gh"]
                    ),
                    request_data={"guardrail_to_apply": guardrail},
                )
            )
            text = "".join(c.choices[0].delta.content or "" for c in out)
            # Stream emitted "AB" + "CD" through mid-stream samples.
            # EOS shrink: nothing more is emitted. The raw tail "efgh"
            # MUST NOT appear.
            assert "ef" not in text and "gh" not in text, (
                f"raw upstream tail leaked at EOS shrink: {text!r}"
            )
            assert text == "ABCD", text
            # Stream still terminates with finish_reason.
            assert out[-1].choices[0].finish_reason == "stop"

        @pytest.mark.asyncio
        async def test_action_mode_chunk_straddling_surrogate(self):
            """Surrogate token spans chunks: WAIT until complete, then INTERVENED."""
            handler = UnifiedLLMGuardrails()
            # Upstream produces "before [EMA" / "IL_1] after" — surrogate split.
            # Guardrail returns WAIT on partial, GUARDRAIL_INTERVENED on complete.
            guardrail = TestUnifiedLLMGuardrails.TestActionMode._ScriptedActionGuardrail(
                [
                    ("WAIT", None, None),
                    (
                        "GUARDRAIL_INTERVENED",
                        "before [EMAIL_1] after",
                        None,
                    ),
                    (
                        "GUARDRAIL_INTERVENED",
                        "before [EMAIL_1] after",
                        None,
                    ),
                ],
                sampling_rate=1,
            )
            user = UserAPIKeyAuth(
                api_key="k", request_route="/v1/chat/completions"
            )
            out = await TestUnifiedLLMGuardrails.TestActionMode._collect(
                handler.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=user,
                    response=TestUnifiedLLMGuardrails.TestActionMode._content_chunks(
                        ["before [EMA", "IL_1] after"]
                    ),
                    request_data={"guardrail_to_apply": guardrail},
                )
            )
            text = "".join(c.choices[0].delta.content or "" for c in out)
            # Client never sees the leaky partial; full surrogate emitted as one delta.
            assert text == "before [EMAIL_1] after"
            assert "[EMA" not in "".join(
                c.choices[0].delta.content
                for c in out
                if c.choices[0].delta.content
                and len(c.choices[0].delta.content) < 10
            ), "no partial-surrogate sub-emit"

        @pytest.mark.asyncio
        async def test_action_mode_tool_calls_surfaced_to_guardrail(self):
            """Accumulated tool_calls reach the guardrail via inputs.tool_calls."""
            handler = UnifiedLLMGuardrails()
            guardrail = TestUnifiedLLMGuardrails.TestActionMode._ScriptedActionGuardrail(
                [
                    ("NONE", None, None),
                    ("NONE", None, None),
                    ("NONE", None, None),
                ],
                sampling_rate=2,
            )

            async def upstream():
                _make = TestUnifiedLLMGuardrails.TestActionMode._make_chunk
                yield _make(content="Hi! ")
                yield _make(
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"loc":"',
                            },
                        }
                    ]
                )
                yield _make(
                    tool_calls=[
                        {
                            "index": 0,
                            "function": {"arguments": 'NYC"}'},
                        }
                    ]
                )
                yield _make(content="Done.", finish_reason="stop")

            user = UserAPIKeyAuth(
                api_key="k", request_route="/v1/chat/completions"
            )
            out = await TestUnifiedLLMGuardrails.TestActionMode._collect(
                handler.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=user,
                    response=upstream(),
                    request_data={"guardrail_to_apply": guardrail},
                )
            )
            # Final guardrail call sees the complete accumulated tool_call.
            final_call = guardrail.calls[-1]
            assert final_call["is_final"] is True
            assert final_call["tool_calls"] is not None
            assert len(final_call["tool_calls"]) == 1
            tc = final_call["tool_calls"][0]
            assert tc["id"] == "call_1"
            assert tc["function"]["name"] == "get_weather"
            assert tc["function"]["arguments"] == '{"loc":"NYC"}'

            # Client sees both tool_call delta chunks (replayed at emit / EOS).
            tc_chunks = [
                c
                for c in out
                if c.choices[0].delta and c.choices[0].delta.tool_calls
            ]
            assert len(tc_chunks) == 2

            # Text content reaches client correctly.
            text = "".join(c.choices[0].delta.content or "" for c in out)
            assert text == "Hi! Done."

        @pytest.mark.asyncio
        async def test_action_mode_tool_calls_blockable(self):
            """BLOCKED based on tool_call args terminates the stream."""
            from litellm.exceptions import GuardrailRaisedException

            handler = UnifiedLLMGuardrails()
            guardrail = TestUnifiedLLMGuardrails.TestActionMode._ScriptedActionGuardrail(
                [("BLOCKED", None, "denied tool_call")],
                sampling_rate=2,
            )

            async def upstream():
                _make = TestUnifiedLLMGuardrails.TestActionMode._make_chunk
                yield _make(content="Hi")
                yield _make(
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "x",
                            "function": {
                                "name": "rm",
                                "arguments": '{"path":"/"}',
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                )

            user = UserAPIKeyAuth(
                api_key="k", request_route="/v1/chat/completions"
            )
            with pytest.raises(
                GuardrailRaisedException, match="denied tool_call"
            ):
                await TestUnifiedLLMGuardrails.TestActionMode._collect(
                    handler.async_post_call_streaming_iterator_hook(
                        user_api_key_dict=user,
                        response=upstream(),
                        request_data={"guardrail_to_apply": guardrail},
                    )
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

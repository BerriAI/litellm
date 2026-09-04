from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import Request, Response

import litellm
from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.llm_shield.llm_shield import (
    GUARDRAIL_NAME,
    LLMShieldGuardrail,
)
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import Choices, Delta, Message, ModelResponse, ModelResponseStream, StreamingChoices


def _guardrail(**overrides: object) -> LLMShieldGuardrail:
    params: dict[str, object] = {
        "api_key": "test-key",
        "api_base": "http://shield.test",
        "guardrail_name": GUARDRAIL_NAME,
        "event_hook": "pre_call",
        "default_on": True,
    }
    params.update(overrides)
    return LLMShieldGuardrail(**params)


def _response(payload: dict, status_code: int = 200) -> Response:
    return Response(
        status_code=status_code,
        json=payload,
        request=Request("POST", "http://shield.test/v1/guard/redact"),
    )


def _mock_post(guardrail: LLMShieldGuardrail, *payloads: dict) -> AsyncMock:
    """Queues one shield response per expected call."""
    mock = AsyncMock(side_effect=[_response(p) for p in payloads])
    guardrail.async_handler.post = mock  # type: ignore[method-assign]
    return mock


def _chunk(content: str | None, finish_reason: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(content=content), finish_reason=finish_reason)]
    )


async def _drain(generator) -> list:
    return [chunk async for chunk in generator]


def test_llm_shield_guardrail_config(monkeypatch: pytest.MonkeyPatch):
    """Should register through init_guardrails_v2 like any other provider."""
    monkeypatch.setattr(litellm, "guardrail_name_config_map", {})
    monkeypatch.setenv("LLM_SHIELD_API_KEY", "test-key")

    init_guardrails_v2(
        all_guardrails=[
            {
                "guardrail_name": "llm_shield",
                "litellm_params": {"guardrail": "llm_shield", "mode": "pre_call", "default_on": True},
            }
        ],
        config_file_path="",
    )

    registered = [cb for cb in litellm.callbacks if isinstance(cb, LLMShieldGuardrail)]
    assert len(registered) == 1
    assert registered[0].guardrail_name == "llm_shield"


class TestLLMShieldInitialization:
    def test_api_base_defaults_to_localhost(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LLM_SHIELD_API_BASE", raising=False)
        assert _guardrail(api_base=None).api_base == "http://localhost:8000"

    def test_api_base_reads_environment(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LLM_SHIELD_API_BASE", "http://shield.internal:9000")
        assert _guardrail(api_base=None).api_base == "http://shield.internal:9000"

    def test_trailing_slash_is_stripped(self):
        assert _guardrail(api_base="http://shield.test/").api_base == "http://shield.test"

    def test_both_modes_can_be_enabled_on_one_entry(self):
        """Redaction and restoration are two halves of one config entry.

        A deployment that lists only pre_call would redact the request and then hand
        the placeholders straight back to the end user.
        """
        guardrail = _guardrail(event_hook=["pre_call", "post_call"])
        data: dict = {"messages": []}

        assert guardrail.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call) is True
        assert guardrail.should_run_guardrail(data=data, event_type=GuardrailEventHooks.post_call) is True
        assert guardrail.should_run_guardrail(data=data, event_type=GuardrailEventHooks.during_call) is False


class TestRedaction:
    @pytest.mark.asyncio
    async def test_string_content_is_redacted(self):
        guardrail = _guardrail()
        _mock_post(guardrail, {"texts": ["Email [EMAIL_1] about it"]})

        data = {"messages": [{"role": "user", "content": "Email a@b.com about it"}]}
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="completion"
        )

        assert result["messages"][0]["content"] == "Email [EMAIL_1] about it"

    @pytest.mark.asyncio
    async def test_multimodal_text_parts_are_redacted(self):
        """The list content shape is a historical bypass; text parts must be covered."""
        guardrail = _guardrail()
        _mock_post(guardrail, {"texts": ["call [PHONE_1]"]})

        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "call 555-0100"},
                        {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
                    ],
                }
            ]
        }
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

        assert data["messages"][0]["content"][0]["text"] == "call [PHONE_1]"
        assert data["messages"][0]["content"][1]["image_url"]["url"] == "http://x/y.png"

    @pytest.mark.asyncio
    async def test_request_without_text_is_untouched(self):
        """No text to redact means no call to LLM Shield.

        This deliberately uses a request with no caller text at all. An earlier
        version used a Responses-API `input`, which asserted the very bypass that
        let `input` reach the provider unredacted.
        """
        guardrail = _guardrail()
        mock = _mock_post(guardrail)
        data = {"model": "gpt-4o", "temperature": 0.2}

        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

        mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_id_is_reused_across_hooks(self):
        """Rehydration can only resolve tokens minted under the same session."""
        guardrail = _guardrail()
        mock = _mock_post(guardrail, {"texts": ["[EMAIL_1]"]}, {"texts": ["a@b.com"]})

        data = {"messages": [{"role": "user", "content": "a@b.com"}]}
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")
        await guardrail._rehydrate(["[EMAIL_1]"], guardrail._session_id(data))

        sessions = {call.kwargs["headers"]["X-Session-ID"] for call in mock.call_args_list}
        assert len(sessions) == 1


class TestRequestCoverage:
    """Every request shape that carries caller text must be redacted.

    A shape missed here is not a cosmetic gap: the guardrail reports as enabled
    while the raw value goes to the provider.
    """

    @pytest.mark.asyncio
    async def test_responses_api_string_input_is_redacted(self):
        """Measured against a live provider: `input` reached the model unredacted."""
        guardrail = _guardrail()
        mock = _mock_post(guardrail, {"texts": ["Email [EMAIL_1] the invoice"]})

        data = {"input": "Email jane.doe@example.com the invoice"}
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="aresponses")

        assert mock.call_args_list[0].kwargs["json"]["texts"] == ["Email jane.doe@example.com the invoice"]
        assert data["input"] == "Email [EMAIL_1] the invoice"

    @pytest.mark.asyncio
    async def test_responses_api_list_input_is_redacted(self):
        guardrail = _guardrail()
        _mock_post(guardrail, {"texts": ["[EMAIL_1]", "[PHONE_1]"]})

        data = {
            "input": [
                {"role": "user", "content": "jane.doe@example.com"},
                {"role": "user", "content": [{"type": "input_text", "text": "555-0100"}]},
            ]
        }
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="aresponses")

        assert data["input"][0]["content"] == "[EMAIL_1]"
        assert data["input"][1]["content"][0]["text"] == "[PHONE_1]"

    @pytest.mark.asyncio
    async def test_tool_call_arguments_are_redacted(self):
        """Tool arguments carry the values the user asked the model to act on."""
        guardrail = _guardrail()
        _mock_post(guardrail, {"texts": ['{"email": "[EMAIL_1]"}']})

        data = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "send", "arguments": '{"email": "jane.doe@example.com"}'},
                        }
                    ],
                }
            ]
        }
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

        assert data["messages"][0]["tool_calls"][0]["function"]["arguments"] == '{"email": "[EMAIL_1]"}'

    @pytest.mark.asyncio
    async def test_responses_api_instructions_are_redacted(self):
        """`instructions` is provider-bound text that sits outside `messages`."""
        guardrail = _guardrail()
        mock = _mock_post(guardrail, {"texts": ["contact [EMAIL_1]"]})

        data = {"instructions": "contact jane.doe@example.com", "input": ""}
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="aresponses")

        assert mock.call_args_list[0].kwargs["json"]["texts"] == ["contact jane.doe@example.com"]
        assert data["instructions"] == "contact [EMAIL_1]"

    @pytest.mark.asyncio
    async def test_legacy_function_call_arguments_are_redacted(self):
        guardrail = _guardrail()
        _mock_post(guardrail, {"texts": ['{"email": "[EMAIL_1]"}']})

        data = {
            "messages": [
                {
                    "role": "assistant",
                    "function_call": {"name": "send", "arguments": '{"email": "jane.doe@example.com"}'},
                }
            ]
        }
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

        assert data["messages"][0]["function_call"]["arguments"] == '{"email": "[EMAIL_1]"}'

    @pytest.mark.asyncio
    async def test_every_shape_in_one_request_is_redacted(self):
        guardrail = _guardrail()
        mock = _mock_post(guardrail, {"texts": ["a", "b", "c", "d"]})

        data = {
            "messages": [
                {"role": "user", "content": "one"},
                {"role": "user", "content": [{"type": "text", "text": "two"}]},
                {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "f", "arguments": "three"}}],
                },
            ],
            "input": "four",
        }
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

        assert mock.call_args_list[0].kwargs["json"]["texts"] == ["one", "two", "three", "four"]
        assert data["messages"][0]["content"] == "a"
        assert data["messages"][1]["content"][0]["text"] == "b"
        assert data["messages"][2]["tool_calls"][0]["function"]["arguments"] == "c"
        assert data["input"] == "d"


class TestRestoration:
    @pytest.mark.asyncio
    async def test_openai_shape_is_restored(self):
        guardrail = _guardrail(event_hook="post_call")
        _mock_post(guardrail, {"texts": ["a@b.com"]})

        response = ModelResponse(choices=[Choices(index=0, message=Message(role="assistant", content="[EMAIL_1]"))])
        result = await guardrail.async_post_call_success_hook(
            data={"messages": []}, user_api_key_dict=None, response=response
        )

        assert result.choices[0].message.content == "a@b.com"

    @pytest.mark.asyncio
    async def test_responses_api_shape_is_restored(self):
        """The Responses API reply carries output items, not choices.

        Measured against a live provider: once the request side was fixed the reply
        came back still holding the placeholder, because this shape has no choices
        to walk.
        """
        guardrail = _guardrail(event_hook="post_call")
        _mock_post(guardrail, {"texts": ["a@b.com"]})

        response = SimpleNamespace(output=[SimpleNamespace(content=[{"type": "output_text", "text": "[EMAIL_1]"}])])
        result = await guardrail.async_post_call_success_hook(
            data={"messages": []}, user_api_key_dict=None, response=response
        )

        assert result.output[0].content[0]["text"] == "a@b.com"

    @pytest.mark.asyncio
    async def test_responses_api_object_blocks_are_restored(self):
        """Blocks arrive as objects too, depending on how far the reply is parsed."""
        guardrail = _guardrail(event_hook="post_call")
        _mock_post(guardrail, {"texts": ["a@b.com"]})

        block = SimpleNamespace(text="[EMAIL_1]")
        response = SimpleNamespace(output=[SimpleNamespace(content=[block])])
        await guardrail.async_post_call_success_hook(data={"messages": []}, user_api_key_dict=None, response=response)

        assert block.text == "a@b.com"

    @pytest.mark.asyncio
    async def test_anthropic_message_shape_is_restored(self):
        """The /v1/messages reply is a plain dict with no choices.

        Measured against a live provider: without its own branch the reply went
        back to the caller still carrying the placeholder, even though the
        request had been redacted correctly.
        """
        guardrail = _guardrail(event_hook="post_call")
        _mock_post(guardrail, {"texts": ["a@b.com"]})

        response = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "[EMAIL_1]"}],
        }
        result = await guardrail.async_post_call_success_hook(
            data={"messages": []}, user_api_key_dict=None, response=response
        )

        assert result["content"][0]["text"] == "a@b.com"

    @pytest.mark.asyncio
    async def test_anthropic_non_text_blocks_are_left_alone(self):
        guardrail = _guardrail(event_hook="post_call")
        _mock_post(guardrail, {"texts": ["a@b.com"]})

        response = {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "[EMAIL_1]"},
                {"type": "tool_use", "id": "t1", "name": "lookup", "input": {}},
            ],
        }
        result = await guardrail.async_post_call_success_hook(
            data={"messages": []}, user_api_key_dict=None, response=response
        )

        assert result["content"][0]["text"] == "a@b.com"
        assert result["content"][1] == {"type": "tool_use", "id": "t1", "name": "lookup", "input": {}}


class TestVaultIsolation:
    """The vault id must never be something a caller can choose.

    The vault holds the plaintext behind every placeholder. If a caller could name
    the vault, they could send a placeholder, have the model echo it back, and get
    another caller's value restored into their own reply.
    """

    @pytest.mark.asyncio
    async def test_caller_supplied_session_id_is_not_used(self):
        guardrail = _guardrail()
        mock = _mock_post(guardrail, {"texts": ["[EMAIL_1]"]})

        data = {
            "messages": [{"role": "user", "content": "a@b.com"}],
            "metadata": {"llm_shield_session_id": "victim-session"},
            "litellm_session_id": "victim-session",
        }
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

        used = mock.call_args_list[0].kwargs["headers"]["X-Session-ID"]
        assert used != "victim-session"
        assert data["metadata"]["llm_shield_session_id"] == used

    @pytest.mark.asyncio
    async def test_restore_ignores_a_foreign_session_id(self):
        """A reply is left unrestored rather than resolved against another vault."""
        guardrail = _guardrail(event_hook="post_call")
        mock = _mock_post(guardrail, {"texts": ["[EMAIL_1]"]})

        data = {"metadata": {"llm_shield_session_id": "victim-session"}}
        response = ModelResponse(choices=[Choices(index=0, message=Message(role="assistant", content="[EMAIL_1]"))])
        await guardrail.async_post_call_success_hook(data=data, user_api_key_dict=None, response=response)

        assert mock.call_args_list[0].kwargs["headers"]["X-Session-ID"] != "victim-session"

    @pytest.mark.asyncio
    async def test_each_request_gets_its_own_vault(self):
        guardrail = _guardrail()
        mock = _mock_post(guardrail, {"texts": ["[EMAIL_1]"]}, {"texts": ["[EMAIL_1]"]})

        for _ in range(2):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=None,
                cache=None,
                data={"messages": [{"role": "user", "content": "a@b.com"}]},
                call_type="completion",
            )

        seen = {call.kwargs["headers"]["X-Session-ID"] for call in mock.call_args_list}
        assert len(seen) == 2


class TestFailClosed:
    @pytest.mark.asyncio
    async def test_unreachable_shield_blocks_the_request(self):
        """Failing open would send the PII upstream, defeating the guardrail."""
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(side_effect=ConnectionError("refused"))

        with pytest.raises(GuardrailRaisedException):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=None,
                cache=None,
                data={"messages": [{"role": "user", "content": "a@b.com"}]},
                call_type="completion",
            )

    @pytest.mark.asyncio
    async def test_error_status_blocks_the_request(self):
        guardrail = _guardrail()
        guardrail.async_handler.post = AsyncMock(return_value=_response({"error": "nope"}, status_code=500))

        with pytest.raises(GuardrailRaisedException):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=None,
                cache=None,
                data={"messages": [{"role": "user", "content": "a@b.com"}]},
                call_type="completion",
            )

    @pytest.mark.asyncio
    async def test_short_payload_blocks_the_request(self):
        """A response that loses an entry would silently misalign the write-back."""
        guardrail = _guardrail()
        _mock_post(guardrail, {"texts": []})

        with pytest.raises(GuardrailRaisedException):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=None,
                cache=None,
                data={"messages": [{"role": "user", "content": "a@b.com"}]},
                call_type="completion",
            )


class TestStreamingRehydration:
    @pytest.mark.asyncio
    async def test_split_placeholder_is_not_emitted_in_fragments(self):
        """The window holds back a partial placeholder and releases it once complete."""
        guardrail = _guardrail(event_hook="post_call")
        # Shield holds "[EMAIL" back, then releases the restored value.
        _mock_post(
            guardrail,
            {"text": "Email ", "carry": "[EMAIL"},
            {"text": "a@b.com about it", "carry": ""},
        )

        async def stream():
            yield _chunk("Email [EMAIL")
            yield _chunk("_1] about it", finish_reason="stop")

        chunks = await _drain(
            guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=None, response=stream(), request_data={"messages": []}
            )
        )

        emitted = [c.choices[0].delta.content for c in chunks]
        assert emitted == ["Email ", "a@b.com about it"]
        # No fragment of the placeholder ever reached the client.
        assert not any("[EMAIL" in (text or "") for text in emitted)

    @pytest.mark.asyncio
    async def test_carry_is_returned_to_the_next_call(self):
        guardrail = _guardrail(event_hook="post_call")
        mock = _mock_post(
            guardrail,
            {"text": "", "carry": "hold"},
            {"text": "held-and-more", "carry": ""},
        )

        async def stream():
            yield _chunk("hold")
            yield _chunk("-and-more", finish_reason="stop")

        await _drain(
            guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=None, response=stream(), request_data={"messages": []}
            )
        )

        assert mock.call_args_list[0].kwargs["json"]["carry"] == ""
        assert mock.call_args_list[1].kwargs["json"]["carry"] == "hold"
        assert mock.call_args_list[1].kwargs["json"]["final"] is True

    @pytest.mark.asyncio
    async def test_chunks_are_forwarded_as_they_arrive(self):
        """Restoration must not buffer the stream into a single terminal chunk."""
        guardrail = _guardrail(event_hook="post_call")
        _mock_post(
            guardrail,
            {"text": "one ", "carry": ""},
            {"text": "two ", "carry": ""},
            {"text": "three", "carry": ""},
        )

        async def stream():
            yield _chunk("one ")
            yield _chunk("two ")
            yield _chunk("three", finish_reason="stop")

        chunks = await _drain(
            guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=None, response=stream(), request_data={"messages": []}
            )
        )

        assert len(chunks) == 3
        assert [c.choices[0].delta.content for c in chunks] == ["one ", "two ", "three"]

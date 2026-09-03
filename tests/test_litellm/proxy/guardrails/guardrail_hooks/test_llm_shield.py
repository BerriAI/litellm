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
    async def test_request_without_messages_is_untouched(self):
        guardrail = _guardrail()
        mock = _mock_post(guardrail)
        data = {"input": "no messages here"}

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

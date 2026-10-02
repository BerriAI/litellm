import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import Request, Response

import litellm
from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.llm_shield_proxy.llm_shield_proxy import (
    GUARDRAIL_NAME,
    LLMShieldProxyGuardrail,
)
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.llms.openai import (
    FunctionCallArgumentsDeltaEvent,
    OutputTextDeltaEvent,
    OutputTextDoneEvent,
    ResponsesAPIStreamEvents,
)
from litellm.types.utils import Choices, Delta, Message, ModelResponse, ModelResponseStream, StreamingChoices


def _guardrail(**overrides: object) -> LLMShieldProxyGuardrail:
    params: dict[str, object] = {
        "api_key": "test-key",
        "api_base": "http://shield.test",
        "guardrail_name": GUARDRAIL_NAME,
        "event_hook": "pre_call",
        "default_on": True,
    }
    params.update(overrides)
    return LLMShieldProxyGuardrail(**params)


def _response(payload: dict, status_code: int = 200) -> Response:
    return Response(
        status_code=status_code,
        json=payload,
        request=Request("POST", "http://shield.test/v1/guard/redact"),
    )


def _mock_post(guardrail: LLMShieldProxyGuardrail, *payloads: dict) -> AsyncMock:
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


def _tool_chunk(arguments: str, finish_reason: str | None = None) -> ModelResponseStream:
    """One streamed fragment of tool call 0's arguments."""
    tool_call = {"index": 0, "id": "call_1", "type": "function", "function": {"name": "send", "arguments": arguments}}
    return ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(tool_calls=[tool_call]), finish_reason=finish_reason)]
    )


def _field(holder: object, name: str) -> object:
    """Reads a field from a dict or a model; the guardrail emits both shapes."""
    return holder.get(name) if isinstance(holder, dict) else getattr(holder, name)


class _FakeShield:
    """The three guard endpoints over one fixed vault, placeholder -> original.

    The stream endpoint holds back a trailing `[` that has not closed yet, which is the
    behaviour that makes a placeholder split across two chunks come out whole.
    """

    def __init__(self, vault: dict[str, str]) -> None:
        self.vault = vault
        self.urls: list[str] = []

    def _restore(self, text: str) -> str:
        for placeholder, original in self.vault.items():
            text = text.replace(placeholder, original)
        return text

    async def post(self, url: str, headers: dict, json: dict, timeout: float) -> Response:
        self.urls.append(url)
        if url.endswith("/rehydrate/stream"):
            text = self._restore(json["carry"] + json["text"])
            opening = text.rfind("[")
            if json["final"] or opening == -1 or "]" in text[opening:]:
                return _response({"text": text, "carry": ""})
            return _response({"text": text[:opening], "carry": text[opening:]})
        return _response({"texts": [self._restore(text) for text in json["texts"]]})


def _shielded(vault: dict[str, str]) -> tuple[LLMShieldProxyGuardrail, _FakeShield]:
    guardrail = _guardrail(event_hook="post_call")
    shield = _FakeShield(vault)
    guardrail.async_handler.post = shield.post  # type: ignore[method-assign]
    return guardrail, shield


def _sse(event: dict) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def _sse_events(frames: list) -> list[dict]:
    """Parses emitted SSE output, whatever its chunking, back into event payloads."""
    raw = b"".join(frame.encode() if isinstance(frame, str) else frame for frame in frames).decode()
    return [
        json.loads(line[len("data:") :])
        for event in raw.split("\n\n")
        for line in event.split("\n")
        if line.startswith("data:")
    ]


def _text_block_stream(*deltas: str) -> list[bytes]:
    """An Anthropic /v1/messages stream with one text block made of `deltas`."""
    return [
        _sse({"type": "message_start", "message": {"id": "msg_1", "role": "assistant", "content": []}}),
        _sse({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        *(
            _sse({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": d}})
            for d in deltas
        ),
        _sse({"type": "content_block_stop", "index": 0}),
        _sse({"type": "message_stop"}),
    ]


async def _restore_stream(guardrail: LLMShieldProxyGuardrail, chunks: list) -> list:
    async def stream():
        for chunk in chunks:
            yield chunk

    return await _drain(
        guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=None, response=stream(), request_data={"messages": []}
        )
    )


def test_llm_shield_guardrail_config(monkeypatch: pytest.MonkeyPatch):
    """Should register through init_guardrails_v2 like any other provider."""
    monkeypatch.setattr(litellm, "guardrail_name_config_map", {})
    monkeypatch.setenv("LLM_SHIELD_PROXY_API_KEY", "test-key")

    init_guardrails_v2(
        all_guardrails=[
            {
                "guardrail_name": "llm_shield_proxy",
                "litellm_params": {"guardrail": "llm_shield_proxy", "mode": "pre_call", "default_on": True},
            }
        ],
        config_file_path="",
    )

    registered = [cb for cb in litellm.callbacks if isinstance(cb, LLMShieldProxyGuardrail)]
    assert len(registered) == 1
    assert registered[0].guardrail_name == "llm_shield_proxy"


class TestLLMShieldProxyInitialization:
    def test_api_base_defaults_to_localhost(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LLM_SHIELD_PROXY_API_BASE", raising=False)
        assert _guardrail(api_base=None).api_base == "http://localhost:8000"

    def test_api_base_reads_environment(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LLM_SHIELD_PROXY_API_BASE", "http://shield.internal:9000")
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
        """No text to redact means no call to LLM Shield Proxy.

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
    async def test_completions_prompt_is_redacted(self):
        """/v1/completions puts its text in a top-level `prompt`, not in messages."""
        guardrail = _guardrail()
        mock = _mock_post(guardrail, {"texts": ["Email [EMAIL_1]"]})

        data = {"prompt": "Email jane.doe@example.com"}
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="atext_completion")

        assert mock.call_args_list[0].kwargs["json"]["texts"] == ["Email jane.doe@example.com"]
        assert data["prompt"] == "Email [EMAIL_1]"

    @pytest.mark.asyncio
    async def test_completions_prompt_array_is_redacted(self):
        """`prompt` also accepts an array, and each entry is provider-bound."""
        guardrail = _guardrail()
        _mock_post(guardrail, {"texts": ["[EMAIL_1]", "[PHONE_1]"]})

        data = {"prompt": ["jane.doe@example.com", "555-0100"]}
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="atext_completion")

        assert data["prompt"] == ["[EMAIL_1]", "[PHONE_1]"]

    @pytest.mark.asyncio
    async def test_responses_function_call_items_are_redacted(self):
        """Responses input items hold tool data in `arguments` and `output`."""
        guardrail = _guardrail()
        _mock_post(guardrail, {"texts": ['{"email": "[EMAIL_1]"}', "sent to [EMAIL_1]"]})

        data = {
            "input": [
                {"type": "function_call", "name": "send", "arguments": '{"email": "jane.doe@example.com"}'},
                {"type": "function_call_output", "call_id": "c1", "output": "sent to jane.doe@example.com"},
            ]
        }
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="aresponses")

        assert data["input"][0]["arguments"] == '{"email": "[EMAIL_1]"}'
        assert data["input"][1]["output"] == "sent to [EMAIL_1]"

    @pytest.mark.asyncio
    async def test_anthropic_system_prompt_is_redacted(self):
        """/v1/messages carries its system prompt at the top level, not in messages."""
        guardrail = _guardrail()
        mock = _mock_post(guardrail, {"texts": ["the user is [EMAIL_1]"]})

        data = {"system": "the user is jane.doe@example.com", "messages": []}
        await guardrail.async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="anthropic_messages"
        )

        assert mock.call_args_list[0].kwargs["json"]["texts"] == ["the user is jane.doe@example.com"]
        assert data["system"] == "the user is [EMAIL_1]"

    @pytest.mark.asyncio
    async def test_anthropic_system_blocks_are_redacted(self):
        """`system` also accepts a list of text blocks."""
        guardrail = _guardrail()
        _mock_post(guardrail, {"texts": ["[EMAIL_1]"]})

        data = {"system": [{"type": "text", "text": "jane.doe@example.com"}], "messages": []}
        await guardrail.async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="anthropic_messages"
        )

        assert data["system"][0]["text"] == "[EMAIL_1]"

    @pytest.mark.asyncio
    async def test_string_array_input_is_redacted(self):
        """Embeddings and moderations send `input` as an array of bare strings."""
        guardrail = _guardrail()
        _mock_post(guardrail, {"texts": ["[EMAIL_1]", "[PHONE_1]"]})

        data = {"input": ["jane.doe@example.com", "555-0100"]}
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="aembedding")

        assert data["input"] == ["[EMAIL_1]", "[PHONE_1]"]

    @pytest.mark.asyncio
    async def test_participant_name_is_redacted(self):
        """`name` on a user turn identifies a person."""
        guardrail = _guardrail()
        mock = _mock_post(guardrail, {"texts": ["hi", "[PERSON_1]"]})

        data = {"messages": [{"role": "user", "name": "Jane Doe", "content": "hi"}]}
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

        assert mock.call_args_list[0].kwargs["json"]["texts"] == ["hi", "Jane Doe"]
        assert data["messages"][0]["name"] == "[PERSON_1]"

    @pytest.mark.asyncio
    async def test_tool_function_name_is_left_alone(self):
        """On a tool turn the same field is the function name.

        Redacting it would stop the call routing, so this asserts it is never sent
        to the shield at all.
        """
        guardrail = _guardrail()
        mock = _mock_post(guardrail, {"texts": ["result"]})

        data = {"messages": [{"role": "tool", "name": "get_weather", "content": "result"}]}
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

        assert data["messages"][0]["name"] == "get_weather"
        assert mock.call_args_list[0].kwargs["json"]["texts"] == ["result"]

    @pytest.mark.asyncio
    async def test_anthropic_tool_result_content_is_redacted(self):
        """A tool_result nests its own content, as a string or as more blocks."""
        guardrail = _guardrail()
        _mock_post(guardrail, {"texts": ["[EMAIL_1]", "[EMAIL_2]"]})

        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "found jane.doe@example.com"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "t2",
                            "content": [{"type": "text", "text": "also bob@example.com"}],
                        },
                    ],
                }
            ]
        }
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

        assert data["messages"][0]["content"][0]["content"] == "[EMAIL_1]"
        assert data["messages"][0]["content"][1]["content"][0]["text"] == "[EMAIL_2]"

    @pytest.mark.asyncio
    async def test_nesting_past_the_bound_blocks_the_request(self):
        """Nesting is caller controlled, so the descent has to stop somewhere -- and where
        it stops, the request must not go out.

        This test used to assert the opposite: that text past the bound was skipped. That
        sent `past-the-bound@example.com` to the provider unredacted while the guardrail
        reported as enabled.
        """
        guardrail = _guardrail()
        mock = _mock_post(guardrail)

        deep: dict = {"type": "tool_result", "content": "past-the-bound@example.com"}
        for _ in range(200):
            deep = {"type": "tool_result", "content": [deep]}
        data = {"messages": [{"role": "user", "content": [{"type": "text", "text": "shallow"}, deep]}]}

        with pytest.raises(GuardrailRaisedException):
            await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")
        mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_deep_tool_input_blocks_the_request(self):
        """A tool_use input past the JSON bound must not be forwarded half-redacted."""
        guardrail = _guardrail()
        mock = _mock_post(guardrail)

        deep: dict = {"email": "past-the-bound@example.com"}
        for _ in range(100):
            deep = {"next": deep}
        block = {"type": "tool_use", "id": "t1", "name": "f", "input": deep}
        data = {"messages": [{"role": "assistant", "content": [block]}]}

        with pytest.raises(GuardrailRaisedException):
            await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")
        mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_realistic_nesting_is_redacted_in_full(self):
        """The bounds are far past real payloads: a tool input nested inside a tool result,
        several JSON levels deep, is redacted whole rather than refused."""
        guardrail = _guardrail()
        _mock_post(guardrail, {"texts": ["[EMAIL_1]"]})

        tool_use = {
            "type": "tool_use",
            "id": "t1",
            "name": "f",
            "input": {"a": {"b": {"c": {"d": {"to": "x@example.com"}}}}},
        }
        data = {"messages": [{"role": "user", "content": [{"type": "tool_result", "content": [tool_use]}]}]}
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

        assert tool_use["input"]["a"]["b"]["c"]["d"]["to"] == "[EMAIL_1]"

    @pytest.mark.asyncio
    async def test_responses_prompt_object_variables_are_redacted(self):
        """A PromptObject's variables are substituted into the prompt provider side.

        The id and version pick which stored prompt to run and have to arrive
        unchanged; the variables are caller text.
        """
        guardrail = _guardrail()
        _mock_post(guardrail, {"texts": ["[EMAIL_1]"]})

        data = {"prompt": {"id": "pmpt_123", "version": "2", "variables": {"customer": "jane.doe@example.com"}}}
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="aresponses")

        assert data["prompt"]["variables"]["customer"] == "[EMAIL_1]"
        assert data["prompt"]["id"] == "pmpt_123"
        assert data["prompt"]["version"] == "2"

    @pytest.mark.asyncio
    async def test_completions_suffix_is_redacted(self):
        """LiteLLM forwards the legacy `suffix` to providers that support it."""
        guardrail = _guardrail()
        _mock_post(guardrail, {"texts": ["signed [EMAIL_1]", "write to [EMAIL_1]"]})

        data = {"prompt": "write to jane.doe@example.com", "suffix": "signed jane.doe@example.com"}
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="atext_completion")

        assert data["suffix"] == "signed [EMAIL_1]"

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

    @pytest.mark.asyncio
    async def test_anthropic_tool_use_input_is_redacted(self):
        """A replayed tool_use block carries its arguments as a JSON object, not a string."""
        guardrail = _guardrail()
        mock = _mock_post(guardrail, {"texts": ["[EMAIL_1]", "[PHONE_1]"]})

        data = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "send",
                            "input": {"to": "jane.doe@example.com", "meta": {"phone": "555-0100"}},
                        }
                    ],
                }
            ]
        }
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

        assert mock.call_args_list[0].kwargs["json"]["texts"] == ["jane.doe@example.com", "555-0100"]
        block = data["messages"][0]["content"][0]
        assert block["input"] == {"to": "[EMAIL_1]", "meta": {"phone": "[PHONE_1]"}}
        assert block["name"] == "send", "the tool name has to arrive unchanged for the call to route"

    @pytest.mark.asyncio
    async def test_responses_reasoning_summary_is_redacted(self):
        """A replayed reasoning item quotes the conversation in its summary parts."""
        guardrail = _guardrail()
        _mock_post(guardrail, {"texts": ["user asked about [EMAIL_1]"]})

        data = {
            "input": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "user asked about jane.doe@example.com"}],
                }
            ]
        }
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="aresponses")

        assert data["input"][0]["summary"][0]["text"] == "user asked about [EMAIL_1]"

    def test_tool_schemas_give_up_their_free_text_and_nothing_else(self):
        """Every string is collected except what must reach the model verbatim: names,
        types, formats, patterns and required lists."""
        data = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "top",
                        "parameters": {
                            "type": "object",
                            "title": "title",
                            "properties": {
                                # A property that is itself named "description".
                                "description": {"type": "string", "description": "named"},
                                "kind": {"type": "string", "enum": ["a", "b"], "const": "a", "description": "enum"},
                                "deep": {"type": "array", "items": {"type": "object", "description": "nested"}},
                                "to": {
                                    "type": "string",
                                    "format": "email",
                                    "pattern": "^.+@.+$",
                                    "examples": ["example"],
                                    "default": "default",
                                },
                                "choice": {"anyOf": [{"type": "object", "default": {"type": "object-default"}}]},
                                # Property names that collide with keywords are subschemas all the same.
                                "type": {"type": "string", "description": "named-type"},
                            },
                            "required": ["to"],
                            "$defs": {"shared": {"description": "defined"}},
                            # Keywords nobody listed: scanned by default.
                            "dependencies": {"mode": {"description": "dependent"}},
                            "$comment": "comment",
                            "x-note": "vendor",
                        },
                    },
                }
            ]
        }
        caller, privileged = LLMShieldProxyGuardrail._locate_request_texts(data)

        assert sorted(text for text, _ in caller) == ["a", "a", "b"], "enum and const go to the caller vault"
        assert sorted(text for text, _ in privileged) == [
            "comment",
            "default",
            "defined",
            "dependent",
            "enum",
            "example",
            "named",
            "named-type",
            "nested",
            "object-default",
            "title",
            "top",
            "vendor",
        ]

    @pytest.mark.asyncio
    async def test_enum_values_are_redacted_and_restored_in_the_tool_call(self):
        """An enum value holding PII is redacted, and the model's use of the stand-in is
        restored in its tool arguments, so the call still carries a value the schema allows."""
        guardrail = _guardrail(event_hook=["pre_call", "post_call"])
        shield = _FakeShield({"[EMAIL_1]": "ops@example.com"})
        redact_mock = _mock_post(guardrail, {"texts": ["[EMAIL_1]"]})
        data = {
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "notify",
                        "parameters": {"properties": {"to": {"type": "string", "enum": ["ops@example.com"]}}},
                    },
                }
            ],
        }

        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")
        assert redact_mock.call_args_list[0].kwargs["json"]["texts"] == ["ops@example.com"]
        assert data["tools"][0]["function"]["parameters"]["properties"]["to"]["enum"] == ["[EMAIL_1]"]

        guardrail.async_handler.post = shield.post  # type: ignore[method-assign]
        call = SimpleNamespace(function=SimpleNamespace(name="notify", arguments='{"to": "[EMAIL_1]"}'))
        reply = ModelResponse(choices=[Choices(message=Message(content=None, tool_calls=None))])
        reply.choices[0].message.tool_calls = [call]
        await guardrail.async_post_call_success_hook(data=data, user_api_key_dict=None, response=reply)

        assert json.loads(call.function.arguments) == {"to": "ops@example.com"}

    def test_schema_nesting_past_the_bound_is_refused(self):
        schema: dict = {"type": "object", "description": "past-the-bound@example.com"}
        for _ in range(100):
            schema = {"type": "object", "properties": {"next": schema}}
        data = {"tools": [{"type": "function", "function": {"name": "f", "parameters": schema}}]}

        with pytest.raises(Exception, match="schema"):
            LLMShieldProxyGuardrail._locate_request_texts(data)


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
        assert data["litellm_metadata"]["llm_shield_session_id"] == used

    @pytest.mark.asyncio
    async def test_session_id_is_not_forwarded_to_the_provider(self):
        """The vault id is a capability, so it must stay out of provider-visible metadata.

        `metadata` is forwarded upstream on /v1/responses; `litellm_metadata` is not. A
        provider holding both the placeholders and the session id could call the shield's
        rehydrate endpoint and read back exactly what this guardrail withholds.
        """
        guardrail = _guardrail()
        mock = _mock_post(guardrail, {"texts": ["[EMAIL_1]"]})

        data = {"messages": [{"role": "user", "content": "a@b.com"}], "metadata": {}}
        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

        used = mock.call_args_list[0].kwargs["headers"]["X-Session-ID"]
        assert "llm_shield_session_id" not in data["metadata"]
        assert data["litellm_metadata"]["llm_shield_session_id"] == used

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


    @pytest.mark.parametrize(
        "data",
        [
            pytest.param(
                {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]},
                id="system-turn",
            ),
            pytest.param(
                {"messages": [{"role": "developer", "content": "S"}, {"role": "user", "content": "U"}]},
                id="developer-turn",
            ),
            pytest.param(
                {"system": "S", "messages": [{"role": "user", "content": "U"}]},
                id="anthropic-top-level-system",
            ),
            pytest.param({"instructions": "S", "input": "U"}, id="responses-instructions"),
            pytest.param(
                {
                    "messages": [{"role": "user", "content": "U"}],
                    "tools": [{"type": "function", "function": {"name": "f", "description": "S"}}],
                },
                id="chat-tool-description",
            ),
            pytest.param(
                {"input": "U", "tools": [{"type": "function", "name": "f", "description": "S"}]},
                id="responses-tool-description",
            ),
            pytest.param(
                {"messages": [{"role": "user", "content": "U"}], "functions": [{"name": "f", "description": "S"}]},
                id="legacy-function-description",
            ),
            pytest.param(
                {
                    "messages": [{"role": "user", "content": "U"}],
                    "tools": [{"name": "f", "input_schema": {"properties": {"to": {"description": "S"}}}}],
                },
                id="anthropic-schema-description",
            ),
            pytest.param(
                {
                    "messages": [{"role": "user", "content": "U"}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "n", "description": "S", "schema": {"type": "object"}},
                    },
                },
                id="chat-response-format",
            ),
            pytest.param(
                {
                    "input": "U",
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "n",
                            "schema": {"properties": {"a": {"description": "S"}}},
                        }
                    },
                },
                id="responses-text-format",
            ),
            pytest.param({"prediction": {"type": "content", "content": "U"}, "instructions": "S"}, id="prediction"),
            pytest.param(
                {"prediction": {"type": "content", "content": [{"type": "text", "text": "U"}]}, "instructions": "S"},
                id="prediction-parts",
            ),
            pytest.param(
                {
                    "messages": [{"role": "user", "content": "U"}],
                    "web_search_options": {"user_location": {"type": "approximate", "approximate": {"city": "S"}}},
                },
                id="chat-web-search-location",
            ),
            pytest.param(
                {
                    "input": "U",
                    "tools": [{"type": "web_search", "user_location": {"type": "approximate", "region": "S"}}],
                },
                id="responses-web-search-location",
            ),
            pytest.param({"messages": [{"role": "user", "content": "U"}], "user": "S"}, id="end-user-id"),
            pytest.param({"input": "U", "safety_identifier": "S"}, id="safety-identifier"),
        ],
    )
    def test_server_authored_text_is_split_from_the_callers(self, data: dict) -> None:
        """Every request shape must sort its server-authored spans out of the caller's."""
        caller, privileged = LLMShieldProxyGuardrail._locate_request_texts(data)

        assert [text for text, _ in caller] == ["U"]
        assert [text for text, _ in privileged] == ["S"]

    @pytest.mark.asyncio
    async def test_a_system_prompt_gets_a_vault_of_its_own(self) -> None:
        """The reply is restored against the caller's vault, so the two cannot be one."""
        guardrail = _guardrail()
        mock = _mock_post(guardrail, {"texts": ["[EMAIL_1]"]}, {"texts": ["[EMAIL_2]"]})

        data = {
            "messages": [
                {"role": "system", "content": "escalate to admin@corp.internal"},
                {"role": "user", "content": "email a@b.com"},
            ]
        }
        await guardrail.async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="completion"
        )

        privileged_id, caller_id = (
            call.kwargs["headers"]["X-Session-ID"] for call in mock.call_args_list
        )
        assert privileged_id != caller_id
        assert guardrail._session_id(data) == caller_id

    @pytest.mark.asyncio
    async def test_the_system_prompt_vault_id_is_never_stored(self) -> None:
        """Nothing can restore against the system vault later, because its id is not kept.

        This is what stops a caller from having the model echo a placeholder out of a
        system prompt they cannot see and receiving the plaintext behind it.
        """
        guardrail = _guardrail()
        mock = _mock_post(guardrail, {"texts": ["[EMAIL_1]"]}, {"texts": ["[EMAIL_2]"]})

        data = {
            "messages": [
                {"role": "system", "content": "escalate to admin@corp.internal"},
                {"role": "user", "content": "email a@b.com"},
            ]
        }
        await guardrail.async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="completion"
        )

        privileged_id = mock.call_args_list[0].kwargs["headers"]["X-Session-ID"]
        assert privileged_id not in json.dumps(data, default=str)


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
    async def test_every_choice_is_restored(self):
        """With n>1 a later choice must not be handed back still holding a placeholder."""
        guardrail = _guardrail(event_hook="post_call")
        _mock_post(
            guardrail,
            {"text": "first@example.com", "carry": ""},
            {"text": "second@example.com", "carry": ""},
        )

        async def stream():
            yield ModelResponseStream(
                choices=[
                    StreamingChoices(index=0, delta=Delta(content="[EMAIL_1]"), finish_reason="stop"),
                    StreamingChoices(index=1, delta=Delta(content="[EMAIL_2]"), finish_reason="stop"),
                ]
            )

        chunks = await _drain(
            guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=None, response=stream(), request_data={"messages": []}
            )
        )

        restored = [choice.delta.content for choice in chunks[0].choices]
        assert restored == ["first@example.com", "second@example.com"]

    @pytest.mark.asyncio
    async def test_choice_windows_do_not_cross_contaminate(self):
        """Each choice is its own token stream, so each carries its own window.

        One shared window would send the characters held back for choice 0 up
        against choice 1's next delta and splice the two streams together.
        """
        guardrail = _guardrail(event_hook="post_call")
        mock = _mock_post(
            guardrail,
            {"text": "", "carry": "A-held"},
            {"text": "", "carry": "B-held"},
            {"text": "a-done", "carry": ""},
            {"text": "b-done", "carry": ""},
        )

        async def stream():
            yield ModelResponseStream(
                choices=[
                    StreamingChoices(index=0, delta=Delta(content="a1")),
                    StreamingChoices(index=1, delta=Delta(content="b1")),
                ]
            )
            yield ModelResponseStream(
                choices=[
                    StreamingChoices(index=0, delta=Delta(content="a2"), finish_reason="stop"),
                    StreamingChoices(index=1, delta=Delta(content="b2"), finish_reason="stop"),
                ]
            )

        await _drain(
            guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=None, response=stream(), request_data={"messages": []}
            )
        )

        sent = [call.kwargs["json"] for call in mock.call_args_list]
        assert sent[2]["carry"] == "A-held", "choice 0 must get its own window back"
        assert sent[3]["carry"] == "B-held", "choice 1 must get its own window back"

    @pytest.mark.asyncio
    async def test_a_choice_missing_from_the_last_chunk_still_flushes(self):
        """Held text must not be dropped because its choice ended earlier.

        Choice 1 finishes and stops appearing, then the stream ends without a
        finish_reason for choice 0. Flushing only the terminal chunk's choices would
        discard whatever choice 1 was still holding and truncate its answer.
        """
        guardrail = _guardrail(event_hook="post_call")
        _mock_post(
            guardrail,
            {"text": "", "carry": "held-0"},
            {"text": "", "carry": "held-1"},
            {"text": "zero-done", "carry": ""},
            {"text": "one-done", "carry": ""},
        )

        async def stream():
            yield ModelResponseStream(
                choices=[
                    StreamingChoices(index=0, delta=Delta(content="a")),
                    StreamingChoices(index=1, delta=Delta(content="b")),
                ]
            )
            yield ModelResponseStream(choices=[StreamingChoices(index=0, delta=Delta(content=None))])

        chunks = await _drain(
            guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=None, response=stream(), request_data={"messages": []}
            )
        )

        flushed = {
            choice.index: choice.delta.content for chunk in chunks for choice in chunk.choices if choice.delta.content
        }
        assert flushed.get(1) == "one-done", "choice 1's held text was dropped"
        assert flushed.get(0) == "zero-done"

    @pytest.mark.asyncio
    async def test_held_tool_arguments_land_in_the_finishing_chunk(self):
        """A client parses tool arguments on finish_reason, so the flush must ride that chunk.

        The finishing chunk also carries its own fragment for the same tool call. That
        entry has to survive, with the held text appended after it as a continuation.
        """
        guardrail = _guardrail(event_hook="post_call")
        _mock_post(
            guardrail,
            {"text": '{"to": "', "carry": "[EMA"},
            {"text": "", "carry": '[EMAIL_1]"}'},
            {"text": 'a@example.com"}', "carry": ""},
        )

        async def stream():
            yield _tool_chunk('{"to": "[EMA')
            yield _tool_chunk('IL_1]"}', finish_reason="tool_calls")

        chunks = await _drain(
            guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=None, response=stream(), request_data={"messages": []}
            )
        )

        assert len(chunks) == 2, "the flush must not arrive after the finish_reason chunk"
        final_calls = chunks[1].choices[0].delta.tool_calls
        assert len(final_calls) == 2, "the finishing chunk's own fragment was dropped"
        assert _field(final_calls[1], "index") == 0
        arguments = "".join(
            _field(_field(call, "function"), "arguments") or ""
            for chunk in chunks
            for call in chunk.choices[0].delta.tool_calls
        )
        assert json.loads(arguments) == {"to": "a@example.com"}

    @pytest.mark.asyncio
    async def test_held_tool_arguments_flush_when_the_stream_ends_unfinished(self):
        """No finish_reason at all: a trailing chunk carries the held arguments alone."""
        guardrail = _guardrail(event_hook="post_call")
        _mock_post(
            guardrail,
            {"text": '{"to": "', "carry": "[EMA"},
            {"text": 'a@example.com"}', "carry": ""},
        )

        async def stream():
            yield _tool_chunk('{"to": "[EMA')

        chunks = await _drain(
            guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=None, response=stream(), request_data={"messages": []}
            )
        )

        assert len(chunks) == 2
        trailing = chunks[1].choices[0].delta.tool_calls
        assert trailing == [{"index": 0, "function": {"arguments": 'a@example.com"}'}}], (
            "the copied chunk's own fragment was already delivered and must not repeat"
        )

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


class TestApplyGuardrailToolCalls:
    """The unified entry point the UI's Test button and the translation handlers use."""

    @pytest.mark.asyncio
    async def test_response_tool_call_arguments_are_rehydrated(self):
        """Regression: this path deep-copied tool calls with `copy` never imported.

        47 tests passed with a guaranteed NameError here, because every tool-call test
        covered the request side and this is the only path that reaches the copy.
        """
        guardrail = _guardrail(event_hook="post_call")
        _mock_post(guardrail, {"texts": ["hi", '{"email": "a@b.com"}']})

        data = {"litellm_metadata": {"llm_shield_session_id": "shield-abc"}}
        inputs = {
            "texts": ["hi"],
            "tool_calls": [{"function": {"name": "send", "arguments": '{"email": "[EMAIL_1]"}'}}],
        }

        merged = await guardrail.apply_guardrail(inputs=inputs, request_data=data, input_type="response")

        assert merged["tool_calls"][0]["function"]["arguments"] == '{"email": "a@b.com"}'
        assert inputs["tool_calls"][0]["function"]["arguments"] == '{"email": "[EMAIL_1]"}'


class TestAnthropicStreamRestoration:
    """/v1/messages streams reach the hook as raw SSE frames, with no `choices` to walk."""

    VAULT = {"[EMAIL_1]": "a@example.com"}

    @pytest.mark.asyncio
    async def test_split_placeholder_is_restored_and_never_fragmented(self):
        guardrail, _ = _shielded(self.VAULT)

        out = await _restore_stream(guardrail, _text_block_stream("Mail [EMA", "IL_1] now"))

        deltas = [e["delta"]["text"] for e in _sse_events(out) if e["type"] == "content_block_delta"]
        assert "".join(deltas) == "Mail a@example.com now"
        assert not any("[EMA" in d for d in deltas), "a placeholder fragment reached the client"

    @pytest.mark.asyncio
    async def test_held_text_lands_before_its_block_stops(self):
        """A trailing `[` that never became a placeholder is still part of the answer."""
        guardrail, _ = _shielded(self.VAULT)

        out = await _restore_stream(guardrail, _text_block_stream("Mail [EMAIL_1], x = a["))

        types = [e["type"] for e in _sse_events(out)]
        deltas = [e["delta"]["text"] for e in _sse_events(out) if e["type"] == "content_block_delta"]
        assert "".join(deltas) == "Mail a@example.com, x = a["
        assert types.index("content_block_stop") > max(i for i, t in enumerate(types) if t == "content_block_delta")

    @pytest.mark.asyncio
    async def test_events_split_across_network_chunks_are_restored(self):
        """A chunk can end mid-event; the frame is parsed once it is whole."""
        guardrail, _ = _shielded(self.VAULT)
        raw = b"".join(_text_block_stream("Mail [EMA", "IL_1] now"))

        out = await _restore_stream(guardrail, [raw[i : i + 7] for i in range(0, len(raw), 7)])

        deltas = [e["delta"]["text"] for e in _sse_events(out) if e["type"] == "content_block_delta"]
        assert "".join(deltas) == "Mail a@example.com now"

    @pytest.mark.asyncio
    async def test_str_frames_stay_str(self):
        guardrail, _ = _shielded(self.VAULT)

        out = await _restore_stream(guardrail, [frame.decode() for frame in _text_block_stream("[EMAIL_1]")])

        assert all(isinstance(frame, str) for frame in out)
        assert "a@example.com" in "".join(out)

    @pytest.mark.asyncio
    async def test_tool_input_json_is_restored(self):
        guardrail, _ = _shielded(self.VAULT)
        frames = [
            _sse({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "input": {}}}),
            *(
                _sse(
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "input_json_delta", "partial_json": p},
                    }
                )
                for p in ('{"to": "[EMAI', 'L_1]"}')
            ),
            _sse({"type": "content_block_stop", "index": 1}),
        ]

        out = await _restore_stream(guardrail, frames)

        partial = "".join(e["delta"]["partial_json"] for e in _sse_events(out) if e["type"] == "content_block_delta")
        assert json.loads(partial) == {"to": "a@example.com"}

    @pytest.mark.asyncio
    async def test_signed_thinking_and_foreign_frames_pass_through_byte_for_byte(self):
        """Rewriting a signed thinking block breaks it; other frames are not ours to touch."""
        guardrail, shield = _shielded(self.VAULT)
        thinking = {"type": "thinking_delta", "thinking": "[EMAIL_1]"}
        frames = [
            _sse({"type": "content_block_delta", "index": 0, "delta": thinking}),
            b'data: {"candidates": [{"content": {"parts": [{"text": "[EMAIL_1]"}]}}]}\n\n',
            b"data: not json\n\n",
        ]

        out = await _restore_stream(guardrail, frames)

        assert b"".join(out) == b"".join(frames)
        assert shield.urls == []

    @pytest.mark.asyncio
    async def test_a_raw_stream_that_is_not_sse_is_never_buffered(self):
        """Without event boundaries to wait for, buffering would hold the whole reply."""
        guardrail, _ = _shielded(self.VAULT)
        chunks = [b'[{"candidates": []}', b', {"candidates": []}]']

        out = await _restore_stream(guardrail, chunks)

        assert out == chunks

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cut", [1, 3, 5, 6])
    async def test_a_field_name_split_by_the_first_chunk_still_reads_as_sse(self, cut: int):
        """`b"eve"` then `b"nt: ..."` is still SSE; deciding on the first chunk alone
        would pass the whole stream through with its placeholders."""
        guardrail, _ = _shielded(self.VAULT)
        raw = b"".join(_text_block_stream("Mail [EMAIL_1]"))

        out = await _restore_stream(guardrail, [raw[:cut], raw[cut:]])

        deltas = [e["delta"]["text"] for e in _sse_events(out) if e["type"] == "content_block_delta"]
        assert "".join(deltas) == "Mail a@example.com"


class TestResponsesStreamRestoration:
    """/v1/responses streams are typed events, with no `choices` to walk."""

    VAULT = {"[EMAIL_1]": "a@example.com"}

    @staticmethod
    def _text_delta(delta: str, sequence_number: int, content_index: int = 0) -> OutputTextDeltaEvent:
        return OutputTextDeltaEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
            item_id="msg_1",
            output_index=0,
            content_index=content_index,
            delta=delta,
            sequence_number=sequence_number,
        )

    @pytest.mark.asyncio
    async def test_deltas_and_done_text_are_restored(self):
        guardrail, _ = _shielded(self.VAULT)
        done = OutputTextDoneEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE,
            item_id="msg_1",
            output_index=0,
            content_index=0,
            text="Mail [EMAIL_1] x[",
        )

        out = await _restore_stream(
            guardrail, [self._text_delta("Mail [EMA", 1), self._text_delta("IL_1] x[", 2), done]
        )

        deltas = [e.delta for e in out if isinstance(e, OutputTextDeltaEvent)]
        assert not any("[EMA" in d for d in deltas), "a placeholder fragment reached the client"
        assert "".join(deltas) == "Mail a@example.com x["
        assert out[-1].text == "Mail a@example.com x[", "the done event repeats the full, restored text"
        assert isinstance(out[-2], OutputTextDeltaEvent), "held text must land before the done event"

    @pytest.mark.asyncio
    async def test_function_call_arguments_are_restored(self):
        guardrail, _ = _shielded(self.VAULT)
        events = [
            FunctionCallArgumentsDeltaEvent(
                type=ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA,
                item_id="fc_1",
                output_index=1,
                delta=part,
            )
            for part in ('{"to": "[EMAI', 'L_1]"}')
        ]

        out = await _restore_stream(guardrail, events)

        assert json.loads("".join(e.delta for e in out)) == {"to": "a@example.com"}

    @pytest.mark.asyncio
    async def test_a_truncated_stream_still_flushes(self):
        """No done event at all: whatever the window holds goes out at the end."""
        guardrail, _ = _shielded(self.VAULT)

        out = await _restore_stream(guardrail, [self._text_delta("see [EMAIL_1] a[", 1)])

        assert "".join(e.delta for e in out) == "see a@example.com a["

    @pytest.mark.asyncio
    async def test_completed_response_is_restored(self):
        """The terminal event repeats the whole reply, and clients read it as the answer."""
        guardrail, _ = _shielded(self.VAULT)
        block = {"type": "output_text", "text": "Mail [EMAIL_1]"}
        call = SimpleNamespace(type="function_call", arguments='{"to": "[EMAIL_1]"}')
        completed = SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(output=[SimpleNamespace(content=[block]), call]),
        )

        await _restore_stream(guardrail, [completed])

        assert block["text"] == "Mail a@example.com"
        assert call.arguments == '{"to": "a@example.com"}'

    @pytest.mark.asyncio
    async def test_streams_on_different_parts_do_not_share_a_window(self):
        guardrail, _ = _shielded(self.VAULT)
        events = [
            self._text_delta("one [EMA", 1),
            self._text_delta("two", 2, content_index=1),
            self._text_delta("IL_1]", 3),
        ]

        out = await _restore_stream(guardrail, events)

        by_part: dict[int, str] = {}
        for event in out:
            by_part[event.content_index] = by_part.get(event.content_index, "") + event.delta
        assert by_part == {0: "one a@example.com", 1: "two"}

    @pytest.mark.asyncio
    async def test_reasoning_summary_part_done_is_restored(self):
        """The summary part repeats the whole summary text after its deltas."""
        guardrail, _ = _shielded(self.VAULT)
        part = SimpleNamespace(type="summary_text", text="asked about [EMAIL_1]")
        event = SimpleNamespace(
            type="response.reasoning_summary_part.done", item_id="rs_1", output_index=0, summary_index=0, part=part
        )

        await _restore_stream(guardrail, [event])

        assert part.text == "asked about a@example.com"

    @pytest.mark.asyncio
    async def test_mcp_call_arguments_are_restored(self):
        """A stream family outside the chat-era set: matched by shape, not by name."""
        guardrail, _ = _shielded(self.VAULT)
        deltas = [
            {"type": "response.mcp_call_arguments.delta", "item_id": "mcp_1", "output_index": 0, "delta": d}
            for d in ('{"to": "[EMAI', 'L_1]"}')
        ]
        done = {
            "type": "response.mcp_call_arguments.done",
            "item_id": "mcp_1",
            "output_index": 0,
            "arguments": '{"to": "[EMAIL_1]"}',
        }

        out = await _restore_stream(guardrail, [*deltas, done])

        assert json.loads("".join(e["delta"] for e in out[:-1])) == {"to": "a@example.com"}
        assert json.loads(out[-1]["arguments"]) == {"to": "a@example.com"}
        assert out[-1]["item_id"] == "mcp_1", "identifiers are not text and stay as sent"

    @pytest.mark.asyncio
    async def test_audio_deltas_are_not_sent_to_the_shield(self):
        """Audio arrives base64-encoded; restoring it would cost a round trip for nothing."""
        guardrail, shield = _shielded(self.VAULT)
        audio = {"type": "response.audio.delta", "item_id": "a_1", "output_index": 0, "delta": "UklGRiQAAABXQVZF"}

        out = await _restore_stream(guardrail, [audio])

        assert out == [audio]
        assert shield.urls == []

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
    async def test_deeply_nested_tool_results_are_bounded(self):
        """Nesting is caller controlled, so the descent has to stop somewhere.

        The walk must terminate on a payload built to be pathological, rather than
        following it as far as it goes.
        """
        guardrail = _guardrail()

        captured: list = []

        async def echo(url, headers, json, timeout):  # noqa: ARG001
            captured.append(json["texts"])
            return _response({"texts": list(json["texts"])})

        guardrail.async_handler.post = AsyncMock(side_effect=echo)  # type: ignore[method-assign]

        deep: dict = {"type": "tool_result", "content": "past-the-bound@example.com"}
        for _ in range(200):
            deep = {"type": "tool_result", "content": [deep]}
        data = {"messages": [{"role": "user", "content": [{"type": "text", "text": "shallow"}, deep]}]}

        await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

        sent = captured[0]
        assert "shallow" in sent
        assert "past-the-bound@example.com" not in sent, "the walk followed the chain past its bound"
        assert len(sent) < 200

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

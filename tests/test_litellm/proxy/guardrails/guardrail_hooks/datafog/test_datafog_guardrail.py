"""
Tests for the DataFog PII Guardrail.

All PII values below are synthetic test fixtures (documentation-reserved
domains, test card numbers, invalid SSN ranges).
"""

import os
import re
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath("../../"))  # Adds the parent directory to the system path

from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.datafog.datafog import DataFogGuardrail

EMAIL = "jane.doe@example.com"
CARD = "4242 4242 4242 4242"
SSN = "856-45-6789"
DE_TAX_ID = "12345678901"


def _fake_datafog_module() -> ModuleType:
    fake_datafog = ModuleType("datafog")
    patterns = {
        "EMAIL": re.compile(r"\bjane\.doe@example\.com\b"),
        "PHONE": re.compile(r"\b555-0100\b"),
        "CREDIT_CARD": re.compile(r"\b4242 4242 4242 4242\b"),
        "SSN": re.compile(r"\b856-45-6789\b"),
        "IP_ADDRESS": re.compile(r"\b192\.168\.1\.1\b"),
        "DE_TAX_ID": re.compile(r"\b12345678901\b"),
    }

    def redact(text: str, engine: str, entity_types: list[str], locales: list[str] | None):
        matches = []
        for entity_type in entity_types:
            if entity_type.startswith("DE_") and (not locales or "de" not in locales):
                continue
            pattern = patterns.get(entity_type)
            if pattern is None:
                continue
            for match in pattern.finditer(text):
                matches.append((match.start(), match.end(), entity_type))
        matches.sort(key=lambda item: item[0])

        redacted_parts = []
        entities = []
        cursor = 0
        counts: dict[str, int] = {}
        for start, end, entity_type in matches:
            if start < cursor:
                continue
            counts[entity_type] = counts.get(entity_type, 0) + 1
            redacted_parts.append(text[cursor:start])
            redacted_parts.append(f"[{entity_type}_{counts[entity_type]}]")
            entities.append(SimpleNamespace(type=entity_type))
            cursor = end
        redacted_parts.append(text[cursor:])
        return SimpleNamespace(redacted_text="".join(redacted_parts), entities=entities)

    fake_datafog.redact = redact
    return fake_datafog


@pytest.fixture(autouse=True)
def fake_datafog(monkeypatch):
    monkeypatch.setitem(sys.modules, "datafog", _fake_datafog_module())


def _chat_data(content) -> dict:
    return {"messages": [{"role": "user", "content": content}]}


def _model_response(text: str):
    import litellm

    resp = litellm.ModelResponse()
    resp.choices[0].message.content = text
    return resp


async def _async_iter(items):
    for item in items:
        yield item


async def _collect_async(async_iterable):
    return [item async for item in async_iterable]


@pytest.mark.asyncio
async def test_pre_call_redacts_email():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=_chat_data(f"email the report to {EMAIL} please"),
        call_type="completion",
    )
    content = data["messages"][0]["content"]
    assert EMAIL not in content
    assert "[EMAIL_1]" in content


@pytest.mark.asyncio
async def test_pre_call_clean_message_unchanged():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=_chat_data("summarize this design doc"),
        call_type="completion",
    )
    assert data["messages"][0]["content"] == "summarize this design doc"


@pytest.mark.asyncio
async def test_pre_call_redacts_content_parts_and_skips_images():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=_chat_data(
            [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
                {"type": "text", "text": f"ssn is {SSN}"},
            ]
        ),
        call_type="completion",
    )
    parts = data["messages"][0]["content"]
    assert parts[0]["type"] == "image_url"
    assert SSN not in parts[1]["text"]


@pytest.mark.asyncio
async def test_pre_call_redacts_tool_and_function_call_arguments():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    original = {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": "legacy_lookup",
                    "arguments": f'{{"ssn": "{SSN}"}}',
                },
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "send_email",
                            "arguments": f'{{"recipient": "{EMAIL}"}}',
                        },
                    }
                ],
            }
        ]
    }
    data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=original,
        call_type="completion",
    )

    message = data["messages"][0]
    assert message["content"] is None
    assert SSN not in message["function_call"]["arguments"]
    assert EMAIL not in message["tool_calls"][0]["function"]["arguments"]
    assert SSN in original["messages"][0]["function_call"]["arguments"]
    assert EMAIL in original["messages"][0]["tool_calls"][0]["function"]["arguments"]


@pytest.mark.asyncio
async def test_pre_call_redacts_tool_and_function_definitions():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    original = {
        "messages": [{"role": "user", "content": "look up the customer"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_customer",
                    "description": f"Send the result to {EMAIL}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "card": {
                                "type": "string",
                                "description": f"Customer card number, such as {CARD}",
                            }
                        },
                    },
                },
            },
            {
                "name": "anthropic_lookup",
                "description": f"Look up customer {SSN}",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string", "examples": [EMAIL]},
                    },
                },
            },
        ],
        "functions": [
            {
                "name": "legacy_lookup",
                "description": f"Look up card {CARD}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ssn": {"type": "string", "default": SSN},
                    },
                },
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "customer_record",
                "schema": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string", "description": f"For example, {EMAIL}"},
                    },
                },
            },
        },
    }

    data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=original,
        call_type="completion",
    )

    assert EMAIL not in data["tools"][0]["function"]["description"]
    assert CARD not in data["tools"][0]["function"]["parameters"]["properties"]["card"]["description"]
    assert SSN not in data["tools"][1]["description"]
    assert EMAIL not in data["tools"][1]["input_schema"]["properties"]["email"]["examples"]
    assert CARD not in data["functions"][0]["description"]
    assert SSN not in data["functions"][0]["parameters"]["properties"]["ssn"]["default"]
    assert EMAIL not in data["response_format"]["json_schema"]["schema"]["properties"]["email"]["description"]
    assert EMAIL in original["tools"][0]["function"]["description"]
    assert SSN in original["functions"][0]["parameters"]["properties"]["ssn"]["default"]
    assert EMAIL in original["response_format"]["json_schema"]["schema"]["properties"]["email"]["description"]


@pytest.mark.asyncio
async def test_pre_call_redacts_responses_api_input_string():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    original = {"input": f"Please contact {EMAIL}"}

    data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=original,
        call_type="responses",
    )

    assert EMAIL not in data["input"]
    assert "[EMAIL_1]" in data["input"]
    assert original["input"] == f"Please contact {EMAIL}"


@pytest.mark.asyncio
async def test_pre_call_redacts_responses_api_input_items():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    original = {
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": f"email {EMAIL}"}],
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": {"ssn": SSN},
            },
        ]
    }

    data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=original,
        call_type="responses",
    )

    assert EMAIL not in data["input"][0]["content"][0]["text"]
    assert SSN not in data["input"][1]["output"]["ssn"]
    assert EMAIL in original["input"][0]["content"][0]["text"]
    assert SSN in original["input"][1]["output"]["ssn"]


@pytest.mark.asyncio
async def test_pre_call_redacts_top_level_instructions_and_system_string():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    original = {
        "instructions": f"Route customer follow-up to {EMAIL}",
        "system": f"Billing card on file is {CARD}",
        "input": "summarize the customer account",
    }

    data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=original,
        call_type="responses",
    )

    assert EMAIL not in data["instructions"]
    assert "[EMAIL_1]" in data["instructions"]
    assert CARD not in data["system"]
    assert "[CREDIT_CARD_1]" in data["system"]
    assert original["instructions"] == f"Route customer follow-up to {EMAIL}"
    assert original["system"] == f"Billing card on file is {CARD}"


@pytest.mark.asyncio
async def test_pre_call_redacts_top_level_anthropic_system_content_blocks():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    original = {
        "system": [
            {"type": "text", "text": f"Use tax profile {SSN} for context"},
            {"type": "cache_control", "cache_control": {"type": "ephemeral"}},
        ],
        "messages": [{"role": "user", "content": "hello"}],
    }

    data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=original,
        call_type="completion",
    )

    assert SSN not in data["system"][0]["text"]
    assert "[SSN_1]" in data["system"][0]["text"]
    assert data["system"][1] == original["system"][1]
    assert SSN in original["system"][0]["text"]


@pytest.mark.asyncio
async def test_pre_call_redacts_text_completion_prompt_string_and_list():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    original_string = {"prompt": f"Complete the account note for {EMAIL}"}
    original_list = {"prompt": [f"Email {EMAIL}", f"SSN {SSN}"]}

    string_data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=original_string,
        call_type="completion",
    )
    list_data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=original_list,
        call_type="completion",
    )

    assert EMAIL not in string_data["prompt"]
    assert "[EMAIL_1]" in string_data["prompt"]
    assert EMAIL not in list_data["prompt"][0]
    assert SSN not in list_data["prompt"][1]
    assert original_string["prompt"] == f"Complete the account note for {EMAIL}"
    assert original_list["prompt"] == [f"Email {EMAIL}", f"SSN {SSN}"]


def test_process_content_returns_unmodified_non_text_content():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    content = {"structured": "payload"}
    assert guardrail._process_content(content) == (content, {})


def test_process_tool_payload_handles_nested_lists_and_non_text_values():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    payload = ["plain", {"email": EMAIL}, [SSN], 42]
    redacted, counts = guardrail._process_tool_payload(payload)

    assert EMAIL not in redacted[1]["email"]
    assert SSN not in redacted[2][0]
    assert redacted[3] == 42
    assert counts == {"EMAIL": 1, "SSN": 1}
    assert guardrail._process_tool_payload(42) == (42, {})


def test_process_tool_payload_skips_seen_containers():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    payload = {"self": None, "email": EMAIL}
    payload["self"] = payload
    list_payload = [EMAIL]
    list_payload.append(list_payload)

    redacted, counts = guardrail._process_tool_payload(payload)
    redacted_list, list_counts = guardrail._process_tool_payload(list_payload)

    assert EMAIL not in redacted["email"]
    assert counts == {"EMAIL": 1}
    assert EMAIL not in redacted_list[0]
    assert list_counts == {"EMAIL": 1}


def test_scan_request_tool_calls_preserves_unsupported_entries():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    tool_calls = ["not-a-tool", {"id": "call_1", "type": "function"}]
    assert guardrail._scan_request_tool_calls(tool_calls) == (tool_calls, {})


def test_scan_messages_ignores_non_message_payloads():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    data = {"messages": "not-a-list"}
    assert guardrail._scan_messages(data) == (data, {})

    data = {"messages": ["not-a-dict", {"role": "system"}]}
    assert guardrail._scan_messages(data) == (data, {})


@pytest.mark.asyncio
async def test_block_raises_http_400_without_echoing_pii():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_action="block")
    with pytest.raises(HTTPException) as exc:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=_chat_data(f"send {CARD} to billing"),
            call_type="completion",
        )
    assert exc.value.status_code == 400
    detail = str(exc.value.detail)
    assert "CREDIT_CARD" in detail
    assert CARD not in detail


@pytest.mark.asyncio
async def test_during_call_blocks_when_action_is_block():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_action="block")
    with pytest.raises(HTTPException):
        await guardrail.async_moderation_hook(
            data=_chat_data(f"reach me at {EMAIL}"),
            user_api_key_dict=None,
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_during_call_blocks_tool_call_arguments_when_action_is_block():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_action="block")
    with pytest.raises(HTTPException):
        await guardrail.async_moderation_hook(
            data={
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "send_email",
                                    "arguments": f'{{"recipient": "{EMAIL}"}}',
                                },
                            }
                        ],
                    }
                ]
            },
            user_api_key_dict=None,
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_during_call_blocks_tool_and_function_definitions_when_action_is_block():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_action="block")

    with pytest.raises(HTTPException):
        await guardrail.async_moderation_hook(
            data={
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "notify_customer",
                            "description": f"Notify {EMAIL}",
                        },
                    }
                ]
            },
            user_api_key_dict=None,
            call_type="completion",
        )

    with pytest.raises(HTTPException):
        await guardrail.async_moderation_hook(
            data={
                "functions": [
                    {
                        "name": "legacy_lookup",
                        "parameters": {"type": "object", "description": f"Use card {CARD}"},
                    }
                ]
            },
            user_api_key_dict=None,
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_during_call_blocks_responses_api_input_when_action_is_block():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_action="block")
    with pytest.raises(HTTPException):
        await guardrail.async_moderation_hook(
            data={"input": [{"role": "user", "content": [{"type": "input_text", "text": f"email {EMAIL}"}]}]},
            user_api_key_dict=None,
            call_type="responses",
        )


@pytest.mark.asyncio
async def test_during_call_blocks_top_level_prompt_fields_when_action_is_block():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_action="block")

    with pytest.raises(HTTPException):
        await guardrail.async_moderation_hook(
            data={"instructions": f"Contact billing at {EMAIL}"},
            user_api_key_dict=None,
            call_type="responses",
        )

    with pytest.raises(HTTPException):
        await guardrail.async_moderation_hook(
            data={"system": [{"type": "text", "text": f"Use card {CARD} for lookup"}]},
            user_api_key_dict=None,
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_during_call_blocks_text_completion_prompt_when_action_is_block():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_action="block")

    with pytest.raises(HTTPException):
        await guardrail.async_moderation_hook(
            data={"prompt": [f"Complete customer note for {EMAIL}"]},
            user_api_key_dict=None,
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_during_call_noop_when_action_is_redact():
    # during_call cannot modify content mid-flight; redact mode is a no-op.
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_action="redact")
    data = _chat_data(f"reach me at {EMAIL}")
    result = await guardrail.async_moderation_hook(data=data, user_api_key_dict=None, call_type="completion")
    assert result == data


@pytest.mark.asyncio
async def test_post_call_redacts_model_response():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    response = _model_response(f"the customer is reachable at {EMAIL}")
    await guardrail.async_post_call_success_hook(data={}, user_api_key_dict=None, response=response)
    assert EMAIL not in response.choices[0].message.content


@pytest.mark.asyncio
async def test_post_call_redacts_text_completion_choice_text():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    response = SimpleNamespace(choices=[SimpleNamespace(text=f"the customer email is {EMAIL}")])

    await guardrail.async_post_call_success_hook(data={}, user_api_key_dict=None, response=response)

    assert EMAIL not in response.choices[0].text
    assert "[EMAIL_1]" in response.choices[0].text


@pytest.mark.asyncio
async def test_post_call_redacts_model_response_tool_and_function_arguments():
    import litellm

    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    response = litellm.ModelResponse(
        choices=[
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": None,
                    "role": "assistant",
                    "function_call": {
                        "name": "legacy_lookup",
                        "arguments": f'{{"ssn": "{SSN}"}}',
                    },
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "send_email",
                                "arguments": f'{{"recipient": "{EMAIL}"}}',
                            },
                        }
                    ],
                },
            }
        ]
    )

    await guardrail.async_post_call_success_hook(data={}, user_api_key_dict=None, response=response)

    message = response.choices[0].message
    assert message.content is None
    assert SSN not in message.function_call.arguments
    assert EMAIL not in message.tool_calls[0].function.arguments


@pytest.mark.asyncio
async def test_post_call_redacts_responses_api_output_text_and_arguments():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    response = SimpleNamespace(
        output=[
            {
                "type": "message",
                "content": [{"type": "output_text", "text": f"contact {EMAIL}"}],
            },
            {
                "type": "function_call",
                "arguments": f'{{"card": "{CARD}"}}',
            },
        ]
    )

    await guardrail.async_post_call_success_hook(data={}, user_api_key_dict=None, response=response)

    assert EMAIL not in response.output[0]["content"][0]["text"]
    assert CARD not in response.output[1]["arguments"]


@pytest.mark.asyncio
async def test_post_call_redacts_anthropic_messages_content_and_tool_use_input():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    response = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": f"email {EMAIL}"},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "lookup_customer",
                "input": {"ssn": SSN},
            },
        ],
    }

    await guardrail.async_post_call_success_hook(data={}, user_api_key_dict=None, response=response)

    assert EMAIL not in response["content"][0]["text"]
    assert SSN not in response["content"][1]["input"]["ssn"]


@pytest.mark.asyncio
async def test_post_call_returns_original_when_event_hook_does_not_match():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, event_hook="pre_call")
    response = _model_response(f"the customer is reachable at {EMAIL}")
    result = await guardrail.async_post_call_success_hook(data={}, user_api_key_dict=None, response=response)
    assert result is response
    assert response.choices[0].message.content == f"the customer is reachable at {EMAIL}"


@pytest.mark.asyncio
async def test_post_call_returns_response_without_choices():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    response = SimpleNamespace(choices=[])
    result = await guardrail.async_post_call_success_hook(data={}, user_api_key_dict=None, response=response)
    assert result is response


@pytest.mark.asyncio
async def test_post_call_skips_non_text_response_parts():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=[{"type": "image"}]))])
    result = await guardrail.async_post_call_success_hook(data={}, user_api_key_dict=None, response=response)
    assert result is response


@pytest.mark.asyncio
async def test_post_call_fail_open_returns_response_on_engine_error(monkeypatch):
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    monkeypatch.setattr(
        "litellm.proxy.guardrails.guardrail_hooks.datafog.datafog._redact_text",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    response = _model_response(f"the customer is reachable at {EMAIL}")
    result = await guardrail.async_post_call_success_hook(data={}, user_api_key_dict=None, response=response)
    assert result is response
    assert response.choices[0].message.content == f"the customer is reachable at {EMAIL}"


@pytest.mark.asyncio
async def test_streaming_redacts_model_response_chunks_before_yielding():
    from litellm.types.utils import ModelResponseStream

    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    chunks = [
        ModelResponseStream(choices=[{"index": 0, "delta": {"content": "email jane.doe@"}}]),
        ModelResponseStream(choices=[{"index": 0, "delta": {"content": "example.com"}, "finish_reason": "stop"}]),
    ]

    result = await _collect_async(
        guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=None,
            response=_async_iter(chunks),
            request_data={},
        )
    )

    text = "".join(chunk.choices[0].delta.content or "" for chunk in result)
    assert EMAIL not in text
    assert "[EMAIL_1]" in text


@pytest.mark.asyncio
async def test_streaming_redacts_responses_api_output_text_delta_events():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    chunks = [
        SimpleNamespace(
            type="response.output_text.delta",
            item_id="msg_1",
            output_index=0,
            content_index=0,
            delta="email jane.doe@",
        ),
        SimpleNamespace(
            type="response.output_text.delta",
            item_id="msg_1",
            output_index=0,
            content_index=0,
            delta="example.com",
        ),
    ]

    result = await _collect_async(
        guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=None,
            response=_async_iter(chunks),
            request_data={},
        )
    )

    text = "".join(chunk.delta for chunk in result)
    assert EMAIL not in text
    assert "[EMAIL_1]" in text


@pytest.mark.asyncio
async def test_streaming_redacts_anthropic_sse_text_delta_bytes():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    chunks = [
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"text_delta","text":"email jane.doe@"}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"text_delta","text":"example.com"}}\n\n',
    ]

    result = await _collect_async(
        guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=None,
            response=_async_iter(chunks),
            request_data={},
        )
    )

    raw_text = b"".join(result).decode("utf-8")
    assert EMAIL not in raw_text
    assert "[EMAIL_1]" in raw_text


@pytest.mark.asyncio
async def test_streaming_redacts_text_completion_choice_text_before_yielding():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(index=0, text="email jane.doe@")]),
        SimpleNamespace(choices=[SimpleNamespace(index=0, text="example.com")]),
    ]

    result = await _collect_async(
        guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=None,
            response=_async_iter(chunks),
            request_data={},
        )
    )

    text = "".join(chunk.choices[0].text for chunk in result)
    assert EMAIL not in text
    assert "[EMAIL_1]" in text


@pytest.mark.asyncio
async def test_streaming_block_raises_before_mutating_chunks():
    from litellm.types.utils import ModelResponseStream

    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_action="block")
    chunks = [
        ModelResponseStream(choices=[{"index": 0, "delta": {"content": "email jane.doe@"}}]),
        ModelResponseStream(choices=[{"index": 0, "delta": {"content": "example.com"}, "finish_reason": "stop"}]),
    ]

    with pytest.raises(HTTPException) as exc:
        await _collect_async(
            guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=None,
                response=_async_iter(chunks),
                request_data={},
            )
        )

    assert exc.value.status_code == 400
    assert EMAIL not in str(exc.value.detail)
    assert chunks[0].choices[0].delta.content == "email jane.doe@"
    assert chunks[1].choices[0].delta.content == "example.com"


@pytest.mark.asyncio
async def test_noisy_entities_off_by_default():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=_chat_data("ping 192.168.1.1 about build 2020-01-02"),
        call_type="completion",
    )
    assert data["messages"][0]["content"] == "ping 192.168.1.1 about build 2020-01-02"


@pytest.mark.asyncio
async def test_entity_types_override():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_entity_types=["IP_ADDRESS"])
    data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=_chat_data("ping 192.168.1.1"),
        call_type="completion",
    )
    assert "192.168.1.1" not in data["messages"][0]["content"]


@pytest.mark.asyncio
async def test_german_locale_entities():
    guardrail = DataFogGuardrail(
        guardrail_name="datafog-pii",
        default_on=True,
        datafog_entity_types=["DE_TAX_ID"],
        datafog_locales=["de"],
    )
    data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=_chat_data(f"Steuer-ID {DE_TAX_ID} liegt vor."),
        call_type="completion",
    )
    assert DE_TAX_ID not in data["messages"][0]["content"]


@pytest.mark.asyncio
async def test_fail_open_passes_data_through_on_engine_error(monkeypatch):
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    monkeypatch.setattr(
        "litellm.proxy.guardrails.guardrail_hooks.datafog.datafog._redact_text",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    original = _chat_data(f"reach me at {EMAIL}")
    data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None, cache=None, data=original, call_type="completion"
    )
    assert data["messages"][0]["content"] == f"reach me at {EMAIL}"


@pytest.mark.asyncio
async def test_fail_closed_raises_without_pii_or_cause_chain(monkeypatch):
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_fail_policy="closed")
    monkeypatch.setattr(
        "litellm.proxy.guardrails.guardrail_hooks.datafog.datafog._redact_text",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError(f"parser choked on: reach me at {EMAIL}")),
    )
    with pytest.raises(RuntimeError, match="datafog_fail_policy") as exc:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=_chat_data(f"reach me at {EMAIL}"),
            call_type="completion",
        )
    assert exc.value.__cause__ is None
    assert EMAIL not in str(exc.value)


@pytest.mark.asyncio
async def test_redact_logging_metadata_survives_on_returned_data():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True)
    data = await guardrail.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=_chat_data(f"email {EMAIL}"),
        call_type="completion",
    )
    assert "metadata" in data


@pytest.mark.asyncio
async def test_post_call_block_raises_on_response_pii():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_action="block")
    response = _model_response(f"the customer is reachable at {EMAIL}")
    with pytest.raises(HTTPException) as exc:
        await guardrail.async_post_call_success_hook(data={}, user_api_key_dict=None, response=response)
    assert exc.value.status_code == 400
    assert EMAIL not in str(exc.value.detail)


@pytest.mark.asyncio
async def test_post_call_block_raises_on_response_tool_call_arguments():
    import litellm

    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_action="block")
    response = litellm.ModelResponse(
        choices=[
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "send_email",
                                "arguments": f'{{"recipient": "{EMAIL}"}}',
                            },
                        }
                    ],
                },
            }
        ]
    )

    with pytest.raises(HTTPException) as exc:
        await guardrail.async_post_call_success_hook(data={}, user_api_key_dict=None, response=response)

    assert exc.value.status_code == 400
    assert EMAIL not in str(exc.value.detail)
    assert response.choices[0].message.tool_calls[0].function.arguments == f'{{"recipient": "{EMAIL}"}}'


@pytest.mark.asyncio
async def test_post_call_block_raises_on_text_completion_choice_text():
    guardrail = DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_action="block")
    response = SimpleNamespace(choices=[SimpleNamespace(text=f"the customer email is {EMAIL}")])

    with pytest.raises(HTTPException) as exc:
        await guardrail.async_post_call_success_hook(data={}, user_api_key_dict=None, response=response)

    assert exc.value.status_code == 400
    assert EMAIL not in str(exc.value.detail)
    assert response.choices[0].text == f"the customer email is {EMAIL}"


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_action="explode")
    with pytest.raises(ValueError):
        DataFogGuardrail(guardrail_name="datafog-pii", default_on=True, datafog_fail_policy="maybe")


def test_registry_registration():
    from litellm.proxy.guardrails.guardrail_hooks.datafog import (
        guardrail_class_registry,
        guardrail_initializer_registry,
        initialize_guardrail,
    )
    from litellm.types.guardrails import SupportedGuardrailIntegrations

    assert SupportedGuardrailIntegrations.DATAFOG.value in guardrail_initializer_registry
    assert guardrail_initializer_registry[SupportedGuardrailIntegrations.DATAFOG.value] is initialize_guardrail
    assert guardrail_class_registry[SupportedGuardrailIntegrations.DATAFOG.value] is (DataFogGuardrail)


def test_initialize_guardrail_registers_callback(monkeypatch):
    from litellm.proxy.guardrails.guardrail_hooks.datafog import initialize_guardrail

    registered = []
    monkeypatch.setattr(
        "litellm.logging_callback_manager.add_litellm_callback",
        lambda callback: registered.append(callback),
    )
    params = SimpleNamespace(
        mode="pre_call",
        default_on=True,
        datafog_action="block",
        optional_params=SimpleNamespace(
            datafog_entity_types=["EMAIL"],
            datafog_locales=["de"],
            datafog_fail_policy="closed",
        ),
    )

    guardrail = initialize_guardrail(params, {"guardrail_name": "datafog-pii"})

    assert isinstance(guardrail, DataFogGuardrail)
    assert registered == [guardrail]
    assert guardrail.guardrail_name == "datafog-pii"
    assert guardrail.action == "block"
    assert guardrail.entity_types == ["EMAIL"]
    assert guardrail.locales == ["de"]
    assert guardrail.fail_policy == "closed"


def test_initialize_guardrail_requires_name():
    from litellm.proxy.guardrails.guardrail_hooks.datafog import initialize_guardrail

    params = SimpleNamespace(mode="pre_call", default_on=True, datafog_action="redact", optional_params=None)
    with pytest.raises(ValueError, match="guardrail_name"):
        initialize_guardrail(params, {})


def test_config_model_ui_name():
    assert DataFogGuardrail.get_config_model().ui_friendly_name() == "DataFog PII Guardrail"

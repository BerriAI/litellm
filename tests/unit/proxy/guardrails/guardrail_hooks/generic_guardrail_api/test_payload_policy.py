import copy
import json
import logging
from collections.abc import Callable

import httpx
import pytest

from litellm.exceptions import GuardrailRaisedException
from litellm.llms.anthropic.chat.guardrail_translation.handler import AnthropicMessagesHandler
from litellm.llms.base_llm.guardrail_translation.utils import UnappliableRequestRewrite
from litellm.llms.openai.chat.guardrail_translation.handler import OpenAIChatCompletionsHandler
from litellm.llms.openai.responses.guardrail_translation.handler import OpenAIResponsesHandler
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api import GenericGuardrailAPI
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.payload_policy import (
    IMAGE_OMITTED_PLACEHOLDER,
)
from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import GenericGuardrailAPIRequest

SSN = "123-45-6789"
IMAGE_URL = "data:image/png;base64,SECRETPIXELS"
OTHER_IMAGE_URL = "data:image/png;base64,OTHERPIXELS"


def _answer(body: dict) -> Callable[[dict], dict]:
    return lambda _payload: body


class _FakeGuardrailEndpoint:
    def __init__(self, respond: Callable[[dict], dict] = _answer({"action": "NONE"})):
        self.payloads: list[dict] = []
        self._respond = respond

    async def post(self, *, url: str, json: dict, headers: dict, **_kwargs: object) -> httpx.Response:
        sent = _wire_copy(json)
        self.payloads.append(sent)
        return httpx.Response(200, json=self._respond(sent), request=httpx.Request("POST", url))


class _LoggingObj:
    litellm_call_id = "call-abc"
    litellm_trace_id = "trace-abc"


def _wire_copy(payload: dict) -> dict:
    return json.loads(json.dumps(payload))


def _guardrail(endpoint: _FakeGuardrailEndpoint, **options: object) -> GenericGuardrailAPI:
    return GenericGuardrailAPI(
        api_base="https://guardrail.example",
        guardrail_name="payload-policy-test",
        event_hook="pre_call",
        default_on=True,
        async_handler=endpoint,
        **options,
    )


def _image_message() -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": f"my ssn is {SSN}"},
            {"type": "image_url", "image_url": {"url": IMAGE_URL, "detail": "low"}},
            {"type": "image_url", "image_url": OTHER_IMAGE_URL},
        ],
    }


def _text_message() -> dict:
    return {"role": "user", "content": f"also {SSN}"}


def _texts() -> list[str]:
    return [f"my ssn is {SSN}", f"also {SSN}"]


def _masked_text(text: str) -> str:
    return text.replace(SSN, "[SSN]")


def _masked_part(part: dict) -> dict:
    return {**part, "text": _masked_text(part["text"])} if part.get("type") == "text" else part


def _masked_row(row: dict) -> dict:
    content = row["content"]
    if isinstance(content, str):
        return {**row, "content": _masked_text(content)}
    return {**row, "content": [_masked_part(part) for part in content]}


def _masking_guardrail(payload: dict) -> dict:
    return {
        "action": "GUARDRAIL_INTERVENED",
        "texts": [_masked_text(text) for text in payload.get("texts", [])],
        "structured_messages": [_masked_row(row) for row in payload.get("structured_messages") or []],
    }


def _rewrite_rows(rewrite: Callable[[int, dict], dict]) -> Callable[[dict], dict]:
    def respond(payload: dict) -> dict:
        rows = payload["structured_messages"]
        return {
            "action": "GUARDRAIL_INTERVENED",
            "structured_messages": [rewrite(i, row) for i, row in enumerate(rows)],
        }

    return respond


async def _apply(guardrail: GenericGuardrailAPI, inputs: dict) -> dict:
    return await guardrail.apply_guardrail(
        inputs=inputs,
        request_data={"proxy_server_request": {"headers": {"user-agent": "curl/8"}}},
        input_type="request",
        logging_obj=_LoggingObj(),
    )


@pytest.mark.asyncio
async def test_defaults_send_every_request_field_unchanged():
    endpoint = _FakeGuardrailEndpoint()
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    await _apply(
        _guardrail(endpoint),
        {"texts": ["describe this"], "images": [IMAGE_URL], "tools": tools, "structured_messages": [_image_message()]},
    )

    payload = endpoint.payloads[0]
    assert set(payload) == set(GenericGuardrailAPIRequest.model_fields)
    assert payload["images"] == [IMAGE_URL]
    assert payload["texts"] == ["describe this"]
    assert payload["tools"] == tools
    assert payload["structured_messages"] == [_image_message()]
    assert payload["request_headers"] == {"user-agent": "curl/8"}


@pytest.mark.asyncio
async def test_defaults_still_accept_every_rewrite():
    rewritten_tools = [{"type": "function", "function": {"name": "safe_lookup"}}]
    endpoint = _FakeGuardrailEndpoint(
        _answer(
            {
                "action": "GUARDRAIL_INTERVENED",
                "texts": ["[MASKED]"],
                "images": [OTHER_IMAGE_URL],
                "tools": rewritten_tools,
            }
        )
    )

    result = await _apply(
        _guardrail(endpoint),
        {"texts": ["my ssn is 123"], "images": [IMAGE_URL], "tools": [{"type": "function", "function": {"name": "x"}}]},
    )

    assert result == {"texts": ["[MASKED]"], "images": [OTHER_IMAGE_URL], "tools": rewritten_tools}


@pytest.mark.asyncio
async def test_send_images_false_withholds_every_image_but_keeps_the_parts():
    endpoint = _FakeGuardrailEndpoint()
    messages = [_image_message(), _text_message()]

    await _apply(
        _guardrail(endpoint, send_images=False),
        {"texts": _texts(), "images": [IMAGE_URL], "structured_messages": messages},
    )

    payload = endpoint.payloads[0]
    assert "SECRETPIXELS" not in json.dumps(payload)
    assert "OTHERPIXELS" not in json.dumps(payload)
    assert "images" not in payload
    assert payload["texts"] == _texts()
    assert payload["structured_messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"my ssn is {SSN}"},
                {"type": "image_url", "image_url": {"url": IMAGE_OMITTED_PLACEHOLDER, "detail": "low"}},
                {"type": "image_url", "image_url": IMAGE_OMITTED_PLACEHOLDER},
            ],
        },
        _text_message(),
    ]
    assert messages == [_image_message(), _text_message()]


@pytest.mark.asyncio
async def test_send_images_false_withholds_the_image_in_a_row_the_request_model_cannot_validate():
    endpoint = _FakeGuardrailEndpoint()
    row = {
        "role": "user",
        "content": [
            {"type": "guarded_text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
        ],
    }

    await _apply(_guardrail(endpoint, send_images=False), {"texts": ["hi"], "structured_messages": [row]})

    assert "SECRETPIXELS" not in json.dumps(endpoint.payloads[0])
    assert endpoint.payloads[0]["structured_messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "guarded_text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": IMAGE_OMITTED_PLACEHOLDER}},
            ],
        }
    ]


@pytest.mark.asyncio
async def test_send_images_false_keeps_the_callers_images():
    endpoint = _FakeGuardrailEndpoint(
        _answer({"action": "GUARDRAIL_INTERVENED", "texts": ["[MASKED]"], "images": [OTHER_IMAGE_URL]})
    )

    result = await _apply(_guardrail(endpoint, send_images=False), {"texts": ["ssn 1"], "images": [IMAGE_URL]})

    assert result == {"texts": ["[MASKED]"], "images": [IMAGE_URL]}


@pytest.mark.asyncio
async def test_send_images_false_masks_the_llm_bound_text_and_keeps_the_callers_image():
    guardrail = _guardrail(_FakeGuardrailEndpoint(_masking_guardrail), send_images=False)
    data = {"model": "gpt-x", "messages": [_image_message(), _text_message()]}

    result = await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

    assert result["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "my ssn is [SSN]"},
                {"type": "image_url", "image_url": {"url": IMAGE_URL, "detail": "low"}},
                {"type": "image_url", "image_url": OTHER_IMAGE_URL},
            ],
        },
        {"role": "user", "content": "also [SSN]"},
    ]


@pytest.mark.asyncio
async def test_send_images_false_masks_the_responses_input_and_keeps_the_callers_input_image():
    guardrail = _guardrail(_FakeGuardrailEndpoint(_masking_guardrail), send_images=False)
    data = {
        "model": "gpt-x",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"my ssn is {SSN}"},
                    {"type": "input_image", "image_url": IMAGE_URL},
                ],
            },
            {"role": "user", "content": f"also {SSN}"},
        ],
    }

    result = await OpenAIResponsesHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

    assert result["input"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "my ssn is [SSN]"},
                {"type": "input_image", "image_url": IMAGE_URL, "detail": "auto"},
            ],
        },
        {"role": "user", "content": "also [SSN]"},
    ]


@pytest.mark.asyncio
async def test_send_images_false_masks_the_anthropic_messages_and_keeps_the_callers_base64_image():
    endpoint = _FakeGuardrailEndpoint(_masking_guardrail)
    guardrail = _guardrail(endpoint, send_images=False)
    image_block = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "SECRETPIXELS"}}
    data = {
        "model": "claude-x",
        "max_tokens": 5,
        "system": f"sys {SSN}",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": f"my ssn is {SSN}"}, image_block]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": f"also {SSN}"},
        ],
    }

    result = await AnthropicMessagesHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

    assert "SECRETPIXELS" not in json.dumps(endpoint.payloads[0])
    assert result["system"] == [{"type": "text", "text": "sys [SSN]"}]
    assert result["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "my ssn is [SSN]"}, image_block]},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {"role": "user", "content": [{"type": "text", "text": "also [SSN]"}]},
    ]


@pytest.mark.asyncio
async def test_a_rewritten_image_row_gets_the_callers_images_back():
    messages = [_image_message(), _text_message()]
    endpoint = _FakeGuardrailEndpoint(_rewrite_rows(lambda i, row: _masked_row(row) if i == 0 else row))

    result = await _apply(_guardrail(endpoint, send_images=False), {"texts": _texts(), "structured_messages": messages})

    assert result == {"texts": _texts(), "structured_messages": [_masked_row(_image_message()), messages[1]]}
    assert result["structured_messages"][1] is messages[1]


@pytest.mark.asyncio
async def test_an_echoed_placeholder_row_is_restored_to_the_callers_row():
    messages = [_image_message(), _text_message()]
    endpoint = _FakeGuardrailEndpoint(_rewrite_rows(lambda i, row: _masked_row(row) if i == 1 else row))

    result = await _apply(_guardrail(endpoint, send_images=False), {"texts": _texts(), "structured_messages": messages})

    assert result == {"texts": _texts(), "structured_messages": [messages[0], _masked_row(_text_message())]}
    assert result["structured_messages"][0] is messages[0]


_TEXT_PART = {"type": "text", "text": "my ssn is [SSN]"}
_OBJECT_PLACEHOLDER = {"type": "image_url", "image_url": {"url": IMAGE_OMITTED_PLACEHOLDER, "detail": "low"}}
_BARE_PLACEHOLDER = {"type": "image_url", "image_url": IMAGE_OMITTED_PLACEHOLDER}


def _with_content(content: object) -> Callable[[int, dict], dict]:
    return lambda i, row: {**row, "content": content} if i == 0 else row


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rewrite",
    [
        _with_content([_TEXT_PART, {"type": "image_url", "image_url": {"url": OTHER_IMAGE_URL}}, _BARE_PLACEHOLDER]),
        _with_content([_TEXT_PART, {"type": "text", "text": "y"}, _BARE_PLACEHOLDER]),
        _with_content([_BARE_PLACEHOLDER, _OBJECT_PLACEHOLDER, _BARE_PLACEHOLDER]),
        _with_content([{"type": "text", "text": "x"}]),
        _with_content("my ssn is [SSN]"),
    ],
    ids=[
        "image_part_changed",
        "image_part_replaced_by_text",
        "text_part_replaced_by_placeholder",
        "parts_dropped",
        "content_flattened",
    ],
)
async def test_a_rewrite_that_does_not_line_up_with_a_withheld_image_blocks_the_request(rewrite):
    endpoint = _FakeGuardrailEndpoint(_rewrite_rows(rewrite))

    with pytest.raises(UnappliableRequestRewrite):
        await _apply(
            _guardrail(endpoint, send_images=False),
            {"texts": _texts(), "structured_messages": [_image_message(), _text_message()]},
        )


@pytest.mark.asyncio
async def test_misaligned_rows_block_the_request_when_an_image_was_withheld():
    endpoint = _FakeGuardrailEndpoint(
        _answer({"action": "GUARDRAIL_INTERVENED", "structured_messages": [{"role": "user", "content": "[MASKED]"}]})
    )

    with pytest.raises(UnappliableRequestRewrite):
        await _apply(
            _guardrail(endpoint, send_images=False),
            {"texts": _texts(), "structured_messages": [_text_message(), _image_message()]},
        )


@pytest.mark.asyncio
async def test_a_placeholder_copied_to_a_row_without_images_blocks_the_request():
    endpoint = _FakeGuardrailEndpoint(
        lambda payload: {
            "action": "GUARDRAIL_INTERVENED",
            "structured_messages": [payload["structured_messages"][0], payload["structured_messages"][0]],
        }
    )

    with pytest.raises(UnappliableRequestRewrite):
        await _apply(
            _guardrail(endpoint, send_images=False),
            {"texts": _texts(), "structured_messages": [_image_message(), _text_message()]},
        )


def _row_with_a_part_the_request_model_cannot_validate() -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": f"my ssn is {SSN}"},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "flac"}},
        ],
    }


async def _chat_messages_reaching_the_llm(guardrail: GenericGuardrailAPI, messages: list[dict]) -> list[dict]:
    data = {"model": "gpt-x", "messages": messages}
    return (await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail))[
        "messages"
    ]


def _masked_row_with_a_part_the_request_model_cannot_validate() -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "my ssn is [SSN]"},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "flac"}},
        ],
    }


@pytest.mark.asyncio
async def test_a_row_with_a_part_the_request_model_cannot_validate_reaches_the_llm_masked():
    guardrail = _guardrail(_FakeGuardrailEndpoint(_masking_guardrail), send_images=False)

    messages = await _chat_messages_reaching_the_llm(
        guardrail, [_row_with_a_part_the_request_model_cannot_validate(), _text_message()]
    )

    assert messages == [
        _masked_row_with_a_part_the_request_model_cannot_validate(),
        {"role": "user", "content": "also [SSN]"},
    ]


@pytest.mark.asyncio
async def test_a_row_with_a_part_the_request_model_cannot_validate_reaches_the_llm_masked_at_defaults():
    guardrail = _guardrail(_FakeGuardrailEndpoint(_masking_guardrail))

    messages = await _chat_messages_reaching_the_llm(
        guardrail, [_row_with_a_part_the_request_model_cannot_validate(), _text_message()]
    )

    assert messages == [
        _masked_row_with_a_part_the_request_model_cannot_validate(),
        {"role": "user", "content": "also [SSN]"},
    ]


@pytest.mark.asyncio
async def test_defaults_still_write_back_rewritten_image_rows():
    messages = [_image_message(), _text_message()]
    endpoint = _FakeGuardrailEndpoint(_rewrite_rows(lambda _i, row: {**row, "content": "[MASKED]"}))

    result = await _apply(_guardrail(endpoint), {"texts": _texts(), "structured_messages": messages})

    assert result == {
        "texts": _texts(),
        "structured_messages": [{"role": "user", "content": "[MASKED]"}, {"role": "user", "content": "[MASKED]"}],
    }


@pytest.mark.asyncio
async def test_exclude_payload_fields_drops_only_the_named_fields(caplog):
    endpoint = _FakeGuardrailEndpoint()
    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        guardrail = _guardrail(
            endpoint, exclude_payload_fields=["request_headers", "litellm_version", "input_type", "not_a_field"]
        )

    await _apply(guardrail, {"texts": ["hello"]})

    payload = endpoint.payloads[0]
    assert set(GenericGuardrailAPIRequest.model_fields) - set(payload) == {"request_headers", "litellm_version"}
    assert payload["input_type"] == "request"
    assert payload["litellm_call_id"] == "call-abc"
    assert payload["texts"] == ["hello"]
    assert any("not_a_field" in message for message in caplog.messages)
    assert any("input_type" in message and "still sent" in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_litellm_call_id_cannot_be_excluded():
    endpoint = _FakeGuardrailEndpoint()

    await _apply(_guardrail(endpoint, exclude_payload_fields=["litellm_call_id"]), {"texts": ["hello"]})

    assert endpoint.payloads[0]["litellm_call_id"] == "call-abc"


@pytest.mark.asyncio
async def test_an_exclude_config_naming_only_unknown_or_protected_fields_is_not_lossy(caplog):
    endpoint = _FakeGuardrailEndpoint(_answer({"action": "GUARDRAIL_INTERVENED", "texts": ["[MASKED]"]}))
    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        guardrail = _guardrail(endpoint, exclude_payload_fields=["not_a_field", "input_type"])

    result = await _apply(guardrail, {"texts": ["ssn 123"]})

    assert set(endpoint.payloads[0]) == set(GenericGuardrailAPIRequest.model_fields)
    assert result == {"texts": ["[MASKED]"]}
    assert not any("can only enforce on what it is sent" in message for message in caplog.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "inputs", "response", "expected"),
    [
        ("texts", {"texts": ["ssn 1", "ssn 2"]}, {"texts": ["[M]", "[M]"]}, {"texts": ["ssn 1", "ssn 2"]}),
        ("texts", {"texts": []}, {"texts": ["INVENTED"]}, {"texts": []}),
        (
            "tools",
            {"texts": ["hi"], "tools": [{"type": "function", "function": {"name": "run"}}]},
            {"tools": [{"type": "function"}]},
            {"texts": ["hi"], "tools": [{"type": "function", "function": {"name": "run"}}]},
        ),
        (
            "images",
            {"texts": ["hi"], "images": [IMAGE_URL]},
            {"images": [OTHER_IMAGE_URL]},
            {"texts": ["hi"], "images": [IMAGE_URL]},
        ),
        (
            "structured_messages",
            {"texts": ["hi"], "structured_messages": [{"role": "user", "content": "hi"}]},
            {"structured_messages": [{"role": "user", "content": "INVENTED"}]},
            {"texts": ["hi"]},
        ),
        (
            "structured_messages",
            {"texts": ["hi"], "structured_messages": []},
            {"structured_messages": [{"role": "user", "content": "INVENTED"}]},
            {"texts": ["hi"]},
        ),
        (
            "structured_messages",
            {"texts": ["hi"]},
            {"structured_messages": [{"role": "user", "content": "INVENTED"}]},
            {"texts": ["hi"]},
        ),
    ],
    ids=[
        "texts",
        "texts_empty",
        "tools",
        "images",
        "structured_messages",
        "structured_messages_empty",
        "structured_messages_none",
    ],
)
async def test_an_excluded_field_is_not_sent_and_its_rewrite_is_refused(caplog, field, inputs, response, expected):
    endpoint = _FakeGuardrailEndpoint(_answer({"action": "NONE", **response}))

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        result = await _apply(_guardrail(endpoint, exclude_payload_fields=[field]), inputs)

    assert field not in endpoint.payloads[0]
    assert result == expected
    assert any(f"ignoring the returned {field}" in message for message in caplog.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("excluded", "response"),
    [
        (["texts"], {"texts": ["[MASKED]"]}),
        (["texts"], {"texts": ["[MASKED]"], "structured_messages": [{"role": "user", "content": "ssn 1"}]}),
        (
            ["texts"],
            {
                "texts": ["[MASKED]"],
                "structured_messages": [{"role": "user", "content": "ssn 1"}],
                "tools": [{"type": "function", "function": {"name": "run"}}],
                "images": [IMAGE_URL],
            },
        ),
        (["images"], {"images": [OTHER_IMAGE_URL]}),
        (["tools"], {"tools": [{"type": "function"}]}),
        (["structured_messages"], {"structured_messages": [{"role": "user", "content": "[MASKED]"}]}),
    ],
    ids=[
        "texts",
        "texts_with_rows_echoed",
        "texts_with_every_other_field_echoed",
        "images",
        "tools",
        "structured_messages",
    ],
)
async def test_an_intervention_made_only_through_fields_that_were_not_sent_blocks_the_request(excluded, response):
    endpoint = _FakeGuardrailEndpoint(_answer({"action": "GUARDRAIL_INTERVENED", **response}))
    inputs = {
        "texts": ["ssn 1"],
        "images": [IMAGE_URL],
        "tools": [{"type": "function", "function": {"name": "run"}}],
        "structured_messages": [{"role": "user", "content": "ssn 1"}],
    }

    with pytest.raises(UnappliableRequestRewrite):
        await _apply(_guardrail(endpoint, exclude_payload_fields=excluded), inputs)


@pytest.mark.asyncio
async def test_an_echo_counts_as_no_change_whatever_container_the_caller_used():
    endpoint = _FakeGuardrailEndpoint(
        _answer({"action": "GUARDRAIL_INTERVENED", "texts": ["[MASKED]"], "images": [IMAGE_URL]})
    )

    with pytest.raises(UnappliableRequestRewrite):
        await _apply(
            _guardrail(endpoint, exclude_payload_fields=["texts"]), {"texts": ["ssn 1"], "images": (IMAGE_URL,)}
        )


@pytest.mark.asyncio
async def test_a_chat_request_is_rejected_when_the_only_masking_went_to_texts_that_were_not_sent():
    def echo_rows_and_tools_and_mask_texts(payload: dict) -> dict:
        return {
            "action": "GUARDRAIL_INTERVENED",
            "texts": [_masked_text(row["content"]) for row in payload["structured_messages"]],
            "structured_messages": payload["structured_messages"],
            "tools": payload["tools"],
        }

    guardrail = _guardrail(_FakeGuardrailEndpoint(echo_rows_and_tools_and_mask_texts), exclude_payload_fields=["texts"])
    data = {
        "model": "gpt-x",
        "messages": [_text_message()],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
    }

    with pytest.raises(UnappliableRequestRewrite):
        await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ({"texts": ["[MASKED]"], "images": [OTHER_IMAGE_URL]}, {"texts": ["[MASKED]"], "images": [IMAGE_URL]}),
        (
            {"images": [OTHER_IMAGE_URL], "structured_messages": [{"role": "user", "content": "[MASKED]"}]},
            {
                "texts": ["ssn 1"],
                "images": [IMAGE_URL],
                "structured_messages": [{"role": "user", "content": "[MASKED]"}],
            },
        ),
        (
            {"images": [OTHER_IMAGE_URL], "tools": [{"type": "function", "function": {"name": "safe"}}]},
            {"texts": ["ssn 1"], "images": [IMAGE_URL], "tools": [{"type": "function", "function": {"name": "safe"}}]},
        ),
        ({}, {"texts": ["ssn 1"], "images": [IMAGE_URL]}),
    ],
    ids=["texts_applied", "rows_applied", "tools_applied", "nothing_returned"],
)
async def test_an_intervention_that_does_not_depend_on_an_unsent_field_goes_through(response, expected):
    endpoint = _FakeGuardrailEndpoint(_answer({"action": "GUARDRAIL_INTERVENED", **response}))

    result = await _apply(
        _guardrail(endpoint, exclude_payload_fields=["images"]),
        {"texts": ["ssn 1"], "images": [IMAGE_URL], "structured_messages": [{"role": "user", "content": "ssn 1"}]},
    )

    assert result == expected


@pytest.mark.asyncio
async def test_the_responses_handler_rejects_rows_it_did_not_send_even_when_texts_are_echoed():
    endpoint = _FakeGuardrailEndpoint(
        _answer(
            {
                "action": "GUARDRAIL_INTERVENED",
                "texts": ["hello"],
                "structured_messages": [{"role": "user", "content": "INVENTED"}],
            }
        )
    )
    guardrail = _guardrail(endpoint, exclude_payload_fields=["structured_messages"])
    data = {"model": "gpt-x", "input": [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]}
    original_input = copy.deepcopy(data["input"])

    with pytest.raises(UnappliableRequestRewrite):
        await OpenAIResponsesHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

    assert data["input"] == original_input


@pytest.mark.asyncio
async def test_blocked_still_blocks_a_shaped_payload():
    endpoint = _FakeGuardrailEndpoint(_answer({"action": "BLOCKED", "blocked_reason": "no"}))
    guardrail = _guardrail(endpoint, send_images=False, exclude_payload_fields=["texts", "tools"])

    with pytest.raises(GuardrailRaisedException):
        await _apply(guardrail, {"texts": ["hello"], "structured_messages": [_image_message()]})


@pytest.mark.parametrize(
    "options",
    [{"send_images": False}, {"exclude_payload_fields": ["request_headers"]}],
    ids=["send_images", "exclude_payload_fields"],
)
def test_lossy_options_warn_that_the_guardrail_only_enforces_on_what_it_is_sent(caplog, options):
    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        _guardrail(_FakeGuardrailEndpoint(), **options)

    assert any("can only enforce on what it is sent" in message for message in caplog.messages)


def test_defaults_do_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        _guardrail(_FakeGuardrailEndpoint(), send_images=True, exclude_payload_fields=[])

    assert not any("can only enforce on what it is sent" in message for message in caplog.messages)


@pytest.mark.parametrize(
    "options",
    [{"send_images": "false"}, {"send_images": 0}, {"exclude_payload_fields": "tools"}],
    ids=["send_images_string", "send_images_int", "exclude_payload_fields_string"],
)
def test_a_mistyped_option_fails_at_init(options):
    with pytest.raises(ValueError, match=next(iter(options))):
        _guardrail(_FakeGuardrailEndpoint(), **options)

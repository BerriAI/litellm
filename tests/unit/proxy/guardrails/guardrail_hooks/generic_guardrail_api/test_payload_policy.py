import copy
import json
import logging
from collections.abc import Callable
from typing import Final

import httpx
import pytest

from litellm.exceptions import GuardrailRaisedException
from litellm.llms.anthropic.chat.guardrail_translation.handler import AnthropicMessagesHandler
from litellm.llms.base_llm.guardrail_translation.utils import UnappliableRequestRewrite
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai.chat.guardrail_translation.handler import OpenAIChatCompletionsHandler
from litellm.llms.openai.embeddings.guardrail_translation.handler import OpenAIEmbeddingsHandler
from litellm.llms.openai.responses.guardrail_translation.handler import OpenAIResponsesHandler
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api import GenericGuardrailAPI
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.payload_policy import (
    IMAGE_OMITTED_PLACEHOLDER,
)
from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import GenericGuardrailAPIRequest
from litellm.types.utils import ModelResponse

SSN: Final = "123-45-6789"
IMAGE_URL: Final = "data:image/png;base64,SECRETPIXELS"
OTHER_IMAGE_URL: Final = "data:image/png;base64,OTHERPIXELS"
RUN_TOOL: Final = {"type": "function", "function": {"name": "run"}}


def _answer(body: dict) -> Callable[[dict], dict]:
    return lambda _payload: body


class _GuardrailEndpoint:
    def __init__(self, respond: Callable[[dict], dict] = _answer({"action": "NONE"})):
        self.payloads: Final[list[dict]] = []
        self.handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(self._serve))
        self._respond: Final = respond

    def _serve(self, request: httpx.Request) -> httpx.Response:
        payload: Final = json.loads(request.content)
        self.payloads.append(payload)
        return httpx.Response(200, json=self._respond(payload))


class _LoggingObj:
    litellm_call_id = "call-abc"
    litellm_trace_id = "trace-abc"


def _guardrail(endpoint: _GuardrailEndpoint, **options: object) -> GenericGuardrailAPI:
    return GenericGuardrailAPI(
        api_base="https://guardrail.example",
        guardrail_name="payload-policy-test",
        event_hook="pre_call",
        default_on=True,
        async_handler=endpoint.handler,
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
    content: Final = row["content"]
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
        rows: Final = payload["structured_messages"]
        return {
            "action": "GUARDRAIL_INTERVENED",
            "structured_messages": [rewrite(i, row) for i, row in enumerate(rows)],
        }

    return respond


async def _apply(guardrail: GenericGuardrailAPI, inputs: dict, input_type: str = "request") -> dict:
    return await guardrail.apply_guardrail(
        inputs=inputs,
        request_data={"proxy_server_request": {"headers": {"user-agent": "curl/8"}}},
        input_type=input_type,
        logging_obj=_LoggingObj(),
    )


@pytest.mark.asyncio
async def test_defaults_send_every_request_field_unchanged():
    endpoint: Final = _GuardrailEndpoint()
    tools: Final = [{"type": "function", "function": {"name": "lookup"}}]

    await _apply(
        _guardrail(endpoint),
        {"texts": ["describe this"], "images": [IMAGE_URL], "tools": tools, "structured_messages": [_image_message()]},
    )

    payload: Final = endpoint.payloads[0]
    assert set(payload) == set(GenericGuardrailAPIRequest.model_fields)
    assert payload["images"] == [IMAGE_URL]
    assert payload["texts"] == ["describe this"]
    assert payload["tools"] == tools
    assert payload["structured_messages"] == [_image_message()]
    assert payload["request_headers"] == {"user-agent": "curl/8"}


@pytest.mark.asyncio
async def test_defaults_still_accept_every_rewrite():
    rewritten_tools: Final = [{"type": "function", "function": {"name": "safe_lookup"}}]
    endpoint: Final = _GuardrailEndpoint(
        _answer(
            {
                "action": "GUARDRAIL_INTERVENED",
                "texts": ["[MASKED]"],
                "images": [OTHER_IMAGE_URL],
                "tools": rewritten_tools,
            }
        )
    )

    result: Final = await _apply(
        _guardrail(endpoint),
        {"texts": ["my ssn is 123"], "images": [IMAGE_URL], "tools": [{"type": "function", "function": {"name": "x"}}]},
    )

    assert result == {"texts": ["[MASKED]"], "images": [OTHER_IMAGE_URL], "tools": rewritten_tools}


@pytest.mark.asyncio
async def test_send_images_false_withholds_every_image_but_keeps_the_parts():
    endpoint: Final = _GuardrailEndpoint()
    messages: Final = [_image_message(), _text_message()]

    await _apply(
        _guardrail(endpoint, send_images=False),
        {"texts": _texts(), "images": [IMAGE_URL], "structured_messages": messages},
    )

    payload: Final = endpoint.payloads[0]
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
    endpoint: Final = _GuardrailEndpoint()
    row: Final = {
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
    endpoint: Final = _GuardrailEndpoint(
        _answer({"action": "GUARDRAIL_INTERVENED", "texts": ["[MASKED]"], "images": [OTHER_IMAGE_URL]})
    )

    result: Final = await _apply(_guardrail(endpoint, send_images=False), {"texts": ["ssn 1"], "images": [IMAGE_URL]})

    assert result == {"texts": ["[MASKED]"], "images": [IMAGE_URL]}


@pytest.mark.asyncio
async def test_send_images_false_masks_the_llm_bound_text_and_keeps_the_callers_image():
    endpoint: Final = _GuardrailEndpoint(_masking_guardrail)
    guardrail: Final = _guardrail(endpoint, send_images=False)
    data: Final = {"model": "gpt-x", "messages": [_image_message(), _text_message()]}

    result: Final = await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

    assert "SECRETPIXELS" not in json.dumps(endpoint.payloads[0])
    assert "OTHERPIXELS" not in json.dumps(endpoint.payloads[0])
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
    endpoint: Final = _GuardrailEndpoint(_masking_guardrail)
    guardrail: Final = _guardrail(endpoint, send_images=False)
    data: Final = {
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

    result: Final = await OpenAIResponsesHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

    assert "SECRETPIXELS" not in json.dumps(endpoint.payloads[0])
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
    endpoint: Final = _GuardrailEndpoint(_masking_guardrail)
    guardrail: Final = _guardrail(endpoint, send_images=False)
    image_block: Final = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "SECRETPIXELS"},
    }
    data: Final = {
        "model": "claude-x",
        "max_tokens": 5,
        "system": f"sys {SSN}",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": f"my ssn is {SSN}"}, image_block]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": f"also {SSN}"},
        ],
    }

    result: Final = await AnthropicMessagesHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

    assert "SECRETPIXELS" not in json.dumps(endpoint.payloads[0])
    assert result["system"] == [{"type": "text", "text": "sys [SSN]"}]
    assert result["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "my ssn is [SSN]"}, image_block]},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {"role": "user", "content": [{"type": "text", "text": "also [SSN]"}]},
    ]


@pytest.mark.asyncio
async def test_a_rewritten_image_row_gets_the_callers_images_back():
    messages: Final = [_image_message(), _text_message()]
    endpoint: Final = _GuardrailEndpoint(_rewrite_rows(lambda i, row: _masked_row(row) if i == 0 else row))

    result: Final = await _apply(
        _guardrail(endpoint, send_images=False), {"texts": _texts(), "structured_messages": messages}
    )

    assert result == {"texts": _texts(), "structured_messages": [_masked_row(_image_message()), messages[1]]}
    assert result["structured_messages"][1] is messages[1]


@pytest.mark.asyncio
async def test_a_rewritten_image_row_gets_the_callers_images_back_when_its_parts_are_a_tuple():
    caller_row: Final = {**_image_message(), "content": tuple(_image_message()["content"])}
    endpoint: Final = _GuardrailEndpoint(_rewrite_rows(lambda _i, row: _masked_row(row)))

    result: Final = await _apply(
        _guardrail(endpoint, send_images=False), {"texts": _texts()[:1], "structured_messages": [caller_row]}
    )

    assert result == {"texts": _texts()[:1], "structured_messages": [_masked_row(_image_message())]}


@pytest.mark.asyncio
async def test_an_echoed_placeholder_row_is_restored_to_the_callers_row():
    messages: Final = [_image_message(), _text_message()]
    endpoint: Final = _GuardrailEndpoint(_rewrite_rows(lambda i, row: _masked_row(row) if i == 1 else row))

    result: Final = await _apply(
        _guardrail(endpoint, send_images=False), {"texts": _texts(), "structured_messages": messages}
    )

    assert result == {"texts": _texts(), "structured_messages": [messages[0], _masked_row(_text_message())]}
    assert result["structured_messages"][0] is messages[0]


_TEXT_PART: Final = {"type": "text", "text": "my ssn is [SSN]"}
_OBJECT_PLACEHOLDER: Final = {"type": "image_url", "image_url": {"url": IMAGE_OMITTED_PLACEHOLDER, "detail": "low"}}
_BARE_PLACEHOLDER: Final = {"type": "image_url", "image_url": IMAGE_OMITTED_PLACEHOLDER}


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
    endpoint: Final = _GuardrailEndpoint(_rewrite_rows(rewrite))

    with pytest.raises(UnappliableRequestRewrite):
        await _apply(
            _guardrail(endpoint, send_images=False),
            {"texts": _texts(), "structured_messages": [_image_message(), _text_message()]},
        )


@pytest.mark.asyncio
async def test_misaligned_rows_block_the_request_when_an_image_was_withheld():
    endpoint: Final = _GuardrailEndpoint(
        _answer({"action": "GUARDRAIL_INTERVENED", "structured_messages": [{"role": "user", "content": "[MASKED]"}]})
    )

    with pytest.raises(UnappliableRequestRewrite):
        await _apply(
            _guardrail(endpoint, send_images=False),
            {"texts": _texts(), "structured_messages": [_text_message(), _image_message()]},
        )


@pytest.mark.asyncio
async def test_a_placeholder_copied_to_a_row_without_images_blocks_the_request():
    endpoint: Final = _GuardrailEndpoint(
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
    data: Final = {"model": "gpt-x", "messages": messages}
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
    guardrail: Final = _guardrail(_GuardrailEndpoint(_masking_guardrail), send_images=False)

    messages: Final = await _chat_messages_reaching_the_llm(
        guardrail, [_row_with_a_part_the_request_model_cannot_validate(), _text_message()]
    )

    assert messages == [
        _masked_row_with_a_part_the_request_model_cannot_validate(),
        {"role": "user", "content": "also [SSN]"},
    ]


@pytest.mark.asyncio
async def test_a_row_with_a_part_the_request_model_cannot_validate_reaches_the_llm_masked_at_defaults():
    guardrail: Final = _guardrail(_GuardrailEndpoint(_masking_guardrail))

    messages: Final = await _chat_messages_reaching_the_llm(
        guardrail, [_row_with_a_part_the_request_model_cannot_validate(), _text_message()]
    )

    assert messages == [
        _masked_row_with_a_part_the_request_model_cannot_validate(),
        {"role": "user", "content": "also [SSN]"},
    ]


@pytest.mark.asyncio
async def test_defaults_still_write_back_rewritten_image_rows():
    messages: Final = [_image_message(), _text_message()]
    endpoint: Final = _GuardrailEndpoint(_rewrite_rows(lambda _i, row: {**row, "content": "[MASKED]"}))

    result: Final = await _apply(_guardrail(endpoint), {"texts": _texts(), "structured_messages": messages})

    assert result == {
        "texts": _texts(),
        "structured_messages": [{"role": "user", "content": "[MASKED]"}, {"role": "user", "content": "[MASKED]"}],
    }


@pytest.mark.asyncio
async def test_exclude_payload_fields_drops_only_the_named_fields(caplog):
    endpoint: Final = _GuardrailEndpoint()
    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        guardrail: Final = _guardrail(
            endpoint, exclude_payload_fields=["request_headers", "litellm_version", "input_type", "not_a_field"]
        )

    await _apply(guardrail, {"texts": ["hello"]})

    payload: Final = endpoint.payloads[0]
    assert set(GenericGuardrailAPIRequest.model_fields) - set(payload) == {"request_headers", "litellm_version"}
    assert payload["input_type"] == "request"
    assert payload["litellm_call_id"] == "call-abc"
    assert payload["texts"] == ["hello"]
    assert (
        "Generic Guardrail API (payload-policy-test): ignoring unknown exclude_payload_fields ('not_a_field',). "
        f"Known fields: {sorted(GenericGuardrailAPIRequest.model_fields)}"
    ) in caplog.messages
    assert (
        "Generic Guardrail API (payload-policy-test): exclude_payload_fields cannot drop ('input_type',), "
        "the guardrail needs them to interpret the payload; they are still sent."
    ) in caplog.messages


@pytest.mark.asyncio
async def test_litellm_call_id_cannot_be_excluded():
    endpoint: Final = _GuardrailEndpoint()

    await _apply(_guardrail(endpoint, exclude_payload_fields=["litellm_call_id"]), {"texts": ["hello"]})

    assert endpoint.payloads[0]["litellm_call_id"] == "call-abc"


@pytest.mark.asyncio
async def test_an_exclude_config_naming_only_unknown_or_protected_fields_is_not_lossy(caplog):
    endpoint: Final = _GuardrailEndpoint(_answer({"action": "GUARDRAIL_INTERVENED", "texts": ["[MASKED]"]}))
    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        guardrail: Final = _guardrail(endpoint, exclude_payload_fields=["not_a_field", "input_type"])

    result: Final = await _apply(guardrail, {"texts": ["ssn 123"]})

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
    endpoint: Final = _GuardrailEndpoint(_answer({"action": "NONE", **response}))

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        result: Final = await _apply(_guardrail(endpoint, exclude_payload_fields=[field]), inputs)

    assert field not in endpoint.payloads[0]
    assert result == expected
    assert [message for message in caplog.messages if "ignoring the returned" in message] == [
        f"Generic Guardrail API (payload-policy-test): ignoring the returned {field}, it was not sent to the guardrail."
    ]


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
    endpoint: Final = _GuardrailEndpoint(_answer({"action": "GUARDRAIL_INTERVENED", **response}))
    inputs: Final = {
        "texts": ["ssn 1"],
        "images": [IMAGE_URL],
        "tools": [{"type": "function", "function": {"name": "run"}}],
        "structured_messages": [{"role": "user", "content": "ssn 1"}],
    }

    with pytest.raises(UnappliableRequestRewrite):
        await _apply(_guardrail(endpoint, exclude_payload_fields=excluded), inputs)


@pytest.mark.asyncio
async def test_an_echo_counts_as_no_change_whatever_container_the_caller_used():
    tool: Final = {"type": "function", "function": {"name": "run"}}
    endpoint: Final = _GuardrailEndpoint(
        _answer({"action": "GUARDRAIL_INTERVENED", "texts": ["[MASKED]"], "tools": [tool]})
    )

    with pytest.raises(UnappliableRequestRewrite):
        await _apply(_guardrail(endpoint, exclude_payload_fields=["texts"]), {"texts": ["ssn 1"], "tools": (tool,)})


@pytest.mark.asyncio
async def test_a_changed_images_list_does_not_let_masking_through_unsent_texts_pass_unmasked():
    endpoint: Final = _GuardrailEndpoint(
        lambda payload: {
            "action": "GUARDRAIL_INTERVENED",
            "texts": [_masked_text(row["content"]) for row in payload["structured_messages"]],
            "images": [OTHER_IMAGE_URL],
        }
    )
    guardrail: Final = _guardrail(endpoint, exclude_payload_fields=["texts"])
    data: Final = {"model": "gpt-x", "messages": [_text_message()]}

    with pytest.raises(UnappliableRequestRewrite):
        await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

    assert data["messages"] == [_text_message()]


@pytest.mark.asyncio
async def test_a_refused_response_rewrite_names_the_response():
    endpoint: Final = _GuardrailEndpoint(_answer({"action": "GUARDRAIL_INTERVENED", "texts": ["[MASKED]"]}))

    with pytest.raises(UnappliableRequestRewrite) as refusal:
        await _apply(_guardrail(endpoint, exclude_payload_fields=["texts"]), {"texts": ["ssn 1"]}, "response")

    assert str(refusal.value) == (
        "Guardrail 'payload-policy-test' rewrote the response in a way this endpoint cannot apply, "
        "so the response was rejected rather than sent unrewritten"
    )


_FIELDS_THE_ENDPOINT_NEVER_SUPPLIED: Final = pytest.mark.parametrize(
    "unsupplied_field",
    [{"tools": [RUN_TOOL]}, {"structured_messages": [{"role": "assistant", "content": "[MASKED]"}]}],
    ids=["tools", "structured_messages"],
)


@pytest.mark.asyncio
@_FIELDS_THE_ENDPOINT_NEVER_SUPPLIED
async def test_a_chat_response_is_rejected_when_masking_unsent_texts_comes_with_fields_it_never_supplied(
    unsupplied_field,
):
    endpoint: Final = _GuardrailEndpoint(
        _answer({"action": "GUARDRAIL_INTERVENED", "texts": ["[MASKED]"], **unsupplied_field})
    )
    response: Final = ModelResponse(
        model="gpt-x", choices=[{"index": 0, "message": {"role": "assistant", "content": f"your ssn is {SSN}"}}]
    )

    with pytest.raises(UnappliableRequestRewrite):
        await OpenAIChatCompletionsHandler().process_output_response(
            response=response, guardrail_to_apply=_guardrail(endpoint, exclude_payload_fields=["texts"])
        )

    assert response.choices[0].message.content == f"your ssn is {SSN}"


@pytest.mark.asyncio
@_FIELDS_THE_ENDPOINT_NEVER_SUPPLIED
async def test_an_embeddings_request_is_rejected_when_masking_unsent_texts_comes_with_fields_it_never_supplied(
    unsupplied_field,
):
    endpoint: Final = _GuardrailEndpoint(
        _answer({"action": "GUARDRAIL_INTERVENED", "texts": ["[MASKED]"], **unsupplied_field})
    )
    data: Final = {"model": "text-embedding-x", "input": f"my ssn is {SSN}"}

    with pytest.raises(UnappliableRequestRewrite):
        await OpenAIEmbeddingsHandler().process_input_messages(
            data=data, guardrail_to_apply=_guardrail(endpoint, exclude_payload_fields=["texts"])
        )

    assert data["input"] == f"my ssn is {SSN}"


@pytest.mark.asyncio
async def test_a_chat_request_is_rejected_when_the_only_masking_went_to_texts_that_were_not_sent():
    def echo_rows_and_tools_and_mask_texts(payload: dict) -> dict:
        return {
            "action": "GUARDRAIL_INTERVENED",
            "texts": [_masked_text(row["content"]) for row in payload["structured_messages"]],
            "structured_messages": payload["structured_messages"],
            "tools": payload["tools"],
        }

    guardrail: Final = _guardrail(
        _GuardrailEndpoint(echo_rows_and_tools_and_mask_texts), exclude_payload_fields=["texts"]
    )
    data: Final = {
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
        (
            {"texts": ["[MASKED]"], "images": [OTHER_IMAGE_URL]},
            {"texts": ["[MASKED]"], "images": [IMAGE_URL], "tools": [RUN_TOOL]},
        ),
        (
            {"images": [OTHER_IMAGE_URL], "structured_messages": [{"role": "user", "content": "[MASKED]"}]},
            {
                "texts": ["ssn 1"],
                "images": [IMAGE_URL],
                "tools": [RUN_TOOL],
                "structured_messages": [{"role": "user", "content": "[MASKED]"}],
            },
        ),
        (
            {"images": [OTHER_IMAGE_URL], "tools": [{"type": "function", "function": {"name": "safe"}}]},
            {"texts": ["ssn 1"], "images": [IMAGE_URL], "tools": [{"type": "function", "function": {"name": "safe"}}]},
        ),
        ({}, {"texts": ["ssn 1"], "images": [IMAGE_URL], "tools": [RUN_TOOL]}),
    ],
    ids=["texts_applied", "rows_applied", "tools_applied", "nothing_returned"],
)
async def test_an_intervention_that_does_not_depend_on_an_unsent_field_goes_through(response, expected):
    endpoint: Final = _GuardrailEndpoint(_answer({"action": "GUARDRAIL_INTERVENED", **response}))

    result: Final = await _apply(
        _guardrail(endpoint, exclude_payload_fields=["images"]),
        {
            "texts": ["ssn 1"],
            "images": [IMAGE_URL],
            "tools": [RUN_TOOL],
            "structured_messages": [{"role": "user", "content": "ssn 1"}],
        },
    )

    assert result == expected


@pytest.mark.asyncio
async def test_the_responses_handler_rejects_rows_it_did_not_send_even_when_texts_are_echoed():
    endpoint: Final = _GuardrailEndpoint(
        _answer(
            {
                "action": "GUARDRAIL_INTERVENED",
                "texts": ["hello"],
                "structured_messages": [{"role": "user", "content": "INVENTED"}],
            }
        )
    )
    guardrail: Final = _guardrail(endpoint, exclude_payload_fields=["structured_messages"])
    data: Final = {"model": "gpt-x", "input": [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]}
    original_input: Final = copy.deepcopy(data["input"])

    with pytest.raises(UnappliableRequestRewrite):
        await OpenAIResponsesHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

    assert data["input"] == original_input


@pytest.mark.asyncio
async def test_blocked_still_blocks_a_shaped_payload():
    endpoint: Final = _GuardrailEndpoint(_answer({"action": "BLOCKED", "blocked_reason": "no"}))
    guardrail: Final = _guardrail(endpoint, send_images=False, exclude_payload_fields=["texts", "tools"])

    with pytest.raises(GuardrailRaisedException):
        await _apply(guardrail, {"texts": ["hello"], "structured_messages": [_image_message()]})


@pytest.mark.parametrize(
    "options",
    [{"send_images": False}, {"exclude_payload_fields": ["request_headers"]}],
    ids=["send_images", "exclude_payload_fields"],
)
def test_lossy_options_warn_that_the_guardrail_only_enforces_on_what_it_is_sent(caplog, options):
    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        _guardrail(_GuardrailEndpoint(), **options)

    assert any("can only enforce on what it is sent" in message for message in caplog.messages)


def test_defaults_do_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        _guardrail(_GuardrailEndpoint(), send_images=True, exclude_payload_fields=[])

    assert not any("can only enforce on what it is sent" in message for message in caplog.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("options", "warning"),
    [
        ({"send_images": "maybe"}, "Ignoring send_images='maybe', expected a bool. Images are sent"),
        ({"send_images": [1]}, "Ignoring send_images=[1], expected a bool. Images are sent"),
        (
            {"exclude_payload_fields": "texts"},
            "Ignoring exclude_payload_fields='texts', expected a list of strings. Every field is sent",
        ),
        (
            {"exclude_payload_fields": {"texts": 1}},
            "Ignoring exclude_payload_fields={'texts': 1}, expected a list of strings. Every field is sent",
        ),
    ],
    ids=["send_images_string", "send_images_list", "exclude_payload_fields_string", "exclude_payload_fields_mapping"],
)
async def test_an_invalid_option_is_ignored_with_a_warning_and_the_default_kept(caplog, options, warning):
    endpoint: Final = _GuardrailEndpoint()
    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        guardrail: Final = _guardrail(endpoint, **options)

    await _apply(guardrail, {"texts": ["hello"], "images": [IMAGE_URL]})

    assert warning in caplog.messages
    assert set(endpoint.payloads[0]) == set(GenericGuardrailAPIRequest.model_fields)
    assert endpoint.payloads[0]["images"] == [IMAGE_URL]


@pytest.mark.asyncio
async def test_send_images_accepts_the_string_false(caplog):
    endpoint: Final = _GuardrailEndpoint()
    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        guardrail: Final = _guardrail(endpoint, send_images="false")

    await _apply(guardrail, {"texts": ["hello"], "images": [IMAGE_URL]})

    assert "images" not in endpoint.payloads[0]
    assert not any(message.startswith("Ignoring send_images") for message in caplog.messages)

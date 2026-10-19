import json
import operator
from typing import Final

import httpx
import pytest
import respx
from pydantic import JsonValue, ValidationError

import litellm
from litellm.llms.openai.responses.count_tokens.token_counter import OpenAITokenCounter
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.utils import TokenCountMessagesFormat, TokenCountResponse

COUNT_URL: Final = "https://api.openai.com/v1/responses/input_tokens"


async def _count(monkeypatch: pytest.MonkeyPatch, response: httpx.Response) -> TokenCountResponse | None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with respx.mock:
        respx.post(COUNT_URL).mock(return_value=response)
        return await OpenAITokenCounter().count_tokens(
            model_to_use="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "sk-test"}},
            request_model="openai/gpt-4o",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_total"),
    [
        ({"input_tokens": 15}, 15),
        ({"input_tokens": "15"}, 15),
        ({"input_tokens": 15.0}, 15),
        ({"input_tokens": True}, 1),
        ({"input_tokens": -3}, -3),
        ({}, 0),
        ({"object": "response.input_tokens", "error": {"message": "429 rate limit"}}, 0),
        ({"input_tokens": 7, "object": "response.input_tokens", "nested": {"a": [1, None]}}, 7),
    ],
)
async def test_count_reports_upstream_input_tokens_and_keeps_the_raw_payload(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, object], expected_total: int
) -> None:
    result: Final = await _count(monkeypatch, httpx.Response(200, json=payload))

    assert result is not None
    assert repr(result.model_dump()) == repr(
        {
            "total_tokens": expected_total,
            "request_model": "openai/gpt-4o",
            "model_used": "gpt-4o",
            "tokenizer_type": "openai_api",
            "count_source": None,
            "original_response": payload,
            "error": False,
            "error_message": None,
            "status_code": None,
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rejected_count",
    [
        None,
        15.5,
        "Request timed out",
        "429 Too Many Requests",
        [15],
        [*range(403), "Request timed out"],
        {"value": 429},
    ],
)
async def test_non_integer_input_tokens_becomes_an_error_response_carrying_the_validation_message(
    monkeypatch: pytest.MonkeyPatch, rejected_count: object
) -> None:
    with pytest.raises(ValidationError) as rejection:
        TokenCountResponse.model_validate(
            {
                "total_tokens": rejected_count,
                "request_model": "openai/gpt-4o",
                "model_used": "gpt-4o",
                "tokenizer_type": "openai_api",
            }
        )

    result: Final = await _count(monkeypatch, httpx.Response(200, json={"input_tokens": rejected_count}))

    assert result is not None
    assert [error["loc"] for error in rejection.value.errors()] == [("total_tokens",)]
    assert repr(result.model_dump()) == repr(
        {
            "total_tokens": 0,
            "request_model": "openai/gpt-4o",
            "model_used": "gpt-4o",
            "tokenizer_type": "openai_api",
            "count_source": None,
            "original_response": None,
            "error": True,
            "error_message": str(rejection.value),
            "status_code": 500,
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [[*({"input_tokens": position} for position in range(403)), "429"], "input_tokens", 15, 0, False],
)
async def test_count_payload_that_is_not_an_object_becomes_an_error_response_naming_the_failed_lookup(
    monkeypatch: pytest.MonkeyPatch, payload: object
) -> None:
    with pytest.raises(AttributeError) as failed_lookup:
        operator.attrgetter("get")(payload)

    result: Final = await _count(monkeypatch, httpx.Response(200, json=payload))

    assert result is not None
    assert repr(result.model_dump()) == repr(
        {
            "total_tokens": 0,
            "request_model": "openai/gpt-4o",
            "model_used": "gpt-4o",
            "tokenizer_type": "openai_api",
            "count_source": None,
            "original_response": None,
            "error": True,
            "error_message": str(failed_lookup.value),
            "status_code": 500,
        }
    )


@pytest.mark.asyncio
async def test_null_count_payload_falls_back_to_local_counting(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await _count(monkeypatch, httpx.Response(200, content=b"null")) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("upstream_status", "upstream_text"),
    [(429, "rate limit exceeded"), (403, "forbidden"), (408, "Request timed out")],
)
async def test_upstream_error_status_becomes_an_error_response_with_that_status_and_text(
    monkeypatch: pytest.MonkeyPatch, upstream_status: int, upstream_text: str
) -> None:
    result: Final = await _count(monkeypatch, httpx.Response(upstream_status, text=upstream_text))

    assert result is not None
    assert repr(result.model_dump()) == repr(
        {
            "total_tokens": 0,
            "request_model": "openai/gpt-4o",
            "model_used": "gpt-4o",
            "tokenizer_type": "openai_api",
            "count_source": None,
            "original_response": None,
            "error": True,
            "error_message": upstream_text,
            "status_code": upstream_status,
        }
    )


async def _sent_count_body(
    monkeypatch: pytest.MonkeyPatch,
    messages: list[dict[str, object]],
    system: object,
    messages_format: TokenCountMessagesFormat,
    output_format: JsonValue = None,
) -> object:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with respx.mock:
        route: Final = respx.post(COUNT_URL).mock(return_value=httpx.Response(200, json={"input_tokens": 1}))
        await OpenAITokenCounter().count_tokens(
            model_to_use="gpt-4o",
            messages=messages,
            contents=None,
            deployment={"litellm_params": {"api_key": "sk-test"}},
            request_model="openai/gpt-4o",
            system=system,
            messages_format=messages_format,
            output_format=output_format,
        )
        return json.loads(route.calls.last.request.content)


@pytest.mark.asyncio
async def test_anthropic_format_sends_the_document_image_and_tool_turns_to_the_count_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: Final[list[dict[str, object]]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize."},
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBE"}},
                {"type": "image", "source": {"type": "url", "url": "https://example.com/page.png"}},
            ],
        },
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01", "name": "lookup", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "found"}]},
    ]

    sent: Final = await _sent_count_body(
        monkeypatch, messages, [{"type": "text", "text": "Be brief."}], messages_format="anthropic"
    )

    assert sent == {
        "model": "gpt-4o",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Summarize."},
                    {"type": "input_file", "filename": "document.pdf", "file_data": "data:application/pdf;base64,JVBE"},
                    {"type": "input_image", "image_url": "https://example.com/page.png"},
                ],
            },
            {"type": "function_call", "call_id": "toolu_01", "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "toolu_01", "output": "found"},
        ],
        "instructions": "Be brief.",
    }


@pytest.mark.asyncio
async def test_openai_format_keeps_the_chat_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    messages: Final[list[dict[str, object]]] = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": [{"type": "text", "text": "Summarize."}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "found"},
    ]

    sent: Final = await _sent_count_body(monkeypatch, messages, system=None, messages_format="openai")

    assert sent == {
        "model": "gpt-4o",
        "input": [
            {"role": "user", "content": "Summarize."},
            {"type": "function_call_output", "call_id": "call_1", "output": "found"},
        ],
        "instructions": "Be brief.",
    }


@pytest.mark.asyncio
async def test_anthropic_format_sends_affinity_wrapped_reasoning_unwrapped_as_inference_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapped: Final = ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id("gAAAAoriginal", "deployment-1")  # pyright: ignore[reportPrivateUsage]  # the wrap inference applies
    messages: Final[list[dict[str, object]]] = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "redacted_thinking", "data": f"litellm_encrypted_reasoning:{wrapped}"},
                {"type": "text", "text": "OK"},
            ],
        },
        {"role": "user", "content": "again"},
    ]

    sent: Final = await _sent_count_body(monkeypatch, messages, system=None, messages_format="anthropic")

    assert sent == {
        "model": "gpt-4o",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "reasoning", "summary": [], "encrypted_content": "gAAAAoriginal"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "OK"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "again"}]},
        ],
    }


@pytest.mark.asyncio
async def test_anthropic_format_counts_the_structured_output_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    schema: Final[dict[str, JsonValue]] = {"type": "object", "properties": {"answer": {"type": "string"}}}

    sent: Final = await _sent_count_body(
        monkeypatch,
        [{"role": "user", "content": "Answer in JSON."}],
        system=None,
        messages_format="anthropic",
        output_format={"type": "json_schema", "schema": schema},
    )

    assert sent == {
        "model": "gpt-4o",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Answer in JSON."}]}],
        "text": {"format": {"type": "json_schema", "name": "structured_output", "schema": schema, "strict": False}},
    }


@pytest.mark.asyncio
async def test_anthropic_format_sends_a_replayed_web_search_result_as_its_text(monkeypatch: pytest.MonkeyPatch) -> None:
    snippet: Final = "LiteLLM ships a weekly release candidate and a stable line."
    messages: Final[list[dict[str, object]]] = [
        {"role": "user", "content": "What is new?"},
        {
            "role": "assistant",
            "content": [
                {"type": "server_tool_use", "id": "srvtoolu_01", "name": "web_search", "input": {"query": "news"}},
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_01",
                    "content": [
                        {
                            "type": "web_search_result",
                            "url": "https://docs.litellm.ai/release_notes",
                            "title": "Release notes",
                            "encrypted_content": "",
                            "snippet": snippet,
                        }
                    ],
                },
            ],
        },
        {"role": "user", "content": "Thanks."},
    ]

    sent: Final = json.dumps(await _sent_count_body(monkeypatch, messages, system=None, messages_format="anthropic"))

    assert snippet in sent
    assert "https://docs.litellm.ai/release_notes" in sent


@pytest.mark.asyncio
async def test_anthropic_format_message_without_a_role_answers_400_without_calling_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with respx.mock:
        route: Final = respx.post(COUNT_URL).mock(return_value=httpx.Response(200, json={"input_tokens": 1}))
        result: Final = await OpenAITokenCounter().count_tokens(
            model_to_use="gpt-4o",
            messages=[{"role": "user", "content": "hi"}, {"content": "no role"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "sk-test"}},
            request_model="openai/gpt-4o",
            messages_format="anthropic",
        )

    assert route.called is False
    assert result is not None
    assert (result.error, result.status_code, result.error_message) == (True, 400, "messages.1.role: Field required")


@pytest.mark.asyncio
async def test_openai_format_sends_a_list_system_as_its_text(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: Final = await _sent_count_body(
        monkeypatch,
        [{"role": "user", "content": "hi"}],
        system=[{"type": "text", "text": "Be brief."}, {"type": "text", "text": "Answer in English."}],
        messages_format="openai",
    )

    assert sent == {
        "model": "gpt-4o",
        "input": [{"role": "user", "content": "hi"}],
        "instructions": "Be brief.\nAnswer in English.",
    }


@pytest.mark.asyncio
async def test_undecodable_count_body_becomes_a_500_error_response_carrying_the_decoder_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(json.JSONDecodeError) as undecodable:
        json.loads("not json")

    result: Final = await _count(monkeypatch, httpx.Response(200, text="not json"))

    assert result is not None
    assert repr(result.model_dump()) == repr(
        {
            "total_tokens": 0,
            "request_model": "openai/gpt-4o",
            "model_used": "gpt-4o",
            "tokenizer_type": "openai_api",
            "count_source": None,
            "original_response": None,
            "error": True,
            "error_message": f"CountTokens processing error: {undecodable.value}",
            "status_code": 500,
        }
    )


@pytest.mark.asyncio
async def test_count_uses_the_global_openai_key_chat_would_use(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "openai_key", "sk-from-litellm-settings")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with respx.mock:
        route: Final = respx.post(COUNT_URL).mock(return_value=httpx.Response(200, json={"input_tokens": 9}))
        result: Final = await OpenAITokenCounter().count_tokens(
            model_to_use="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment={"litellm_params": {}},
            request_model="openai/gpt-4o",
        )
    assert result is not None
    assert result.total_tokens == 9
    assert route.calls.last.request.headers["authorization"] == "Bearer sk-from-litellm-settings"


@pytest.mark.asyncio
async def test_count_sends_the_global_key_only_to_the_global_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "openai_key", "sk-private-gateway")
    monkeypatch.setattr(litellm, "api_base", "https://gateway.internal.example/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with respx.mock:
        route: Final = respx.post("https://gateway.internal.example/v1/responses/input_tokens").mock(
            return_value=httpx.Response(200, json={"input_tokens": 4})
        )
        result: Final = await OpenAITokenCounter().count_tokens(
            model_to_use="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment={"litellm_params": {}},
            request_model="openai/gpt-4o",
        )
    assert result is not None
    assert result.total_tokens == 4
    assert route.calls.last.request.headers["authorization"] == "Bearer sk-private-gateway"

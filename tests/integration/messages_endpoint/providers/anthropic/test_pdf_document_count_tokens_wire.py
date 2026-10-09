import io
import json
import uuid
from collections.abc import Sequence
from types import MappingProxyType
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.pdf_document import (
    COUNT_REFUSED,
    COUNT_TOKENS_TARGET,
    LETTER,
    NARROW,
    Page,
    base64_source,
    chat_file,
    document,
    encoded,
    messages_body,
    pdf_bytes,
    pdf_data_url,
    pdf_document,
    rendered_tokens,
    responses_input_file,
    text_document,
)
from integration._support.provider import SharedProvider
from integration._support.wire import Reply
from pydantic import JsonValue, TypeAdapter
from pypdf import PdfReader, PdfWriter

_MODEL: Final = "anthropic/claude-opus-5-5"
_ASK: Final = "Summarize the attached report in one sentence."
_TEXT: Final = "The quarterly report covers revenue, margins and headcount."
_TITLE: Final = "Quarterly report"
_CONTEXT: Final = "Board pack, page one"
_PEER_INPUT_TOKENS: Final = 12
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_PNG_HEADER: Final = b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR" + (1568).to_bytes(4, "big") * 2 + b"\x08\x06\x00\x00\x00"


def _locked(raw: bytes, user_password: str) -> bytes:
    writer: Final = PdfWriter(clone_from=PdfReader(io.BytesIO(raw)))
    writer.encrypt(user_password=user_password, owner_password="owner", algorithm="RC4-128")
    out: Final = io.BytesIO()
    writer.write(out)
    return out.getvalue()


_UNREADABLE: Final[MappingProxyType[str, JsonValue]] = MappingProxyType(
    {
        "garbage-5kb": encoded(bytes(range(256)) * 20),
        "cut-mid-stream": encoded(pdf_bytes((LETTER, LETTER))[:200]),
        "png-bytes": encoded(_PNG_HEADER),
        "user-password": encoded(_locked(pdf_bytes((LETTER, LETTER)), "reader")),
        "empty": "",
        "int": 1234,
    }
)


def _int(value: JsonValue) -> int:
    assert isinstance(value, int), value
    return value


def _turn(blocks: Sequence[JsonValue]) -> list[JsonValue]:
    return messages_body(_MODEL, blocks, _ASK)["messages"]


def _count(gateway: Gateway, provider: SharedProvider, blocks: Sequence[JsonValue]) -> int:
    provider.expect(COUNT_REFUSED)
    response: Final = gateway.request("POST", "/v1/messages/count_tokens", {"model": _MODEL, "messages": _turn(blocks)})
    assert response.status_code == 200, response.text
    assert [(request.method, request.target) for request in provider.received()] == [("POST", COUNT_TOKENS_TARGET)]
    return _int(_JSON_OBJECT.validate_json(response.content)["input_tokens"])


def _local(gateway: Gateway, parts: Sequence[JsonValue]) -> int:
    response: Final = gateway.request(
        "POST", "/utils/token_counter", {"model": _MODEL, "messages": _turn(parts)}, params={"call_endpoint": "false"}
    )
    assert response.status_code == 200, response.text
    return _int(_JSON_OBJECT.validate_json(response.content)["total_tokens"])


def _responses_count(gateway: Gateway, provider: SharedProvider, items: Sequence[JsonValue]) -> int:
    provider.expect(COUNT_REFUSED)
    response: Final = gateway.request(
        "POST",
        "/v1/responses/input_tokens",
        {"model": _MODEL, "input": [{"role": "user", "content": [*items, {"type": "input_text", "text": _ASK}]}]},
    )
    assert response.status_code == 200, response.text
    assert [(request.method, request.target) for request in provider.received()] == [("POST", COUNT_TOKENS_TARGET)]
    payload: Final = _JSON_OBJECT.validate_json(response.content)
    assert payload["object"] == "response.input_tokens", response.text
    return _int(payload["input_tokens"])


def _cost(gateway: Gateway, parts: Sequence[JsonValue]) -> float:
    payload: Final = gateway.post("/spend/calculate", {"model": _MODEL, "messages": _turn(parts)})
    cost: Final = payload["cost"]
    assert isinstance(cost, (int, float)), payload
    return float(cost)


def _input_price(gateway: Gateway) -> float:
    listed: Final = gateway.get("/model/info")["data"]
    assert isinstance(listed, list), listed
    rows: Final = [object_value(row) for row in listed if isinstance(row, dict) and row.get("model_name") == _MODEL]
    assert len(rows) == 1, rows
    price: Final = object_value(rows[0]["model_info"])["input_cost_per_token"]
    assert isinstance(price, float) and price > 0, price
    return price


def _peer_message(identity: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": "One sentence."}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": _PEER_INPUT_TOKENS, "output_tokens": 3},
            }
        ).encode()
    )


def _prompt_tokens(call_id: str) -> int:
    rows: Final = eventually(
        lambda: read_rows('SELECT prompt_tokens FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s', (call_id,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return _int(rows[0]["prompt_tokens"])


@pytest.mark.parametrize("pages", [1, 3, 12])
def test_count_tokens_fallback_prices_each_blank_page_as_a_rendered_image(
    gateway: Gateway, provider: SharedProvider, pages: int
) -> None:
    letter_pages: Final = (LETTER,) * pages
    with_document: Final = _count(gateway, provider, [pdf_document(letter_pages)])
    without: Final = _count(gateway, provider, [])
    assert with_document - without == rendered_tokens(letter_pages), (with_document, without)


def test_count_tokens_fallback_adds_a_page_text_to_its_rendered_image(
    gateway: Gateway, provider: SharedProvider
) -> None:
    text_page: Final = _count(gateway, provider, [pdf_document((Page(text=_TEXT),))])
    blank_page: Final = _count(gateway, provider, [pdf_document((LETTER,))])
    text_source: Final = _count(gateway, provider, [text_document(_TEXT)])
    without: Final = _count(gateway, provider, [])
    assert text_source - without > 0, (text_source, without)
    assert text_page - blank_page == text_source - without, (text_page, blank_page, text_source, without)


def test_count_tokens_fallback_prices_a_page_by_its_rendered_area(gateway: Gateway, provider: SharedProvider) -> None:
    mixed: Final = (LETTER, NARROW, LETTER)
    without: Final = _count(gateway, provider, [])
    narrow: Final = _count(gateway, provider, [pdf_document((NARROW,))])
    assert narrow - without == rendered_tokens((NARROW,)), (narrow, without)
    assert _count(gateway, provider, [pdf_document(mixed)]) - without == rendered_tokens(mixed)


def test_count_tokens_fallback_prices_every_document_in_the_turn(gateway: Gateway, provider: SharedProvider) -> None:
    both: Final = _count(gateway, provider, [pdf_document((LETTER,) * 3), pdf_document((LETTER,))])
    without: Final = _count(gateway, provider, [])
    assert both - without == rendered_tokens((LETTER,) * 4), (both, without)


def test_count_tokens_fallback_prices_a_pdf_without_pages_as_nothing(
    gateway: Gateway, provider: SharedProvider
) -> None:
    assert _count(gateway, provider, [pdf_document(())]) == _count(gateway, provider, [])


def test_count_tokens_fallback_adds_title_and_context_once_per_document(
    gateway: Gateway, provider: SharedProvider
) -> None:
    without: Final = _count(gateway, provider, [])
    plain: Final = _count(gateway, provider, [pdf_document((LETTER,))])
    annotated: Final = _count(gateway, provider, [pdf_document((LETTER,), title=_TITLE, context=_CONTEXT)])
    title: Final = _count(gateway, provider, [text_document(_TITLE)]) - without
    context: Final = _count(gateway, provider, [text_document(_CONTEXT)]) - without
    assert title > 0 and context > 0, (title, context)
    assert annotated - plain == title + context, (annotated, plain, title, context)


@pytest.mark.parametrize("label", list(_UNREADABLE))
def test_count_tokens_fallback_prices_unreadable_pdf_bytes_like_one_image(
    gateway: Gateway, provider: SharedProvider, label: str
) -> None:
    data: Final = _UNREADABLE[label]
    as_pdf: Final = _count(gateway, provider, [document(base64_source(data))])
    as_png: Final = _count(gateway, provider, [document(base64_source(data, media_type="image/png"))])
    without: Final = _count(gateway, provider, [])
    assert as_pdf == as_png, (as_pdf, as_png)
    assert 0 <= as_pdf - without < rendered_tokens((LETTER,)), (as_pdf, without)


def test_count_tokens_fallback_reads_a_list_wrapped_base64_string_like_the_bare_string(
    gateway: Gateway, provider: SharedProvider
) -> None:
    raw: Final = encoded(pdf_bytes((LETTER,)))
    wrapped: Final = _count(gateway, provider, [document(base64_source([raw]))])
    assert wrapped == _count(gateway, provider, [document(base64_source(raw))]), wrapped


def test_count_tokens_fallback_reads_an_owner_locked_pdf(gateway: Gateway, provider: SharedProvider) -> None:
    pages: Final = (LETTER, LETTER)
    locked: Final = document(base64_source(encoded(_locked(pdf_bytes(pages), ""))))
    assert _count(gateway, provider, [locked]) - _count(gateway, provider, []) == rendered_tokens(pages)


def test_count_tokens_fallback_parses_only_the_pdf_media_type(gateway: Gateway, provider: SharedProvider) -> None:
    raw: Final = encoded(pdf_bytes((LETTER,)))
    without: Final = _count(gateway, provider, [])
    labelled: Final = _count(gateway, provider, [document(base64_source(raw))])
    assert labelled - without == rendered_tokens((LETTER,)), (labelled, without)
    mislabelled: Final = _count(gateway, provider, [document(base64_source(raw, media_type="application/x-pdf"))])
    assert mislabelled == _count(gateway, provider, [document(base64_source(raw, media_type="image/png"))])


def test_utils_token_counter_prices_a_chat_file_by_its_pages(gateway: Gateway) -> None:
    one: Final = _local(gateway, [chat_file(pdf_data_url((LETTER,)))])
    three: Final = _local(gateway, [chat_file(pdf_data_url((LETTER,) * 3))])
    assert three - one == rendered_tokens((LETTER,) * 2), (three, one)
    text_page: Final = _local(gateway, [chat_file(pdf_data_url((Page(text=_TEXT),)))])
    text_part: Final = _local(gateway, [chat_file(pdf_data_url((LETTER,))), {"type": "text", "text": _TEXT}])
    assert text_part - one > 0, (text_part, one)
    assert text_page - one == text_part - one, (text_page, text_part, one)


def test_responses_input_tokens_fallback_prices_an_input_file_by_its_pages(
    gateway: Gateway, provider: SharedProvider
) -> None:
    one: Final = _responses_count(gateway, provider, [responses_input_file(pdf_data_url((LETTER,)))])
    three: Final = _responses_count(gateway, provider, [responses_input_file(pdf_data_url((LETTER,) * 3))])
    assert three - one == rendered_tokens((LETTER,) * 2), (three, one)


def test_spend_calculate_prices_a_chat_file_by_its_pages(gateway: Gateway) -> None:
    twelve: Final = _cost(gateway, [chat_file(pdf_data_url((LETTER,) * 12))])
    one: Final = _cost(gateway, [chat_file(pdf_data_url((LETTER,)))])
    assert twelve - one == pytest.approx(rendered_tokens((LETTER,) * 11) * _input_price(gateway)), (twelve, one)


def test_a_cached_pdf_message_keeps_the_peer_usage_on_both_spend_rows(
    gateway: Gateway, provider: SharedProvider
) -> None:
    marker: Final = uuid.uuid4().hex
    body: Final = messages_body(_MODEL, [pdf_document((LETTER,))], f"{_ASK} marker-{marker}")
    provider.expect(_peer_message(f"msg_{marker}"))
    first: Final = gateway.request("POST", "/v1/messages", body)
    second: Final = gateway.request("POST", "/v1/messages", body)
    assert (first.status_code, second.status_code) == (200, 200), (first.text, second.text)
    assert [(request.method, request.target) for request in provider.received()] == [("POST", "/v1/messages")]
    assert _JSON_OBJECT.validate_json(second.content)["id"] == f"msg_{marker}", second.text
    assert first.headers["x-litellm-call-id"] != second.headers["x-litellm-call-id"]
    assert [_prompt_tokens(response.headers["x-litellm-call-id"]) for response in (first, second)] == [
        _PEER_INPUT_TOKENS,
        _PEER_INPUT_TOKENS,
    ]

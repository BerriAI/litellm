import json
from typing import Final
from urllib.parse import urlsplit

import anthropic
import openai
import pytest
from integration._support.client import Gateway, Scenario, eventually, list_value, object_value, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from integration.providers._api_base_trailing_slash_support import (
    KEY,
    TRAILING,
    TRAILING_IDS,
    VECTOR,
    Row,
    assert_control_answers,
    assert_one_request,
    body_of,
    first_item,
    new_marker,
    reply_text,
    request_json,
)
from integration.providers._api_base_url_join_support import COUNTED_TOKENS, RELEVANCE, join_peer
from pydantic import JsonValue

_HAIKU: Final = "anthropic.claude-haiku-5-5"
_ANTHROPIC: Final = Row("anthropic/claude-haiku-5-5", "/an", "/v1/messages")
_GEMINI: Final = Row("gemini/gemini-3.8-flash", "/gm/v1beta", "/models/gemini-3.8-flash:generateContent")
_MISTRAL: Final = Row("mistral/mistral-small-latest", "/mi/v1", "/chat/completions")
_BEDROCK_INVOKE: Final = Row(f"bedrock/invoke/{_HAIKU}", "/bi", f"/model/{_HAIKU}/invoke")
_CHAT_ROWS: Final = (
    _ANTHROPIC,
    _GEMINI,
    _MISTRAL,
    Row("databricks/databricks-meta-llama-3-3-70b-instruct", "/db/serving-endpoints", "/chat/completions"),
    _BEDROCK_INVOKE,
    Row(f"bedrock/converse/{_HAIKU}", "/bc", f"/model/{_HAIKU}/converse"),
)
_STREAM_ROWS: Final = (
    _ANTHROPIC,
    Row("gemini/gemini-3.8-flash", "/gm/v1beta", "/models/gemini-3.8-flash:streamGenerateContent?alt=sse"),
    _MISTRAL,
    Row(f"bedrock/converse/{_HAIKU}", "/bc", f"/model/{_HAIKU}/converse-stream"),
)
_MESSAGES_ROWS: Final = (_ANTHROPIC, _BEDROCK_INVOKE, _GEMINI, _MISTRAL)
_RESPONSES_ROWS: Final = (_ANTHROPIC, _GEMINI)
_EMBEDDING_ROWS: Final = (
    Row("mistral/mistral-embed", "/mi/v1", "/embeddings"),
    Row("gemini/gemini-embedding-2", "/gm/v1beta", "/models/gemini-embedding-2:batchEmbedContents"),
)
_TRANSCRIPTION: Final = Row("mistral/voxtral-mini-latest", "/mi/v1", "/audio/transcriptions")
_FULL_ROUTE_SUFFIXES: Final = ("", "/", "?beta=true")
_FULL_ROUTE_IDS: Final = ("exact", "slash", "query")


def _ids(rows: tuple[Row, ...]) -> tuple[str, ...]:
    return tuple(row.model for row in rows)


def _deploy(scenario: Scenario, row: Row, api_base: str) -> str:
    if row.model.startswith("bedrock/"):
        return scenario.model(model=row.model, api_base=api_base, api_key=KEY, aws_region_name="us-east-1")
    return scenario.model(model=row.model, api_base=api_base, api_key=KEY)


def _openai(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _anthropic(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)


def _chat(gateway: Gateway, model: str, marker: str) -> str | None:
    completion: Final = _openai(gateway).chat.completions.create(
        model=model, messages=[{"role": "user", "content": marker}]
    )
    return completion.choices[0].message.content


def _message_text(gateway: Gateway, model: str, marker: str) -> list[str]:
    message: Final = _anthropic(gateway).messages.create(
        model=model, max_tokens=16, messages=[{"role": "user", "content": marker}]
    )
    return [block.text for block in message.content if block.type == "text"]


def _spend_rows(pattern: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT request_id, api_base, cache_hit FROM "LiteLLM_SpendLogs" WHERE request_id LIKE %s', (pattern,)
    )


def _one_spend_row(pattern: str) -> dict[str, JsonValue]:
    (row,) = eventually(lambda: _spend_rows(pattern), lambda rows: len(rows) == 1, seconds=70)
    return row


def _gemini_contents(marker: str) -> dict[str, JsonValue]:
    return {"contents": [{"role": "user", "parts": [{"text": marker}]}]}


def _gemini_text(payload: dict[str, JsonValue]) -> JsonValue:
    content: Final = object_value(first_item(payload["candidates"])["content"])
    return first_item(content["parts"])["text"]


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
@pytest.mark.parametrize("row", _CHAT_ROWS, ids=_ids(_CHAT_ROWS))
def test_chat_completion_reaches_one_single_slash_url(gateway: Gateway, row: Row, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(join_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, row, f"{wire.url}{row.base}{trailing}")
        assert _chat(gateway, model, marker) == reply_text(marker)
        assert_one_request(wire.drain(), row.target, marker)


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
@pytest.mark.parametrize("row", _STREAM_ROWS, ids=_ids(_STREAM_ROWS))
def test_streamed_chat_reaches_one_single_slash_url(gateway: Gateway, row: Row, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(join_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, row, f"{wire.url}{row.base}{trailing}")
        stream: Final = _openai(gateway).chat.completions.create(
            model=model, messages=[{"role": "user", "content": marker}], stream=True
        )
        text: Final = "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
        assert text == reply_text(marker)
        assert_one_request(wire.drain(), row.target, marker)


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
@pytest.mark.parametrize("row", _MESSAGES_ROWS, ids=_ids(_MESSAGES_ROWS))
def test_messages_reaches_one_single_slash_url(gateway: Gateway, row: Row, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(join_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, row, f"{wire.url}{row.base}{trailing}")
        assert _message_text(gateway, model, marker) == [reply_text(marker)]
        assert_one_request(wire.drain(), row.target, marker)


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
def test_streamed_messages_reach_one_single_slash_url(gateway: Gateway, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(join_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, _ANTHROPIC, f"{wire.url}/an{trailing}")
        with _anthropic(gateway).messages.stream(
            model=model, max_tokens=16, messages=[{"role": "user", "content": marker}]
        ) as stream:
            text: Final = "".join(stream.text_stream)
        assert text == reply_text(marker)
        assert request_json(assert_one_request(wire.drain(), "/an/v1/messages", marker))["stream"] is True


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
@pytest.mark.parametrize("row", _RESPONSES_ROWS, ids=_ids(_RESPONSES_ROWS))
def test_responses_bridge_reaches_one_single_slash_url(gateway: Gateway, row: Row, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(join_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, row, f"{wire.url}{row.base}{trailing}")
        assert _openai(gateway).responses.create(model=model, input=marker).output_text == reply_text(marker)
        assert_one_request(wire.drain(), row.target, marker)


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
@pytest.mark.parametrize("row", _EMBEDDING_ROWS, ids=_ids(_EMBEDDING_ROWS))
def test_embedding_reaches_one_single_slash_url(gateway: Gateway, row: Row, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(join_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, row, f"{wire.url}{row.base}{trailing}")
        response: Final = gateway.request("POST", "/v1/embeddings", {"model": model, "input": marker})
        assert first_item(body_of(response)["data"])["embedding"] == list(VECTOR)
        assert_one_request(wire.drain(), row.target, marker)


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
def test_mistral_transcription_reaches_one_single_slash_url(gateway: Gateway, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(join_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, _TRANSCRIPTION, f"{wire.url}/mi/v1{trailing}")
        response: Final = gateway.request_multipart(
            "/v1/audio/transcriptions",
            {"model": model},
            {"file": ("speech.wav", f"RIFF {marker}".encode(), "audio/wav")},
        )
        assert body_of(response)["text"] == reply_text(marker)
        assert_one_request(wire.drain(), _TRANSCRIPTION.target, marker)


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
def test_google_genai_generate_content_reaches_one_single_slash_url(gateway: Gateway, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(join_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, _GEMINI, f"{wire.url}/gm/v1beta{trailing}")
        response: Final = gateway.request(
            "POST", f"/v1beta/models/{model}:generateContent", _gemini_contents(marker)
        )
        assert _gemini_text(body_of(response)) == reply_text(marker)
        assert_one_request(wire.drain(), _GEMINI.target, marker)


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
def test_google_genai_stream_generate_content_reaches_one_single_slash_url(gateway: Gateway, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(join_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, _GEMINI, f"{wire.url}/gm/v1beta{trailing}")
        response: Final = gateway.request(
            "POST",
            f"/v1beta/models/{model}:streamGenerateContent",
            _gemini_contents(marker),
            params={"alt": "sse"},
        )
        assert response.status_code == 200, response.text
        assert marker in response.text, response.text
        assert_one_request(
            wire.drain(), "/gm/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse", marker
        )


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
def test_count_tokens_reaches_one_single_slash_url(gateway: Gateway, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(join_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, _ANTHROPIC, f"{wire.url}/an{trailing}")
        response: Final = gateway.request(
            "POST", "/v1/messages/count_tokens", {"model": model, "messages": [{"role": "user", "content": marker}]}
        )
        assert body_of(response)["input_tokens"] == COUNTED_TOKENS
        assert_one_request(wire.drain(), "/an/v1/messages/count_tokens", marker)


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
def test_health_check_reaches_one_single_slash_url(gateway: Gateway, trailing: str) -> None:
    with wire_server(join_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, _ANTHROPIC, f"{wire.url}/an{trailing}")
        report: Final = body_of(gateway.request("GET", "/health", params={"model": model}))
        assert (report["healthy_count"], report["unhealthy_count"]) == (1, 0), report
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/an/v1/messages")]


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
def test_spend_log_records_the_single_slash_url_it_called(gateway: Gateway, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(join_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, _ANTHROPIC, f"{wire.url}/an{trailing}")
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": marker}]}
        )
        identity: Final = string_value(body_of(response)["id"])
        assert_one_request(wire.drain(), "/an/v1/messages", marker)
        assert _one_spend_row(identity)["api_base"] == f"{wire.url}/an/v1/messages"


def test_cache_hit_on_a_slash_base_skips_the_upstream_and_logs_its_own_row(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(join_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, _ANTHROPIC, f"{wire.url}/an/")
        body: Final[dict[str, JsonValue]] = {"model": model, "messages": [{"role": "user", "content": marker}]}
        first: Final = body_of(gateway.request("POST", "/v1/chat/completions", body))
        second: Final = body_of(gateway.request("POST", "/v1/chat/completions", body))
        assert first_item(second["choices"])["message"] == first_item(first["choices"])["message"]
        assert_one_request(wire.drain(), "/an/v1/messages", marker)
        identity: Final = string_value(first["id"])
        assert _one_spend_row(identity)["api_base"] == f"{wire.url}/an/v1/messages"
        hit: Final = _one_spend_row(f"{identity}_cache_hit%")
        assert hit["cache_hit"] == "True", hit
        assert "//" not in urlsplit(string_value(hit["api_base"])).path, hit


@pytest.mark.parametrize("suffix", _FULL_ROUTE_SUFFIXES, ids=_FULL_ROUTE_IDS)
def test_chat_base_that_already_names_the_route_keeps_its_path_and_query(gateway: Gateway, suffix: str) -> None:
    marker: Final = new_marker()
    with wire_server(join_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, _ANTHROPIC, f"{wire.url}/an/v1/messages{suffix}")
        assert _chat(gateway, model, marker) == reply_text(marker)
        assert_one_request(wire.drain(), f"/an/v1/messages{suffix.rstrip('/')}", marker)


@pytest.mark.parametrize("suffix", _FULL_ROUTE_SUFFIXES, ids=_FULL_ROUTE_IDS)
def test_messages_base_that_already_names_the_route_keeps_its_path_and_query(gateway: Gateway, suffix: str) -> None:
    marker: Final = new_marker()
    with wire_server(join_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, _ANTHROPIC, f"{wire.url}/an/v1/messages{suffix}")
        assert _message_text(gateway, model, marker) == [reply_text(marker)]
        assert_one_request(wire.drain(), f"/an/v1/messages{suffix.rstrip('/')}", marker)


@pytest.mark.parametrize("suffix", ("", "/", "?tier=a"), ids=("exact", "slash", "query"))
def test_rerank_base_that_already_names_the_route_keeps_its_path_and_query(gateway: Gateway, suffix: str) -> None:
    marker: Final = new_marker()
    with wire_server(join_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, Row("cohere/rerank-v3.5", "/co", ""), f"{wire.url}/co/v2/rerank{suffix}")
        response: Final = gateway.request(
            "POST", "/v1/rerank", {"model": model, "query": marker, "documents": ["only document"]}
        )
        assert [object_value(result)["relevance_score"] for result in list_value(body_of(response)["results"])] == [
            RELEVANCE
        ]
        assert_one_request(wire.drain(), f"/co/v2/rerank{suffix.rstrip('/')}", marker)


@pytest.mark.parametrize(
    ("row", "trailing"),
    (
        (_ANTHROPIC, "//"),
        (Row("anthropic/claude-haiku-5-5", f"/an/{'p' * 5000}", "/v1/messages"), "/"),
        (_GEMINI, "//"),
        (Row("gemini/gemini-3.8-flash", f"/gm/{'p' * 5000}", "/models/gemini-3.8-flash:generateContent"), "/"),
    ),
    ids=("anthropic-two-slashes", "anthropic-5kb-prefix", "gemini-two-slashes", "gemini-5kb-prefix"),
)
def test_base_with_repeated_slashes_or_a_long_prefix_reaches_one_url(gateway: Gateway, row: Row, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(join_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, row, f"{wire.url}{row.base}{trailing}")
        assert _chat(gateway, model, marker) == reply_text(marker)
        assert_one_request(wire.drain(), row.target, marker)


def _anthropic_error(status: int, marker: str) -> bytes:
    return json.dumps(
        {"type": "error", "error": {"type": "authentication_error", "message": f"scripted {status} {marker}"}}
    ).encode()


def _gemini_error(status: int, marker: str) -> bytes:
    error: Final = {"code": status, "message": f"scripted {status} {marker}", "status": "NOT_FOUND"}
    return json.dumps({"error": error}).encode()


@pytest.mark.parametrize(
    ("row", "status"),
    ((_ANTHROPIC, 401), (_GEMINI, 404)),
    ids=("anthropic-401", "gemini-404"),
)
def test_upstream_error_on_a_slash_base_reaches_the_caller_from_one_url(
    gateway: Gateway, row: Row, status: int
) -> None:
    marker: Final = new_marker()
    error: Final = _anthropic_error(status, marker) if row is _ANTHROPIC else _gemini_error(status, marker)

    def respond(_: Request) -> Reply:
        return Reply(status=status, body=error)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, row, f"{wire.url}{row.base}/")
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": marker}]}
        )
        assert response.status_code == status, response.text
        assert marker in response.text, response.text
        assert_one_request(wire.drain(), row.target, marker)
        assert_control_answers(gateway, scenario)


def test_streamed_chat_upstream_error_on_a_slash_base_reaches_the_caller(gateway: Gateway) -> None:
    marker: Final = new_marker()

    def respond(_: Request) -> Reply:
        return Reply(status=401, body=_anthropic_error(401, marker))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, _ANTHROPIC, f"{wire.url}/an/")
        with pytest.raises(openai.AuthenticationError) as raised:
            _openai(gateway).chat.completions.create(
                model=model, messages=[{"role": "user", "content": marker}], stream=True
            )
        assert marker in str(raised.value)
        assert_one_request(wire.drain(), "/an/v1/messages", marker)
        assert_control_answers(gateway, scenario)

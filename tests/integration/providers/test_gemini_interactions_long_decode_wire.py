import json
from collections.abc import Callable
from typing import Final
from uuid import uuid4

import pytest
from integration._support.client import (  # pyright: ignore[reportUnknownVariableType]  # integration fixture module
    Gateway,
    eventually,
)
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_MODEL: Final = "gemini/gemini-4-neon"
_KEY: Final = "synthetic-gemini-interactions-long-decode-key"
_JSON_OBJECT: Final[TypeAdapter[dict[str, JsonValue]]] = TypeAdapter(dict[str, JsonValue])
_INPUT_COST: Final = 0.001
_OUTPUT_COST: Final = 0.002
_FIRST_HOP: Final = {
    "id": "v1_long_decode_first_hop",
    "object": "interaction",
    "status": "requires_continuation",
    "continuation_token": "token-after-first-hop",
    "steps": [{"type": "model_output", "content": [{"type": "text", "text": "The first half, "}]}],
    "usage": {"total_input_tokens": 10, "total_output_tokens": 100, "total_tokens": 110},
}
_SECOND_HOP: Final = {
    "id": "v1_long_decode_second_hop",
    "object": "interaction",
    "status": "completed",
    "steps": [{"type": "model_output", "content": [{"type": "text", "text": "then the second half."}]}],
    "usage": {"total_input_tokens": 10, "total_output_tokens": 50, "total_tokens": 60},
}
_FIRST_REQUEST: Final = {
    "model": "gemini-4-neon",
    "input": [{"type": "user_input", "content": [{"type": "text", "text": "write it all"}]}],
    "store": False,
    "generation_config": {"max_output_tokens": 1000},
}


def _spend_of(request_id: str) -> list[dict[str, JsonValue]]:  # mutable-ok: read_rows return contract
    return read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (request_id,))


def _single_row(rows: list[dict[str, JsonValue]]) -> bool:  # mutable-ok: predicate over read_rows list
    return len(rows) == 1


def _fresh_answer_id() -> str:
    return f"v1_long_decode_{uuid4().hex}"


def _scripted(answer_id: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.target == "/v1beta/interactions"
        hop: Final = (
            _SECOND_HOP if "continuation_token" in json.loads(request.body) else {**_FIRST_HOP, "id": answer_id}
        )
        return Reply(body=json.dumps(hop).encode())

    return respond


def test_gemini_4_chat_answer_google_ends_early_is_continued_into_one_answer_billed_once(gateway: Gateway) -> None:
    answer_id: Final = _fresh_answer_id()
    with wire_server(_scripted(answer_id)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_MODEL,
            api_base=wire.url,
            api_key=_KEY,
            input_cost_per_token=_INPUT_COST,
            output_cost_per_token=_OUTPUT_COST,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "write it all"}], "max_tokens": 1000},
            key=scenario.key(models=[model]),
        )
        assert response.status_code == 200, response.text
        first_body, second_body = (json.loads(wire.received.get(timeout=5).body) for _ in range(2))
        assert first_body == _FIRST_REQUEST
        assert second_body == {**first_body, "continuation_token": "token-after-first-hop"}
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        choice: Final = payload["choices"][0]  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
        assert choice["message"]["content"] == "The first half, then the second half."  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
        assert choice["finish_reason"] == "stop"  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
        assert payload["id"] == answer_id
        assert payload["usage"]["completion_tokens"] == 150  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
        billed: Final = float(response.headers["x-litellm-response-cost"])
        assert billed == pytest.approx(20 * _INPUT_COST + 150 * _OUTPUT_COST)  # pyright: ignore[reportUnknownMemberType]  # pytest.approx untyped overload
        rows: Final[list[dict[str, JsonValue]]] = eventually(  # mutable-ok: read_rows return contract
            lambda: _spend_of(answer_id), _single_row, seconds=70
        )
        spend: Final = rows[0]["spend"]
        assert isinstance(spend, (int, float, str))
        assert float(spend) == pytest.approx(billed), rows  # pyright: ignore[reportUnknownMemberType]  # pytest.approx untyped overload


def test_gemini_4_chat_answer_with_no_output_cap_is_not_continued_and_returns_its_token(gateway: Gateway) -> None:
    answer_id: Final = _fresh_answer_id()
    with wire_server(_scripted(answer_id)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "write it all"}]},
            key=scenario.key(models=[model]),
        )
        assert response.status_code == 200, response.text
        assert json.loads(wire.received.get(timeout=5).body) == {
            key: value for key, value in _FIRST_REQUEST.items() if key != "generation_config"
        }
        assert wire.received.empty()
        choice: Final = _JSON_OBJECT.validate_json(response.content)["choices"][0]  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
        assert choice["finish_reason"] == "length"  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
        assert choice["message"]["content"] == "The first half, "  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
        assert choice["message"]["provider_specific_fields"]["continuation_token"] == "token-after-first-hop"  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys


def _hop_events(hop: dict[str, JsonValue]) -> tuple[bytes, ...]:  # mutable-ok: JSON hop fixture passed as a dict
    steps: Final = hop["steps"]
    events: Final = (
        {"event_type": "interaction.created", "interaction": {"id": hop["id"], "status": "in_progress"}},
        {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}},
        {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": steps[0]["content"][0]["text"]}},  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON fixture indexed by fixed keys
        {
            "event_type": "interaction.completed",
            "interaction": {key: value for key, value in hop.items() if key != "steps"},
        },
    )
    return (*(f"data: {json.dumps(event)}\n\n".encode() for event in events), b"data: [DONE]\n\n")


def _scripted_stream(answer_id: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.target == "/v1beta/interactions?alt=sse"
        hop: Final = (
            _SECOND_HOP if "continuation_token" in json.loads(request.body) else {**_FIRST_HOP, "id": answer_id}
        )
        return Reply(content_type="text/event-stream", chunks=_hop_events(hop))

    return respond


def test_gemini_4_chat_stream_google_ends_early_is_continued_under_one_id_with_summed_usage(gateway: Gateway) -> None:
    answer_id: Final = _fresh_answer_id()
    with wire_server(_scripted_stream(answer_id)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "write it all"}],
                "max_tokens": 1000,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            key=scenario.key(models=[model]),
        )
        assert response.status_code == 200, response.text
        chunks: Final = tuple(
            _JSON_OBJECT.validate_json(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        )
        choices: Final = tuple(chunk["choices"][0] for chunk in chunks if chunk.get("choices"))  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
        assert "".join(choice["delta"].get("content") or "" for choice in choices) == (  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType, reportAttributeAccessIssue, reportOptionalMemberAccess, reportUnknownMemberType, reportUnknownArgumentType]  # JSON payload indexed by fixed keys
            "The first half, then the second half."
        )
        assert tuple(choice["finish_reason"] for choice in choices if choice.get("finish_reason")) == ("stop",)  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
        assert {chunk["id"] for chunk in chunks} == {answer_id}
        usages: Final = tuple(chunk["usage"] for chunk in chunks if chunk.get("usage"))
        assert usages[-1]["completion_tokens"] == 150  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
        first_body, second_body = (json.loads(wire.received.get(timeout=5).body) for _ in range(2))
        assert first_body == {**_FIRST_REQUEST, "stream": True}
        assert second_body == {**first_body, "continuation_token": "token-after-first-hop"}
        assert wire.received.empty()


def test_gemini_4_chat_answer_continued_across_hops_cites_each_source_at_its_joined_text(gateway: Gateway) -> None:
    first_text: Final = "Grüße aus 東京, "
    second_text: Final = "dann Zürich."
    hops: Final = tuple(
        {
            **hop,
            "steps": [
                *searched,
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "start_index": 0,
                                    "end_index": len(text.encode()),
                                    "url": url,
                                    "title": url,
                                }
                            ],
                        }
                    ],
                },
            ],
        }
        for hop, text, url, searched in (
            (
                {**_FIRST_HOP, "id": _fresh_answer_id()},
                first_text,
                "https://tokyo.example",
                ({"type": "google_search_call", "id": "search-1", "arguments": {"queries": ["greetings"]}},),
            ),
            (_SECOND_HOP, second_text, "https://zurich.example", ()),
        )
    )

    def respond(request: Request) -> Reply:
        return Reply(body=json.dumps(hops[1 if "continuation_token" in json.loads(request.body) else 0]).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "write it all"}], "max_tokens": 1000},
            key=scenario.key(models=[model]),
        )
        assert response.status_code == 200, response.text
        first_body, second_body = (json.loads(wire.received.get(timeout=5).body) for _ in range(2))
        assert first_body == _FIRST_REQUEST
        assert second_body == {**first_body, "continuation_token": "token-after-first-hop"}
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        message: Final = payload["choices"][0]["message"]  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
        answer: Final = message["content"]  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
        assert answer == first_text + second_text
        cited: Final = {
            (
                answer[cite["url_citation"]["start_index"] : cite["url_citation"]["end_index"]],  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
                cite["url_citation"]["url"],  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
            )
            for cite in message["annotations"]  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType, reportOptionalIterable, reportUnknownVariableType, reportGeneralTypeIssues]  # JSON payload indexed by fixed keys
        }
        assert cited == {(first_text, "https://tokyo.example"), (second_text, "https://zurich.example")}
        grounding: Final = payload["vertex_ai_grounding_metadata"][0]  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]  # JSON payload indexed by fixed keys
        assert {chunk["web"]["uri"] for chunk in grounding["groundingChunks"]} == {  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType, reportOptionalIterable, reportUnknownVariableType, reportGeneralTypeIssues]  # JSON payload indexed by fixed keys
            "https://tokyo.example",
            "https://zurich.example",
        }

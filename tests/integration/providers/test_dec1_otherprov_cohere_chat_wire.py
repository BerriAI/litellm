"""Contract: chat completions on Cohere (`cohere_chat/...`) through the proxy, against a loopback Cohere peer.

Cohere is not OpenAI-compatible on the wire, so the proxy translates in both directions:

- routing: a model name containing `v1/` uses Cohere's legacy v1 chat body (`message`, `chat_history`); every
  other name uses the v2 chat API, which takes OpenAI-style `messages`. A `v2/` segment in the model name is
  removed before it is sent. The deployment's `api_base` is used VERBATIM as the request URL (no `/v2/chat` is
  appended), which is the most likely thing a port gets wrong;
- auth: `Authorization: Bearer <key>` plus `Request-Source: unspecified:litellm`;
- params: `top_p` is sent as `p`, `stop` as `stop_sequences` (a string becomes a one-item list), `n` as
  `num_generations`, `max_completion_tokens` as `max_tokens`; `tool_choice` is accepted but never sent; params
  outside Cohere's list (`response_format`, `logprobs`) are refused with 400;
- answers: the v2 `message.content` list is joined into one string, `tool_calls` become OpenAI tool calls,
  Cohere finish reasons (`COMPLETE`, `MAX_TOKENS`, `TOOL_CALL`) map to `stop`, `length`, `tool_calls`, usage comes
  from `usage.tokens` (not `billed_units`), and citations become `url_citation` annotations;
- streaming: newline-delimited JSON events (`content-delta`, `tool-call-*`, `message-end`) become OpenAI chunks;
- errors: Cohere's 498 (invalid token) reaches the caller as 400, a 429 keeps its status.

Embeddings and rerank on Cohere are covered by the `test_dec1_embedrerank_*` files; v2 token cost by
`cost_calculation/cases/misc.py` (`cohere-command-a-v2-tokens`).
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_otherprov_support import (
    CALL_COST,
    PROVIDER_KEY,
    REPLY_TEXT,
    TOOL,
    Call,
    chat,
    deployment,
    error_message,
    final_usage,
    finish_reasons,
    first_choice,
    first_message,
    joined_delta,
    json_reply,
    marker,
    ndjson,
    peer,
    served,
    spend_row_by_group,
    stream_chunks,
)
from pydantic import JsonValue

_V2_PATH: Final = "/v2/chat"
_MODEL: Final = "cohere_chat/command-a-03-2025"


def _v2_answer(
    *,
    content: list[JsonValue] | None = None,
    finish_reason: str = "COMPLETE",
    tool_calls: list[JsonValue] | None = None,
    citations: list[JsonValue] | None = None,
    tokens: tuple[int, int] = (11, 7),
) -> dict[str, JsonValue]:
    message: Final[dict[str, JsonValue]] = {"role": "assistant"}
    if content is not None:
        message["content"] = content
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if citations is not None:
        message["citations"] = citations
    return {
        "id": "dec1-cohere-answer",
        "finish_reason": finish_reason,
        "message": message,
        "usage": {
            "billed_units": {"input_tokens": 999, "output_tokens": 999},
            "tokens": {"input_tokens": tokens[0], "output_tokens": tokens[1]},
        },
    }


def _text_answer(_call: Call) -> Reply:
    return json_reply(_v2_answer(content=[{"type": "text", "text": REPLY_TEXT}]))


def test_v2_posts_to_the_api_base_verbatim_with_bearer_and_request_source(gateway: Gateway) -> None:
    """A `cohere_chat/command-a-03-2025` deployment whose `api_base` is `<peer>/v2/chat`.

    The peer receives exactly one `POST /v2/chat` with `authorization: Bearer dec1-otherprov-provider-key` and
    `request-source: unspecified:litellm` (`cohere/common_utils.py:110-118`), and a body whose `model` is
    `command-a-03-2025` and whose `messages` hold the caller's user message unchanged (v2 is OpenAI-shaped,
    `v2_transformation.py:170-183`).
    """
    text: Final = marker()
    with peer(_text_answer) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, api_base_suffix=_V2_PATH)
        served(chat(gateway, model, text=text))
        call: Final = provider.only()
        assert (call.method, call.path) == ("POST", _V2_PATH)
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"
        assert call.headers["request-source"] == "unspecified:litellm"
        assert call.body["model"] == "command-a-03-2025", call.body
        assert list_value(call.body["messages"]) == [{"role": "user", "content": text}], call.body


def test_v2_does_not_append_a_chat_path_to_a_bare_api_base(gateway: Gateway) -> None:
    """A Cohere v2 deployment whose `api_base` is the bare peer URL with a custom path `/dec1/custom`.

    `CohereV2ChatConfig.get_complete_url` returns the base unchanged (`v2_transformation.py:277-292`), so the peer
    is called at `/dec1/custom`, never `/dec1/custom/v2/chat`. A port that appends Cohere's path like the
    OpenAI-compatible providers do would break every deployment that sets a full URL.
    """
    with peer(_text_answer) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, api_base_suffix="/dec1/custom")
        served(chat(gateway, model))
        assert provider.only().path == "/dec1/custom"


def test_v2_segment_in_the_model_name_is_removed_before_sending(gateway: Gateway) -> None:
    """`cohere_chat/v2/command-a-03-2025` routes to v2 and reaches the peer with `model: "command-a-03-2025"`
    (`main.py:3303-3307` removes `v2/`)."""
    with peer(_text_answer) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "cohere_chat/v2/command-a-03-2025", api_base_suffix=_V2_PATH)
        served(chat(gateway, model))
        assert provider.only().body["model"] == "command-a-03-2025"


def test_v2_maps_top_p_stop_n_and_max_completion_tokens_to_cohere_names(gateway: Gateway) -> None:
    """The caller sends `top_p: 0.5`, `stop: "END"`, `n: 2`, `max_completion_tokens: 40`, `seed: 9`.

    The peer body has `p: 0.5`, `stop_sequences: ["END"]`, `num_generations: 2`, `max_tokens: 40`, `seed: 9`, and
    none of `top_p`, `stop`, `n`, `max_completion_tokens` (`CohereChatConfig.map_openai_params`,
    `cohere/chat/transformation.py:184-213`; the proxy uses the v1 mapper for v2 too, `utils.py:5840-5847`).
    """
    with peer(_text_answer) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, api_base_suffix=_V2_PATH)
        served(chat(gateway, model, top_p=0.5, stop="END", n=2, max_completion_tokens=40, seed=9))
        sent: Final = provider.only().body
        assert (sent["p"], sent["stop_sequences"], sent["num_generations"], sent["max_tokens"], sent["seed"]) == (
            0.5,
            ["END"],
            2,
            40,
            9,
        ), sent
        assert not {"top_p", "stop", "n", "max_completion_tokens"} & set(sent), sent


def test_v2_keeps_tools_but_never_sends_tool_choice(gateway: Gateway) -> None:
    """`tool_choice` is in Cohere's supported list but `map_openai_params` has no branch for it
    (`cohere/chat/transformation.py:190-213`), so `tool_choice: "required"` is accepted (200) and silently not
    sent, while the tool list reaches the peer with the function name intact.

    Pinned current behaviour; a port that forwards `tool_choice` would change what Cohere does.
    """
    with peer(_text_answer) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, api_base_suffix=_V2_PATH)
        served(chat(gateway, model, tools=[TOOL], tool_choice="required"))
        sent: Final = provider.only().body
        assert "tool_choice" not in sent, sent
        function: Final = object_value(object_value(list_value(sent["tools"])[0])["function"])
        assert function["name"] == "get_weather", sent


@pytest.mark.parametrize(("name", "value"), [("response_format", {"type": "json_object"}), ("logprobs", True)])
def test_params_outside_the_cohere_list_are_refused_before_the_call(
    gateway: Gateway, name: str, value: JsonValue
) -> None:
    """Cohere's supported list (`cohere/chat/transformation.py:166-181`, used for every cohere route by
    `get_supported_openai_params.py:117-118`) has neither `response_format` nor `logprobs`.

    Each is answered 400 with `cohere_chat does not support parameters: ['<name>']` (`utils.py:5793`) and the peer
    is not called.
    """
    with peer(_text_answer) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, api_base_suffix=_V2_PATH)
        response: Final = chat(gateway, model, **{name: value})
        assert response.status_code == 400, response.text
        assert f"cohere_chat does not support parameters: ['{name}']" in error_message(response)
        assert provider.posts() == ()


def test_v2_content_list_is_joined_and_usage_comes_from_tokens_not_billed_units(gateway: Gateway) -> None:
    """Cohere answers `message.content` `[{"type":"text","text":"dec1 "},{"type":"text","text":"joined"}]` with
    `usage.tokens` 11/7 and `usage.billed_units` 999/999.

    The caller's message content is `dec1 joined` and usage is 11 prompt, 7 completion, 18 total
    (`v2_transformation.py:209-213`, `:247-262`). The deployment prices give `x-litellm-response-cost` 0.025.
    """

    def respond(_call: Call) -> Reply:
        return json_reply(_v2_answer(content=[{"type": "text", "text": "dec1 "}, {"type": "text", "text": "joined"}]))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, api_base_suffix=_V2_PATH)
        response: Final = chat(gateway, model)
        body: Final = served(response)
        assert first_message(body)["content"] == "dec1 joined"
        assert object_value(body["usage"])["prompt_tokens"] == 11
        assert object_value(body["usage"])["completion_tokens"] == 7
        assert object_value(body["usage"])["total_tokens"] == 18
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(CALL_COST)


@pytest.mark.parametrize(
    ("cohere_reason", "openai_reason"),
    [("COMPLETE", "stop"), ("MAX_TOKENS", "length"), ("STOP_SEQUENCE", "stop"), ("ERROR_TOXIC", "content_filter")],
)
def test_v2_finish_reason_is_mapped_to_openai_words(gateway: Gateway, cohere_reason: str, openai_reason: str) -> None:
    """Cohere's upper-case finish reasons reach the caller as OpenAI words via `map_finish_reason`
    (`core_helpers.py:443-447` Cohere block, `MAX_TOKENS` at `:454`)."""

    def respond(_call: Call) -> Reply:
        return json_reply(_v2_answer(content=[{"type": "text", "text": REPLY_TEXT}], finish_reason=cohere_reason))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, api_base_suffix=_V2_PATH)
        assert first_choice(served(chat(gateway, model)))["finish_reason"] == openai_reason


def test_v2_tool_calls_become_openai_tool_calls_with_null_content(gateway: Gateway) -> None:
    """Cohere answers `finish_reason: "TOOL_CALL"` and one `tool_calls` entry `{"id": "get_weather_dec1", "type":
    "function", "function": {"name": "get_weather", "arguments": "{\\"city\\":\\"Oslo\\"}"}}` plus a `tool_plan`.

    The caller's message has `content: null`, one tool call with the same id, name and arguments string, and the
    choice's `finish_reason` is `tool_calls` (`v2_transformation.py:226-241`, `core_helpers.py:445`).
    """
    call_entry: Final[dict[str, JsonValue]] = {
        "id": "get_weather_dec1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city":"Oslo"}'},
    }

    def respond(_call: Call) -> Reply:
        answer: Final = _v2_answer(finish_reason="TOOL_CALL", tool_calls=[call_entry])
        object_value(answer["message"])["tool_plan"] = "I will look up Oslo."
        return json_reply(answer)

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, api_base_suffix=_V2_PATH)
        body: Final = served(chat(gateway, model, tools=[TOOL]))
        message: Final = first_message(body)
        assert message["content"] is None, message
        tool_call: Final = object_value(list_value(message["tool_calls"])[0])
        assert tool_call["id"] == "get_weather_dec1"
        assert object_value(tool_call["function"]) == {"name": "get_weather", "arguments": '{"city":"Oslo"}'}
        assert first_choice(body)["finish_reason"] == "tool_calls"


def test_v2_document_citations_become_one_url_citation_per_source(gateway: Gateway) -> None:
    """Cohere answers one citation spanning characters 0..4 with two document sources, the first with a `url`
    and the second with only `id: "doc-2"`.

    The caller's message has two `url_citation` annotations, both with `start_index` 0 and `end_index` 4: the first
    carries the source URL and title, the second the synthetic URL `source:doc-2`
    (`_translate_citations_to_openai_annotations`, `v2_transformation.py:328-356`).
    """
    citation: Final[dict[str, JsonValue]] = {
        "start": 0,
        "end": 4,
        "text": "dec1",
        "sources": [
            {"type": "document", "id": "doc-1", "url": "https://example.test/one", "document": {"title": "One"}},
            {"type": "document", "id": "doc-2", "document": {"title": "Two"}},
        ],
    }

    def respond(_call: Call) -> Reply:
        return json_reply(_v2_answer(content=[{"type": "text", "text": "dec1 cited"}], citations=[citation]))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, api_base_suffix=_V2_PATH)
        annotations: Final = list_value(first_message(served(chat(gateway, model)))["annotations"])
        assert [object_value(item)["url_citation"] for item in annotations] == [
            {"start_index": 0, "end_index": 4, "title": "One", "url": "https://example.test/one"},
            {"start_index": 0, "end_index": 4, "title": "Two", "url": "source:doc-2"},
        ]


def test_v2_stream_of_content_deltas_reaches_the_caller_as_openai_chunks(gateway: Gateway) -> None:
    """Cohere streams newline-delimited events `message-start`, `content-start`, two `content-delta` events
    ("dec1 " and "stream") and `content-end`.

    The caller's SSE chunks concatenate to `dec1 stream` and the last finish reason is `stop`
    (`CohereV2ModelResponseIterator.chunk_parser`, `cohere/common_utils.py:309-365`).
    """

    def respond(_call: Call) -> Reply:
        return Reply(
            content_type="application/x-ndjson",
            chunks=ndjson(
                {"type": "message-start", "id": "dec1", "delta": {"message": {"role": "assistant"}}},
                {"type": "content-start", "index": 0, "delta": {"message": {"content": {"type": "text", "text": ""}}}},
                {"type": "content-delta", "index": 0, "delta": {"message": {"content": {"text": "dec1 "}}}},
                {"type": "content-delta", "index": 0, "delta": {"message": {"content": {"text": "stream"}}}},
                {"type": "content-end", "index": 0},
            ),
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, api_base_suffix=_V2_PATH)
        chunks: Final = stream_chunks(gateway, model)
        assert joined_delta(chunks, "content") == "dec1 stream"
        assert finish_reasons(chunks)[-1] == "stop"
        assert provider.only().body["stream"] is True


def test_v2_stream_message_end_usage_reaches_the_caller(gateway: Gateway) -> None:
    """Cohere's v2 stream ends with `{"type": "message-end", "delta": {"finish_reason": "MAX_TOKENS", "usage":
    {"tokens": {"input_tokens": 40, "output_tokens": 30}}}}`.

    Intended: the caller's final usage is 40 prompt and 30 completion tokens and the last finish reason is
    `length`. Today `chunk_parser` only reads usage for chunks whose `event` key is `message-end` and reads it from
    `chunk["data"]["delta"]` (`cohere/common_utils.py:290-307`, `:346-347`), while the v2 event carries `type` and a
    top-level `delta`, so the reported usage is dropped and the proxy estimates tokens instead.
    """
    pytest.skip("BUG: Cohere v2 stream message-end usage and finish_reason are ignored (parser reads event/data keys)")

    def respond(_call: Call) -> Reply:
        return Reply(
            content_type="application/x-ndjson",
            chunks=ndjson(
                {"type": "content-delta", "index": 0, "delta": {"message": {"content": {"text": "dec1"}}}},
                {
                    "type": "message-end",
                    "delta": {
                        "finish_reason": "MAX_TOKENS",
                        "usage": {"tokens": {"input_tokens": 40, "output_tokens": 30}},
                    },
                },
            ),
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, api_base_suffix=_V2_PATH)
        chunks: Final = stream_chunks(gateway, model, stream_options={"include_usage": True})
        found: Final = final_usage(chunks)
        assert (found["prompt_tokens"], found["completion_tokens"]) == (40, 30), found
        assert finish_reasons(chunks)[-1] == "length"


def test_v2_stream_in_sse_framing_is_parsed(gateway: Gateway) -> None:
    """Cohere's v2 API documents its stream as server-sent events (`event: content-delta` then `data: {...}`).

    Intended: an SSE-framed stream with two content deltas reaches the caller as `dec1 sse`. UNVERIFIED that the
    live API frames v2 this way; the v2 iterator calls `json.loads` on each raw `str` line without removing a
    `data:` prefix (`cohere/common_utils.py:387-405`, lines come from `iter_sse_lines`,
    `llm_http_handler.py:1234-1236`), so an `event:` line would raise.
    """
    pytest.skip("BUG: Cohere v2 iterator json-decodes raw SSE lines; event:/data: framing fails the stream")

    def respond(_call: Call) -> Reply:
        return Reply(
            content_type="text/event-stream",
            chunks=(
                (
                    b'event: content-delta\ndata: {"type":"content-delta","index":0,'
                    b'"delta":{"message":{"content":{"text":"dec1 "}}}}\n\n'
                ),
                (
                    b'event: content-delta\ndata: {"type":"content-delta","index":0,'
                    b'"delta":{"message":{"content":{"text":"sse"}}}}\n\n'
                ),
            ),
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, api_base_suffix=_V2_PATH)
        assert joined_delta(stream_chunks(gateway, model), "content") == "dec1 sse"


def test_v2_stream_tool_call_without_arguments_is_closed_with_an_empty_object(gateway: Gateway) -> None:
    """Cohere streams `tool-call-start` for `get_time` (id `call_dec1`) with no arguments, then `tool-call-end`.

    The caller's chunks carry the tool call id and name, and the arguments pieces join to `{}`
    (`_parse_tool_call_end`, `cohere/common_utils.py:257-261`), so OpenAI clients can `json.loads` them.
    UNVERIFIED: the exact `tool-call-start` delta shape read by `CohereV2StreamedToolCall`.
    """
    start: Final[dict[str, JsonValue]] = {
        "type": "tool-call-start",
        "index": 0,
        "delta": {
            "message": {
                "tool_calls": {
                    "id": "call_dec1",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": ""},
                }
            }
        },
    }

    def respond(_call: Call) -> Reply:
        return Reply(content_type="application/x-ndjson", chunks=ndjson(start, {"type": "tool-call-end", "index": 0}))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, api_base_suffix=_V2_PATH)
        chunks: Final = stream_chunks(gateway, model, tools=[TOOL])
        calls: Final = [
            object_value(call)
            for chunk in chunks
            for choice in list_value(chunk.get("choices") or [])
            for call in list_value(object_value(object_value(choice).get("delta") or {}).get("tool_calls") or [])
        ]
        assert any(call.get("id") == "call_dec1" for call in calls), calls
        arguments: Final = "".join(
            str(object_value(call["function"]).get("arguments") or "") for call in calls if call.get("function")
        )
        assert arguments == "{}", calls


def test_v1_route_sends_the_legacy_message_body(gateway: Gateway) -> None:
    """A `cohere_chat/v1/command-r` deployment (the `v1/` segment selects the legacy API, `common_utils.py:83-88`).

    The peer body carries the last user turn as `message` and no OpenAI `messages` list
    (`CohereChatConfig.transform_request`, `cohere/chat/transformation.py:231-249`). Cohere v1 answers
    `{"text": ...}` with `meta.billed_units` 11/7 and the caller gets that text and those token counts
    (`:265-312`). UNVERIFIED: whether the body also carries `model` (no code path adds it was found).
    """
    text: Final = marker()

    def respond(_call: Call) -> Reply:
        return json_reply(
            {
                "text": REPLY_TEXT,
                "generation_id": "dec1",
                "meta": {"billed_units": {"input_tokens": 11, "output_tokens": 7}},
            }
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "cohere_chat/v1/command-r", api_base_suffix="/v1/chat")
        body: Final = served(chat(gateway, model, text=text))
        sent: Final = provider.only().body
        assert sent["message"] == text and "messages" not in sent, sent
        assert first_message(body)["content"] == REPLY_TEXT
        assert object_value(body["usage"])["completion_tokens"] == 7


def test_498_invalid_token_from_cohere_reaches_the_caller_as_400(gateway: Gateway) -> None:
    """Cohere answers HTTP 498 `{"message": "dec1 bad token"}`.

    `_map_cohere_exception` maps 400 and 498 to `BadRequestError` (`exception_mapping_utils.py:1567-1573`), so the
    caller gets 400 with `CohereException` and `dec1 bad token` in the message. A port that passes 498 through
    would send clients a non-standard status.
    """

    def respond(_call: Call) -> Reply:
        return json_reply({"message": "dec1 bad token"}, status=498)

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, api_base_suffix=_V2_PATH)
        response: Final = chat(gateway, model)
        assert response.status_code == 400, response.text
        assert "dec1 bad token" in error_message(response)
        assert len(provider.posts()) == 1


def test_429_from_cohere_keeps_its_status(gateway: Gateway) -> None:
    """Cohere answers 429 `{"message": "dec1 slow down"}`; no specific branch matches, so the final `APIError`
    keeps `status_code` 429 (`exception_mapping_utils.py:1610-1618`) and the caller gets 429 with the text."""

    def respond(_call: Call) -> Reply:
        return json_reply({"message": "dec1 slow down"}, status=429)

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, api_base_suffix=_V2_PATH)
        response: Final = chat(gateway, model)
        assert response.status_code == 429, response.text
        assert "dec1 slow down" in error_message(response)


def test_v2_call_is_billed_from_usage_tokens_in_spend_logs(gateway: Gateway) -> None:
    """With deployment prices 0.001/0.002 and `usage.tokens` 11/7, the one SpendLogs row has spend 0.025,
    prompt 11, completion 7, and `custom_llm_provider` `cohere_chat`. `billed_units` (999/999) must not be used."""
    with peer(_text_answer) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, api_base_suffix=_V2_PATH)
        served(chat(gateway, model))
        row: Final = spend_row_by_group(gateway, model)
        assert float(str(row["spend"])) == pytest.approx(CALL_COST), row
        assert (row["prompt_tokens"], row["completion_tokens"]) == (11, 7), row
        assert row["custom_llm_provider"] == "cohere_chat", row

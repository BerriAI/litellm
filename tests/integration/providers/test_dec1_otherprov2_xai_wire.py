"""Contract: xAI (Grok) chat tool calls, image input, server-side web search fees, reported cost and the Responses API.

An `xai/<model>` deployment is an OpenAI-compatible chat client (`XAIChatConfig`, `litellm/llms/xai/chat/
transformation.py`) that posts to `<api_base>/chat/completions` with `Authorization: Bearer <key>`. What the
caller and xAI observe beyond the shared OpenAI shape:

- tools: `strict` is stripped (covered elsewhere), `tool_choice`, `parallel_tool_calls` and `user` are forwarded,
  a tool call answer reaches the caller unchanged, and an assistant tool call plus tool result replay as sent;
- vision: `image_url` parts stay a content list;
- reasoning: `reasoning_effort` is only accepted on models the price map marks `supports_reasoning`;
- cost: `usage.server_side_tool_usage_details.web_search_calls` adds a per-call fee ($5 per 1000 calls when the
  model sets no `search_context_cost_per_query`), and `usage.cost_in_usd_ticks` (1e10 ticks per dollar) replaces
  the token math entirely and suppresses the search fee (`litellm/llms/xai/cost_calculator.py`);
- `/v1/responses` on an xAI deployment posts to `<api_base>/responses` and rewrites tools: `code_interpreter`
  loses `container`, `web_search` loses `search_context_size` and keeps domain filters, `x_search` keeps its
  handle and date fields, and `metadata` is dropped (`litellm/llms/xai/responses/transformation.py`).

Chat `web_search_options` (routed to `/responses`), usage detail passthrough, reasoning token folding and the
param refusals of grok-3-mini and grok-4 already have tests (`test_xai_web_search_wire.py`,
`test_xai_server_tool_usage_wire.py`, `test_dec1_otherprov_reasoning_and_fireworks_wire.py`).
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_otherprov2_support import (
    TOOL_ARGUMENTS,
    TOOL_CALL_ID,
    chunk,
    delta_choice,
    raw_stream,
    responses,
    responses_reply,
    tool_call,
    tool_call_reply,
    upstream_error,
    usage_with,
)
from integration.providers.dec1_otherprov_support import (
    CALL_COST,
    PROVIDER_KEY,
    TOOL,
    Call,
    chat,
    chat_json,
    default_respond,
    deployment,
    error_message,
    first_choice,
    first_message,
    json_reply,
    marker,
    peer,
    served,
    stream_chunks,
)
from pydantic import JsonValue

_GROK: Final = "xai/grok-4.3"
_IMAGE_URL: Final = "https://images.example.com/dec1-otherprov2.png"


def _xai_respond(call: Call) -> Reply:
    """Chat answers for chat calls, Responses answers for `/responses` calls."""
    if call.path.endswith("/responses"):
        return responses_reply(call.body)
    return default_respond(call)


def test_xai_chat_posts_json_with_the_bearer_key_to_the_v1_chat_route(gateway: Gateway) -> None:
    """An `xai/grok-4.3` deployment whose `api_base` is the peer plus `/v1`.

    xAI receives one `POST /v1/chat/completions` with `authorization: Bearer dec1-otherprov-provider-key` and
    `content-type: application/json` (`XAIModelInfo.validate_environment`, `litellm/llms/xai/common_utils.py:41-58`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GROK, api_base_suffix="/v1")
        served(chat(gateway, model))
        call: Final = provider.only()
        assert (call.path, call.headers["authorization"]) == ("/v1/chat/completions", f"Bearer {PROVIDER_KEY}")
        assert call.headers["content-type"].startswith("application/json"), call.headers


def test_xai_grok_3_forwards_stop_and_frequency_penalty(gateway: Gateway) -> None:
    """`grok-3` is neither grok-3-mini, grok-4 nor grok-code-fast, so both `stop` and `frequency_penalty` are in
    its supported list (`xai/chat/transformation.py:178-185`, `:198-219`) and reach xAI as sent: `stop: ["END"]`
    and `frequency_penalty: 0.3`. A port that applied the grok-4 refusals to every model would 400 here."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "xai/grok-3", api_base_suffix="/v1")
        served(chat(gateway, model, stop=["END"], frequency_penalty=0.3))
        sent: Final = provider.only().body
        assert (sent["stop"], sent["frequency_penalty"]) == (["END"], 0.3), sent


def test_xai_tool_choice_parallel_tool_calls_and_user_are_forwarded(gateway: Gateway) -> None:
    """The caller sends one tool, `tool_choice` naming it, `parallel_tool_calls: false` and `user: "dec1-user"`.

    All three are in `XAIChatConfig.get_supported_openai_params` (`xai/chat/transformation.py:153-173`) and reach
    xAI unchanged; the tool keeps its name and parameters.
    """
    choice: Final[dict[str, JsonValue]] = {"type": "function", "function": {"name": "get_weather"}}
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GROK, api_base_suffix="/v1")
        served(chat(gateway, model, tools=[TOOL], tool_choice=choice, parallel_tool_calls=False, user="dec1-user"))
        sent: Final = provider.only().body
        assert (sent["tool_choice"], sent["parallel_tool_calls"], sent["user"]) == (choice, False, "dec1-user"), sent
        function: Final = object_value(object_value(list_value(sent["tools"])[0])["function"])
        assert function["name"] == "get_weather" and "parameters" in function, sent


def test_xai_tool_call_answer_reaches_the_caller_with_its_id_and_arguments(gateway: Gateway) -> None:
    """xAI answers one tool call (`call_dec1otherprov2`, `get_weather`, `{"city": "Oslo"}`) with finish reason
    `tool_calls`. The caller's first choice carries the same id, function name and argument string, `content`
    is null, and the finish reason stays `tool_calls`."""
    with peer(tool_call_reply) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GROK, api_base_suffix="/v1")
        body: Final = served(chat(gateway, model, tools=[TOOL]))
        choice: Final = first_choice(body)
        message: Final = object_value(choice["message"])
        assert choice["finish_reason"] == "tool_calls", body
        assert message["content"] is None, body
        found: Final = object_value(list_value(message["tool_calls"])[0])
        assert (found["id"], object_value(found["function"])["arguments"]) == (TOOL_CALL_ID, TOOL_ARGUMENTS), found


def test_xai_tool_result_turn_is_replayed_with_its_tool_call_id(gateway: Gateway) -> None:
    """The caller replays an assistant turn with one tool call and a `tool` message answering it.

    xAI receives the assistant message with that tool call (same id and function) and the tool message with
    `tool_call_id: call_dec1otherprov2` and its content, in order after the user turn.
    """
    text: Final = marker()
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": text},
        {"role": "assistant", "content": None, "tool_calls": [tool_call()]},
        {"role": "tool", "tool_call_id": TOOL_CALL_ID, "content": "sunny"},
    ]
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GROK, api_base_suffix="/v1")
        served(chat(gateway, model, messages=messages, tools=[TOOL]))
        sent: Final = list_value(provider.only().body["messages"])
        assert [object_value(m)["role"] for m in sent] == ["user", "assistant", "tool"], sent
        replayed: Final = object_value(list_value(object_value(sent[1])["tool_calls"])[0])
        assert (replayed["id"], object_value(replayed["function"])["name"]) == (TOOL_CALL_ID, "get_weather"), sent
        assert (object_value(sent[2])["tool_call_id"], object_value(sent[2])["content"]) == (TOOL_CALL_ID, "sunny")


def test_xai_image_url_parts_reach_xai_as_a_content_list(gateway: Gateway) -> None:
    """A user message with a text part and an `image_url` part (`detail: "high"`) on the vision model grok-4.3.

    xAI receives the content as the same two-part list, the image part keeping its URL and detail (the chat
    transform only strips `name`, `xai/chat/transformation.py:260-278`). UNVERIFIED: that no generic step
    downloads or inlines the remote URL as base64 for xAI; the test pins the URL as sent.
    """
    text: Final = marker()
    parts: Final[list[JsonValue]] = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": _IMAGE_URL, "detail": "high"}},
    ]
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GROK, api_base_suffix="/v1")
        served(chat(gateway, model, messages=[{"role": "user", "content": parts}]))
        content: Final = list_value(object_value(list_value(provider.only().body["messages"])[0])["content"])
        assert object_value(content[0]) == {"type": "text", "text": text}, content
        assert object_value(object_value(content[1])["image_url"]) == {"url": _IMAGE_URL, "detail": "high"}, content


def test_xai_reasoning_effort_reaches_a_reasoning_model(gateway: Gateway) -> None:
    """`xai/grok-4.3` has `supports_reasoning: true` in the price map, so `reasoning_effort` joins its supported
    params (`xai/chat/transformation.py:190-192`) and `reasoning_effort: "low"` reaches xAI as sent."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GROK, api_base_suffix="/v1")
        served(chat(gateway, model, reasoning_effort="low"))
        assert provider.only().body["reasoning_effort"] == "low"


def test_xai_reasoning_effort_is_refused_on_a_non_reasoning_model(gateway: Gateway) -> None:
    """`xai/grok-4.20-non-reasoning` has no `supports_reasoning` flag, so `reasoning_effort: "high"` is not a
    supported param and the call is answered 400 with `xai does not support parameters: ['reasoning_effort']`
    (`litellm/utils.py:5789-5794`) before xAI is called."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "xai/grok-4.20-non-reasoning", api_base_suffix="/v1")
        response: Final = chat(gateway, model, reasoning_effort="high")
        assert response.status_code == 400, response.text
        assert "xai does not support parameters: ['reasoning_effort']" in error_message(response)
        assert provider.posts() == ()


def test_xai_stream_tool_call_deltas_reach_the_caller_with_id_and_arguments(gateway: Gateway) -> None:
    """xAI streams one tool call in a single delta (id, name and full arguments) and then finish reason
    `tool_calls`. The caller's chunks carry a tool call delta with id `call_dec1otherprov2`, name `get_weather`
    and arguments `{"city": "Oslo"}`, and the last finish reason is `tool_calls`."""

    def respond(call: Call) -> Reply:
        streamed: Final[dict[str, JsonValue]] = {"index": 0, **tool_call()}
        return raw_stream(
            chunk(call.body, choices=delta_choice({"role": "assistant", "tool_calls": [streamed]})),
            chunk(call.body, choices=delta_choice({}, finish_reason="tool_calls")),
            chunk(call.body, choices=[], usage=usage_with()),
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GROK, api_base_suffix="/v1")
        chunks: Final = stream_chunks(gateway, model, tools=[TOOL])
        deltas: Final = [
            object_value(object_value(list_value(c["choices"])[0])["delta"])
            for c in chunks
            if isinstance(c.get("choices"), list) and list_value(c["choices"])
        ]
        calls: Final = [object_value(list_value(d["tool_calls"])[0]) for d in deltas if d.get("tool_calls")]
        assert calls and calls[0]["id"] == TOOL_CALL_ID, chunks
        function: Final = object_value(calls[0]["function"])
        assert (function["name"], function["arguments"]) == ("get_weather", TOOL_ARGUMENTS), chunks
        reasons: Final = [
            object_value(list_value(c["choices"])[0]).get("finish_reason")
            for c in chunks
            if isinstance(c.get("choices"), list) and list_value(c["choices"])
        ]
        assert [reason for reason in reasons if reason][-1] == "tool_calls", chunks


def test_xai_web_search_calls_add_the_default_per_call_fee(gateway: Gateway) -> None:
    """xAI answers 11 prompt and 7 completion tokens plus `server_side_tool_usage_details: {"web_search_calls": 2}`.

    The token cost at the deployment prices is 0.025 and grok-4.3 sets no `search_context_cost_per_query`, so
    each search costs the default 0.005 (`_DEFAULT_WEB_SEARCH_COST_PER_CALL`, `xai/cost_calculator.py:19`,
    `:139-159`); `x-litellm-response-cost` is 0.035. UNVERIFIED: that the built-in tool tracker resolves model
    info for `grok-4.3` from the price map (`tool_call_cost_tracking.py:150-163`) rather than the deployment's.
    """

    def respond(call: Call) -> Reply:
        details: Final[dict[str, JsonValue]] = {"web_search_calls": 2}
        return json_reply(chat_json(call.body, usage_block=usage_with(server_side_tool_usage_details=details)))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GROK, api_base_suffix="/v1")
        response: Final = chat(gateway, model)
        served(response)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(CALL_COST + 2 * 0.005)


def test_xai_cost_in_usd_ticks_is_the_response_cost(gateway: Gateway) -> None:
    """xAI answers `usage.cost_in_usd_ticks: 420000000`. At 10,000,000,000 ticks per dollar
    (`USD_TICKS_PER_DOLLAR`, `xai/common_utils.py:14-26`) that is $0.042, which becomes `usage.cost`
    (`xai/chat/transformation.py:63-67`, `:344-346`) and the cost reported to the caller, instead of the 0.025 the
    deployment prices would give (`xai/cost_calculator.py:65-80`). The deployment is unpriced: deployment prices win
    over a provider-reported cost (`litellm_logging.py:2544-2552`, `cost_calculator.py:2284`)."""

    def respond(call: Call) -> Reply:
        return json_reply(chat_json(call.body, usage_block=usage_with(cost_in_usd_ticks=420000000)))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GROK, api_base_suffix="/v1", priced=False)
        response: Final = chat(gateway, model)
        served(response)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.042), response.headers


def test_xai_reported_ticks_suppress_the_web_search_fee(gateway: Gateway) -> None:
    """xAI answers `cost_in_usd_ticks: 420000000` and two web search calls in one usage block. The reported $0.042
    already includes the searches, so `cost_per_web_search_request` returns 0 when xAI reported a cost
    (`xai/cost_calculator.py:147-148`) and the caller is charged 0.042, not 0.052. The deployment is unpriced: a
    priced one ignores the ticks and billed 0.035 (tokens plus the search fee) in run 44176682961."""

    def respond(call: Call) -> Reply:
        details: Final[dict[str, JsonValue]] = {"web_search_calls": 2}
        block: Final = usage_with(cost_in_usd_ticks=420000000, server_side_tool_usage_details=details)
        return json_reply(chat_json(call.body, usage_block=block))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GROK, api_base_suffix="/v1", priced=False)
        response: Final = chat(gateway, model)
        served(response)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.042), response.headers


@pytest.mark.parametrize("status", [401, 429, 503])
def test_xai_error_status_reaches_the_caller(gateway: Gateway, status: int) -> None:
    """xAI answers HTTP 401, 429 or 503 with an OpenAI error body whose message is `dec1 xai failure`.

    xAI is an OpenAI-compatible provider, so `_map_openai_exception` keeps the status
    (`exception_mapping_utils.py:2733-2749`): the caller gets the same status with the provider message inside
    the error, after exactly one upstream call (`num_retries: 0`).
    """
    with peer(lambda call: upstream_error(status, "dec1 xai failure")) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GROK, api_base_suffix="/v1")
        response: Final = chat(gateway, model)
        assert response.status_code == status, response.text
        assert "dec1 xai failure" in error_message(response)
        assert len(provider.posts()) == 1


def test_xai_responses_api_posts_to_v1_responses_with_the_bearer_key(gateway: Gateway) -> None:
    """`POST /v1/responses` on an xAI deployment whose base is the peer plus `/v1`.

    xAI receives `POST /v1/responses` (`XAIResponsesAPIConfig.get_complete_url`, `xai/responses/
    transformation.py:238-257`) with `authorization: Bearer dec1-otherprov-provider-key` and model `grok-4.3`;
    the caller gets an `object: response` whose output text is the reply and whose usage is 13 in, 5 out.
    """
    with peer(_xai_respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GROK, api_base_suffix="/v1")
        body: Final = served(responses(gateway, model))
        call: Final = provider.only()
        assert (call.path, call.headers["authorization"]) == ("/v1/responses", f"Bearer {PROVIDER_KEY}")
        assert call.body["model"] == "grok-4.3", call.body
        assert body["object"] == "response", body
        assert object_value(body["usage"])["input_tokens"] == 13, body


def test_xai_responses_drops_metadata_and_the_code_interpreter_container(gateway: Gateway) -> None:
    """The caller sends `metadata: {"team": "dec1"}` and a `code_interpreter` tool with `container: {"type":
    "auto"}`. xAI receives no `metadata` key and the tool as exactly `{"type": "code_interpreter"}`
    (`xai/responses/transformation.py:157-174`)."""
    with peer(_xai_respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GROK, api_base_suffix="/v1")
        served(
            responses(
                gateway,
                model,
                metadata={"team": "dec1"},
                tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
            )
        )
        sent: Final = provider.only().body
        assert "metadata" not in sent, sent
        assert sent["tools"] == [{"type": "code_interpreter"}], sent


def test_xai_responses_web_search_tool_keeps_flat_domain_filters_and_drops_context_size(gateway: Gateway) -> None:
    """The caller's `web_search` tool has `search_context_size: "high"`, flat `allowed_domains` and
    `excluded_domains`, and `enable_image_understanding: true`.

    xAI receives `{"type": "web_search", "filters": {"allowed_domains": [...], "excluded_domains": [...]},
    "enable_image_understanding": true}` with no `search_context_size` (`_transform_web_search_tool`,
    `xai/responses/transformation.py:67-99`).
    """
    tool: Final[dict[str, JsonValue]] = {
        "type": "web_search",
        "search_context_size": "high",
        "allowed_domains": ["docs.example.com"],
        "excluded_domains": ["spam.example.com"],
        "enable_image_understanding": True,
    }
    with peer(_xai_respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GROK, api_base_suffix="/v1")
        served(responses(gateway, model, tools=[tool]))
        assert provider.only().body["tools"] == [
            {
                "type": "web_search",
                "filters": {"allowed_domains": ["docs.example.com"], "excluded_domains": ["spam.example.com"]},
                "enable_image_understanding": True,
            }
        ]


def test_xai_responses_x_search_tool_keeps_only_its_documented_fields(gateway: Gateway) -> None:
    """The caller's `x_search` tool has `allowed_x_handles`, `from_date`, `to_date`, `enable_video_understanding`
    and an unknown `dec1_extra` key. xAI receives the four documented fields with the same values and no
    `dec1_extra` (`_transform_x_search_tool`, `xai/responses/transformation.py:101-139`)."""
    tool: Final[dict[str, JsonValue]] = {
        "type": "x_search",
        "allowed_x_handles": ["dec1handle"],
        "from_date": "2026-11-01",
        "to_date": "2026-11-17",
        "enable_video_understanding": False,
        "dec1_extra": "dropped",
    }
    with peer(_xai_respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GROK, api_base_suffix="/v1")
        served(responses(gateway, model, tools=[tool]))
        assert provider.only().body["tools"] == [
            {
                "type": "x_search",
                "allowed_x_handles": ["dec1handle"],
                "from_date": "2026-11-01",
                "to_date": "2026-11-17",
                "enable_video_understanding": False,
            }
        ]


def test_xai_chat_answer_message_has_no_reasoning_when_xai_sends_none(gateway: Gateway) -> None:
    """A plain xAI answer (no reasoning fields) reaches the caller with the reply text as `content` and no
    `reasoning_content` key or a null one, so a port does not invent empty reasoning for non-reasoning answers.
    UNVERIFIED: whether the key is absent or null; both are accepted."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GROK, api_base_suffix="/v1")
        message: Final = first_message(served(chat(gateway, model)))
        assert message["content"] == "dec1 otherprov reply", message
        assert message.get("reasoning_content") in (None, ""), message
        assert provider.only().path == "/v1/chat/completions"

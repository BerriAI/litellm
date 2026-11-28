"""Contract: `POST /v1/responses` served by an `openai/` deployment, request to wire and answer to caller.

For OpenAI the Responses API is passed through almost untouched (`OpenAIResponsesAPIConfig`,
`litellm/llms/openai/responses/transformation.py`): the body reaches `<api_base>/responses` with the upstream
model name, and every Responses parameter (`store`, `background`, `include`, `reasoning`, `metadata`,
`max_output_tokens`, `text`) keeps its name and value. The proxy does change four things a port must keep:

- the response `id` the caller sees is a litellm-managed id: `resp_` plus base64 of
  `litellm:custom_llm_provider:openai;model_id:<deployment id>;response_id:<provider id>`
  (`ResponsesAPIRequestUtils._build_responses_api_response_id`), which the proxy then encrypts
  (`proxy/hooks/responses_id_security.py:372-406`), so follow-up calls route back to the same deployment;
- `max_output_tokens` under 16 is raised to 16, the OpenAI minimum (`_enforce_min_max_output_tokens`), so a
  Claude Code style `max_tokens: 1` probe does not fail;
- `reasoning.effort` on a model whose cost map row lacks `supports_reasoning` is refused (or dropped under
  `drop_params`);
- the gpt-5 temperature and top_p rules from chat apply here too, keyed on `reasoning.effort`.

Streaming is relayed event by event in the provider's order; the ids inside the events are rewritten to the
managed id consistently (#51243). Spend is billed from the `response.completed` usage.
The previous_response_id and lifecycle (retrieve, delete, cancel) contracts already live in
`test_responses_core_wire.py` and `test_responses_lifecycle_wire.py` and are not repeated here.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value, string_value
from integration._support.wire import Reply
from integration.providers.dec1_openai_support import (
    CALL_COST,
    RESPONSES_EVENT_ORDER,
    Call,
    data_events,
    deployment,
    error_message,
    json_reply,
    marker,
    openai_error,
    peer,
    responses,
    responses_object,
    served,
    spend_row_by_group,
    stream_lines,
)
from pydantic import JsonValue

_MODEL: Final = "openai/gpt-4o-mini"
_REASONING_MODEL: Final = "openai/gpt-5"


def test_a_plain_request_reaches_the_responses_path_with_the_upstream_model(gateway: Gateway) -> None:
    """The caller sends `{"model": <alias>, "input": "<marker>"}` to an `openai/gpt-4o-mini` deployment whose
    `api_base` ends in `/v1`.

    The provider receives one `POST /v1/responses` with `authorization: Bearer <deployment key>` and a body of
    exactly `{"model": "gpt-4o-mini", "input": "<marker>"}`.
    """
    text: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(responses(gateway, model, input=text))
        call: Final = provider.only()
        assert (call.method, call.path) == ("POST", "/v1/responses")
        assert call.headers["authorization"] == "Bearer dec1-openai-provider-key"
        assert call.body == {"model": "gpt-4o-mini", "input": text}


def test_the_caller_gets_a_managed_id_that_wraps_the_provider_id(gateway: Gateway) -> None:
    """The provider answers `id: resp_dec1provider`.

    The caller's `id` starts with `resp_` and differs from the provider id. It is opaque: the proxy encrypts the
    base64 `litellm:custom_llm_provider:...;response_id:...` id (`_encrypt_response_id`,
    `proxy/hooks/responses_id_security.py:372-406` on main), so it no longer base64-decodes (run 44176682961). A
    follow-up naming it reaches the provider as `previous_response_id: resp_dec1provider`. The rest of the object
    (`object`, `status`) is the provider's.
    """

    def respond(call: Call) -> Reply:
        return json_reply(responses_object(call.body, identity="resp_dec1provider"))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        body: Final = served(responses(gateway, model))
        managed: Final = string_value(body["id"])
        assert managed.startswith("resp_") and managed != "resp_dec1provider", body
        assert "resp_dec1provider" not in managed, managed
        assert (body["object"], body["status"]) == ("response", "completed"), body
        served(responses(gateway, model, previous_response_id=managed))
        assert provider.posts()[-1].body["previous_response_id"] == "resp_dec1provider"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("store", False),
        ("background", True),
        ("include", ["reasoning.encrypted_content", "message.output_text.logprobs"]),
        ("metadata", {"dec1": "yes"}),
        ("instructions", "Answer briefly."),
        ("text", {"format": {"type": "json_object"}}),
        ("parallel_tool_calls", False),
        ("truncation", "auto"),
        ("user", "dec1-end-user"),
        ("service_tier", "flex"),
        ("max_output_tokens", 64),
        ("prompt_cache_key", "dec1-key"),
    ],
)
def test_each_responses_param_is_forwarded_unchanged(gateway: Gateway, name: str, value: JsonValue) -> None:
    """One Responses parameter is sent to `openai/gpt-4o-mini`.

    The provider body carries `<name>: <value>` exactly as sent. `store: false` matters most: dropping it would
    make OpenAI retain data the caller asked it not to keep.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(responses(gateway, model, **{name: value}))
        assert provider.only().body[name] == value


def test_a_background_request_returns_the_queued_object(gateway: Gateway) -> None:
    """The caller sends `background: true` and the provider answers `status: "queued"` with no output and null usage.

    The caller gets HTTP 200 with `status == "queued"`, an empty `output` list and a managed `resp_` id, so it can
    poll with GET. No spend is computed from a null usage. UNVERIFIED: a null `usage` on a queued object is
    accepted by the response model (`ResponsesAPIResponse.usage` is optional).
    """

    def respond(call: Call) -> Reply:
        return json_reply(responses_object(call.body, status="queued", background=True))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        body: Final = served(responses(gateway, model, background=True))
        assert provider.only().body["background"] is True
        assert (body["status"], body["output"]) == ("queued", []), body
        assert string_value(body["id"]).startswith("resp_")


def test_a_max_output_tokens_below_sixteen_is_raised_to_sixteen(gateway: Gateway) -> None:
    """`max_output_tokens: 1` is below OpenAI's Responses minimum of 16, which OpenAI rejects for every model.

    The provider receives `max_output_tokens: 16` (`OPENAI_RESPONSES_API_MIN_MAX_OUTPUT_TOKENS`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(responses(gateway, model, max_output_tokens=1))
        assert provider.only().body["max_output_tokens"] == 16


def test_reasoning_settings_reach_a_reasoning_model_unchanged(gateway: Gateway) -> None:
    """`reasoning: {"effort": "low", "summary": "auto"}` with `include: ["reasoning.encrypted_content"]` and
    `store: false` on `openai/gpt-5`.

    The provider body carries the same `reasoning` object and `include` list. This is the shape stateless
    clients use to carry reasoning across turns.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _REASONING_MODEL)
        served(
            responses(
                gateway,
                model,
                reasoning={"effort": "low", "summary": "auto"},
                include=["reasoning.encrypted_content"],
                store=False,
            )
        )
        sent: Final = provider.only().body
        assert (sent["reasoning"], sent["include"], sent["store"]) == (
            {"effort": "low", "summary": "auto"},
            ["reasoning.encrypted_content"],
            False,
        ), sent


def test_reasoning_effort_on_a_non_reasoning_model_is_refused(gateway: Gateway) -> None:
    """`reasoning: {"effort": "high"}` on `openai/gpt-4o-mini`, whose row lacks `supports_reasoning`.

    The caller gets 400 with `gpt-4o-mini doesn't support `reasoning.effort` (its model cost map entry lacks
    `supports_reasoning`)` and the provider is not called.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        response: Final = responses(gateway, model, reasoning={"effort": "high"})
        assert response.status_code == 400, response.text
        assert "doesn't support `reasoning.effort`" in error_message(response), response.text
        assert provider.posts() == ()


def test_reasoning_on_a_non_reasoning_model_is_dropped_under_drop_params(gateway: Gateway) -> None:
    """With `drop_params: true` on the `gpt-4o-mini` deployment the same request is served and the provider body
    has no `reasoning` key."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, drop_params=True)
        served(responses(gateway, model, reasoning={"effort": "high"}))
        assert "reasoning" not in provider.only().body


def test_gpt_5_responses_refuses_temperature_while_reasoning(gateway: Gateway) -> None:
    """`temperature: 0.3` on `openai/gpt-5` over Responses is answered 400 with `gpt-5 doesn't support
    temperature=0.3 while reasoning is active`, the same rule as chat."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _REASONING_MODEL)
        response: Final = responses(gateway, model, temperature=0.3)
        assert response.status_code == 400, response.text
        assert "doesn't support temperature=0.3 while reasoning is active" in error_message(response)
        assert provider.posts() == ()


def test_gpt_5_responses_refuses_top_p_while_reasoning(gateway: Gateway) -> None:
    """`top_p: 0.5` on `openai/gpt-5` over Responses is answered 400 with `only supports top_p when
    reasoning.effort resolves to 'none'`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _REASONING_MODEL)
        response: Final = responses(gateway, model, top_p=0.5)
        assert response.status_code == 400, response.text
        assert "only supports top_p when reasoning.effort resolves to 'none'" in error_message(response)


def test_gpt_5_responses_drops_temperature_and_top_p_under_drop_params(gateway: Gateway) -> None:
    """With `drop_params: true`, `temperature: 0.3` and `top_p: 0.5` on `gpt-5` are both removed and the call is
    served."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _REASONING_MODEL, drop_params=True)
        served(responses(gateway, model, temperature=0.3, top_p=0.5))
        sent: Final = provider.only().body
        assert "temperature" not in sent and "top_p" not in sent, sent


def test_gpt_5_1_responses_accepts_temperature_with_effort_none(gateway: Gateway) -> None:
    """`reasoning: {"effort": "none"}` with `temperature: 0.3` on `openai/gpt-5.1` (which supports `none`) is
    forwarded with both fields unchanged."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-5.1")
        served(responses(gateway, model, temperature=0.3, reasoning={"effort": "none"}))
        sent: Final = provider.only().body
        assert (sent["temperature"], sent["reasoning"]) == (0.3, {"effort": "none"}), sent


def test_a_reasoning_input_item_is_forwarded_with_its_encrypted_content(gateway: Gateway) -> None:
    """A stateless second turn replays a reasoning item `{"type": "reasoning", "id": "rs_dec1", "summary": [],
    "encrypted_content": "enc-dec1"}` before the next user message, with `store: false`.

    The provider's `input` list keeps that item with the same `encrypted_content` and in the same position, so
    OpenAI can resume the hidden chain of thought.
    """
    reasoning_item: Final[dict[str, JsonValue]] = {
        "type": "reasoning",
        "id": "rs_dec1",
        "summary": [],
        "encrypted_content": "enc-dec1",
    }
    items: Final[list[JsonValue]] = [
        {"role": "user", "content": marker()},
        reasoning_item,
        {"role": "user", "content": marker()},
    ]
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _REASONING_MODEL)
        served(responses(gateway, model, input=items, store=False, include=["reasoning.encrypted_content"]))
        sent_items: Final = list_value(provider.only().body["input"])
        assert object_value(sent_items[1])["type"] == "reasoning", sent_items
        assert object_value(sent_items[1])["encrypted_content"] == "enc-dec1", sent_items


def test_a_reasoning_output_item_comes_back_before_the_message(gateway: Gateway) -> None:
    """The provider answers an `output` of a reasoning item with one `summary_text` part `thought dec1`, then the
    assistant message.

    The caller's `output` keeps both items in that order, with the reasoning summary text unchanged.
    """

    def respond(call: Call) -> Reply:
        payload: Final = responses_object(call.body)
        reasoning: Final[dict[str, JsonValue]] = {
            "id": "rs_dec1out",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "thought dec1"}],
        }
        return json_reply({**payload, "output": [reasoning, *list_value(payload["output"])]})

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _REASONING_MODEL)
        body: Final = served(responses(gateway, model))
        output: Final = list_value(body["output"])
        assert [object_value(item)["type"] for item in output] == ["reasoning", "message"], output
        summary: Final = list_value(object_value(output[0])["summary"])
        assert object_value(summary[0])["text"] == "thought dec1"


def test_the_stream_relays_every_event_in_the_provider_order(gateway: Gateway) -> None:
    """The caller streams; the provider sends the eleven canonical events from `response.created` to
    `response.completed`.

    The caller receives the same event types in the same order, and every `response.*` id it sees in
    `response.created` and `response.completed` is the same managed `resp_` id. A port that buffers and
    re-emits events must keep both the order and the id rewrite consistent.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        _, lines = stream_lines(gateway, "/v1/responses", {"model": model, "input": marker(), "stream": True})
        events: Final = data_events(lines)
        assert tuple(string_value(event["type"]) for event in events) == RESPONSES_EVENT_ORDER, events
        created: Final = object_value(events[0]["response"])["id"]
        completed: Final = object_value(events[-1]["response"])["id"]
        assert created == completed and string_value(created).startswith("resp_"), (created, completed)
        assert provider.only().body["stream"] is True


def test_the_streamed_text_deltas_add_up_to_the_final_text(gateway: Gateway) -> None:
    """The three `response.output_text.delta` events carry `dec1 `, `openai ` and `reply`.

    Joined they equal the `text` of `response.output_text.done`, `dec1 openai reply`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        _, lines = stream_lines(gateway, "/v1/responses", {"model": model, "input": marker(), "stream": True})
        events: Final = data_events(lines)
        deltas: Final = "".join(string_value(e["delta"]) for e in events if e["type"] == "response.output_text.delta")
        done: Final = next(e for e in events if e["type"] == "response.output_text.done")
        assert deltas == done["text"] == "dec1 openai reply", events


def test_a_responses_call_is_billed_from_its_usage(gateway: Gateway) -> None:
    """One non-streamed call whose usage reports 11 input and 7 output tokens, priced at 0.001 and 0.002.

    One SpendLogs row lands for the model group with `spend = 0.025`, `prompt_tokens = 11`,
    `completion_tokens = 7` and `custom_llm_provider = 'openai'`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(responses(gateway, model))
        row: Final = spend_row_by_group(gateway, model)
        assert float(str(row["spend"])) == pytest.approx(CALL_COST), row
        assert (row["prompt_tokens"], row["completion_tokens"], row["custom_llm_provider"]) == (11, 7, "openai"), row


def test_a_streamed_responses_call_is_billed_from_the_completed_event(gateway: Gateway) -> None:
    """The same usage arrives only inside `response.completed` of a stream; one row lands with `spend = 0.025`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        stream_lines(gateway, "/v1/responses", {"model": model, "input": marker(), "stream": True})
        row: Final = spend_row_by_group(gateway, model)
        assert float(str(row["spend"])) == pytest.approx(CALL_COST), row


def test_a_provider_429_on_responses_keeps_its_status(gateway: Gateway) -> None:
    """The provider answers the Responses call with 429 `dec1 responses throttled`.

    The caller gets 429 with the OpenAI error envelope and the provider text in the message; the provider is
    called once.
    """
    with peer(lambda call: openai_error(429, "dec1 responses throttled")) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        response: Final = responses(gateway, model)
        assert response.status_code == 429, response.text
        assert "dec1 responses throttled" in error_message(response)
        assert len(provider.posts()) == 1


def test_previous_response_id_of_a_managed_id_reaches_openai_as_the_raw_id(gateway: Gateway) -> None:
    """The caller continues with `previous_response_id` set to the managed id it received for provider id
    `resp_dec1first`.

    The second provider request carries `previous_response_id: "resp_dec1first"`, never the managed wrapper,
    which OpenAI would not recognise. (Also covered on other aliases in `test_responses_core_wire.py`; kept here
    for the plain `openai/` deployment.)
    """

    def respond(call: Call) -> Reply:
        first: Final = "previous_response_id" not in call.body
        return json_reply(responses_object(call.body, identity="resp_dec1first" if first else "resp_dec1second"))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        first_id: Final = served(responses(gateway, model))["id"]
        served(responses(gateway, model, previous_response_id=first_id))
        assert provider.posts()[-1].body["previous_response_id"] == "resp_dec1first"


def test_an_unknown_top_level_responses_field_is_not_forwarded(gateway: Gateway) -> None:
    """A field the Responses request type does not declare, `dec1_vendor_flag: true`, sent at the top level does
    not reach the provider: the body is exactly `{"model", "input"}` (run 44176682961). Pinned current behaviour;
    the optional params are taken from the declared Responses request keys only, and `extra_body` is the way to
    send a vendor field."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(responses(gateway, model, dec1_vendor_flag=True))
        assert "dec1_vendor_flag" not in provider.only().body, provider.only().body


def test_a_function_tool_is_forwarded_in_the_responses_flat_shape(gateway: Gateway) -> None:
    """A Responses function tool `{"type": "function", "name": "lookup", "parameters": {...}, "strict": true}` is
    forwarded unchanged (flat, not wrapped in `function`), with `strict` kept."""
    tool: Final[dict[str, JsonValue]] = {
        "type": "function",
        "name": "lookup",
        "description": "Look something up",
        "parameters": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
            "additionalProperties": False,
        },
        "strict": True,
    }
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(responses(gateway, model, tools=[tool]))
        assert json.dumps(provider.only().body["tools"], sort_keys=True) == json.dumps([tool], sort_keys=True)

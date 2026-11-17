"""`previous_response_id`, the resource routes and background mode for responses the bridge served.

A bridged provider keeps no conversation, so the proxy keeps it: each turn's request (`proxy_server_request`, only
stored when `store_prompts_in_spend_logs` is on) and answer land in `LiteLLM_SpendLogs`, and a follow up that names
`previous_response_id` is answered by `ResponsesSessionHandler.get_all_spend_logs_for_previous_response_id`
(`litellm/responses/litellm_completion_transformation/session_handler.py:610`): it finds the referenced row,
then every row of that row's `session_id` with the same owner (same user and team, or the same key when the key
has neither), oldest first, and replays them as chat history before the new input. `previous_response_id` itself
never reaches the chat provider. The previous turn's `instructions` are carried only when the follow up sends
none (`transformation.py:1641`), and the follow up's spend row joins the first turn's session through
`litellm_trace_id`. The client facing id is encrypted with its owner (`proxy/hooks/responses_id_security.py`), so
a key of another user is refused with 403 before any lookup.

The resource routes (`GET`, `DELETE`, `POST .../cancel`, `GET .../input_items`) decode the provider from the id
and need a native Responses config (`responses/main.py:2250`), which Anthropic does not have. Background mode with
`litellm_settings.responses.background_mode.polling_via_cache: all` answers `litellm_poll_` ids from Redis for any
provider, bridged ones included.

This module boots one owned proxy with prompt storage and cache polling on, so the shared gateway is left alone.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    list_value,
    object_value,
    string_value,
)
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply
from integration.providers.dec1_respapi_support import (
    MESSAGES_PATH,
    REPLY_TEXT,
    Call,
    anthropic,
    chat_bridge,
    chat_json,
    chat_messages,
    json_reply,
    marker,
    output_text,
    peer,
    responses,
    served,
    spend_rows,
    tool_call,
)
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: OwnedProxy

    @property
    def gateway(self) -> Gateway:
        return self.proxy.gateway


def _config(directory: Path) -> Path:
    stock: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    path: Final = directory / "dec1-respapi.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **stock,
                "general_settings": {**object_value(stock["general_settings"]), "store_prompts_in_spend_logs": True},
                "litellm_settings": {
                    **object_value(stock["litellm_settings"]),
                    "responses": {"background_mode": {"polling_via_cache": "all"}},
                },
            }
        )
    )
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("dec1-respapi-chaining")
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(gateway, directory, {}, config=_config(directory)) as owned,
    ):
        yield _Rig(owned)


def _landed(gateway: Gateway, model: str, count: int) -> list[dict[str, JsonValue]]:
    """Wait until `count` spend rows of the test's model group exist, so the next turn's lookup finds them."""
    return spend_rows(gateway, model, count)


def test_a_follow_up_replays_the_first_turn_before_the_new_input(rig: _Rig) -> None:
    """Turn one sends `<marker>` and is answered `REPLY_TEXT`; turn two sends `and then?` with
    `previous_response_id` set to turn one's id.

    The second chat body's `messages` must be exactly `[user <marker>, assistant REPLY_TEXT, user "and then?"]`.
    """
    first: Final = marker()
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        first_id: Final = served(responses(rig.gateway, model, input=first))["id"]
        _landed(rig.gateway, model, 1)
        served(responses(rig.gateway, model, input="and then?", previous_response_id=first_id))
        replayed: Final = chat_messages(provider.last())
        assert [(message["role"], message.get("content")) for message in replayed] == [
            ("user", first),
            ("assistant", REPLY_TEXT),
            ("user", "and then?"),
        ]


def test_previous_response_id_never_reaches_the_chat_provider(rig: _Rig) -> None:
    """The follow up names `previous_response_id`. A chat provider has no such field, so the second chat body
    must not contain the key `previous_response_id` and must not contain the first response id anywhere."""
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        first_id: Final = string_value(served(responses(rig.gateway, model))["id"])
        _landed(rig.gateway, model, 1)
        served(responses(rig.gateway, model, previous_response_id=first_id))
        body: Final = provider.last().body
        assert "previous_response_id" not in body
        assert first_id not in json.dumps(body)


def test_three_turns_replay_in_order(rig: _Rig) -> None:
    """Turns one, two and three chain through `previous_response_id`, each sending its own marker.

    The third chat body must list the three user markers in turn order, each followed by an assistant
    `REPLY_TEXT` except the last: `[u1, a, u2, a, u3]`.
    """
    prompts: Final = (marker(), marker(), marker())
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        previous: str | None = None
        for turn, prompt in enumerate(prompts):
            extra: dict[str, JsonValue] = {"previous_response_id": previous} if previous else {}
            previous = string_value(served(responses(rig.gateway, model, input=prompt, **extra))["id"])
            _landed(rig.gateway, model, turn + 1)
        replayed: Final = chat_messages(provider.last())
        assert [message.get("content") for message in replayed] == [
            prompts[0],
            REPLY_TEXT,
            prompts[1],
            REPLY_TEXT,
            prompts[2],
        ]


def test_the_first_turns_instructions_are_carried_when_the_follow_up_sends_none(rig: _Rig) -> None:
    """Turn one sends `instructions: "Speak like a pirate."`; turn two sends no instructions.

    `_latest_replayable_instructions` carries the stored instructions, so the second chat body must start with
    `{"role": "system", "content": "Speak like a pirate."}` exactly once.
    """
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        first_id: Final = served(responses(rig.gateway, model, instructions="Speak like a pirate."))["id"]
        _landed(rig.gateway, model, 1)
        served(responses(rig.gateway, model, previous_response_id=first_id))
        messages: Final = chat_messages(provider.last())
        assert messages[0] == {"role": "system", "content": "Speak like a pirate."}
        assert [message for message in messages if message["role"] == "system"] == [messages[0]]


def test_the_follow_ups_own_instructions_replace_the_stored_ones(rig: _Rig) -> None:
    """Turn one sends `instructions: "old rules"`; turn two sends `instructions: "new rules"`.

    The second chat body must start with the `new rules` system message, and no message may contain `old rules`:
    replayed history is rebuilt without the stored instructions (`session_handler.py:470`).
    """
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        first_id: Final = served(responses(rig.gateway, model, instructions="old rules"))["id"]
        _landed(rig.gateway, model, 1)
        served(responses(rig.gateway, model, instructions="new rules", previous_response_id=first_id))
        messages: Final = chat_messages(provider.last())
        assert messages[0] == {"role": "system", "content": "new rules"}
        assert "old rules" not in json.dumps(messages)


def test_a_tool_call_turn_is_replayed_so_the_tool_output_has_its_call(rig: _Rig) -> None:
    """Turn one is answered with tool call `call_dec1chain` to `get_weather`; turn two sends only a
    `function_call_output` for `call_dec1chain` (`sunny`) with `previous_response_id`.

    The second chat body must hold, after the first user turn, an assistant message whose `tool_calls[0].id` is
    `call_dec1chain`, then `{"role": "tool", "tool_call_id": "call_dec1chain", "content": "sunny"}`. Without the
    replay OpenAI rejects a tool message with no preceding call.
    """

    def respond(call: Call) -> Reply:
        if any(object_value(message).get("role") == "tool" for message in list_value(call.body["messages"])):
            return json_reply(chat_json(call.body))
        return json_reply(
            chat_json(
                call.body,
                content=None,
                finish_reason="tool_calls",
                message={"tool_calls": [tool_call("call_dec1chain", "get_weather", '{"city":"Oslo"}')]},
            )
        )

    with peer(respond) as provider, rig.gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        first_id: Final = served(responses(rig.gateway, model, tools=[WEATHER_TOOL]))["id"]
        _landed(rig.gateway, model, 1)
        served(
            responses(
                rig.gateway,
                model,
                tools=[WEATHER_TOOL],
                previous_response_id=first_id,
                input=[{"type": "function_call_output", "call_id": "call_dec1chain", "output": "sunny"}],
            )
        )
        messages: Final = chat_messages(provider.last())
        assert [message["role"] for message in messages] == ["user", "assistant", "tool"]
        assert object_value(list_value(messages[1]["tool_calls"])[0])["id"] == "call_dec1chain"
        assert (messages[2]["tool_call_id"], messages[2]["content"]) == ("call_dec1chain", "sunny")


def test_a_follow_up_on_anthropic_replays_the_turn_as_alternating_messages(rig: _Rig) -> None:
    """The same two turn chain on an Anthropic deployment.

    The second `/v1/messages` body must hold three messages with roles `user, assistant, user`, the assistant
    turn reading `REPLY_TEXT`.
    """
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = anthropic(scenario, provider)
        first_id: Final = served(responses(rig.gateway, model))["id"]
        _landed(rig.gateway, model, 1)
        served(responses(rig.gateway, model, input="and then?", previous_response_id=first_id))
        call: Final = provider.last()
        assert call.path == MESSAGES_PATH
        messages: Final = chat_messages(call)
        assert [message["role"] for message in messages] == ["user", "assistant", "user"]
        assert REPLY_TEXT in json.dumps(messages[1])


def test_the_follow_ups_spend_row_joins_the_first_turns_session(rig: _Rig) -> None:
    """Two chained bridged turns. The bridge sets `litellm_trace_id` to the first turn's session
    (`transformation.py:1694`), so both SpendLogs rows must share one non empty `session_id`."""
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        first_id: Final = served(responses(rig.gateway, model))["id"]
        _landed(rig.gateway, model, 1)
        served(responses(rig.gateway, model, previous_response_id=first_id))
        rows: Final = _landed(rig.gateway, model, 2)
        sessions: Final = {row["session_id"] for row in rows}
        assert len(sessions) == 1 and next(iter(sessions)), rows


def test_each_bridged_turn_is_filed_under_the_id_its_caller_saw(rig: _Rig) -> None:
    """Two chained turns. Each SpendLogs row's `request_id` must equal the managed `id` the caller got for that
    turn, oldest first, and both rows have `call_type: aresponses`. The session lookup searches by that id."""
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        first_id: Final = served(responses(rig.gateway, model))["id"]
        _landed(rig.gateway, model, 1)
        second_id: Final = served(responses(rig.gateway, model, previous_response_id=first_id))["id"]
        rows: Final = _landed(rig.gateway, model, 2)
        assert [row["request_id"] for row in rows] == [first_id, second_id]
        assert {row["call_type"] for row in rows} == {"aresponses"}


def test_a_failed_turn_is_not_replayed(rig: _Rig) -> None:
    """Turn one succeeds; turn two (chained on one) is answered 400 by the provider; turn three chains on turn one.

    Failed rows are not conversation turns (`_is_a_conversation_turn`, `session_handler.py:82`), so the third
    chat body must not contain turn two's input marker.
    """
    failing: Final = marker()

    def respond(call: Call) -> Reply:
        if failing in json.dumps(call.body):
            return json_reply({"error": {"message": "dec1 refused", "type": "invalid_request_error"}}, 400)
        return json_reply(chat_json(call.body))

    with peer(respond) as provider, rig.gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        first_id: Final = served(responses(rig.gateway, model))["id"]
        _landed(rig.gateway, model, 1)
        assert responses(rig.gateway, model, input=failing, previous_response_id=first_id).status_code == 400
        _landed(rig.gateway, model, 2)
        served(responses(rig.gateway, model, previous_response_id=first_id))
        assert failing not in json.dumps(provider.last().body)


def test_another_users_key_is_refused_403_for_a_previous_response_id(rig: _Rig) -> None:
    """User A's key creates a response; user B's key sends a follow up with that `previous_response_id`.

    The encrypted id names user A, so `ResponsesIDSecurity.check_user_access_to_response_id` answers 403 with
    `Forbidden. The response id is not associated with the user, who this key belongs to.` and the provider gets
    no second call.
    """
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        owner: Final = scenario.key(models=[model], user_id=scenario.user())
        other: Final = scenario.key(models=[model], user_id=scenario.user())
        first_id: Final = served(responses(rig.gateway, model, key=owner))["id"]
        refused: Final = responses(rig.gateway, model, key=other, previous_response_id=first_id)
        assert refused.status_code == 403, refused.text
        assert "The response id is not associated with the user, who this key belongs to" in refused.text
        assert len(provider.posts()) == 1


def test_a_second_key_of_the_same_user_continues_the_conversation(rig: _Rig) -> None:
    """User A creates a response with key one, then continues it with key two (same user).

    The owner match is on the user, not the key (`session_handler.py:640`), so the second chat body must replay
    turn one's input marker.
    """
    first: Final = marker()
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        user: Final = scenario.user()
        key_one: Final = scenario.key(models=[model], user_id=user)
        key_two: Final = scenario.key(models=[model], user_id=user)
        first_id: Final = served(responses(rig.gateway, model, key=key_one, input=first))["id"]
        _landed(rig.gateway, model, 1)
        served(responses(rig.gateway, model, key=key_two, previous_response_id=first_id))
        assert chat_messages(provider.last())[0] == {"role": "user", "content": first}


def test_retrieving_an_anthropic_bridged_response_is_refused_without_calling_anthropic(rig: _Rig) -> None:
    """`GET /v1/responses/{id}` for a response an Anthropic deployment served.

    `get_responses` finds no Responses config for `anthropic` and raises `GET responses is not supported for
    anthropic` (`responses/main.py:2251`), so the caller gets an error status and the peer receives no GET.
    UNVERIFIED: the mapped status code (a ValueError through `exception_type`, likely 500); only `>= 400` and
    the message are asserted.
    """
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = anthropic(scenario, provider)
        identity: Final = string_value(served(responses(rig.gateway, model))["id"])
        fetched: Final = rig.gateway.request("GET", f"/v1/responses/{identity}")
        assert fetched.status_code >= 400, fetched.text
        assert "GET responses is not supported for anthropic" in fetched.text
        assert [call.method for call in provider.all()] == ["POST"]


@pytest.mark.parametrize(
    ("method", "suffix", "message"),
    [
        ("DELETE", "", "DELETE responses is not supported for anthropic"),
        ("POST", "/cancel", "CANCEL responses is not supported for anthropic"),
        ("GET", "/input_items", "list_input_items is not supported for anthropic"),
    ],
)
def test_resource_routes_on_an_anthropic_bridged_response_are_refused(
    rig: _Rig, method: str, suffix: str, message: str
) -> None:
    """`DELETE /v1/responses/{id}`, `POST /v1/responses/{id}/cancel` or `GET /v1/responses/{id}/input_items` for an
    Anthropic bridged response.

    Each handler raises `<VERB> ... is not supported for anthropic` (`responses/main.py:2020`, `:2574`, `:2416`),
    so the caller gets an error carrying that message and Anthropic receives only the original POST.
    UNVERIFIED: status code mapping, as for GET.
    """
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = anthropic(scenario, provider)
        identity: Final = string_value(served(responses(rig.gateway, model))["id"])
        answer: Final = rig.gateway.request(method, f"/v1/responses/{identity}{suffix}")
        assert answer.status_code >= 400, answer.text
        assert message in answer.text
        assert len(provider.all()) == 1


def test_retrieving_a_chat_bridged_openai_response_goes_to_the_openai_responses_route(rig: _Rig) -> None:
    """`GET /v1/responses/{id}` for a response the `use_chat_completions_api` OpenAI deployment served.

    The id decodes to provider `openai`, which has a native Responses config, so the proxy sends `GET
    /v1/responses/<chat completion id>` to the peer even though the response was made by `/chat/completions`.
    This pins today's behaviour (OpenAI itself would answer 404 for a chat completion id); a port that answers
    from the spend logs instead must change this test deliberately. UNVERIFIED: path shape; read
    `litellm/llms/openai/responses/transformation.py` `transform_get_response_api_request`.
    """
    with (
        peer(lambda call: json_reply(chat_json(call.body, identity="chatcmpl-dec1get"))) as provider,
        rig.gateway.scenario() as scenario,
    ):
        model: Final = chat_bridge(scenario, provider)
        identity: Final = string_value(served(responses(rig.gateway, model))["id"])
        rig.gateway.request("GET", f"/v1/responses/{identity}")
        gets: Final = [call for call in provider.all() if call.method == "GET"]
        assert [call.path for call in gets] == ["/v1/responses/chatcmpl-dec1get"]


def test_a_background_request_on_a_bridged_deployment_returns_a_queued_polling_id(rig: _Rig) -> None:
    """`background: true` on the chat bridge with cache polling on for every model.

    `should_use_polling_for_request` is true, so the caller immediately gets `id` starting `litellm_poll_`,
    `object: response`, `status: queued`, `output: []` and `usage: null` (`polling_handler.py:59`).
    """
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        body: Final = served(responses(rig.gateway, model, background=True))
        assert string_value(body["id"]).startswith("litellm_poll_"), body
        assert (body["object"], body["status"], body["output"], body.get("usage")) == ("response", "queued", [], None)


def test_polling_a_bridged_background_response_ends_completed_with_the_answer(rig: _Rig) -> None:
    """The same background request, then `GET /v1/responses/<litellm_poll_ id>` until it settles.

    The background task streams the bridged call into Redis, so polling must reach `status: completed` with
    `REPLY_TEXT` as output text (the canned stream's pieces join to it), and the peer receives one streamed chat
    call (`stream: true`).
    """
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        polling_id: Final = string_value(served(responses(rig.gateway, model, background=True))["id"])
        final: Final = eventually(
            lambda: rig.gateway.request("GET", f"/v1/responses/{polling_id}").json(),
            lambda state: state.get("status") == "completed",
            seconds=60,
        )
        assert output_text(final) == REPLY_TEXT
        assert provider.only().body["stream"] is True


def test_an_unknown_polling_id_is_404(rig: _Rig) -> None:
    """`GET /v1/responses/litellm_poll_dec1missing` with polling on. No Redis state exists, so the proxy answers 404
    with `Polling response litellm_poll_dec1missing not found or expired` (`endpoints.py:879`)."""
    answer: Final = rig.gateway.request("GET", "/v1/responses/litellm_poll_dec1missing")
    assert answer.status_code == 404, answer.text
    assert "Polling response litellm_poll_dec1missing not found or expired" in answer.text

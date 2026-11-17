"""`previous_response_id` across bridged providers, owners and deployments, round 2.

A bridged provider keeps no conversation, so the proxy rebuilds it from `LiteLLM_SpendLogs`
(`ResponsesSessionHandler.get_all_spend_logs_for_previous_response_id`,
`litellm/responses/litellm_completion_transformation/session_handler.py:610`): it finds the row whose
`request_id` is the referenced response, then every row of that row's `session_id` with the same owner, and
replays each row's stored request (`proxy_server_request`, only kept with `store_prompts_in_spend_logs`) and
stored answer as chat history in front of the new input (`transformation.py:1578`). The history is provider
agnostic, so each carrier's own chat transform turns it into its native turn shape (Gemini `contents` with
`user` and `model` roles, Converse `messages`, plain chat messages for Groq).

The id the caller holds is the proxy's encryption of `resp_<base64 "litellm:custom_llm_provider:<p>;model_id:<id>;
response_id:<chat id>">` plus its owner (`proxy/hooks/responses_id_security.py:372`). The owner is the key's
user, or for a team key with no user only the team, or for a key with neither the key itself
(`resolve_resource_owner_id`, `llms/base_llm/managed_resources/isolation.py:25`). With
`responses_api_deployment_check` on, the decoded `model_id` pins a follow up to the deployment that served the
referenced turn (`router_utils/pre_call_checks/deployment_affinity_check.py:401`).

This module boots one owned proxy with `store_prompts_in_spend_logs: true`, background polling through the cache
for every model, and `router_settings.optional_pre_call_checks: ["responses_api_deployment_check"]`, so the shared
gateway is left alone. Round 1 (`test_dec1_respapi_chaining_owned_proxy.py`) pinned the chat, DeepSeek and
Anthropic replay, instructions carry over, tool call replay, failed turns, cross user refusal and background
polling on the chat carrier.
"""

from __future__ import annotations

import json
import time
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
    object_value,
    string_value,
)
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply
from integration.providers.dec1_respapi2_support import (
    bedrock,
    converse_texts,
    every_respond,
    gemini,
    gemini_stream,
    gemini_texts,
    groq,
    unique_text,
)
from integration.providers.dec1_respapi_support import (
    REPLY_TEXT,
    STREAM_PIECES,
    Call,
    anthropic,
    anthropic_message,
    chat_bridge,
    chat_messages,
    deepseek,
    error_message,
    output_text,
    peer,
    responses,
    served,
    spend_rows,
    stream_events,
)
from pydantic import JsonValue

_GROQ_SERVICE_TIER_BUG: Final = (
    "BUG: a Groq reply without service_tier fails with AttributeError on ModelResponse.service_tier"
    " (groq/chat/transformation.py:329 calls getattr without a default)"
)

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

FOLLOW_UP: Final = "and then?"
UNMANAGED_DETAIL_START: Final = "Forbidden. This response id was not issued by this proxy"
OTHER_USER_DETAIL_START: Final = "Forbidden. The response id is not associated with the user"


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: OwnedProxy

    @property
    def gateway(self) -> Gateway:
        return self.proxy.gateway


def _config(directory: Path) -> Path:
    stock: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    path: Final = directory / "dec1-respapi2.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **stock,
                "general_settings": {**object_value(stock["general_settings"]), "store_prompts_in_spend_logs": True},
                "litellm_settings": {
                    **object_value(stock["litellm_settings"]),
                    "responses": {"background_mode": {"polling_via_cache": "all"}},
                },
                "router_settings": {
                    **object_value(stock["router_settings"]),
                    "optional_pre_call_checks": ["responses_api_deployment_check"],
                },
            }
        )
    )
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("dec1-respapi2-chaining")
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(gateway, directory, {}, config=_config(directory)) as owned,
    ):
        yield _Rig(owned)


def _first_turn(gateway: Gateway, model: str, prompt: str, *, key: str | None = None) -> str:
    """Serve turn one and wait until its spend row lands, so the follow up's lookup can find it."""
    first_id: Final = string_value(served(responses(gateway, model, input=prompt, key=key))["id"])
    spend_rows(gateway, model, 1)
    return first_id


def test_a_gemini_follow_up_replays_the_turn_as_user_and_model_contents(rig: _Rig) -> None:
    """Turn one on a Gemini deployment sends `<prompt>` and is answered `REPLY_TEXT`; turn two sends `and then?`
    with `previous_response_id`. The second `generateContent` body's `contents` must be exactly
    `[(user, <prompt>), (model, REPLY_TEXT), (user, "and then?")]`: the stored assistant turn becomes a Gemini
    `model` turn, not a second user turn."""
    prompt: Final = unique_text()
    with peer(every_respond) as provider, rig.gateway.scenario() as scenario:
        model: Final = gemini(scenario, provider)
        first_id: Final = _first_turn(rig.gateway, model, prompt)
        served(responses(rig.gateway, model, input=FOLLOW_UP, previous_response_id=first_id))
        assert gemini_texts(provider.last()) == [("user", prompt), ("model", REPLY_TEXT), ("user", FOLLOW_UP)]


def test_a_bedrock_follow_up_replays_the_turn_as_converse_messages(rig: _Rig) -> None:
    """The same two turn chain on a Bedrock Converse deployment. The second `converse` body's `messages` must be
    `[(user, <prompt>), (assistant, REPLY_TEXT), (user, "and then?")]`, with each text in a `{"text": ...}`
    content block."""
    prompt: Final = unique_text()
    with peer(every_respond) as provider, rig.gateway.scenario() as scenario:
        model: Final = bedrock(scenario, provider)
        first_id: Final = _first_turn(rig.gateway, model, prompt)
        served(responses(rig.gateway, model, input=FOLLOW_UP, previous_response_id=first_id))
        assert converse_texts(provider.last()) == [
            ("user", prompt),
            ("assistant", REPLY_TEXT),
            ("user", FOLLOW_UP),
        ]


def test_a_groq_follow_up_replays_the_turn_as_chat_messages(rig: _Rig) -> None:
    """The same chain on a Groq deployment. The second chat body's `messages` must be
    `[(user, <prompt>), (assistant, REPLY_TEXT), (user, "and then?")]` and must not carry `previous_response_id`.
    Skipped as a product bug until fixed, see `_GROQ_SERVICE_TIER_BUG`."""
    pytest.skip(_GROQ_SERVICE_TIER_BUG)
    prompt: Final = unique_text()
    with peer(every_respond) as provider, rig.gateway.scenario() as scenario:
        model: Final = groq(scenario, provider)
        first_id: Final = _first_turn(rig.gateway, model, prompt)
        served(responses(rig.gateway, model, input=FOLLOW_UP, previous_response_id=first_id))
        last: Final = provider.last()
        assert [(message["role"], message.get("content")) for message in chat_messages(last)] == [
            ("user", prompt),
            ("assistant", REPLY_TEXT),
            ("user", FOLLOW_UP),
        ]
        assert "previous_response_id" not in last.body


def test_a_conversation_can_continue_on_a_different_bridged_provider(rig: _Rig) -> None:
    """Turn one is served by an Anthropic deployment; turn two names it as `previous_response_id` but asks a
    DeepSeek deployment (another model group, so the affinity pin finds no deployment and normal routing
    applies). The history is read from spend logs, not from the provider, so the DeepSeek chat body must replay
    `[(user, <prompt>), (assistant, REPLY_TEXT), (user, "and then?")]` and the Anthropic peer gets one POST."""
    prompt: Final = unique_text()
    with (
        peer(every_respond) as claude_peer,
        peer(every_respond) as deepseek_peer,
        rig.gateway.scenario() as scenario,
    ):
        claude: Final = anthropic(scenario, claude_peer)
        chat_only: Final = deepseek(scenario, deepseek_peer)
        first_id: Final = _first_turn(rig.gateway, claude, prompt)
        served(responses(rig.gateway, chat_only, input=FOLLOW_UP, previous_response_id=first_id))
        assert [(message["role"], message.get("content")) for message in chat_messages(deepseek_peer.only())] == [
            ("user", prompt),
            ("assistant", REPLY_TEXT),
            ("user", FOLLOW_UP),
        ]
        assert len(claude_peer.posts()) == 1


def test_referencing_an_earlier_turn_still_replays_the_whole_session(rig: _Rig) -> None:
    """Turns one (`<a>`) and two (`<b>`, chained on one) are served; turn three chains on turn ONE again with
    `<c>`. Without a compaction point the handler replays every stored turn of the session
    (`replayed_turns = len(all_spend_logs)`, `session_handler.py:337`), so turn three's chat body is
    `[a, REPLY, b, REPLY, c]`, not the branch `[a, REPLY, c]` OpenAI would build. Pinned as current behaviour: a
    rewrite that starts branching would change what clients that retry from an older id see."""
    first, second, third = unique_text(), unique_text(), unique_text()
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        first_id: Final = _first_turn(rig.gateway, model, first)
        served(responses(rig.gateway, model, input=second, previous_response_id=first_id))
        spend_rows(rig.gateway, model, 2)
        served(responses(rig.gateway, model, input=third, previous_response_id=first_id))
        assert [(message["role"], message.get("content")) for message in chat_messages(provider.last())] == [
            ("user", first),
            ("assistant", REPLY_TEXT),
            ("user", second),
            ("assistant", REPLY_TEXT),
            ("user", third),
        ]


def test_a_streamed_first_turn_is_replayed_with_its_joined_text(rig: _Rig) -> None:
    """Turn one is streamed on the chat carrier (pieces `dec1 `, `bridged `, `reply`); turn two, not streamed,
    names the id from `response.completed`. The replayed assistant turn must be the joined streamed text
    `dec1 bridged reply`, so the stream's spend row stores the assembled answer."""
    prompt: Final = unique_text()
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        events: Final = stream_events(rig.gateway, model, input=prompt)
        first_id: Final = string_value(object_value(events[-1]["response"])["id"])
        spend_rows(rig.gateway, model, 1)
        served(responses(rig.gateway, model, input=FOLLOW_UP, previous_response_id=first_id))
        assert [(message["role"], message.get("content")) for message in chat_messages(provider.last())] == [
            ("user", prompt),
            ("assistant", "".join(STREAM_PIECES)),
            ("user", FOLLOW_UP),
        ]


def test_a_second_key_of_the_same_team_continues_the_conversation(rig: _Rig) -> None:
    """Two keys of one team, neither with a user. The response id names only the team (the owner id is empty for
    a team key, `isolation.py:41`), and the session query matches on team, so key two's follow up must be served
    and replay key one's turn: `[(user, <prompt>), (assistant, REPLY_TEXT), (user, "and then?")]`."""
    prompt: Final = unique_text()
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        team: Final = scenario.team(models=[model])
        key_one: Final = scenario.key(team_id=team, models=[model])
        key_two: Final = scenario.key(team_id=team, models=[model])
        first_id: Final = _first_turn(rig.gateway, model, prompt, key=key_one)
        served(responses(rig.gateway, model, input=FOLLOW_UP, previous_response_id=first_id, key=key_two))
        assert [(message["role"], message.get("content")) for message in chat_messages(provider.last())] == [
            ("user", prompt),
            ("assistant", REPLY_TEXT),
            ("user", FOLLOW_UP),
        ]


def test_a_key_with_no_user_or_team_keeps_its_responses_to_itself(rig: _Rig) -> None:
    """Two keys with neither a user nor a team. Key one's response id is owned by `key:<its hash>`
    (`isolation.py:44`), so key two's follow up must be refused with 403 whose message starts
    `Forbidden. The response id is not associated with the user` (`responses_id_security.py:293`), and the
    peer must have received only turn one."""
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        key_one: Final = scenario.key(models=[model])
        key_two: Final = scenario.key(models=[model])
        first_id: Final = string_value(served(responses(rig.gateway, model, key=key_one))["id"])
        refused: Final = responses(rig.gateway, model, previous_response_id=first_id, key=key_two)
        assert len(provider.posts()) == 1
    assert refused.status_code == 403, refused.text
    assert error_message(refused).startswith(OTHER_USER_DETAIL_START)


def test_a_key_with_no_user_or_team_continues_its_own_conversation(rig: _Rig) -> None:
    """The ownerless key that created the response continues it. The session query falls back to matching the
    same `api_key` when the row has no user and no team (`session_handler.py:670`), so the follow up replays
    `[(user, <prompt>), (assistant, REPLY_TEXT), (user, "and then?")]`."""
    prompt: Final = unique_text()
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        key: Final = scenario.key(models=[model])
        first_id: Final = _first_turn(rig.gateway, model, prompt, key=key)
        served(responses(rig.gateway, model, input=FOLLOW_UP, previous_response_id=first_id, key=key))
        assert [(message["role"], message.get("content")) for message in chat_messages(provider.last())] == [
            ("user", prompt),
            ("assistant", REPLY_TEXT),
            ("user", FOLLOW_UP),
        ]


def test_an_id_this_proxy_never_issued_is_refused_before_the_bridge(rig: _Rig) -> None:
    """A non admin key sends `previous_response_id: resp_dec1respapi2notissued`, which does not decrypt to a
    managed id. With a master key configured and `allow_unmanaged_response_ids` unset, the proxy answers 403 whose
    message starts `Forbidden. This response id was not issued by this proxy` (`responses_id_security.py:56`,
    `:253`) and the bridged peer receives no POST."""
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        key: Final = scenario.key(models=[model])
        refused: Final = responses(rig.gateway, model, previous_response_id="resp_dec1respapi2notissued", key=key)
        assert provider.posts() == ()
    assert refused.status_code == 403, refused.text
    assert error_message(refused).startswith(UNMANAGED_DETAIL_START)


def test_the_follow_up_echoes_the_previous_response_id_the_caller_sent(rig: _Rig) -> None:
    """The follow up's body echoes `previous_response_id` (`transformation.py:3651`). It should be the id the
    caller sent, but the pre call hook replaces the field with the decrypted internal id before the bridge
    (`responses_id_security.py:193`) and nothing maps it back, so the caller receives an id it never held."""
    pytest.skip("BUG: bridged follow up echoes the decrypted internal previous_response_id, not the caller's id")
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        key: Final = scenario.key(models=[model])
        first_id: Final = _first_turn(rig.gateway, model, unique_text(), key=key)
        body: Final = served(responses(rig.gateway, model, previous_response_id=first_id, key=key))
    assert body["previous_response_id"] == first_id


def test_a_follow_up_is_routed_to_the_deployment_that_served_the_first_turn(rig: _Rig) -> None:
    """One model group has two chat bridged deployments on two peers. Turn one lands on one of them; four follow
    ups that each name turn one's id must all reach that same peer and none the other, because the decoded
    `model_id` pins the deployment (`deployment_affinity_check.py:401-417`)."""
    with peer() as left, peer() as right, rig.gateway.scenario() as scenario:
        group: Final = chat_bridge(scenario, left)
        chat_bridge(scenario, right, name=group)
        first_id: Final = string_value(served(responses(rig.gateway, group))["id"])
        first_peer: Final = left if left.posts() else right
        other_peer: Final = right if first_peer is left else left
        for _ in range(4):
            served(responses(rig.gateway, group, previous_response_id=first_id))
        assert (len(first_peer.posts()), len(other_peer.posts())) == (5, 0)


def test_a_bridged_response_id_is_not_the_raw_chat_id_and_differs_per_turn(rig: _Rig) -> None:
    """Two turns on one Gemini deployment return two different `resp_` ids, and neither contains the Gemini
    `responseId` or a `chatcmpl` id, because the client facing id is the encrypted managed id
    (`responses_id_security.py:406`)."""
    with peer(every_respond) as provider, rig.gateway.scenario() as scenario:
        model: Final = gemini(scenario, provider)
        ids: Final = [string_value(served(responses(rig.gateway, model))["id"]) for _ in range(2)]
    assert len(set(ids)) == 2
    assert all(identity.startswith("resp_") and "chatcmpl" not in identity for identity in ids), ids


def _anthropic_stream(call: Call) -> Reply:
    """The Anthropic SSE answer to a `stream: true` Messages call: `STREAM_PIECES` as text deltas, then usage."""
    message: Final = anthropic_message(call.body)
    start: Final = {**message, "content": [], "stop_reason": None}
    events: Final[list[tuple[str, dict[str, JsonValue]]]] = [
        ("message_start", {"type": "message_start", "message": start}),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        *(
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece}},
            )
            for piece in STREAM_PIECES
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": object_value(message["usage"])["output_tokens"]},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {name}\ndata: {json.dumps(event)}\n\n".encode() for name, event in events),
    )


def _streaming_respond(call: Call) -> Reply:
    """`every_respond`, but a streamed Anthropic or Gemini call is answered with `STREAM_PIECES` as SSE."""
    if call.method == "POST" and call.path.endswith("/v1/messages") and call.body.get("stream") is True:
        return _anthropic_stream(call)
    if call.method == "POST" and ":streamGenerateContent" in call.path:
        return gemini_stream(STREAM_PIECES)
    return every_respond(call)


@pytest.mark.parametrize("carrier", ["anthropic", "gemini"])
def test_a_background_request_on_a_non_openai_provider_polls_to_completed(rig: _Rig, carrier: str) -> None:
    """`background: true` on an Anthropic or Gemini deployment with cache polling on for every model answers a
    queued `litellm_poll_` id (`proxy/response_polling/polling_handler.py:59`); polling
    `GET /v1/responses/<id>` must settle on `status: completed` whose output text is `REPLY_TEXT`, and the
    provider peer receives exactly one POST (the background task, not one per poll).

    Test mistake fixed (run 44176682961): the background task always streams from the provider
    (`data["stream"] = True`, litellm/proxy/response_polling/background_streaming.py:181 on main), so the peer
    must answer the streamed call. Before, Gemini settled on the text of `gemini_stream`'s default pieces
    ("dec1 gemini reply") and Anthropic on "" because the peer answered a stream with a plain JSON message. The
    peer now streams `STREAM_PIECES` for both, and the one POST must be the streamed form."""
    with peer(_streaming_respond) as provider, rig.gateway.scenario() as scenario:
        model: Final = anthropic(scenario, provider) if carrier == "anthropic" else gemini(scenario, provider)
        queued: Final = served(responses(rig.gateway, model, background=True))
        poll_id: Final = string_value(queued["id"])
        assert poll_id.startswith("litellm_poll_"), poll_id
        settled: Final = eventually(
            lambda: JSON_OBJECT.validate_json(rig.gateway.request("GET", f"/v1/responses/{poll_id}").content),
            lambda body: body.get("status") in {"completed", "failed", "incomplete"},
            seconds=60,
        )
        assert settled["status"] == "completed", settled
        assert output_text(settled) == REPLY_TEXT
        (call,) = provider.posts()
        if carrier == "anthropic":
            assert call.body.get("stream") is True, call.body
        else:
            assert ":streamGenerateContent" in call.path, call.path


def test_a_follow_up_waits_for_a_just_finished_turn(rig: _Rig) -> None:
    """Turn two is sent immediately after turn one returns, without waiting for its spend row. The intended
    contract (`get_all_spend_logs_for_previous_response_id` docstring, `session_handler.py:610`) is that a
    just finished turn is still replayed: `[user, assistant, user]`. The retry window is only
    `RESPONSES_SESSION_LOOKUP_MAX_ATTEMPTS` (3) times `RESPONSES_SESSION_LOOKUP_RETRY_INTERVAL` (0.2 s)
    (`litellm/constants.py:1944-1945`, `session_handler.py:678`), shorter than the spend writer's
    `proxy_batch_write_at` (1 s on this proxy, 10 s by default), so the history is usually empty and only the new
    input is sent, silently dropping the conversation."""
    pytest.skip("BUG: session lookup retries for 0.4 s, shorter than the spend log batch write interval")
    prompt: Final = unique_text()
    with peer() as provider, rig.gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        first_id: Final = string_value(served(responses(rig.gateway, model, input=prompt))["id"])
        started: Final = time.monotonic()
        served(responses(rig.gateway, model, input=FOLLOW_UP, previous_response_id=first_id))
        elapsed: Final = time.monotonic() - started
        roles: Final[list[JsonValue]] = [message["role"] for message in chat_messages(provider.last())]
        assert roles == ["user", "assistant", "user"], (roles, elapsed)

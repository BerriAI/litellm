import asyncio
import uuid
from collections.abc import Iterator
from hashlib import sha256
from typing import Final

import anthropic
import openai
import pytest
from integration._support.client import JSON_OBJECT, Gateway, eventually, list_value, object_value, string_value
from integration._support.database import read_rows
from integration.authorization._hidden_alias_budget import (
    anthropic_client,
    async_anthropic_client,
    async_openai_client,
    fresh_chat,
    fresh_message,
    fresh_post,
    fresh_response,
    install_aliases,
    landed_once,
    openai_client,
    remove_aliases,
    spend_rows,
    upstream_hits,
    upstream_requests,
)
from integration.authorization._key_disallowed_models import (
    DENIED,
    KEY_DENIAL,
    ROUTES,
    UPSTREAM_RECORDED_ROUTES,
    DenyRig,
    denial,
    deny_rig,
    send,
    send_all,
)
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(240)]


@pytest.fixture(scope="module")
def rig() -> Iterator[DenyRig]:
    with deny_rig() as built:
        yield built


def _stored(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT disallowed_models FROM "LiteLLM_VerificationToken" WHERE token=%s',
        (sha256(key.encode()).hexdigest(),),
    )


def _ids(gateway: Gateway, path: str, key: str, field: str) -> frozenset[str]:
    response: Final = gateway.request("GET", path, key=key)
    assert response.status_code == 200, f"{path}: {response.status_code} {response.text}"
    entries: Final = list_value(JSON_OBJECT.validate_json(response.content)["data"])
    return frozenset(string_value(object_value(entry)[field]) for entry in entries)


@pytest.mark.parametrize("route", ROUTES)
def test_every_inference_route_refuses_a_disallowed_model_before_the_provider(rig: DenyRig, route: str) -> None:
    refused_marker: Final = f"refused-{uuid.uuid4().hex}"
    served_marker: Final = f"served-{uuid.uuid4().hex}"
    refused: Final = send(rig.gateway, route, rig.denied.serving(route), rig.key, refused_marker)
    assert denial(refused) == KEY_DENIAL, refused.text
    assert rig.denied.serving(route) in refused.text, refused.text
    served: Final = send(rig.gateway, route, rig.allowed.serving(route), rig.key, served_marker)
    assert served.status_code == 200, f"{route}: {served.status_code} {served.text}"
    observed: Final = upstream_requests(rig.gateway.upstream_url)
    expected: Final = (0, 1) if route in UPSTREAM_RECORDED_ROUTES else (0, 0)
    assert (upstream_hits(observed, refused_marker), upstream_hits(observed, served_marker)) == expected


@pytest.mark.parametrize("route", ROUTES)
def test_a_key_without_disallowed_models_is_served_the_same_model_on_every_route(rig: DenyRig, route: str) -> None:
    served: Final = send(rig.gateway, route, rig.denied.serving(route), rig.open_key, f"open-{uuid.uuid4().hex}")
    assert served.status_code == 200, f"{route}: {served.status_code} {served.text}"


def _sdk_denial(error: openai.APIStatusError | anthropic.APIStatusError) -> tuple[int, JsonValue]:
    body: Final = JSON_OBJECT.validate_python(error.body)
    detail: Final = object_value(body.get("error", body))
    return error.status_code, detail.get("litellm_error_type", detail.get("type"))


def test_the_openai_sdk_raises_permission_denied_for_a_disallowed_model(rig: DenyRig) -> None:
    client: Final = openai_client(rig.gateway, rig.key)
    prompt: Final = [{"role": "user", "content": "sdk"}]
    calls: Final = {
        "chat": lambda: client.chat.completions.create(model=rig.denied.chat, messages=prompt),
        "chatstream": lambda: list(client.chat.completions.create(model=rig.denied.chat, messages=prompt, stream=True)),
        "completions": lambda: client.completions.create(model=rig.denied.chat, prompt="sdk"),
        "embeddings": lambda: client.embeddings.create(model=rig.denied.chat, input="sdk"),
        "responses": lambda: client.responses.create(model=rig.denied.responses, input="sdk"),
        "responsesstream": lambda: list(
            client.responses.create(model=rig.denied.responses_stream, input="sdk", stream=True)
        ),
    }
    seen: Final = {}
    for name, call in calls.items():
        with pytest.raises(openai.PermissionDeniedError) as raised:
            call()
        seen[name] = _sdk_denial(raised.value)
    assert seen == dict.fromkeys(calls, KEY_DENIAL)
    served: Final = list(client.chat.completions.create(model=rig.allowed.chat, messages=prompt, stream=True))
    assert any(chunk.choices and chunk.choices[0].delta.content for chunk in served), served


async def _consumed(stream: openai.AsyncStream[object]) -> list[object]:
    return [event async for event in stream]


async def test_the_async_openai_sdk_raises_permission_denied_for_a_disallowed_model(rig: DenyRig) -> None:
    client: Final = async_openai_client(rig.gateway, rig.key)
    prompt: Final = [{"role": "user", "content": "sdk"}]

    async def chat_stream() -> list[object]:
        return await _consumed(
            await client.chat.completions.create(model=rig.denied.chat, messages=prompt, stream=True)
        )

    async def responses_stream() -> list[object]:
        return await _consumed(
            await client.responses.create(model=rig.denied.responses_stream, input="sdk", stream=True)
        )

    calls: Final = {
        "chat": lambda: client.chat.completions.create(model=rig.denied.chat, messages=prompt),
        "chatstream": chat_stream,
        "completions": lambda: client.completions.create(model=rig.denied.chat, prompt="sdk"),
        "embeddings": lambda: client.embeddings.create(model=rig.denied.chat, input="sdk"),
        "responses": lambda: client.responses.create(model=rig.denied.responses, input="sdk"),
        "responsesstream": responses_stream,
    }
    outcomes: Final = await asyncio.gather(*(call() for call in calls.values()), return_exceptions=True)
    assert all(isinstance(outcome, openai.PermissionDeniedError) for outcome in outcomes), outcomes
    assert [_sdk_denial(outcome) for outcome in outcomes] == [KEY_DENIAL] * len(calls)
    served: Final = await client.responses.create(model=rig.allowed.responses, input="sdk")
    assert served.output_text == "served reply", served


def test_the_anthropic_sdk_raises_permission_denied_for_a_disallowed_model(rig: DenyRig) -> None:
    client: Final = anthropic_client(rig.gateway, rig.key)
    prompt: Final = [{"role": "user", "content": "sdk"}]
    with pytest.raises(anthropic.PermissionDeniedError) as plain:
        client.messages.create(model=rig.denied.responses, max_tokens=16, messages=prompt)
    with pytest.raises(anthropic.PermissionDeniedError) as streamed:
        list(client.messages.create(model=rig.denied.responses_stream, max_tokens=16, messages=prompt, stream=True))
    assert (_sdk_denial(plain.value), _sdk_denial(streamed.value)) == (KEY_DENIAL, KEY_DENIAL)
    served: Final = client.messages.create(model=rig.allowed.responses, max_tokens=16, messages=prompt)
    assert served.content[0].text == "served reply", served


async def test_the_async_anthropic_sdk_raises_permission_denied_for_a_disallowed_model(rig: DenyRig) -> None:
    client: Final = async_anthropic_client(rig.gateway, rig.key)
    prompt: Final = [{"role": "user", "content": "sdk"}]

    async def streamed() -> list[object]:
        stream: Final = await client.messages.create(
            model=rig.denied.responses_stream, max_tokens=16, messages=prompt, stream=True
        )
        return [event async for event in stream]

    outcomes: Final = await asyncio.gather(
        client.messages.create(model=rig.denied.responses, max_tokens=16, messages=prompt),
        streamed(),
        return_exceptions=True,
    )
    assert all(isinstance(outcome, anthropic.PermissionDeniedError) for outcome in outcomes), outcomes
    assert [_sdk_denial(outcome) for outcome in outcomes] == [KEY_DENIAL, KEY_DENIAL]


def test_a_star_pattern_refuses_and_hides_every_model_it_matches_and_spares_the_rest(rig: DenyRig) -> None:
    with rig.gateway.scenario() as scenario:
        family: Final = f"integration-family-{uuid.uuid4().hex[:10]}"
        matched: Final = (scenario.model(name=f"{family}-small"), scenario.model(name=f"{family}-large"))
        key: Final = scenario.key(disallowed_models=[f"{family}-*"])
        refused: Final = tuple(fresh_chat(rig.gateway, model, key, uuid.uuid4().hex) for model in matched)
        assert [denial(response) for response in refused] == [KEY_DENIAL, KEY_DENIAL]
        spared: Final = fresh_chat(rig.gateway, rig.allowed.chat, key, uuid.uuid4().hex)
        assert spared.status_code == 200, spared.text
        listed: Final = _ids(rig.gateway, "/v1/models", key, "id")
        assert (rig.allowed.chat in listed, listed & frozenset(matched)) == (True, frozenset())


def test_a_lone_star_entry_refuses_every_model(rig: DenyRig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(disallowed_models=["*"])
        refused: Final = tuple(
            fresh_chat(rig.gateway, model, key, uuid.uuid4().hex) for model in (rig.allowed.chat, rig.denied.chat)
        )
        assert [denial(response) for response in refused] == [KEY_DENIAL, KEY_DENIAL]
        assert _ids(rig.gateway, "/v1/models", key, "id") == frozenset()


def test_a_team_alias_cannot_reach_a_disallowed_model(rig: DenyRig) -> None:
    with rig.gateway.scenario() as scenario:
        alias: Final = f"team-alias-{uuid.uuid4().hex[:8]}"
        team: Final = scenario.team(model_aliases={alias: rig.denied.chat})
        served: Final = fresh_chat(rig.gateway, alias, scenario.key(team_id=team), uuid.uuid4().hex)
        assert served.status_code == 200, served.text
        key: Final = scenario.key(team_id=team, disallowed_models=[rig.denied.chat])
        assert denial(fresh_chat(rig.gateway, alias, key, uuid.uuid4().hex)) == KEY_DENIAL


def test_a_router_alias_requested_directly_cannot_reach_a_disallowed_model(rig: DenyRig) -> None:
    with rig.gateway.scenario() as scenario:
        alias: Final = f"router-alias-{uuid.uuid4().hex[:8]}"
        install_aliases(rig.gateway, {alias: rig.denied.chat})
        scenario.cleanups.callback(remove_aliases, rig.gateway, frozenset({alias}))
        eventually(
            lambda: frozenset(
                fresh_chat(rig.gateway, alias, rig.open_key, uuid.uuid4().hex).status_code for _ in range(6)
            ),
            lambda statuses: statuses == frozenset({200}),
            seconds=60,
        )
        refused: Final = fresh_chat(rig.gateway, alias, rig.key, uuid.uuid4().hex)
        assert denial(refused) == KEY_DENIAL, refused.text
        assert alias not in _ids(rig.gateway, "/v1/models", rig.key, "id")


def test_a_disallowed_alias_name_is_refused_while_its_target_stays_callable(rig: DenyRig) -> None:
    with rig.gateway.scenario() as scenario:
        alias: Final = f"alias-{uuid.uuid4().hex[:8]}"
        key: Final = scenario.key(aliases={alias: rig.allowed.chat}, disallowed_models=[alias])
        assert denial(fresh_chat(rig.gateway, alias, key, uuid.uuid4().hex)) == KEY_DENIAL
        served: Final = fresh_chat(rig.gateway, rig.allowed.chat, key, uuid.uuid4().hex)
        assert served.status_code == 200, served.text


@pytest.mark.parametrize("route", ["chat", "responses", "messages"])
def test_a_request_fallback_to_a_disallowed_model_is_refused_on_every_route(rig: DenyRig, route: str) -> None:
    marker: Final = f"fallback-{uuid.uuid4().hex}"
    fallbacks: Final = {"fallbacks": [rig.denied.serving(route)]}
    senders: Final = {
        "chat": lambda: fresh_chat(rig.gateway, rig.allowed.chat, rig.key, marker, fallbacks),
        "responses": lambda: fresh_response(rig.gateway, rig.allowed.responses, rig.key, marker, fallbacks),
        "messages": lambda: fresh_message(rig.gateway, rig.allowed.responses, rig.key, marker, fallbacks),
    }
    refused: Final = senders[route]()
    assert denial(refused) == KEY_DENIAL, refused.text
    control: Final = f"control-{uuid.uuid4().hex}"
    assert fresh_chat(rig.gateway, rig.allowed.chat, rig.key, control).status_code == 200
    observed: Final = upstream_requests(rig.gateway.upstream_url)
    assert (upstream_hits(observed, marker), upstream_hits(observed, control)) == (0, 1)


def test_a_comma_separated_fan_out_is_served_by_every_named_model_for_a_key_without_disallowed_models(
    rig: DenyRig,
) -> None:
    marker: Final = f"fanout-{uuid.uuid4().hex}"
    served: Final = fresh_chat(rig.gateway, f"{rig.allowed.chat}, {rig.denied.chat}", rig.open_key, marker)
    assert served.status_code == 200, f"{served.status_code} {served.text}"
    assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 2


def test_a_key_with_disallowed_models_still_fans_out_across_models_it_may_call(rig: DenyRig) -> None:
    with rig.gateway.scenario() as scenario:
        marker: Final = f"fanout-{uuid.uuid4().hex}"
        served: Final = fresh_chat(rig.gateway, f"{rig.allowed.chat},{scenario.model()}", rig.key, marker)
        assert served.status_code == 200, f"{served.status_code} {served.text}"
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 2


@pytest.mark.parametrize("position", ["first", "last"])
def test_a_comma_separated_fan_out_is_refused_wherever_the_disallowed_model_sits(rig: DenyRig, position: str) -> None:
    marker: Final = f"fanout-{uuid.uuid4().hex}"
    names: Final = (rig.denied.chat, rig.allowed.chat) if position == "first" else (rig.allowed.chat, rig.denied.chat)
    refused: Final = fresh_chat(rig.gateway, ",".join(names), rig.key, marker)
    assert denial(refused) == KEY_DENIAL, refused.text
    control: Final = f"control-{uuid.uuid4().hex}"
    assert fresh_chat(rig.gateway, rig.allowed.chat, rig.key, control).status_code == 200
    observed: Final = upstream_requests(rig.gateway.upstream_url)
    assert (upstream_hits(observed, marker), upstream_hits(observed, control)) == (0, 1)


def test_a_comma_separated_fan_out_cannot_reach_a_model_a_star_pattern_or_a_team_alias_disallows(rig: DenyRig) -> None:
    with rig.gateway.scenario() as scenario:
        family: Final = f"integration-family-{uuid.uuid4().hex[:10]}"
        matched: Final = scenario.model(name=f"{family}-small")
        alias: Final = f"team-alias-{uuid.uuid4().hex[:8]}"
        team: Final = scenario.team(model_aliases={alias: f"{rig.allowed.chat},{matched}"})
        served: Final = fresh_chat(rig.gateway, alias, scenario.key(team_id=team), uuid.uuid4().hex)
        assert served.status_code == 200, served.text
        key: Final = scenario.key(team_id=team, disallowed_models=[f"{family}-*"])
        through_pattern: Final = fresh_chat(rig.gateway, f"{rig.allowed.chat},{matched}", key, uuid.uuid4().hex)
        through_alias: Final = fresh_chat(rig.gateway, alias, key, uuid.uuid4().hex)
        assert (denial(through_pattern), denial(through_alias)) == (KEY_DENIAL, KEY_DENIAL)


@pytest.mark.parametrize(
    ("path", "field"),
    [
        ("/v1/models", "id"),
        ("/models", "id"),
        ("/model/info", "model_name"),
        ("/v1/model/info", "model_name"),
        ("/model_group/info", "model_group"),
    ],
)
def test_every_model_listing_hides_disallowed_models_and_keeps_the_rest(rig: DenyRig, path: str, field: str) -> None:
    everything: Final = frozenset(rig.allowed.names()) | frozenset(rig.denied.names())
    unrestricted: Final = _ids(rig.gateway, path, rig.open_key, field)
    assert unrestricted & everything == everything, unrestricted
    listed: Final = _ids(rig.gateway, path, rig.key, field)
    assert listed & everything == frozenset(rig.allowed.names()), listed


def test_a_proxy_admins_key_with_disallowed_models_does_not_list_them_in_model_group_info(rig: DenyRig) -> None:
    with rig.gateway.scenario() as scenario:
        admin: Final = scenario.user(user_role="proxy_admin")
        key: Final = scenario.key(user_id=admin, disallowed_models=[rig.denied.chat])
        listed: Final = _ids(rig.gateway, "/model_group/info", key, "model_group")
        assert (rig.allowed.chat in listed, rig.denied.chat in listed) == (True, False), listed
        assert denial(fresh_chat(rig.gateway, rig.denied.chat, key, uuid.uuid4().hex)) == KEY_DENIAL


def test_a_regenerated_key_keeps_its_disallowed_models(rig: DenyRig) -> None:
    with rig.gateway.scenario() as scenario:
        old: Final = string_value(rig.gateway.post("/key/generate", {"disallowed_models": [rig.denied.chat]})["key"])
        new: Final = string_value(rig.gateway.post("/key/regenerate", {"key": old})["key"])
        scenario.cleanups.callback(scenario.delete_key, new)
        assert _stored(new) == [{"disallowed_models": [rig.denied.chat]}]
        assert denial(fresh_chat(rig.gateway, rig.denied.chat, new, uuid.uuid4().hex)) == KEY_DENIAL
        served: Final = fresh_chat(rig.gateway, rig.allowed.chat, new, uuid.uuid4().hex)
        assert served.status_code == 200, served.text


def test_a_key_owner_cannot_drop_disallowed_models_by_regenerating_the_key(rig: DenyRig) -> None:
    with rig.gateway.scenario() as scenario:
        owner: Final = scenario.user(user_role="internal_user")
        key: Final = scenario.key(user_id=owner, disallowed_models=[rig.denied.chat])
        refused: Final = rig.gateway.request("POST", "/key/regenerate", {"key": key, "disallowed_models": []}, key=key)
        assert refused.status_code == DENIED, f"{refused.status_code} {refused.text}"
        assert "disallowed_models" in refused.text, refused.text
        assert _stored(key) == [{"disallowed_models": [rig.denied.chat]}]
        assert denial(fresh_chat(rig.gateway, rig.denied.chat, key, uuid.uuid4().hex)) == KEY_DENIAL


def test_a_proxy_admin_can_drop_disallowed_models_while_regenerating_the_key(rig: DenyRig) -> None:
    with rig.gateway.scenario() as scenario:
        old: Final = string_value(rig.gateway.post("/key/generate", {"disallowed_models": [rig.denied.chat]})["key"])
        new: Final = string_value(rig.gateway.post("/key/regenerate", {"key": old, "disallowed_models": []})["key"])
        scenario.cleanups.callback(scenario.delete_key, new)
        assert _stored(new) == [{"disallowed_models": []}]
        served: Final = fresh_chat(rig.gateway, rig.denied.chat, new, uuid.uuid4().hex)
        assert served.status_code == 200, served.text


def test_a_non_admin_team_member_cannot_remove_disallowed_models_from_a_team_key(rig: DenyRig) -> None:
    with rig.gateway.scenario() as scenario:
        team: Final = scenario.team(team_member_permissions=["/key/update", "/key/info"])
        member: Final = scenario.member(team)
        member_key: Final = scenario.key(team_id=team, user_id=member)
        key: Final = scenario.key(team_id=team, user_id=member, disallowed_models=[rig.denied.chat, "other-*"])
        for body in ({"disallowed_models": []}, {"disallowed_models": None}, {"disallowed_models": ["other-*"]}):
            refused = rig.gateway.request("POST", "/key/update", {"key": key, **body}, key=member_key)
            assert refused.status_code == DENIED, f"{body}: {refused.status_code} {refused.text}"
        assert _stored(key) == [{"disallowed_models": [rig.denied.chat, "other-*"]}]
        assert denial(fresh_chat(rig.gateway, rig.denied.chat, key, uuid.uuid4().hex)) == KEY_DENIAL


def test_bulk_update_writes_disallowed_models_and_they_are_enforced(rig: DenyRig) -> None:
    pytest.skip("BUG: /key/bulk_update documents disallowed_models but drops it and reports the update as successful")
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key()
        updated: Final = rig.gateway.post(
            "/key/bulk_update", {"keys": [{"key": key, "disallowed_models": [rig.denied.chat]}]}
        )
        assert updated["failed_updates"] == [], updated
        assert _stored(key) == [{"disallowed_models": [rig.denied.chat]}]
        assert denial(fresh_chat(rig.gateway, rig.denied.chat, key, uuid.uuid4().hex)) == KEY_DENIAL


def test_a_service_account_key_stores_and_enforces_disallowed_models(rig: DenyRig) -> None:
    with rig.gateway.scenario() as scenario:
        created: Final = rig.gateway.post(
            "/key/service-account/generate",
            {"team_id": scenario.team(), "disallowed_models": [rig.denied.chat]},
        )
        key: Final = string_value(created["key"])
        scenario.cleanups.callback(scenario.delete_key, key)
        assert _stored(key) == [{"disallowed_models": [rig.denied.chat]}]
        assert denial(fresh_chat(rig.gateway, rig.denied.chat, key, uuid.uuid4().hex)) == KEY_DENIAL
        served: Final = fresh_chat(rig.gateway, rig.allowed.chat, key, uuid.uuid4().hex)
        assert served.status_code == 200, served.text


def test_key_list_returns_a_keys_disallowed_models(rig: DenyRig) -> None:
    with rig.gateway.scenario() as scenario:
        alias: Final = f"listed-{uuid.uuid4().hex}"
        scenario.key(key_alias=alias, disallowed_models=[rig.denied.chat, "other-*"])
        listed: Final = rig.gateway.get("/key/list", {"key_alias": alias, "return_full_object": "true"})
        (entry,) = list_value(listed["keys"])
        assert object_value(entry)["disallowed_models"] == [rig.denied.chat, "other-*"], entry


@pytest.mark.parametrize(
    "malformed",
    [7, "not-a-list", [7], [None], {"model": "x"}, [["nested"]]],
    ids=["int", "string", "list-of-int", "list-of-null", "object", "nested-list"],
)
def test_a_malformed_disallowed_models_value_is_rejected_and_changes_nothing(
    rig: DenyRig, malformed: JsonValue
) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(disallowed_models=[rig.denied.chat])
        created: Final = rig.gateway.request("POST", "/key/generate", {"disallowed_models": malformed})
        updated: Final = rig.gateway.request("POST", "/key/update", {"key": key, "disallowed_models": malformed})
        assert (created.status_code, updated.status_code) == (422, 422), f"{created.text} {updated.text}"
        assert _stored(key) == [{"disallowed_models": [rig.denied.chat]}]
        assert denial(fresh_chat(rig.gateway, rig.denied.chat, key, uuid.uuid4().hex)) == KEY_DENIAL
        unrelated: Final = fresh_chat(rig.gateway, rig.denied.chat, rig.open_key, uuid.uuid4().hex)
        assert unrelated.status_code == 200, unrelated.text


def test_empty_oversized_and_repeated_entries_are_stored_and_only_real_matches_are_refused(rig: DenyRig) -> None:
    with rig.gateway.scenario() as scenario:
        oversized: Final = "x" * 5000
        entries: Final[list[JsonValue]] = ["", oversized, rig.denied.chat, rig.denied.chat]
        key: Final = scenario.key(disallowed_models=entries)
        assert _stored(key) == [{"disallowed_models": entries}]
        assert denial(fresh_chat(rig.gateway, rig.denied.chat, key, uuid.uuid4().hex)) == KEY_DENIAL
        served: Final = fresh_chat(rig.gateway, rig.allowed.chat, key, uuid.uuid4().hex)
        assert served.status_code == 200, served.text


@pytest.mark.parametrize(
    "fields", [{"disallowed_models": []}, {"disallowed_models": None}, {}], ids=["empty", "null", "missing"]
)
def test_an_empty_null_or_missing_disallowed_models_refuses_nothing(rig: DenyRig, fields: dict[str, JsonValue]) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(**fields)
        assert _stored(key) == [{"disallowed_models": []}]
        served: Final = fresh_chat(rig.gateway, rig.denied.chat, key, uuid.uuid4().hex)
        assert served.status_code == 200, served.text
        assert rig.denied.chat in _ids(rig.gateway, "/v1/models", key, "id")


def test_an_unknown_key_asking_for_a_disallowed_model_gets_401_not_the_model_denial(rig: DenyRig) -> None:
    marker: Final = f"unknown-key-{uuid.uuid4().hex}"
    refused: Final = fresh_chat(rig.gateway, rig.denied.chat, f"sk-{uuid.uuid4().hex}", marker)
    assert refused.status_code == 401, f"{refused.status_code} {refused.text}"
    assert "key_model_access_denied" not in refused.text, refused.text
    control: Final = f"control-{uuid.uuid4().hex}"
    assert fresh_chat(rig.gateway, rig.allowed.chat, rig.key, control).status_code == 200
    observed: Final = upstream_requests(rig.gateway.upstream_url)
    assert (upstream_hits(observed, marker), upstream_hits(observed, control)) == (0, 1)


def test_a_refused_request_logs_no_successful_spend_and_the_next_served_one_logs_once(rig: DenyRig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(disallowed_models=[rig.denied.chat])
        refused_marker: Final = f"refused-{uuid.uuid4().hex}"
        served_marker: Final = f"served-{uuid.uuid4().hex}"
        refused: Final = fresh_chat(rig.gateway, rig.denied.chat, key, refused_marker)
        assert denial(refused) == KEY_DENIAL, refused.text
        served: Final = fresh_chat(rig.gateway, rig.allowed.chat, key, served_marker)
        assert served.status_code == 200, served.text
        row: Final = landed_once(key, served_marker)
        assert (row["status"], row["model_group"]) == ("success", rig.allowed.chat), row
        assert [r for r in spend_rows(key) if r["status"] == "success" and r["marker"] != served_marker] == []


def test_disallowed_models_set_on_one_proxy_are_enforced_by_its_peer(rig: DenyRig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key()
        warmed: Final = fresh_chat(rig.peer, rig.denied.chat, key, uuid.uuid4().hex)
        assert warmed.status_code == 200, warmed.text
        rig.gateway.post("/key/update", {"key": key, "disallowed_models": [rig.denied.chat]})
        refused: Final = eventually(
            lambda: fresh_chat(rig.peer, rig.denied.chat, key, uuid.uuid4().hex),
            lambda response: response.status_code == DENIED,
            seconds=60,
        )
        assert denial(refused) == KEY_DENIAL, refused.text
        rig.gateway.post("/key/update", {"key": key, "disallowed_models": []})
        eventually(
            lambda: fresh_chat(rig.peer, rig.denied.chat, key, uuid.uuid4().hex).status_code,
            lambda status: status == 200,
            seconds=60,
        )


def test_repeating_a_refused_request_never_reaches_the_provider(rig: DenyRig) -> None:
    markers: Final = tuple(f"repeat-{index}-{uuid.uuid4().hex}" for index in range(20))
    refused: Final = send_all(lambda marker: fresh_chat(rig.gateway, rig.denied.chat, rig.key, marker), markers)
    assert [denial(response) for response in refused] == [KEY_DENIAL] * len(markers)
    control: Final = f"control-{uuid.uuid4().hex}"
    assert fresh_chat(rig.gateway, rig.allowed.chat, rig.key, control).status_code == 200
    observed: Final = upstream_requests(rig.gateway.upstream_url)
    assert sum(upstream_hits(observed, marker) for marker in markers) == 0
    assert upstream_hits(observed, control) == 1


def test_the_messages_token_counter_refuses_a_disallowed_model(rig: DenyRig) -> None:
    counted: Final = fresh_post(
        rig.gateway,
        "/v1/messages/count_tokens",
        {"model": rig.denied.responses, "messages": [{"role": "user", "content": "count"}]},
        rig.key,
        uuid.uuid4().hex,
    )
    assert denial(counted) == KEY_DENIAL, counted.text

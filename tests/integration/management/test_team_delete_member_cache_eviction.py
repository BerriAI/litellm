"""
`/team/delete` cache eviction across both proxies: member user objects, the team object and the
team's keys must stop being served by every worker once the team rows are gone.

Auth caches the user object under the Redis key `<user_id>`, the team under `team_id:<team_id>`
and the key under its sha256; `enable_redis_auth_cache` is on, so Redis is the observable and
the pubsub channel carries the in-memory eviction to the peer proxy.
"""

import asyncio
import json
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Final

import anthropic
import httpx
import openai
import pytest
from pydantic import JsonValue
from redis import Redis

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    delete_key_if_present,
    eventually,
    string_value,
)
from tests.integration._support.database import read_rows, write_rows
from tests.integration._support.wire import Reply, Request, wire_server

_USAGE: Final = {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}
_CACHE_KEY_HEADER: Final = "x-litellm-cache-key"


def _redis() -> Redis:
    return Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))


def _cached_user(cache: Redis, user_id: str) -> dict[str, JsonValue] | None:
    raw: Final = cache.get(user_id)
    if raw is None:
        return None
    assert isinstance(raw, bytes), raw
    return JSON_OBJECT.validate_json(raw)


def _warmed_user(cache: Redis, user_id: str) -> dict[str, JsonValue]:
    """The cached user once its Redis SET has landed: auth writes memory at once but sends the Redis
    SET on the request's pipeline, so the entry can trail the response that warmed it."""
    cached: Final = eventually(lambda: _cached_user(cache, user_id), lambda value: value is not None, seconds=10)
    assert cached is not None
    return cached


def _delete_team_if_present(gateway: Gateway, team_id: str) -> None:
    if read_rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,)):
        gateway.post("/team/delete", {"team_ids": [team_id]})


def _team(gateway: Gateway, scenario: Scenario) -> str:
    """A team the test deletes itself; cleanup removes it only if the test failed before that delete."""
    created: Final = gateway.post("/team/new", {"team_alias": f"integration-{uuid.uuid4().hex}"})
    team_id: Final = string_value(created["team_id"])
    scenario.cleanups.callback(_delete_team_if_present, gateway, team_id)
    return team_id


def _team_key(gateway: Gateway, scenario: Scenario, team_id: str, model: str) -> str:
    """A key `/team/delete` removes; cleanup deletes it only if the team delete never ran."""
    created: Final = gateway.post("/key/generate", {"team_id": team_id, "models": [model]})
    token: Final = string_value(created["key"])
    scenario.cleanups.callback(delete_key_if_present, gateway, token)
    return token


def _delete_team(gateway: Gateway, team_id: str) -> None:
    deleted: Final = gateway.post("/team/delete", {"team_ids": [team_id]})
    assert deleted == {"deleted_teams": [team_id]}, deleted
    assert read_rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,)) == []


def _chat_body(model: str, text: str, stream: bool = False) -> dict[str, JsonValue]:
    body: dict[str, JsonValue] = {"model": model, "messages": [{"role": "user", "content": text}]}
    if stream:
        body["stream"] = True
    return body


def _chat(proxy: Gateway, model: str, key: str, text: str) -> httpx.Response:
    return proxy.request("POST", "/v1/chat/completions", _chat_body(model, text), key=key)


def _team_info(proxy: Gateway, team_id: str) -> httpx.Response:
    return proxy.request("GET", "/team/info", params={"team_id": team_id})


@pytest.mark.parametrize("roster_case", ("exact", "lower"), ids=("exact-case", "different-case"))
def test_team_delete_evicts_legacy_email_only_member_from_redis(gateway: Gateway, roster_case: str) -> None:
    """A roster entry carrying only an email (pre-backfill legacy shape) still names a cached user; the
    delete has to resolve it, in whatever case the roster stored it, and drop that user's cache entry."""
    with gateway.scenario() as scenario, _redis() as cache:
        model: Final = scenario.model()
        email: Final = f"Legacy-{uuid.uuid4().hex[:12]}@Example.com"
        user: Final = scenario.user(user_email=email)
        key: Final = scenario.key(user_id=user, models=[model])
        team: Final = _team(gateway, scenario)
        roster_email: Final = email if roster_case == "exact" else email.lower()
        assert (roster_email == email) is (roster_case == "exact"), (email, roster_email)
        write_rows(
            'UPDATE "LiteLLM_TeamTable" SET members_with_roles = %s::jsonb WHERE team_id = %s',
            (json.dumps([{"role": "user", "user_id": None, "user_email": roster_email}]), team),
        )
        write_rows('UPDATE "LiteLLM_UserTable" SET teams = array_append(teams, %s) WHERE user_id = %s', (team, user))
        warm: Final = _chat(gateway, model, key, "warm legacy member " + uuid.uuid4().hex)
        assert warm.status_code == 200, warm.text
        warmed: Final = _warmed_user(cache, user)
        assert warmed["teams"] == [team], warmed

        _delete_team(gateway, team)

        eventually(lambda: _cached_user(cache, user), lambda cached: cached is None, seconds=10)
        rows: Final = read_rows('SELECT teams FROM "LiteLLM_UserTable" WHERE user_id = %s', (user,))
        assert rows == [{"teams": []}], rows


def test_team_delete_evicts_member_cached_on_peer_and_peer_rehydrates_without_the_team(
    gateway: Gateway, peer: Gateway
) -> None:
    """The peer's in-memory copy of the member is evicted over pubsub: its next request misses locally
    and re-caches the user from the db, whose `teams` no longer holds the deleted team."""
    with gateway.scenario() as scenario, _redis() as cache:
        model: Final = scenario.model()
        user: Final = scenario.user(user_role="internal_user")
        team: Final = _team(gateway, scenario)
        gateway.post("/team/member_add", {"team_id": team, "member": {"role": "user", "user_id": user}})
        key: Final = scenario.key(user_id=user, models=[model])
        warm: Final = _chat(peer, model, key, "warm member on peer " + uuid.uuid4().hex)
        assert warm.status_code == 200, warm.text
        warmed: Final = _warmed_user(cache, user)
        assert warmed["teams"] == [team], warmed

        _delete_team(gateway, team)

        eventually(lambda: _cached_user(cache, user), lambda cached: cached is None, seconds=10)

        def rehydrate() -> dict[str, JsonValue] | None:
            # A peer worker still holding the stale in-memory copy answers from it and never
            # rewrites Redis, so each poll issues a fresh request rather than re-reading Redis alone.
            response: Final = _chat(peer, model, key, "rehydrate member on peer " + uuid.uuid4().hex)
            assert response.status_code == 200, response.text
            return _cached_user(cache, user)

        rehydrated: Final = eventually(rehydrate, lambda cached: cached is not None, seconds=10)
        assert rehydrated is not None and rehydrated["teams"] == [], rehydrated


def _sse(events: Sequence[object]) -> tuple[bytes, ...]:
    return tuple(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + (b"data: [DONE]\n\n",)


def _chat_reply(stream: bool) -> Reply:
    identity: Final = "chatcmpl-" + uuid.uuid4().hex
    if not stream:
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "team probe"}, "finish_reason": "stop"}
                    ],
                    "usage": _USAGE,
                }
            ).encode()
        )
    head: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
    return Reply(
        content_type="text/event-stream",
        chunks=_sse(
            (
                {**head, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "team "}}]},
                {**head, "choices": [{"index": 0, "delta": {"content": "probe"}}]},
                {**head, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {**head, "choices": [], "usage": _USAGE},
            )
        ),
    )


def _responses_reply(stream: bool) -> Reply:
    identity: Final = uuid.uuid4().hex
    completed: Final = {
        "id": "resp_" + identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "id": "msg_" + identity,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "team probe", "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
    }
    if not stream:
        return Reply(body=json.dumps(completed).encode())
    events: Final = (
        {"type": "response.created", "response": {**completed, "status": "in_progress", "output": [], "usage": None}},
        {
            "type": "response.output_text.delta",
            "item_id": "msg_" + identity,
            "output_index": 0,
            "content_index": 0,
            "delta": "team probe",
        },
        {"type": "response.completed", "response": completed},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
    )


def _upstream(request: Request) -> Reply:
    stream: Final = json.loads(request.body).get("stream") is True
    if request.target.endswith("/responses"):
        return _responses_reply(stream)
    return _chat_reply(stream)


def _v1(proxy: Gateway) -> str:
    return str(proxy.client.base_url).rstrip("/") + "/v1"


def _sdk_status(error: openai.APIStatusError | anthropic.APIStatusError) -> int | str:
    if isinstance(error, (openai.AuthenticationError, anthropic.AuthenticationError)):
        return error.status_code
    return f"{type(error).__name__}:{error.status_code}"


def _httpx_chat(proxy: Gateway, model: str, key: str, stream: bool, text: str) -> int | str:
    return proxy.request("POST", "/v1/chat/completions", _chat_body(model, text, stream), key=key).status_code


def _httpx_messages(proxy: Gateway, model: str, key: str, stream: bool, text: str) -> int | str:
    body: Final = {"model": model, "max_tokens": 64, "messages": [{"role": "user", "content": text}], "stream": stream}
    return proxy.request("POST", "/v1/messages", body, key=key).status_code


def _httpx_responses(proxy: Gateway, model: str, key: str, stream: bool, text: str) -> int | str:
    return proxy.request(
        "POST", "/v1/responses", {"model": model, "input": text, "stream": stream}, key=key
    ).status_code


def _openai_sync(proxy: Gateway, model: str, key: str, stream: bool, text: str) -> int | str:
    with openai.OpenAI(
        api_key=key, base_url=_v1(proxy), max_retries=0, http_client=httpx.Client(timeout=15, trust_env=False)
    ) as client:
        try:
            if stream:
                for _ in client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": text}], stream=True
                ):
                    pass
            else:
                client.chat.completions.create(model=model, messages=[{"role": "user", "content": text}])
        except openai.APIStatusError as error:
            return _sdk_status(error)
        return 200


def _openai_async(proxy: Gateway, model: str, key: str, stream: bool, text: str) -> int | str:
    async def call() -> int | str:
        async with openai.AsyncOpenAI(
            api_key=key, base_url=_v1(proxy), max_retries=0, http_client=httpx.AsyncClient(timeout=15, trust_env=False)
        ) as client:
            try:
                if stream:
                    async for _ in await client.chat.completions.create(
                        model=model, messages=[{"role": "user", "content": text}], stream=True
                    ):
                        pass
                else:
                    await client.chat.completions.create(model=model, messages=[{"role": "user", "content": text}])
            except openai.APIStatusError as error:
                return _sdk_status(error)
            return 200

    return asyncio.run(call())


def _anthropic_sync(proxy: Gateway, model: str, key: str, stream: bool, text: str) -> int | str:
    with anthropic.Anthropic(
        api_key=key,
        base_url=str(proxy.client.base_url),
        max_retries=0,
        http_client=httpx.Client(timeout=15, trust_env=False),
    ) as client:
        try:
            if stream:
                for _ in client.messages.create(
                    model=model, max_tokens=64, messages=[{"role": "user", "content": text}], stream=True
                ):
                    pass
            else:
                client.messages.create(model=model, max_tokens=64, messages=[{"role": "user", "content": text}])
        except anthropic.APIStatusError as error:
            return _sdk_status(error)
        return 200


@dataclass(frozen=True, slots=True)
class _Client:
    name: str
    call: Callable[[Gateway, str, str, bool, str], int | str]
    stream: bool


_CLIENTS: Final = (
    _Client("httpx-chat", _httpx_chat, False),
    _Client("httpx-chat-stream", _httpx_chat, True),
    _Client("httpx-messages", _httpx_messages, False),
    _Client("httpx-messages-stream", _httpx_messages, True),
    _Client("httpx-responses", _httpx_responses, False),
    _Client("httpx-responses-stream", _httpx_responses, True),
    _Client("openai-sync", _openai_sync, False),
    _Client("openai-sync-stream", _openai_sync, True),
    _Client("openai-async", _openai_async, False),
    _Client("openai-async-stream", _openai_async, True),
    _Client("anthropic-sync", _anthropic_sync, False),
    _Client("anthropic-sync-stream", _anthropic_sync, True),
)


def _observe(proxies: Mapping[str, Gateway], model: str, key: str) -> dict[str, int | str]:
    """One cell per proxy and client; unique text per cell keeps the response cache out of the picture."""
    return {
        f"{proxy_name}/{client.name}": client.call(
            proxy, model, key, client.stream, f"{client.name} {uuid.uuid4().hex}"
        )
        for proxy_name, proxy in proxies.items()
        for client in _CLIENTS
    }


def _off(observed: Mapping[str, int | str], expected: int) -> dict[str, int | str]:
    return {cell: status for cell, status in observed.items() if status != expected}


def test_team_delete_refuses_the_team_key_for_every_client_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """Every surface a deleted team's key can reach, on the primary and on the peer, answers 401
    once the team is gone; every cell is checked and every failing cell is reported at once."""
    proxies: Final = {"primary": gateway, "peer": peer}
    with wire_server(_upstream) as upstream, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=upstream.url + "/v1")
        team: Final = _team(gateway, scenario)
        key: Final = _team_key(gateway, scenario, team, model)
        before: Final = _observe(proxies, model, key)
        assert _off(before, 200) == {}, _off(before, 200)

        _delete_team(gateway, team)

        eventually(
            lambda: _httpx_chat(peer, model, key, False, "deleted team key on peer"),
            lambda status: status == 401,
            seconds=10,
        )
        after: Final = _observe(proxies, model, key)
        assert _off(after, 401) == {}, _off(after, 401)


def test_team_delete_rejects_the_deleted_key_before_the_response_cache(gateway: Gateway) -> None:
    """A request the response cache already answers for this key is refused at auth after the delete:
    401, and the upstream never sees it, so the cache-hit path cannot outlive the key."""
    with wire_server(_upstream) as upstream, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=upstream.url + "/v1")
        team: Final = _team(gateway, scenario)
        key: Final = _team_key(gateway, scenario, team, model)
        marker: Final = "cache twin " + uuid.uuid4().hex
        body: Final = _chat_body(model, marker)
        first: Final = gateway.request("POST", "/v1/chat/completions", body, key=key)
        assert first.status_code == 200, first.text
        assert first.headers.get(_CACHE_KEY_HEADER) is None, dict(first.headers)
        second: Final = gateway.request("POST", "/v1/chat/completions", body, key=key)
        assert second.status_code == 200, second.text
        assert second.headers.get(_CACHE_KEY_HEADER), dict(second.headers)
        assert second.json()["id"] == first.json()["id"], (first.text, second.text)
        received: Final = upstream.drain()
        assert len(received) == 1 and marker.encode() in received[0].body, received

        _delete_team(gateway, team)

        third: Final = gateway.request("POST", "/v1/chat/completions", body, key=key)
        assert third.status_code == 401, third.text
        assert "token_not_found_in_db" in third.text, third.text
        assert upstream.drain() == (), "upstream saw a request for the deleted key"


def test_team_delete_evicts_team_object_and_key_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """Team object and key warm on both proxies before the delete: `/team/info` is 404 and the key is
    401 on both afterwards, and neither the team nor the key entry is left in Redis."""
    with gateway.scenario() as scenario, _redis() as cache:
        model: Final = scenario.model()
        team: Final = _team(gateway, scenario)
        key: Final = _team_key(gateway, scenario, team, model)
        hashed: Final = sha256(key.encode()).hexdigest()
        for proxy in (gateway, peer):
            info: httpx.Response = _team_info(proxy, team)
            assert info.status_code == 200 and info.json()["team_id"] == team, info.text
            warm: httpx.Response = _chat(proxy, model, key, "warm team key " + uuid.uuid4().hex)
            assert warm.status_code == 200, warm.text
        # Both SETs ride the warming request's Redis pipeline and can land after its response.
        eventually(lambda: cache.exists(f"team_id:{team}"), lambda present: present == 1, seconds=10)
        eventually(lambda: cache.exists(hashed), lambda present: present == 1, seconds=10)

        _delete_team(gateway, team)

        eventually(lambda: _team_info(peer, team).status_code, lambda status: status == 404, seconds=10)
        eventually(
            lambda: _chat(peer, model, key, "deleted team key on peer").status_code,
            lambda status: status == 401,
            seconds=10,
        )
        for proxy in (gateway, peer):
            gone: httpx.Response = _team_info(proxy, team)
            assert gone.status_code == 404 and "Team not found" in gone.text, gone.text
            refused: httpx.Response = _chat(proxy, model, key, "deleted team key " + uuid.uuid4().hex)
            assert refused.status_code == 401 and "token_not_found_in_db" in refused.text, refused.text
        assert cache.exists(f"team_id:{team}") == 0, cache.keys(f"*{team}*")
        assert cache.exists(hashed) == 0, cache.keys(f"*{hashed}*")

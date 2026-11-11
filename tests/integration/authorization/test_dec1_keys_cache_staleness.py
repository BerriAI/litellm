"""How soon a change to a key, team or user is enforced for a key that both proxies have already cached.

Contract (litellm/proxy/auth/auth_checks.py `delete_cache_key_object`, `delete_cache_key_objects`;
litellm/proxy/common_utils/auth_cache_invalidation_pubsub.py):

* Auth serves key objects cache first. The in memory entry lives 60 seconds by default
  (`general_settings.user_api_key_cache_ttl`), and with `enable_redis_auth_cache` there is a shared Redis copy.
* Every management write that changes what a key may do evicts the entry on the proxy that handled the write
  and publishes the eviction on the Redis channel `litellm_proxy.auth_cache_invalidation`, so the other
  proxies drop their copy without waiting for the TTL.
* The tests therefore warm the key on two proxies (`gateway` and `peer`, which share Postgres and Redis),
  make the change through `gateway`, and require the new behaviour on the writer within 5 seconds and on the
  peer within 10 seconds. Both bounds are far below the 60 second TTL, so a port that drops the eviction or
  the broadcast fails here.
* Nothing is cached for a key that does not exist, so a key created after a failed attempt works at once.

Not covered here because other files already pin it: block and unblock of a key
(`test_key_block_and_unblock_reach_both_warmed_workers`), deletion
(`test_a_deleted_key_gets_the_generic_401_on_both_proxies`) and team block (`management/test_team_block.py`).
"""

from __future__ import annotations

from typing import Final

import httpx

from tests.integration._support.client import Gateway, delete_key_if_present, eventually, list_value, string_value
from tests.integration.authorization.dec1_keys_support import (
    UNKNOWN_KEY_MESSAGE,
    assert_denied,
    assert_model_denied,
    assert_route_denied,
    assert_served,
    chat,
    marker,
    spend_rows,
)

_WRITER_SECONDS: Final = 5
_PEER_SECONDS: Final = 10


def _warm(key: str, model: str, *proxies: Gateway) -> None:
    for proxy in proxies:
        assert_served(chat(proxy, key, model))


def _settled(proxy: Gateway, key: str, model: str, status: int, seconds: float) -> httpx.Response:
    return eventually(lambda: chat(proxy, key, model), lambda response: response.status_code == status, seconds=seconds)


def test_removing_a_model_from_a_warmed_key_is_enforced_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A key allowed two models is warmed on both proxies, then `/key/update` on one proxy removes the second.

    The second model must answer 403 `key_model_access_denied` on the writer within 5 seconds and on the
    peer within 10 seconds, while the first model keeps being served on both.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        key: Final = scenario.key(models=[first, second])
        _warm(key, second, gateway, peer)
        gateway.post("/key/update", {"key": key, "models": [first]})
        assert_model_denied(_settled(gateway, key, second, 403, _WRITER_SECONDS), second, "key_model_access_denied")
        assert_model_denied(_settled(peer, key, second, 403, _PEER_SECONDS), second, "key_model_access_denied")
        _warm(key, first, gateway, peer)


def test_adding_a_model_to_a_warmed_key_is_served_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A key allowed one model is refused a second model on both proxies, then `/key/update` adds it.

    The second model must be served on the writer within 5 seconds and on the peer within 10 seconds. A
    stale denial is as much a bug as a stale grant.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        key: Final = scenario.key(models=[first])
        _warm(key, first, gateway, peer)
        for proxy in (gateway, peer):
            assert chat(proxy, key, second).status_code == 403
        gateway.post("/key/update", {"key": key, "models": [first, second]})
        assert_served(_settled(gateway, key, second, 200, _WRITER_SECONDS))
        assert_served(_settled(peer, key, second, 200, _PEER_SECONDS))


def test_restricting_the_routes_of_a_warmed_key_is_enforced_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A key without route restrictions is warmed on both proxies, then `/key/update` sets
    `allowed_routes: ["/v1/embeddings"]`.

    Chat completions must answer 403 with the "Virtual key is not allowed to call this route" message on the
    writer within 5 seconds and on the peer within 10 seconds.

    Pinned current behaviour: the route refusal body is the bare `detail` shape (see `assert_route_denied`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        _warm(key, model, gateway, peer)
        gateway.post("/key/update", {"key": key, "allowed_routes": ["/v1/embeddings"]})
        for proxy, seconds in ((gateway, _WRITER_SECONDS), (peer, _PEER_SECONDS)):
            assert_route_denied(_settled(proxy, key, model, 403, seconds), ["/v1/embeddings"], "/v1/chat/completions")


def test_shortening_the_expiry_of_a_warmed_key_expires_it_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A key without expiry is warmed on both proxies, then `/key/update` sets `duration: "1s"`.

    Both proxies must answer 401 `expired_key` within 10 seconds. The cached copy on the peer carries no
    expiry, so it must be evicted for the new expiry to be seen.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        _warm(key, model, gateway, peer)
        gateway.post("/key/update", {"key": key, "duration": "1s"})
        for proxy in (gateway, peer):
            refused = _settled(proxy, key, model, 401, _PEER_SECONDS)
            assert refused.json()["error"]["type"] == "expired_key", refused.text


def test_extending_an_expired_key_restores_it_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A key created with `duration: "1s"` is refused as expired on both proxies, then `/key/update` sets
    `duration: "1h"`.

    The key must be served again on the writer within 5 seconds and on the peer within 10 seconds.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(duration="1s")
        for proxy in (gateway, peer):
            _settled(proxy, key, model, 401, _PEER_SECONDS)
        gateway.post("/key/update", {"key": key, "duration": "1h"})
        assert_served(_settled(gateway, key, model, 200, _WRITER_SECONDS))
        assert_served(_settled(peer, key, model, 200, _PEER_SECONDS))


def test_a_regenerated_key_replaces_the_old_value_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A key is warmed on both proxies and then regenerated through `/key/regenerate` on one of them.

    The new value must be served on both proxies, and the old value must answer the generic 401
    `token_not_found_in_db` on the writer within 5 seconds and on the peer within 10 seconds.
    UNVERIFIED: regeneration is assumed to have no grace period by default; see `/key/regenerate`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        old: Final = string_value(gateway.post("/key/generate", {"models": [model]})["key"])
        scenario.cleanups.callback(delete_key_if_present, gateway, old)
        _warm(old, model, gateway, peer)
        new: Final = string_value(gateway.post("/key/regenerate", {"key": old})["key"])
        scenario.cleanups.callback(scenario.delete_key, new)
        assert new != old
        _warm(new, model, gateway, peer)
        for proxy, seconds in ((gateway, _WRITER_SECONDS), (peer, _PEER_SECONDS)):
            assert_denied(
                _settled(proxy, old, model, 401, seconds),
                status=401,
                error_type="token_not_found_in_db",
                param="key",
                message=UNKNOWN_KEY_MESSAGE,
            )


def test_a_key_created_after_a_failed_attempt_with_the_same_value_works_at_once(
    gateway: Gateway, peer: Gateway
) -> None:
    """A not yet issued key value is refused with 401 on both proxies, and then an admin creates a key with
    exactly that value.

    The key must be served on the writer within 5 seconds and on the peer within 10 seconds. A miss must
    not be cached in a way that keeps a freshly created key out.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        value: Final = f"sk-{marker()}"
        for proxy in (gateway, peer):
            assert chat(proxy, value, model).status_code == 401
        assert scenario.key(key=value, models=[model]) == value
        assert_served(_settled(gateway, value, model, 200, _WRITER_SECONDS))
        assert_served(_settled(peer, value, model, 200, _PEER_SECONDS))


def test_a_key_created_on_one_proxy_is_usable_on_the_other_without_warming(gateway: Gateway, peer: Gateway) -> None:
    """A key is created through one proxy and its first ever request goes to the other proxy.

    The peer has never seen the key, so it must load it from the shared database and answer 200 on the
    first request, with no polling.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        assert_served(chat(peer, key, model))


def test_a_deleted_then_recreated_key_value_serves_again_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A key with an admin chosen value is warmed, deleted, refused on both proxies, and created again with
    the same value.

    After the second creation the value must be served on both proxies within 10 seconds. The refusal seen
    while the key was deleted must not stick.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        value: Final = f"sk-{marker()}"
        gateway.post("/key/generate", {"key": value, "models": [model]})
        scenario.cleanups.callback(delete_key_if_present, gateway, value)
        _warm(value, model, gateway, peer)
        gateway.post("/key/delete", {"keys": [value]})
        for proxy in (gateway, peer):
            _settled(proxy, value, model, 401, _PEER_SECONDS)
        assert scenario.key(key=value, models=[model]) == value
        for proxy in (gateway, peer):
            assert_served(_settled(proxy, value, model, 200, _PEER_SECONDS))


def test_narrowing_a_users_models_reaches_their_warmed_personal_key_on_both_proxies(
    gateway: Gateway, peer: Gateway
) -> None:
    """A user allowed two models owns a personal key warmed on both proxies, then `/user/update` removes the
    second model.

    The key itself is unchanged, but it must answer 403 `user_model_access_denied` for the second model on
    both proxies within 10 seconds.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        user: Final = scenario.user(models=[first, second])
        key: Final = scenario.key(user_id=user)
        _warm(key, second, gateway, peer)
        gateway.post("/user/update", {"user_id": user, "models": [first]})
        for proxy in (gateway, peer):
            assert_model_denied(_settled(proxy, key, second, 403, _PEER_SECONDS), second, "user_model_access_denied")


def test_narrowing_a_teams_models_reaches_its_warmed_key_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A team allowed two models has a key warmed on both proxies, then `/team/update` removes the second.

    The key must answer 403 `team_model_access_denied` for the second model on both proxies within 10
    seconds, without any write to the key.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        team: Final = scenario.team(models=[first, second])
        key: Final = scenario.key(team_id=team)
        _warm(key, second, gateway, peer)
        gateway.post("/team/update", {"team_id": team, "models": [first]})
        for proxy in (gateway, peer):
            assert_model_denied(_settled(proxy, key, second, 403, _PEER_SECONDS), second, "team_model_access_denied")


def test_restricting_a_team_member_reaches_their_warmed_key_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A team member's key is warmed on both proxies for two team models, then `/team/member_update` sets
    `allowed_models` to the first only.

    The member's key must answer 403 `team_model_access_denied` for the second model on both proxies
    within 10 seconds.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        team: Final = scenario.team(models=[first, second])
        member: Final = scenario.member(team)
        key: Final = scenario.key(team_id=team, user_id=member)
        _warm(key, second, gateway, peer)
        gateway.post("/team/member_update", {"team_id": team, "user_id": member, "allowed_models": [first]})
        for proxy in (gateway, peer):
            assert_model_denied(_settled(proxy, key, second, 403, _PEER_SECONDS), second, "team_model_access_denied")


def test_moving_a_warmed_key_to_another_team_applies_the_new_teams_models(gateway: Gateway, peer: Gateway) -> None:
    """A key in a team that allows model A is warmed on both proxies, then `/key/update` moves it to a team
    that allows only model B.

    Within 10 seconds both proxies must serve B and answer 403 `team_model_access_denied` for A.
    UNVERIFIED: `/key/update` with a new `team_id` is assumed to be accepted for an admin when the key has
    no model list of its own.
    """
    with gateway.scenario() as scenario:
        model_a: Final = scenario.model()
        model_b: Final = scenario.model()
        team_a: Final = scenario.team(models=[model_a])
        team_b: Final = scenario.team(models=[model_b])
        key: Final = scenario.key(team_id=team_a)
        _warm(key, model_a, gateway, peer)
        gateway.post("/key/update", {"key": key, "team_id": team_b})
        for proxy in (gateway, peer):
            assert_served(_settled(proxy, key, model_b, 200, _PEER_SECONDS))
            assert_model_denied(chat(proxy, key, model_a), model_a, "team_model_access_denied")


def test_a_tag_added_to_a_warmed_keys_metadata_appears_on_its_next_requests(gateway: Gateway) -> None:
    """A key serves one request, then `/key/update` adds `metadata.tags` and the key serves more requests.

    A later spend row for the key must carry the new tag in `request_tags`. The cached key object must
    not keep injecting the old metadata.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        assert_served(chat(gateway, key, model))
        tag: Final = f"late-tag-{marker()}"
        gateway.post("/key/update", {"key": key, "metadata": {"tags": [tag]}})
        assert_served(chat(gateway, key, model))
        rows: Final = spend_rows(gateway, key, at_least=2)
        assert [row for row in rows if tag in list_value(row["request_tags"])] != [], rows


def test_unblocking_a_key_restores_it_on_the_writer_within_seconds(gateway: Gateway) -> None:
    """A warmed key is blocked, refused, and then unblocked through `/key/unblock` on the same proxy.

    The key must be served again within 5 seconds. A cached blocked copy must not outlive the unblock.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        _warm(key, model, gateway)
        gateway.post("/key/block", {"key": key})
        _settled(gateway, key, model, 401, _WRITER_SECONDS)
        gateway.post("/key/unblock", {"key": key})
        assert_served(_settled(gateway, key, model, 200, _WRITER_SECONDS))


def test_deleting_a_team_stops_its_warmed_keys_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A team key is warmed on both proxies and then its team is deleted through `/team/delete`.

    Deleting a team deletes its keys, so the key must answer 401 on both proxies within 10 seconds and its
    row must be gone. UNVERIFIED: the refusal is expected to be the generic `token_not_found_in_db` body.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = string_value(gateway.post("/team/new", {"team_alias": marker()})["team_id"])
        key: Final = string_value(gateway.post("/key/generate", {"team_id": team})["key"])
        scenario.cleanups.callback(gateway.request, "POST", "/team/delete", {"team_ids": [team]})
        scenario.cleanups.callback(delete_key_if_present, gateway, key)
        _warm(key, model, gateway, peer)
        gateway.post("/team/delete", {"team_ids": [team]})
        for proxy in (gateway, peer):
            refused = _settled(proxy, key, model, 401, _PEER_SECONDS)
            assert refused.json()["error"]["type"] == "token_not_found_in_db", refused.text

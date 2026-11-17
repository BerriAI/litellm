"""Model and config changes made through one proxy instance reach its peers through Redis pub/sub.

With `store_model_in_db: true` every write to a config table (`LiteLLM_ProxyModelTable` and the other tables in
`_CONFIG_SYNCED_TABLE_NAMES`) is followed by a `PUBLISH` on the Redis channel `litellm_proxy.config_change`
(prefixed `<namespace>:` when the coordination Redis has a namespace). The message is JSON
`{"object_type": "<lowercase table name>", "model_ids": [...]}`, and a model create, update or upsert names
the written `model_id`. Every instance subscribes to that channel at startup. On a message a peer applies the
named model ids at once and then, after a debounce of 1 second plus up to 5 seconds of jitter and at most one
resync per 10 seconds, reloads its whole model list and router settings from the database. A write to the
`router_settings`, `general_settings` or `litellm_settings` row of `LiteLLM_Config` is published the same way
with that param name as the `object_type`. A publish failure never fails the write, and a subscriber that lost
Redis reconnects with a backoff of 5 seconds doubling to 60 and then does a full resync.

Without Redis the only path is the periodic `add_deployment` job, every `PROXY_CONFIG_RELOAD_INTERVAL_SECONDS`
(default 30). The fleet here sets that to 3600 unless a test says otherwise, so a peer that picks a change up in
seconds proves the pub/sub path. The Rust Redis layer and a Rust config loader must keep the channel name, the
message shape and these timings, since Python and Rust pods share them during a rolling deploy.

Read from `litellm/proxy/common_utils/config_sync_pubsub.py` and `litellm/proxy/proxy_server.py`
(`start_config_sync_subscriber`, the `add_deployment_job` scheduling near line 12046) and
`litellm/proxy/utils.py` (`invalidate_config_param`, `_config_cache_key`).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue
from redis import Redis
from redis.client import PubSub

from tests.integration._support.client import JSON_OBJECT, eventually, held, object_value
from tests.integration.routing.dec1_multiinst_support import (
    CONFIG_SYNC_CHANNEL,
    MODEL_TABLE,
    PEER_PICKUP_SECONDS,
    Fleet,
    add_model,
    ask,
    decoded,
    fleet,
    listed_models,
    redis_stopped,
    router_values,
    served,
    unique,
)
from tests.integration.routing.dec1_routing_support import INVALID_MODEL, PROVIDER_KEY, deployment, error_message

pytestmark: Final = [pytest.mark.timeout(400), pytest.mark.shares_worker_state]

RECONNECT_SECONDS: Final = 90  # backoff 5 s doubling to 60 s, plus the debounce and jitter
POLL_RELOAD_SECONDS: Final = 5
CONFIG_PARAM_TTL_SECONDS: Final = 60  # LITELLM_CONFIG_CACHE_TTL_SECONDS


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Fleet]:
    with fleet(
        tmp_path_factory.mktemp("dec1-multiinst-sync"),
        models=lambda api_base: (deployment("sync-static", "ok-sync-static", api_base),),
        router_settings={"num_retries": 0, "disable_cooldowns": True},
        environment={"REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT": "1"},
    ) as value:
        yield value


def _subscribe(client: Redis) -> PubSub:
    pubsub = client.pubsub()
    pubsub.subscribe(CONFIG_SYNC_CHANNEL)
    eventually(lambda: pubsub.get_message(timeout=0.2), lambda message: message is not None, seconds=5)
    return pubsub


def _drain(pubsub: PubSub, seconds: float) -> tuple[dict[str, JsonValue], ...]:
    found: list[dict[str, JsonValue]] = []  # mutable-ok: appended while polling

    def read() -> int:
        message = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.2)
        if message is not None and isinstance(message.get("data"), bytes | str):
            found.append(JSON_OBJECT.validate_json(message["data"]))
        return len(found)

    eventually(read, lambda count: count >= 1, seconds=seconds, return_last_on_timeout=True)
    return tuple(found)


def test_every_instance_subscribes_to_the_config_change_channel_at_startup(pair: Fleet) -> None:
    """Both instances boot with `store_model_in_db: true` and a coordination Redis.

    `PUBSUB NUMSUB litellm_proxy.config_change` on that Redis must report at least 2 subscribers, one per
    instance. A Rust config loader that forgets to subscribe leaves every peer stale for the full reload interval.
    """
    assert pair.subscribers() >= 2


def test_model_new_publishes_the_model_table_and_the_new_model_id(pair: Fleet) -> None:
    """A test client subscribes to `litellm_proxy.config_change`, then the first instance adds a model with
    `POST /model/new` and a chosen `model_info.id`.

    A message must arrive whose JSON body has `object_type` `"litellm_proxymodeltable"` and whose `model_ids`
    list contains that id. This is the wire format Python and Rust pods exchange during a rolling deploy.
    """
    with pair.store() as client:
        pubsub = _subscribe(client)
        model_id: Final = add_model(pair.first, unique("sync-publish"), "ok-sync-publish", pair.upstream.api_base)
        messages: Final = _drain(pubsub, 10)
        pubsub.close()
    model_messages: Final = [message for message in messages if message.get("object_type") == MODEL_TABLE]
    assert model_messages, messages
    assert any(model_id in (message.get("model_ids") or []) for message in model_messages), messages  # type: ignore[operator]


def test_a_model_added_on_one_instance_is_served_by_the_peer_well_before_the_reload_interval(pair: Fleet) -> None:
    """The first instance adds group `sync-added-*` with `POST /model/new`; the periodic reload is 3600 seconds.

    Within 20 seconds the second instance must answer a chat request for that group with 200 from the new
    deployment's upstream (`served ok-sync-added`). Only the pub/sub message can have told it.
    """
    group: Final = unique("sync-added")
    add_model(pair.first, group, "ok-sync-added", pair.upstream.api_base)
    response: Final = eventually(
        lambda: ask(pair.second, group, unique("q")),
        lambda answer: answer.status_code == 200,
        seconds=PEER_PICKUP_SECONDS,
    )
    assert served(response) == "ok-sync-added"


def test_a_model_added_on_one_instance_is_listed_by_the_peers_models_route(pair: Fleet) -> None:
    """The first instance adds group `sync-listed-*`.

    Within 20 seconds `GET /v1/models` on the second instance must list that group name. Listing and routing come
    from the same reloaded router, so a peer that serves a model also lists it.
    """
    group: Final = unique("sync-listed")
    add_model(pair.first, group, "ok-sync-listed", pair.upstream.api_base)
    listed = eventually(lambda: listed_models(pair.second), lambda listed: group in listed, seconds=PEER_PICKUP_SECONDS)
    assert group in listed


def test_a_model_added_on_one_instance_keeps_its_deployment_id_on_the_peer(pair: Fleet) -> None:
    """The first instance adds a model with `model_info.id` set to a chosen id.

    Within 20 seconds `GET /v1/model/info` on the second instance must list a deployment whose `model_info.id` is
    that id. Cooldown, usage and health keys in Redis are named by this id, so it must be the same on every pod.
    """
    group: Final = unique("sync-id")
    model_id: Final = add_model(pair.first, group, "ok-sync-id", pair.upstream.api_base)

    def ids() -> frozenset[str]:
        data = pair.second.get("/v1/model/info")["data"]
        assert isinstance(data, list)
        return frozenset(str(object_value(object_value(entry)["model_info"]).get("id")) for entry in data)

    eventually(ids, lambda found: model_id in found, seconds=PEER_PICKUP_SECONDS)


def test_a_model_deleted_on_one_instance_stops_being_served_by_the_peer(pair: Fleet) -> None:
    """A model added on the first instance is served by the second; then the first deletes it with
    `POST /model/delete {"id": <id>}`.

    Within 20 seconds the second instance must answer a chat request for the group with 400 and an error message
    containing `Invalid model name passed in model=`. A delete publishes with an empty `model_ids` list, so the
    peer only learns it through the debounced full resync.
    """
    group: Final = unique("sync-deleted")
    model_id: Final = add_model(pair.first, group, "ok-sync-deleted", pair.upstream.api_base)
    eventually(lambda: ask(pair.second, group, unique("q")).status_code, lambda status: status == 200, seconds=20)
    pair.first.post("/model/delete", {"id": model_id})
    response: Final = eventually(
        lambda: ask(pair.second, group, unique("q")),
        lambda answer: answer.status_code == 400,
        seconds=PEER_PICKUP_SECONDS + 15,
    )
    assert INVALID_MODEL in error_message(response)


def test_a_model_updated_on_one_instance_sends_the_peers_traffic_to_the_new_upstream_model(pair: Fleet) -> None:
    """A model added on the first instance points at upstream model `ok-sync-before`; after the second serves it,
    the first changes it with `PATCH /model/{id}/update` to `openai/ok-sync-after`.

    Within 20 seconds every chat request to the second instance must be served by `ok-sync-after`. UNVERIFIED:
    the PATCH body shape `{"litellm_params": {...}}`; see `update_model` in
    `litellm/proxy/management_endpoints/model_management_endpoints.py:1745`.
    """
    group: Final = unique("sync-updated")
    model_id: Final = add_model(pair.first, group, "ok-sync-before", pair.upstream.api_base)
    eventually(lambda: ask(pair.second, group, unique("q")).status_code, lambda status: status == 200, seconds=20)
    response: Final = pair.first.request(
        "PATCH",
        f"/model/{model_id}/update",
        {
            "model_name": group,
            "litellm_params": {
                "model": "openai/ok-sync-after",
                "api_key": PROVIDER_KEY,
                "api_base": pair.upstream.api_base,
            },
        },
    )
    assert response.status_code == 200, response.text
    eventually(
        lambda: served(ask(pair.second, group, unique("q"))),
        lambda name: name == "ok-sync-after",
        seconds=PEER_PICKUP_SECONDS,
    )


def test_two_deployments_added_to_one_group_on_one_instance_are_both_used_by_the_peer(pair: Fleet) -> None:
    """The first instance adds two deployments to one group, upstream models `ok-sync-twin-a` and `ok-sync-twin-b`.

    Once the second instance serves the group, twenty requests through it must reach both upstream models at
    least once. A peer that applied only the first published id would serve the group from one deployment.
    """
    group: Final = unique("sync-twin")
    add_model(pair.first, group, "ok-sync-twin-a", pair.upstream.api_base)
    add_model(pair.first, group, "ok-sync-twin-b", pair.upstream.api_base)
    names = eventually(
        lambda: frozenset(served(ask(pair.second, group, unique("q"))) for _ in range(20)),
        lambda names: names == {"ok-sync-twin-a", "ok-sync-twin-b"},
        seconds=PEER_PICKUP_SECONDS + 15,
    )
    assert names == {"ok-sync-twin-a", "ok-sync-twin-b"}


def test_a_blocked_model_stops_serving_on_the_peer_and_an_unblock_restores_it(pair: Fleet) -> None:
    """A model added on the first instance is blocked there with `POST /model/block {"model_id": <id>}`.

    Within 30 seconds a chat request for its group on the second instance must stop returning 200, and after
    `POST /model/unblock` on the first instance the second must serve the group again within 30 seconds.
    UNVERIFIED: the exact status a blocked model answers with (treated here as anything but 200); see
    `_set_model_blocked_status` in `model_management_endpoints.py`.
    """
    group: Final = unique("sync-blocked")
    model_id: Final = add_model(pair.first, group, "ok-sync-blocked", pair.upstream.api_base)
    eventually(lambda: ask(pair.second, group, unique("q")).status_code, lambda status: status == 200, seconds=20)
    pair.first.post("/model/block", {"model_id": model_id})
    eventually(lambda: ask(pair.second, group, unique("q")).status_code, lambda status: status != 200, seconds=30)
    pair.first.post("/model/unblock", {"model_id": model_id})
    restored = eventually(
        lambda: ask(pair.second, group, unique("q")).status_code, lambda status: status == 200, seconds=30
    )
    assert restored == 200


def test_router_settings_changed_through_config_update_reach_the_peer(pair: Fleet) -> None:
    """The first instance stores `router_settings.num_retries: 4` with `POST /config/update`.

    Within 30 seconds `GET /router/settings` on the second instance must report `num_retries` 4 in
    `current_values`. The write publishes the param name `router_settings` on the config sync channel and the
    peer's resync reads the row straight from the database.
    """
    pair.first.post("/config/update", {"router_settings": {"num_retries": 4}})
    retries = eventually(lambda: router_values(pair.second).get("num_retries"), lambda value: value == 4, seconds=30)
    assert retries == 4


def test_config_update_publishes_the_param_name_as_the_object_type(pair: Fleet) -> None:
    """A test client subscribes to the config sync channel, then the first instance stores
    `router_settings.num_retries: 2` with `POST /config/update`.

    A message with `object_type` `"router_settings"` and an empty `model_ids` list must arrive. Only the params
    in `_RESYNC_APPLIED_CONFIG_PARAM_NAMES` are published.
    """
    with pair.store() as client:
        pubsub = _subscribe(client)
        pair.first.post("/config/update", {"router_settings": {"num_retries": 2}})
        messages: Final = _drain(pubsub, 10)
        pubsub.close()
    assert {"object_type": "router_settings", "model_ids": []} in messages, messages


def test_config_update_evicts_the_cached_config_row_from_redis(pair: Fleet) -> None:
    """The first instance stores `router_settings.num_retries: 3` with `POST /config/update`.

    Afterwards the Redis key `litellm_config:param:router_settings` must either be absent or hold a JSON object
    whose `param_value.num_retries` is 3; it must never keep a value from before the write. Every pod reads config
    rows through this key for up to 60 seconds, so a write that forgets to evict it serves stale settings fleet-wide.
    """
    pair.first.post("/config/update", {"router_settings": {"num_retries": 3}})
    raw: Final = pair.raw("litellm_config:param:router_settings")
    if raw is not None:
        value = object_value(decoded(raw))
        assert object_value(value["param_value"]).get("num_retries") == 3, value


def test_a_cached_config_row_in_redis_expires_within_sixty_seconds(pair: Fleet) -> None:
    """Both instances read `general_settings` through the config param cache when they reload.

    Once Redis holds `litellm_config:param:general_settings`, its value must be JSON with `param_name`
    `"general_settings"` and its TTL must be between 1 and 60 seconds (`LITELLM_CONFIG_CACHE_TTL_SECONDS`).
    UNVERIFIED: that a reload reads `general_settings` through `get_config_param` soon enough for the key to
    exist; see `add_deployment_locked` in `proxy_server.py:9195`.
    """
    pair.first.post("/config/update", {"router_settings": {"num_retries": 0}})
    raw: Final = eventually(
        lambda: pair.raw("litellm_config:param:general_settings"), lambda value: value is not None, seconds=40
    )
    assert object_value(decoded(raw))["param_name"] == "general_settings"
    assert 0 < pair.ttl("litellm_config:param:general_settings") <= CONFIG_PARAM_TTL_SECONDS


def test_a_malformed_message_on_the_channel_does_not_stop_the_peer_from_syncing(pair: Fleet) -> None:
    """A test client publishes `not json` and then `{"object_type": 5}` on `litellm_proxy.config_change`.

    A model added on the first instance afterwards must still be served by the second instance within 20 seconds.
    `parse_config_change` drops what it cannot validate; a Rust subscriber that panics or exits on a bad message
    would leave the pod stale until restart.
    """
    with pair.store() as client:
        client.publish(CONFIG_SYNC_CHANNEL, "not json")
        client.publish(CONFIG_SYNC_CHANNEL, '{"object_type": 5}')
    group: Final = unique("sync-after-garbage")
    add_model(pair.first, group, "ok-sync-after-garbage", pair.upstream.api_base)
    status = eventually(
        lambda: ask(pair.second, group, unique("q")).status_code,
        lambda status: status == 200,
        seconds=PEER_PICKUP_SECONDS,
    )
    assert status == 200


def test_a_model_new_while_redis_is_down_still_succeeds_and_serves_on_the_writer(pair: Fleet) -> None:
    """The fleet's Redis is stopped, then the first instance adds group `sync-redis-down-*`.

    `POST /model/new` must return 200 and the first instance must serve the group within 10 seconds. The publish
    is best effort (`publish_config_change` logs a warning) and never fails the database write.
    """
    group: Final = unique("sync-redis-down")
    with redis_stopped(pair):
        add_model(pair.first, group, "ok-sync-redis-down", pair.upstream.api_base)
        response = eventually(
            lambda: ask(pair.first, group, unique("q")), lambda answer: answer.status_code == 200, seconds=10
        )
    assert served(response) == "ok-sync-redis-down"


def test_a_peer_catches_up_on_a_change_made_while_redis_was_down_once_redis_returns(pair: Fleet) -> None:
    """With Redis stopped the first instance adds group `sync-missed-*`; the message is lost. Redis is then
    started again on the same port.

    Within 90 seconds the second instance must serve that group with 200. Its subscriber reconnects with backoff
    and, because it lost the connection, runs a full resync from the database on reconnect
    (`ConfigSyncSubscriber._run`, `_resync_required`). The reload interval is 3600 seconds, so nothing else can.
    """
    group: Final = unique("sync-missed")
    with redis_stopped(pair):
        add_model(pair.first, group, "ok-sync-missed", pair.upstream.api_base)
    response: Final = eventually(
        lambda: ask(pair.second, group, unique("q")),
        lambda answer: answer.status_code == 200,
        seconds=RECONNECT_SECONDS,
    )
    assert served(response) == "ok-sync-missed"
    eventually(pair.subscribers, lambda count: count >= 2, seconds=RECONNECT_SECONDS)


def test_without_redis_a_peer_does_not_see_a_new_model_before_the_reload_interval(tmp_path: Path) -> None:
    """Two instances share only the database (no Redis) and reload every 3600 seconds.

    After the first instance adds group `sync-noredis-*`, the second must keep answering 400 for that group for
    10 seconds. This pins that pub/sub is the only fast path: a rewrite that silently drops the subscriber is
    only caught by a test like the ones above, and a rewrite that polls faster changes database load.
    """
    with fleet(tmp_path, redis=False, router_settings={"num_retries": 0}) as pair_without_redis:
        group: Final = unique("sync-noredis")
        add_model(pair_without_redis.first, group, "ok-sync-noredis", pair_without_redis.upstream.api_base)
        refused = held(
            lambda: ask(pair_without_redis.second, group, unique("q")).status_code,
            lambda status: status == 400,
            holding=10,
        )
        assert refused == 400


def test_without_redis_a_peer_picks_up_a_new_model_on_the_periodic_reload(tmp_path: Path) -> None:
    """Two instances share only the database and run with `PROXY_CONFIG_RELOAD_INTERVAL_SECONDS=5`.

    After the first instance adds group `sync-poll-*`, the second must serve it with 200 within 30 seconds, from
    the `add_deployment_job` that reloads `LiteLLM_ProxyModelTable` every interval. A value that is not a
    positive integer falls back to 30 seconds (`proxy_server.py:12022`).
    """
    with fleet(
        tmp_path,
        redis=False,
        router_settings={"num_retries": 0},
        environment={"PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": str(POLL_RELOAD_SECONDS)},
    ) as polled:
        group: Final = unique("sync-poll")
        add_model(polled.first, group, "ok-sync-poll", polled.upstream.api_base)
        response = eventually(
            lambda: ask(polled.second, group, unique("q")), lambda answer: answer.status_code == 200, seconds=30
        )
        assert served(response) == "ok-sync-poll"


def test_a_model_from_the_config_file_is_not_removed_by_a_peers_resync(pair: Fleet) -> None:
    """Group `sync-static` comes from the config file of both instances, not from the database.

    After the first instance adds and deletes a database model (which makes the second run a full resync), the
    second instance must still serve `sync-static` with 200 from `ok-sync-static`. A resync that replaced the
    whole model list with the database rows would drop every config file model on every pod.
    """
    model_id: Final = add_model(pair.first, unique("sync-churn"), "ok-sync-churn", pair.upstream.api_base)
    pair.first.post("/model/delete", {"id": model_id})
    held(
        lambda: ask(pair.second, "sync-static", unique("q")).status_code,
        lambda status: status == 200,
        holding=15,
        seconds=20,
    )
    assert served(ask(pair.second, "sync-static", unique("q"))) == "ok-sync-static"

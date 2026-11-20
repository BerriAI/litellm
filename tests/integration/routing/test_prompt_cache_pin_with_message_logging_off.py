import collections
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.client import eventually, gateway_from_environment, object_value
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.redis_process import owned_redis
from integration._support.wire import Reply, Wire, wire_server
from integration.routing._prompt_cache_pin_support import (
    DEFAULT_MARK,
    anthropic_deployment,
    ask_turn,
    chat_body,
    claude_peer,
    new_marker,
    one_new_pin,
    pin_config,
    post_chat,
    read_pins,
    system_turn,
)
from pydantic import JsonValue, TypeAdapter
from redis import Redis

from litellm.constants import REDACTED_BY_LITELLM

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 300)

_GROUP: Final = "pin-logging-off"
_LEFT: Final = "pin-logging-off-left"
_RIGHT: Final = "pin-logging-off-right"
_FOLLOW_UPS: Final = 20
_REDACTED_MESSAGES: Final = [{"role": "user", "content": REDACTED_BY_LITELLM}]
_LOGGED_EVENT: Final = TypeAdapter(dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class _Rig:
    owned: OwnedProxy
    redis: "Redis[bytes]"
    logger_sink: Wire


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("prompt-cache-pin-message-logging-off")
    with (
        gateway_from_environment() as gateway,
        owned_redis(directory) as cache,
        wire_server(claude_peer) as left,
        wire_server(claude_peer) as right,
        wire_server(lambda request: Reply()) as logger_sink,
    ):
        config: Final = pin_config(
            directory,
            [anthropic_deployment(_GROUP, _LEFT, left.url), anthropic_deployment(_GROUP, _RIGHT, right.url)],
            litellm_settings={
                "turn_off_message_logging": True,
                "callbacks": ["integration._support.redaction_probe.probe"],
                "cache": False,
                "enable_redis_auth_cache": False,
            },
        )
        environment: Final = {
            "REDIS_HOST": cache.host,
            "REDIS_PORT": str(cache.port),
            "REDACTION_PROBE_URL": logger_sink.url,
        }
        with (
            owned_proxy_process(gateway, directory, environment, config=config, workers=1) as owned,
            Redis(host=cache.host, port=cache.port, socket_timeout=2) as redis,
        ):
            yield _Rig(owned, redis, logger_sink)


def test_follow_ups_stay_on_the_first_deployment_while_the_logger_sees_redacted_messages(rig: _Rig) -> None:
    before: Final = frozenset(read_pins(rig.redis))
    body: Final = chat_body(_GROUP, [system_turn(new_marker(), DEFAULT_MARK), ask_turn(new_marker())])

    _, first_served = post_chat(rig.owned.gateway, body)
    _, pin = one_new_pin(rig.redis, before)
    assert pin.model_id == first_served, (pin, first_served)

    served: Final = tuple(post_chat(rig.owned.gateway, body)[1] for _ in range(_FOLLOW_UPS))
    assert served == (first_served,) * _FOLLOW_UPS, collections.Counter(served)

    eventually(rig.logger_sink.received.qsize, lambda count: count >= 1 + _FOLLOW_UPS, seconds=30)
    logged: Final = tuple(
        object_value(_LOGGED_EVENT.validate_json(request.body)["standard"])["messages"]
        for request in rig.logger_sink.drain()
    )
    assert len(logged) == 1 + _FOLLOW_UPS, len(logged)
    assert all(messages == _REDACTED_MESSAGES for messages in logged), logged

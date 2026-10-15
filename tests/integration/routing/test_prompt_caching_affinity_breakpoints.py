from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows, scratch_database
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Wire, wire_server
from integration.providers._cache_control_marks_support import (
    anthropic_deployment,
    anthropic_peer,
    final_text,
    marker_of,
    new_marker,
    owned_config,
    post_chat,
)
from pydantic import JsonValue

_MODEL: Final = "affinity-breakpoint-claude"
_FOLLOW_UPS: Final = 16
_LONG_SYSTEM: Final = "Answer from the tool results below. " + "lorem ipsum " * 1500


def _first_turn(session: str, marker: str) -> list[JsonValue]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": f"{_LONG_SYSTEM} session {session}",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        },
        {"role": "user", "content": final_text(marker)},
    ]


def _follow_up(first_turn: list[JsonValue], marker: str) -> list[JsonValue]:
    return [*first_turn, {"role": "assistant", "content": "sunny"}, {"role": "user", "content": final_text(marker)}]


def _served(wire: Wire) -> frozenset[str]:
    return frozenset(marker_of(request) for request in wire.drain())


def _body(messages: list[JsonValue]) -> dict[str, JsonValue]:
    return {"model": _MODEL, "messages": messages, "max_tokens": 64}


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_follow_ups_return_to_the_deployment_that_cached_the_breakpoint_prefix(gateway: Gateway, tmp_path: Path) -> None:
    session: Final = new_marker()
    first_marker: Final = new_marker()
    follow_up_markers: Final = frozenset(new_marker() for _ in range(_FOLLOW_UPS))
    first_turn: Final = _first_turn(session, first_marker)
    with scratch_database() as database_url, wire_server(anthropic_peer) as left, wire_server(anthropic_peer) as right:
        config: Final = owned_config(
            tmp_path,
            [
                {**anthropic_deployment(_MODEL, left.url), "model_info": {"id": f"breakpoint-left-{session}"}},
                {**anthropic_deployment(_MODEL, right.url), "model_info": {"id": f"breakpoint-right-{session}"}},
            ],
            router_settings={"optional_pre_call_checks": ["prompt_caching"]},
        )
        with owned_proxy_process(
            gateway,
            tmp_path,
            {"DATABASE_URL": database_url},
            config=config,
            remove_environment=("DATABASE_URL_READ_REPLICA",),
            workers=2,
        ) as owned:
            status, response_id, text = post_chat(owned.gateway, _body(first_turn))
            assert status == 200, text
            eventually(
                lambda: read_rows(
                    'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                    (response_id,),
                    database_url=database_url,
                ),
                lambda rows: len(rows) == 1,
                seconds=70,
            )
            follow_ups: Final = tuple(
                post_chat(owned.gateway, _body(_follow_up(first_turn, marker))) for marker in follow_up_markers
            )
        served: Final = (_served(left), _served(right))
    assert [status for status, _, _ in follow_ups] == [200] * _FOLLOW_UPS, [text for _, _, text in follow_ups]
    (pinned,) = tuple(side for side in served if first_marker in side)
    assert pinned == {first_marker, *follow_up_markers}, served

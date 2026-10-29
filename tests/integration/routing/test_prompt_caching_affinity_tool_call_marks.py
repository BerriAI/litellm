from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows, scratch_database
from integration._support.process import owned_proxy_process
from integration._support.wire import Wire, wire_server
from integration.providers._cache_control_marks_support import (
    CITIES,
    anthropic_deployment,
    anthropic_peer,
    chat_body,
    conversation,
    final_text,
    marked_calls,
    marker_of,
    new_marker,
    owned_config,
    post_chat,
    tool_call,
)
from pydantic import JsonValue

_MODEL: Final = "affinity-claude"
_FOLLOW_UPS: Final = 24
_LONG_SYSTEM: Final = "Answer from the tool results below. " + "lorem ipsum " * 1500


def _first_turn(session: str, marker: str) -> list[JsonValue]:
    calls: Final = [*marked_calls(cities=CITIES[:2]), tool_call(CITIES[2])]
    return [
        {"role": "system", "content": f"{_LONG_SYSTEM} session {session}"},
        *conversation(marker, calls, ask_marked=False)[1:],
    ]


def _follow_up(first_turn: list[JsonValue], marker: str) -> list[JsonValue]:
    return [*first_turn, {"role": "assistant", "content": "sunny"}, {"role": "user", "content": final_text(marker)}]


def _served(wire: Wire) -> frozenset[str]:
    return frozenset(marker_of(request) for request in wire.drain())


@pytest.mark.timeout(240)
def test_router_affinity_is_lost_when_auto_caching_stands_down_for_tool_call_marks(
    gateway: Gateway, tmp_path: Path
) -> None:
    session: Final = new_marker()
    first_marker: Final = new_marker()
    follow_up_markers: Final = tuple(new_marker() for _ in range(_FOLLOW_UPS))
    first_turn: Final = _first_turn(session, first_marker)
    with scratch_database() as database_url, wire_server(anthropic_peer) as left, wire_server(anthropic_peer) as right:
        config: Final = owned_config(
            tmp_path,
            [
                {**anthropic_deployment(_MODEL, left.url), "model_info": {"id": f"affinity-left-{session}"}},
                {**anthropic_deployment(_MODEL, right.url), "model_info": {"id": f"affinity-right-{session}"}},
            ],
            router_settings={"optional_pre_call_checks": ["prompt_caching"]},
        )
        with (
            owned_proxy_process(
                gateway,
                tmp_path,
                {"DATABASE_URL": database_url},
                config=config,
                remove_environment=("DATABASE_URL_READ_REPLICA",),
                workers=2,
            ) as owned,
            owned.gateway.scenario() as scenario,
        ):
            key: Final = scenario.key(metadata={"enable_prompt_caching": True})
            status, response_id, text = post_chat(owned.gateway, chat_body(_MODEL, first_turn), key=key)
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
                post_chat(owned.gateway, chat_body(_MODEL, _follow_up(first_turn, marker)), key=key)
                for marker in follow_up_markers
            )
        served: Final = {"left": _served(left), "right": _served(right)}
    assert [status for status, _, _ in follow_ups] == [200] * _FOLLOW_UPS, [text for _, _, text in follow_ups]
    assert {first_marker, *follow_up_markers} == served["left"] | served["right"]
    assert all(served[side] & set(follow_up_markers) for side in served), served

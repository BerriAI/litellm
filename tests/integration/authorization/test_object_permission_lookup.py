import time
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.database import read_rows

PLAIN_REQUESTS: Final = 10
STATS_FLUSH_WINDOW_SECONDS: Final = 11.0


def _object_permission_reads() -> int:
    rows: Final = read_rows(
        "SELECT seq_scan + idx_scan AS reads FROM pg_stat_user_tables WHERE relname = %s",
        ("LiteLLM_ObjectPermissionTable",),
    )
    reads: Final = rows[0]["reads"]
    assert isinstance(reads, int), rows
    return reads


def _settled_object_permission_reads(previous: int, unchanged_since: float) -> int:
    changed: Final = eventually(
        lambda: _object_permission_reads() != previous,
        lambda drifted: drifted,
        seconds=STATS_FLUSH_WINDOW_SECONDS - (time.monotonic() - unchanged_since),
        return_last_on_timeout=True,
    )
    if not changed:
        return previous
    return _settled_object_permission_reads(_object_permission_reads(), time.monotonic())


def _assert_plain_chat_served(gateway: Gateway, model: str, key: str) -> None:
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "no vector stores"}]},
        key=key,
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == (
        "Hello! This is a mock response from the fake OpenAI endpoint."
    )


def _assert_forbidden_vector_store_denied(gateway: Gateway, model: str, key: str) -> None:
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "forbidden"}], "vector_store_ids": ["vs_forbidden"]},
        key=key,
    )
    assert response.status_code == 401, response.text
    assert response.json()["error"]["type"] == "key_vector_store_access_denied", response.text


@pytest.mark.covers("authorization.vector_store.plain_request_skips_object_permission_lookup")
def test_chat_request_without_vector_stores_does_not_read_object_permission_table(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model], object_permission={"vector_stores": ["vs_allowed"]})
        _assert_plain_chat_served(gateway, model, key)
        before_control: Final = _object_permission_reads()
        _assert_forbidden_vector_store_denied(gateway, model, key)
        eventually(_object_permission_reads, lambda reads: reads > before_control, seconds=15)
        baseline: Final = _settled_object_permission_reads(_object_permission_reads(), time.monotonic())
        for _ in range(PLAIN_REQUESTS):
            _assert_plain_chat_served(gateway, model, key)
        after: Final = _settled_object_permission_reads(_object_permission_reads(), time.monotonic())
        assert after - baseline < PLAIN_REQUESTS, (
            f"{PLAIN_REQUESTS} plain chat requests added {after - baseline} object permission reads"
        )

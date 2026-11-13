from typing import Final

import pytest

from litellm.proxy.agent_endpoints.external_cloud_agents.inventory import (
    Block,
    Message,
    SourceCache,
    message_tools,
    parse_frame,
)
from tests.unit.proxy.agent_endpoints.external_cloud_agents.conftest import DatabaseDouble, Harness


FRAME: Final = {"session_id": "bot", "tenant_id": "tenant", "type": "prompt", "actor": {"id": "member"}}


def test_authenticated_frame_is_always_allowed_and_unknown_fields_are_ignored(harness: Harness) -> None:
    response: Final = harness.post_hook(
        {**FRAME, "unknown": {"future": "field"}, "messages": [{"content": "not persisted"}]}
    )
    assert response.status_code == 200
    assert response.json() == {"action": "allow"}
    assert harness.db.writes[-1][1:4] == ("bot", "member", None)
    assert "not persisted" not in str(harness.db.writes)


@pytest.mark.parametrize("header", [None, "Bearer wrong", "Basic wrong", "Bearer "])
def test_missing_or_bad_bearer_does_not_write_inventory(harness: Harness, header: str | None) -> None:
    headers: Final = {"Authorization": header} if header is not None else {}
    response: Final = harness.client.post(f"/grok_bot/inference_hook/{harness.source_id}", json=FRAME, headers=headers)
    assert response.status_code == 401
    assert harness.db.writes == ()


def test_unknown_source_is_unauthorized(harness: Harness) -> None:
    response: Final = harness.client.post(
        "/grok_bot/inference_hook/unknown", json=FRAME, headers={"Authorization": "Bearer key"}
    )
    assert response.status_code == 401
    assert harness.db.writes == ()


def test_tenant_mismatch_is_unauthorized_without_recording(harness: Harness) -> None:
    response: Final = harness.post_hook({**FRAME, "tenant_id": "other-tenant"})
    assert response.status_code == 401
    assert harness.db.writes == ()


@pytest.mark.parametrize("extra", [{"type": "config-test"}, {"source": {"application": "config-test"}}])
def test_config_test_updates_connection_without_registering_bot(harness: Harness, extra: dict[str, object]) -> None:
    response: Final = harness.post_hook({**FRAME, **extra})
    assert response.status_code == 200
    assert response.json() == {"action": "allow"}
    assert harness.db.writes == ((harness.source_id,),)


def test_first_frame_binds_tenant(harness: Harness) -> None:
    assert harness.db.source is not None
    harness.db.source = harness.db.source.model_copy(update={"tenant_id": None})
    response: Final = harness.post_hook(FRAME)
    assert response.status_code == 200
    assert harness.db.source.tenant_id == FRAME["tenant_id"]
    assert harness.db.writes[-1][1] == FRAME["session_id"]


def test_tool_and_mcp_names_are_distinct_and_bounded_before_persistence(harness: Harness) -> None:
    blocks: Final = [
        {"type": "tool_use", "name": f"tool-{i:02}", "tool_info": {"toolset_name": "mcp-server"}} for i in range(70)
    ]
    response: Final = harness.post_hook({**FRAME, "type": "tool_call", "messages": [{"content": blocks + blocks}]})
    assert response.status_code == 200
    assert harness.db.writes[-1][-1] == ["mcp-server", *(f"tool-{i:02}" for i in range(49))]


def test_prompt_text_is_not_a_tool() -> None:
    frame: Final = parse_frame(b'{"messages":[{"content":[{"type":"text","name":"secret","text":"private"}]}]}')
    assert frame is not None
    assert message_tools(frame.messages[0]) == ()


def test_tool_extraction_caps_a_forty_thousand_block_frame() -> None:
    message: Final = Message(content=tuple(Block(type="tool_use", name=f"tool-{i:05}") for i in range(40000)))
    assert message_tools(message) == tuple(f"tool-{i:05}" for i in range(50))


@pytest.mark.parametrize("reads_fail", [False, True])
def test_internal_database_errors_fail_open(harness: Harness, reads_fail: bool) -> None:
    harness.db.fail_reads = reads_fail
    harness.db.fail_writes = True
    response: Final = harness.post_hook(FRAME)
    assert response.status_code == 200
    assert response.json() == {"action": "allow"}


@pytest.mark.parametrize("body", [b"not-json", b'{"actor":42}'])
def test_malformed_authenticated_frame_still_updates_connection(harness: Harness, body: bytes) -> None:
    response: Final = harness.client.post(
        f"/grok_bot/inference_hook/{harness.source_id}",
        content=body,
        headers={"Authorization": "Bearer test-ingest-credential"},
    )
    assert response.status_code == 200
    assert response.json() == {"action": "allow"}
    assert harness.db.writes == ((harness.source_id,),)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def read(self) -> float:
        return self.now

    def advance(self) -> None:
        self.now += 21


@pytest.mark.asyncio
async def test_missing_sources_are_cached_briefly_and_bounded(db: DatabaseDouble) -> None:
    clock: Final = Clock()
    cache: Final = SourceCache(clock.read)
    assert await cache.source(db, "missing") is None
    db.fail_reads = True
    assert await cache.source(db, "missing") is None
    clock.advance()
    with pytest.raises(RuntimeError, match="database offline"):
        await cache.source(db, "missing")
    db.fail_reads = False
    for index in range(257):
        assert await cache.source(db, f"missing-{index}") is None
    db.fail_reads = True
    assert await cache.source(db, "missing-256") is None
    with pytest.raises(RuntimeError, match="database offline"):
        await cache.source(db, "missing-0")


@pytest.mark.asyncio
async def test_cache_is_worker_local_expires_and_can_be_invalidated(db: DatabaseDouble) -> None:
    clock: Final = Clock()
    cache: Final = SourceCache(clock.read)
    assert db.source is not None
    source_id: Final = db.source.source_id
    original: Final = await cache.source(db, source_id)
    db.source = db.source.model_copy(update={"tenant_id": "new"})
    assert await cache.source(db, source_id) == original
    clock.advance()
    assert await cache.source(db, source_id) == db.source
    cache.discard(source_id)
    db.source = None
    assert await cache.source(db, source_id) is None

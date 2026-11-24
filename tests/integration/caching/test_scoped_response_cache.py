import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.openai_wire import chat_reply
from integration._support.provider import SharedProvider
from integration.caching.response_cache_case import await_entry, entry_key, redis_store
from pydantic import JsonValue

pytestmark: Final = pytest.mark.shares_worker_state


def test_team_disables_cache_reads_and_writes_and_key_can_enable_with_its_own_ttl(
    gateway: Gateway, provider: SharedProvider
) -> None:
    case: Final = uuid.uuid4().hex
    replies: Final = tuple(f"{case}-{index}" for index in range(3))
    model: Final = "gpt-6.1-sol"
    provider.expect(*(chat_reply(f"chatcmpl-{reply}", model, reply, stream=False) for reply in replies))
    with gateway.scenario() as scenario, redis_store() as store:
        team: Final = scenario.team(metadata={"cache_params": {"enabled": False}})
        team_key: Final = scenario.key(team_id=team)
        override_key: Final = scenario.key(
            team_id=team,
            metadata={"cache": {"ttl": 120}, "cache_params": {"enabled": True, "ttl": 30}},
        )
        body: Final[dict[str, JsonValue]] = {
            "model": f"openai/{model}",
            "messages": [{"role": "user", "content": case}],
        }
        path: Final = "/v1/chat/completions"
        first: Final = gateway.request("POST", path, body, key=team_key)
        assert first.status_code == 200, first.text
        assert first.json()["id"] == f"chatcmpl-{replies[0]}"
        assert first.headers.get("x-litellm-cache-scope") == "team"

        enabled: Final = gateway.request("POST", path, body, key=override_key)
        assert enabled.status_code == 200, enabled.text
        assert enabled.json()["id"] == f"chatcmpl-{replies[1]}"
        assert enabled.headers.get("x-litellm-cache-scope") == "key"
        cache_key: Final = await_entry(store, replies[1])
        assert 0 < store.ttl(cache_key) <= 30

        bypassed: Final = gateway.request("POST", path, body, key=team_key)
        assert bypassed.status_code == 200, bypassed.text
        assert bypassed.json()["id"] == f"chatcmpl-{replies[2]}"
        assert gateway.post(path, body, key=override_key)["id"] == f"chatcmpl-{replies[1]}"
        assert entry_key(store, replies[0]) is None
        assert entry_key(store, replies[2]) is None
        assert len(provider.received()) == 3

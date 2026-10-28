"""``a2a_protocol_version`` is agent-only metadata: the completion bridge must strip it before
``litellm.acompletion`` so it never reaches a provider, and paths that do not build an a2a-sdk client (card
discovery, the ``a2a/<agent>`` chat-completions provider) must not change for a pinned agent."""

from __future__ import annotations

import json
import uuid
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually
from integration.compatibility._a2a_peers import legacy_send, peer, posts, register, spend_rows, sse_frames
from pydantic import JsonValue


def _observations(gateway: Gateway) -> list[dict[str, JsonValue]]:
    response: Final = httpx.get(gateway.upstream_url + "/__observations", trust_env=False, timeout=15)
    response.raise_for_status()
    return list(response.json()["requests"])


@pytest.mark.parametrize("method", ("message/send", "message/stream"))
def test_completion_bridge_drops_protocol_version_before_the_provider(gateway: Gateway, method: str) -> None:
    marker: Final = "bridge" + uuid.uuid4().hex
    key: Final = "sk-scripted-" + marker
    with gateway.scenario() as scenario:
        identity: Final = register(
            gateway,
            scenario.cleanups.callback,
            marker,
            gateway.upstream_url + "/v1",
            {"custom_llm_provider": "openai", "model": "gpt-4o-mini", "api_key": key, "a2a_protocol_version": "0.3"},
        )
        text: Final = "bridge ping " + marker
        response: Final = gateway.request("POST", f"/a2a/{identity}", legacy_send(marker, text=text, method=method))
        assert response.status_code == 200, response.text
        if method == "message/send":
            body: Final = response.json()
            assert body["id"] == marker and "error" not in body, response.text
            parts = body["result"]["message"]["parts"]
            assert parts == [{"text": "Hello! This is a mock response from the fake OpenAI endpoint."}], response.text
        else:
            frames: Final = sse_frames(response.text)
            assert frames and all("error" not in frame and frame["id"] == marker for frame in frames), response.text
            assert "TASK_STATE_COMPLETED" in response.text, response.text
        mine: Final = [item for item in _observations(gateway) if item["authorization"] == "Bearer " + key]
        assert len(mine) == 1, mine
        sent: Final = mine[0]["body"]
        assert isinstance(sent, dict), mine
        assert "a2a_protocol_version" not in sent and "a2a_protocol_version" not in json.dumps(sent), sent
        assert sent["model"] == "gpt-4o-mini" and sent["messages"] == [{"role": "user", "content": text}], sent
        assert sent.get("stream", False) is (method == "message/stream"), sent
        if method == "message/send":
            rows: Final = eventually(lambda: spend_rows(marker), lambda values: len(values) == 1, seconds=70)
            assert rows[0]["call_type"] == "asend_message" and rows[0]["status"] == "success", rows


def test_card_discovery_of_pinned_agent_is_unchanged(gateway: Gateway) -> None:
    with peer("lgcard", "langgraph") as agent, gateway.scenario() as scenario:
        identity: Final = register(
            gateway, scenario.cleanups.callback, agent.marker, agent.url, {"a2a_protocol_version": "0.3"}
        )
        discovered: Final = gateway.request("GET", f"/a2a/{identity}/.well-known/agent-card.json")
        assert discovered.status_code == 200, discovered.text
        card: Final = discovered.json()
        assert card["protocolVersion"] == "1.0" and card["name"] == agent.marker, discovered.text
        assert "a2a_protocol_version" not in discovered.text, discovered.text
        assert agent.wire.drain() == ()  # type: ignore[union-attr]


def test_chat_completions_a2a_provider_unaffected_by_pin(gateway: Gateway) -> None:
    with peer("lgchat", "langgraph") as agent, gateway.scenario() as scenario:
        register(gateway, scenario.cleanups.callback, agent.marker, agent.url, {"a2a_protocol_version": "0.3"})
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": "a2a/" + agent.marker, "messages": [{"role": "user", "content": "chat ping " + agent.marker}]},
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["choices"][0]["message"]["content"] == "echo: user: chat ping " + agent.marker, response.text
        sent: Final = posts(agent.wire)  # type: ignore[arg-type]
        assert len(sent) == 1 and sent[0]["method"] == "message/send", sent
        assert "a2a_protocol_version" not in json.dumps(sent), sent

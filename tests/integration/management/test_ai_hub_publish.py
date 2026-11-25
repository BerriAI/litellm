import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, list_value, object_value

AGENTS_PER_RACE: Final = 8
PUBLISHES: Final[MappingProxyType[str, Callable[[Gateway, str], httpx.Response]]] = MappingProxyType({
    "single-agent-route": lambda proxy, agent_id: proxy.request("POST", f"/v1/agents/{agent_id}/make_public"),
    "dashboard-modal": lambda proxy, agent_id: proxy.request(
        "POST", "/v1/agents/make_public", {"agent_ids": [agent_id], "previous_agent_ids": []}
    ),
})


def _agent(gateway: Gateway, scenario: Scenario) -> str:
    name: Final = f"hub-publish-{uuid.uuid4().hex}"
    card: Final = {
        "protocolVersion": "0.3",
        "name": name,
        "description": "Synthetic hub agent",
        "version": "1.0.0",
        "url": "http://127.0.0.1:9/",
        "capabilities": {"streaming": False},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [],
    }
    created: Final = gateway.request("POST", "/v1/agents", {"agent_name": name, "agent_card_params": card})
    assert created.status_code == 200, created.text
    agent_id: Final = str(created.json()["agent_id"])
    scenario.cleanups.callback(lambda: gateway.request("DELETE", f"/v1/agents/{agent_id}"))
    return agent_id


def _stored_public_agents(gateway: Gateway) -> frozenset[str]:
    rows: Final = gateway.rows(
        'SELECT param_value FROM "LiteLLM_Config" WHERE param_name=%s', ("litellm_settings",)
    )
    return frozenset(str(item) for item in list_value(object_value(rows[0]["param_value"])["public_agent_groups"]))


@pytest.mark.parametrize("publish", tuple(PUBLISHES), ids=tuple(PUBLISHES))
def test_concurrent_publishes_through_two_proxies_keep_every_agent_and_a_delete_unpublishes(
    gateway: Gateway, peer: Gateway, scenario: Scenario, publish: str
) -> None:
    agent_ids: Final = tuple(_agent(gateway, scenario) for _ in range(AGENTS_PER_RACE))
    proxies: Final = (gateway, peer)

    with ThreadPoolExecutor(max_workers=AGENTS_PER_RACE) as pool:
        replies: Final = tuple(
            pool.map(
                lambda indexed: PUBLISHES[publish](proxies[indexed[0] % 2], indexed[1]),
                enumerate(agent_ids),
            )
        )

    assert [reply.status_code for reply in replies] == [200] * AGENTS_PER_RACE, [reply.text for reply in replies]
    assert frozenset(agent_ids) <= _stored_public_agents(gateway)

    deleted: Final = peer.request("DELETE", f"/v1/agents/{agent_ids[0]}")

    assert deleted.status_code == 200, deleted.text
    assert agent_ids[0] not in _stored_public_agents(gateway)
    assert frozenset(agent_ids[1:]) <= _stored_public_agents(gateway)

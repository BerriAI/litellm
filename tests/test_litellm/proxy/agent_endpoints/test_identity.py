from collections.abc import Mapping
from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy.agent_endpoints.identity import agent_identity, preserve_identity, validate_identity_binding
from litellm.types.agents import AgentResponse

TENANT: Final = "11111111-1111-4111-8111-111111111111"
CLIENT: Final = "22222222-2222-4222-8222-222222222222"
ISSUER: Final = f"https://login.microsoftonline.com/{TENANT}/v2.0"


def binding() -> dict[str, object]:
    return {"provider": "microsoft_entra", "tenant_id": TENANT, "client_id": CLIENT}


def registered_agent(agent_id: str = "agent-one") -> AgentResponse:
    return AgentResponse(
        agent_id=agent_id, agent_name="Readable agent", agent_card_params={}, litellm_params={"identity": binding()}
    )


@pytest.mark.parametrize("params", [None, {}, {"identity": None}])
def test_unbound_agents_keep_legacy_configuration(params: Mapping[str, object] | None) -> None:
    assert agent_identity(params) is None
    validate_identity_binding(params, (), ())


@pytest.mark.parametrize("identity", [{}, {"provider": "other"}, {"provider": "microsoft_entra", "tenant_id": TENANT, "client_id": "bad"}])
def test_rejects_invalid_identity_configuration(identity: Mapping[str, object]) -> None:
    with pytest.raises(HTTPException) as failure:
        validate_identity_binding({"identity": identity}, (), (ISSUER,))
    assert failure.value.status_code == 400


def test_binding_requires_trusted_tenant_and_unique_agent() -> None:
    agent: Final = registered_agent()
    with pytest.raises(HTTPException) as untrusted:
        validate_identity_binding(agent.litellm_params, (), ())
    assert untrusted.value.status_code == 400
    with pytest.raises(HTTPException) as duplicate:
        validate_identity_binding(agent.litellm_params, (agent,), (ISSUER,))
    assert duplicate.value.status_code == 409
    validate_identity_binding(agent.litellm_params, (agent,), (ISSUER,), agent.agent_id)
    assert agent_identity(agent.litellm_params).issuer == ISSUER


@pytest.mark.parametrize("incoming", [{"model": "new-model"}, {}])
def test_runtime_updates_preserve_identity(incoming: Mapping[str, object]) -> None:
    existing: Final = {"identity": binding(), "model": "old-model"}
    merged: Final = preserve_identity(incoming, existing)
    assert merged["identity"] == binding()
    assert dict(incoming) == {key: value for key, value in merged.items() if key != "identity"}
    assert existing["model"] == "old-model"


def test_identity_can_be_explicitly_removed() -> None:
    agent: Final = registered_agent()
    assert preserve_identity({"identity": None}, agent.litellm_params or {}) == {"identity": None}
    assert preserve_identity({"model": "new"}, {}) == {"model": "new"}

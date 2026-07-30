"""
Unit tests for the policy list endpoints: config-defined policies and attachments
must be listed alongside DB-backed ones.
"""

import pytest
from fastapi import HTTPException

import litellm.proxy.proxy_server as proxy_server
from litellm.proxy.policy_engine.attachment_registry import get_attachment_registry
from litellm.proxy.policy_engine.policy_endpoints import (
    delete_policy,
    delete_policy_attachment,
    get_policy,
    list_policies,
    list_policy_attachments,
)
from litellm.proxy.policy_engine.policy_registry import get_policy_registry


@pytest.fixture
def config_policy_engine(monkeypatch):
    """Load one config policy + attachment into the registries, with no DB attached."""
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    policy_registry = get_policy_registry()
    attachment_registry = get_attachment_registry()
    policy_registry.load_policies({"config-policy": {"description": "from config", "guardrails": {"add": ["tooling"]}}})
    attachment_registry.load_attachments([{"policy": "config-policy", "scope": "*"}])
    yield
    policy_registry.clear()
    attachment_registry.clear()


@pytest.mark.asyncio
async def test_list_policies_returns_config_policies(config_policy_engine):
    response = await list_policies()

    assert response.total_count == 1
    policy = response.policies[0]
    assert policy.policy_name == "config-policy"
    assert policy.policy_id == "config-policy"
    assert policy.guardrails_add == ["tooling"]
    assert policy.policy_definition_location == "config"


@pytest.mark.asyncio
async def test_list_policies_excludes_config_policies_for_version_filters(config_policy_engine):
    assert (await list_policies(version_status="draft")).policies == []
    assert len((await list_policies(version_status="production")).policies) == 1


@pytest.mark.asyncio
async def test_list_policy_attachments_returns_config_attachments(config_policy_engine):
    response = await list_policy_attachments()

    assert response.total_count == 1
    attachment = response.attachments[0]
    assert attachment.policy_name == "config-policy"
    assert attachment.scope == "*"
    assert attachment.policy_definition_location == "config"


@pytest.mark.asyncio
async def test_get_policy_returns_config_policy(config_policy_engine):
    policy = await get_policy("config-policy")

    assert policy.policy_name == "config-policy"
    assert policy.policy_definition_location == "config"


@pytest.mark.asyncio
async def test_mutating_config_policy_is_rejected(config_policy_engine):
    with pytest.raises(HTTPException) as exc_info:
        await delete_policy("config-policy")
    assert exc_info.value.status_code == 400
    assert "config.yaml" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_deleting_config_attachment_is_rejected(config_policy_engine):
    attachment_id = (await list_policy_attachments()).attachments[0].attachment_id

    with pytest.raises(HTTPException) as exc_info:
        await delete_policy_attachment(attachment_id)
    assert exc_info.value.status_code == 400
    assert "config.yaml" in str(exc_info.value.detail)

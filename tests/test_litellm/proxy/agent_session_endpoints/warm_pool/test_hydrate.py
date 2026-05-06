"""Unit tests for `litellm/proxy/agent_session_endpoints/warm_pool/hydrate.py`.

Verify the payload builder:
* respects scope filtering — out-of-scope secrets never land in the payload
* tolerates a missing `LiteLLM_AgentVMConfig` row (defaults to allow_all)
* falls back to a sensible model when the agent has none
* stringifies env_vars correctly so the daemon's env file is well-formed
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest


# Tests need a stable salt key BEFORE the encryption module is imported by
# the production code we're calling.
os.environ.setdefault("LITELLM_SALT_KEY", "sk-test-salt-key-do-not-use-in-prod")


from litellm.proxy.agent_session_endpoints.warm_pool.hydrate import (
    build_hydrate_payload,
)
from litellm.proxy.agent_settings_endpoints.encryption import encrypt_optional


class _Row:
    def __init__(self, **fields: Any) -> None:
        for k, v in fields.items():
            setattr(self, k, v)


class _Table:
    def __init__(self, rows: Optional[List[_Row]] = None) -> None:
        self.rows = rows or []

    async def find_many(self, where=None) -> List[_Row]:
        if where is None:
            return list(self.rows)
        return [
            r
            for r in self.rows
            if all(getattr(r, k, None) == v for k, v in where.items())
        ]

    async def find_unique(self, where: Dict[str, Any]) -> Optional[_Row]:
        for r in self.rows:
            if all(getattr(r, k, None) == v for k, v in where.items()):
                return r
        return None


class _DB:
    def __init__(self) -> None:
        self.litellm_agentsecret = _Table()
        self.litellm_agentvmconfig = _Table()


class _Prisma:
    def __init__(self) -> None:
        self.db = _DB()


@pytest.fixture
def prisma():
    return _Prisma()


@pytest.fixture
def jwt_expires_at():
    return datetime(2026, 5, 7, 0, 0, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_payload_with_no_secrets_no_config(prisma, jwt_expires_at):
    """No secrets, no VMConfig row -> defaults all the way through."""
    payload = await build_hydrate_payload(
        prisma_client=prisma,
        session_id="sess-1",
        agent_id="agent-1",
        team_id="team-A",
        jwt="jwt-x",
        jwt_expires_at=jwt_expires_at,
        repos=[{"url": "https://github.com/a/b", "ref": "main"}],
        env_vars={"NODE_ENV": "test"},
        agent_row=_Row(model="gpt-4o", system_prompt="hi"),
    )

    assert payload.session_id == "sess-1"
    assert payload.agent_id == "agent-1"
    assert payload.jwt == "jwt-x"
    assert payload.jwt_expires_at == jwt_expires_at.isoformat()
    assert payload.secrets == {}
    assert payload.network_access.mode == "allow_all"
    assert payload.agent_config.model == "gpt-4o"
    assert payload.agent_config.system_prompt == "hi"
    assert len(payload.repos) == 1
    assert payload.repos[0].url == "https://github.com/a/b"
    assert payload.env_vars == {"NODE_ENV": "test"}


@pytest.mark.asyncio
async def test_in_scope_secrets_decrypted(prisma, jwt_expires_at):
    """`scope='all'` and matching repo scope land in the payload."""
    prisma.db.litellm_agentsecret.rows = [
        _Row(
            team_id="team-A",
            name="GLOBAL_TOKEN",
            value_enc=encrypt_optional("global-secret-value"),
            scope="all",
        ),
        _Row(
            team_id="team-A",
            name="REPO_TOKEN",
            value_enc=encrypt_optional("repo-specific-value"),
            scope=["foo/bar"],
        ),
    ]

    payload = await build_hydrate_payload(
        prisma_client=prisma,
        session_id="sess-1",
        agent_id="agent-1",
        team_id="team-A",
        jwt="jwt",
        jwt_expires_at=jwt_expires_at,
        repos=[{"url": "https://github.com/foo/bar"}],
        env_vars=None,
        agent_row=None,
    )

    assert payload.secrets == {
        "GLOBAL_TOKEN": "global-secret-value",
        "REPO_TOKEN": "repo-specific-value",
    }


@pytest.mark.asyncio
async def test_out_of_scope_secrets_excluded(prisma, jwt_expires_at):
    """A secret scoped to repo `x/y` is dropped when session repos don't match."""
    prisma.db.litellm_agentsecret.rows = [
        _Row(
            team_id="team-A",
            name="SECRET_FOR_X",
            value_enc=encrypt_optional("must-not-leak"),
            scope=["x/y"],
        ),
    ]

    payload = await build_hydrate_payload(
        prisma_client=prisma,
        session_id="sess-1",
        agent_id="agent-1",
        team_id="team-A",
        jwt="jwt",
        jwt_expires_at=jwt_expires_at,
        repos=[{"url": "https://github.com/foo/bar"}],
        env_vars=None,
        agent_row=None,
    )

    assert "SECRET_FOR_X" not in payload.secrets


@pytest.mark.asyncio
async def test_other_team_secrets_excluded(prisma, jwt_expires_at):
    """A secret on a different team_id is invisible to this team."""
    prisma.db.litellm_agentsecret.rows = [
        _Row(
            team_id="other-team",
            name="OTHER_TEAM_SECRET",
            value_enc=encrypt_optional("must-not-leak"),
            scope="all",
        ),
    ]

    payload = await build_hydrate_payload(
        prisma_client=prisma,
        session_id="sess-1",
        agent_id="agent-1",
        team_id="team-A",
        jwt="jwt",
        jwt_expires_at=jwt_expires_at,
        repos=[],
        env_vars=None,
        agent_row=None,
    )

    assert payload.secrets == {}


@pytest.mark.asyncio
async def test_network_access_loaded_from_vm_config(prisma, jwt_expires_at):
    """`LiteLLM_AgentVMConfig.network_access` is reflected in the payload."""
    prisma.db.litellm_agentvmconfig.rows = [
        _Row(
            team_id="team-A",
            network_access={
                "mode": "allowlist",
                "allowlist": ["github.com", "api.openai.com"],
            },
        ),
    ]

    payload = await build_hydrate_payload(
        prisma_client=prisma,
        session_id="sess-1",
        agent_id="agent-1",
        team_id="team-A",
        jwt="jwt",
        jwt_expires_at=jwt_expires_at,
        repos=[],
        env_vars=None,
        agent_row=None,
    )

    assert payload.network_access.mode == "allowlist"
    assert payload.network_access.allowlist == ["github.com", "api.openai.com"]


@pytest.mark.asyncio
async def test_decrypt_failure_skips_secret_does_not_crash(prisma, jwt_expires_at):
    """A bad ciphertext is skipped — payload still builds, secret is absent."""
    good_ciphertext = encrypt_optional("good-value")
    bad_ciphertext = "not-valid-ciphertext"
    prisma.db.litellm_agentsecret.rows = [
        _Row(
            team_id="team-A",
            name="GOOD_SECRET",
            value_enc=good_ciphertext,
            scope="all",
        ),
        _Row(
            team_id="team-A",
            name="BAD_SECRET",
            value_enc=bad_ciphertext,
            scope="all",
        ),
    ]

    def fake_decrypt(value, key):
        if value == good_ciphertext:
            return "good-value"
        raise ValueError("simulated decrypt fail")

    with patch(
        "litellm.proxy.agent_session_endpoints.warm_pool.hydrate.decrypt_optional",
        new=fake_decrypt,
    ):
        payload = await build_hydrate_payload(
            prisma_client=prisma,
            session_id="sess-1",
            agent_id="agent-1",
            team_id="team-A",
            jwt="jwt",
            jwt_expires_at=jwt_expires_at,
            repos=[],
            env_vars=None,
            agent_row=None,
        )

    assert "GOOD_SECRET" in payload.secrets
    assert "BAD_SECRET" not in payload.secrets


@pytest.mark.asyncio
async def test_default_model_when_agent_row_missing(prisma, jwt_expires_at):
    """Builder uses 'gpt-4o-mini' when agent_row is None and no model passed."""
    payload = await build_hydrate_payload(
        prisma_client=prisma,
        session_id="sess-1",
        agent_id="agent-1",
        team_id="team-A",
        jwt="jwt",
        jwt_expires_at=jwt_expires_at,
        repos=[],
        env_vars=None,
        agent_row=None,
    )
    assert payload.agent_config.model == "gpt-4o-mini"

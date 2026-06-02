"""
Tests for `team_config.get_team_vm_config` — BYOC creds resolution.

Covers:
- Validation #11: missing creds → InvalidCredentialsError fast (no instance launched)
- Validation #12: cross-team isolation — two teams resolve to two distinct creds
- Encryption round-trip (encrypt_aws_creds + decrypt_aws_creds)
- env-var fallback for local dev
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest


def _set_master_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-do-not-use-in-prod-1234567890")
    # `_get_salt_key` falls through to `master_key` when LITELLM_SALT_KEY is unset
    # but for tests we set the env var directly so we don't need to import
    # proxy_server.master_key.


def test_encrypt_decrypt_aws_creds_round_trip(monkeypatch):
    _set_master_key(monkeypatch)
    from litellm.proxy.agent_session_endpoints.vm_providers.base import AwsCreds
    from litellm.proxy.agent_session_endpoints.vm_providers.team_config import (
        decrypt_aws_creds,
        encrypt_aws_creds,
    )

    creds = AwsCreds(
        access_key_id="AKIAROUNDTRIPTESTKEY",
        secret_access_key="round-trip-secret-do-not-leak",
        session_token=None,
        region="us-west-2",
    )
    blob = encrypt_aws_creds(creds)

    # The encrypted blob must NOT contain the plaintext key or secret.
    assert "AKIAROUNDTRIPTESTKEY" not in blob
    assert "round-trip-secret-do-not-leak" not in blob

    decrypted = decrypt_aws_creds(blob)
    assert decrypted.access_key_id == creds.access_key_id
    assert decrypted.secret_access_key == creds.secret_access_key
    assert decrypted.region == creds.region


def test_decrypt_invalid_blob_raises_invalid_credentials(monkeypatch):
    _set_master_key(monkeypatch)
    from litellm.proxy.agent_session_endpoints.vm_providers.base import (
        InvalidCredentialsError,
    )
    from litellm.proxy.agent_session_endpoints.vm_providers.team_config import (
        decrypt_aws_creds,
    )

    with pytest.raises(InvalidCredentialsError):
        decrypt_aws_creds("not-a-json-string-{{{")


@pytest.mark.asyncio
async def test_no_db_no_env_raises_invalid_credentials(monkeypatch):
    """Validation #11: missing creds must fail-fast, not launch any instance."""
    _set_master_key(monkeypatch)
    monkeypatch.delenv("LITELLM_AGENT_AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("LITELLM_AGENT_AWS_SECRET_ACCESS_KEY", raising=False)

    from litellm.proxy.agent_session_endpoints.vm_providers.base import (
        InvalidCredentialsError,
    )
    from litellm.proxy.agent_session_endpoints.vm_providers.team_config import (
        get_team_vm_config,
    )

    # Prisma returns None for the team.
    prisma = MagicMock()
    prisma.db = MagicMock()
    prisma.db.litellm_agentvmconfig = MagicMock()
    prisma.db.litellm_agentvmconfig.find_unique = AsyncMock(return_value=None)

    with pytest.raises(InvalidCredentialsError):
        await get_team_vm_config("team-no-creds", prisma_client=prisma)


@pytest.mark.asyncio
async def test_env_fallback_when_no_db_row(monkeypatch):
    _set_master_key(monkeypatch)
    monkeypatch.setenv("LITELLM_AGENT_AWS_ACCESS_KEY_ID", "AKIAFROMENV")
    monkeypatch.setenv("LITELLM_AGENT_AWS_SECRET_ACCESS_KEY", "secret-from-env")
    monkeypatch.setenv("LITELLM_AGENT_AWS_REGION", "us-east-1")

    from litellm.proxy.agent_session_endpoints.vm_providers.team_config import (
        get_team_vm_config,
    )

    prisma = MagicMock()
    prisma.db = MagicMock()
    prisma.db.litellm_agentvmconfig = MagicMock()
    prisma.db.litellm_agentvmconfig.find_unique = AsyncMock(return_value=None)

    cfg = await get_team_vm_config("team-env", prisma_client=prisma)
    assert cfg.aws_creds.access_key_id == "AKIAFROMENV"
    assert cfg.aws_creds.secret_access_key == "secret-from-env"
    assert cfg.aws_creds.region == "us-east-1"


@pytest.mark.asyncio
async def test_falls_back_to_env_when_prisma_table_missing(monkeypatch):
    """If the LiteLLM_AgentVMConfig table doesn't exist yet (Epic G not run),
    the resolver falls back to env vars instead of crashing."""
    _set_master_key(monkeypatch)
    monkeypatch.setenv("LITELLM_AGENT_AWS_ACCESS_KEY_ID", "AKIAFROMENVFALLBACK")
    monkeypatch.setenv("LITELLM_AGENT_AWS_SECRET_ACCESS_KEY", "secret-fallback")

    from litellm.proxy.agent_session_endpoints.vm_providers.team_config import (
        get_team_vm_config,
    )

    prisma = MagicMock()
    prisma.db = MagicMock()
    prisma.db.litellm_agentvmconfig = MagicMock()
    prisma.db.litellm_agentvmconfig.find_unique = AsyncMock(
        side_effect=Exception("relation does not exist")
    )

    cfg = await get_team_vm_config("team-no-table", prisma_client=prisma)
    assert cfg.aws_creds.access_key_id == "AKIAFROMENVFALLBACK"


def _row_with_creds(blob: str, region: str = "us-west-2") -> Any:
    """Build a fake `LiteLLM_AgentVMConfig` Prisma row with attribute access."""
    row = MagicMock()
    row.aws_creds_enc = blob
    row.region = region
    row.subnet_id = "subnet-1"
    row.security_group_id = "sg-1"
    row.iam_instance_profile = "litellm-ec2-poc"
    row.instance_type = "t3.large"
    row.use_spot = True
    row.ami_id = "ami-deadbeef"
    return row


@pytest.mark.asyncio
async def test_byoc_cross_team_isolation(monkeypatch):
    """Validation #12: two teams resolve to two distinct creds objects."""
    _set_master_key(monkeypatch)
    from litellm.proxy.agent_session_endpoints.vm_providers.base import AwsCreds
    from litellm.proxy.agent_session_endpoints.vm_providers.team_config import (
        encrypt_aws_creds,
        get_team_vm_config,
    )

    creds_a = AwsCreds(
        access_key_id="AKIATEAMA000000000",
        secret_access_key="teama-secret",
        region="us-west-2",
    )
    creds_b = AwsCreds(
        access_key_id="AKIATEAMB000000000",
        secret_access_key="teamb-secret",
        region="us-east-1",
    )
    blob_a = encrypt_aws_creds(creds_a)
    blob_b = encrypt_aws_creds(creds_b)

    rows: Dict[str, Any] = {
        "team-a": _row_with_creds(blob_a, region="us-west-2"),
        "team-b": _row_with_creds(blob_b, region="us-east-1"),
    }

    async def find_unique(where: Dict[str, Any]) -> Optional[Any]:
        return rows.get(where["team_id"])

    prisma = MagicMock()
    prisma.db = MagicMock()
    prisma.db.litellm_agentvmconfig = MagicMock()
    prisma.db.litellm_agentvmconfig.find_unique = AsyncMock(side_effect=find_unique)

    cfg_a = await get_team_vm_config("team-a", prisma_client=prisma)
    cfg_b = await get_team_vm_config("team-b", prisma_client=prisma)

    assert cfg_a.aws_creds.access_key_id == "AKIATEAMA000000000"
    assert cfg_a.aws_creds.region == "us-west-2"
    assert cfg_b.aws_creds.access_key_id == "AKIATEAMB000000000"
    assert cfg_b.aws_creds.region == "us-east-1"

    # Sanity: they cannot accidentally be the same object.
    assert cfg_a.aws_creds.access_key_id != cfg_b.aws_creds.access_key_id


@pytest.mark.asyncio
async def test_db_row_with_no_creds_raises(monkeypatch):
    _set_master_key(monkeypatch)
    from litellm.proxy.agent_session_endpoints.vm_providers.base import (
        InvalidCredentialsError,
    )
    from litellm.proxy.agent_session_endpoints.vm_providers.team_config import (
        get_team_vm_config,
    )

    row = _row_with_creds("")  # empty creds blob
    row.aws_creds_enc = None
    prisma = MagicMock()
    prisma.db = MagicMock()
    prisma.db.litellm_agentvmconfig = MagicMock()
    prisma.db.litellm_agentvmconfig.find_unique = AsyncMock(return_value=row)

    with pytest.raises(InvalidCredentialsError):
        await get_team_vm_config("team-empty", prisma_client=prisma)

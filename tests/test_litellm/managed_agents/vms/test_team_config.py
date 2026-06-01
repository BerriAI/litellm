"""
Tests for `team_config.get_team_vm_config` — BYOC creds resolution against
Epic G's `LiteLLM_AgentVMConfig` schema (per-field encrypted columns).

Covers:
- Validation #11: missing creds → InvalidCredentialsError fast (no instance launched)
- Validation #12: cross-team isolation — two teams resolve to two distinct creds
- Per-field encryption round-trip (encrypt_aws_field + decrypt_aws_field)
- env-var fallback for local dev
- Falls back to env vars when the AgentVMConfig table doesn't exist yet

Reconciliation note: this file was originally written against Epic B's
single-blob `aws_creds_enc` shape. Rewritten on the integration branch to
match Epic G's per-field encrypted columns (`aws_access_key_id_enc`,
`aws_secret_access_key_enc`, `aws_region`, ...).
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest


def _set_master_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-do-not-use-in-prod-1234567890")


def test_encrypt_decrypt_aws_field_round_trip(monkeypatch):
    _set_master_key(monkeypatch)
    from litellm.managed_agents.vms.team_config import (
        decrypt_aws_field,
        encrypt_aws_field,
    )

    plain = "AKIAROUNDTRIPTESTKEY"
    blob = encrypt_aws_field(plain, "aws_access_key_id")
    assert blob is not None
    # The encrypted blob must NOT contain the plaintext.
    assert plain not in blob
    decrypted = decrypt_aws_field(blob, "aws_access_key_id")
    assert decrypted == plain


def test_encrypt_aws_field_returns_none_for_empty():
    from litellm.managed_agents.vms.team_config import (
        encrypt_aws_field,
    )

    assert encrypt_aws_field(None, "aws_access_key_id") is None
    assert encrypt_aws_field("", "aws_access_key_id") is None


def test_decrypt_aws_field_returns_none_for_empty():
    from litellm.managed_agents.vms.team_config import (
        decrypt_aws_field,
    )

    assert decrypt_aws_field(None, "aws_access_key_id") is None
    assert decrypt_aws_field("", "aws_access_key_id") is None


@pytest.mark.asyncio
async def test_no_db_no_env_raises_invalid_credentials(monkeypatch):
    """Validation #11: missing creds must fail-fast, not launch any instance."""
    _set_master_key(monkeypatch)
    monkeypatch.delenv("LITELLM_AGENT_AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("LITELLM_AGENT_AWS_SECRET_ACCESS_KEY", raising=False)

    from litellm.managed_agents.vms.base import (
        InvalidCredentialsError,
    )
    from litellm.managed_agents.vms.team_config import (
        get_team_vm_config,
    )

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

    from litellm.managed_agents.vms.team_config import (
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
    """If the LiteLLM_AgentVMConfig table doesn't exist yet, the resolver falls
    back to env vars instead of crashing."""
    _set_master_key(monkeypatch)
    monkeypatch.setenv("LITELLM_AGENT_AWS_ACCESS_KEY_ID", "AKIAFROMENVFALLBACK")
    monkeypatch.setenv("LITELLM_AGENT_AWS_SECRET_ACCESS_KEY", "secret-fallback")

    from litellm.managed_agents.vms.team_config import (
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


def _row_with_encrypted_creds(
    access_key_id_plain: str,
    secret_access_key_plain: str,
    region: str = "us-west-2",
) -> Any:
    """Build a fake `LiteLLM_AgentVMConfig` Prisma row with G's column shape."""
    from litellm.managed_agents.vms.team_config import (
        encrypt_aws_field,
    )

    row = MagicMock()
    row.provider = "ec2"
    row.aws_auth_method = "access_keys"
    row.aws_access_key_id_enc = encrypt_aws_field(
        access_key_id_plain, "aws_access_key_id"
    )
    row.aws_secret_access_key_enc = encrypt_aws_field(
        secret_access_key_plain, "aws_secret_access_key"
    )
    row.aws_role_arn_enc = None
    row.aws_region = region
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
    from litellm.managed_agents.vms.team_config import (
        get_team_vm_config,
    )

    rows: Dict[str, Any] = {
        "team-a": _row_with_encrypted_creds(
            "AKIATEAMA000000000", "teama-secret", region="us-west-2"
        ),
        "team-b": _row_with_encrypted_creds(
            "AKIATEAMB000000000", "teamb-secret", region="us-east-1"
        ),
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

    assert cfg_a.aws_creds.access_key_id != cfg_b.aws_creds.access_key_id


@pytest.mark.asyncio
async def test_db_row_with_no_creds_raises(monkeypatch):
    _set_master_key(monkeypatch)
    from litellm.managed_agents.vms.base import (
        InvalidCredentialsError,
    )
    from litellm.managed_agents.vms.team_config import (
        get_team_vm_config,
    )

    row = MagicMock()
    row.provider = "ec2"
    row.aws_auth_method = "access_keys"
    row.aws_access_key_id_enc = None
    row.aws_secret_access_key_enc = None
    row.aws_role_arn_enc = None
    row.aws_region = "us-west-2"
    row.subnet_id = None
    row.security_group_id = None
    row.iam_instance_profile = None
    row.instance_type = None
    row.use_spot = True
    row.ami_id = None

    prisma = MagicMock()
    prisma.db = MagicMock()
    prisma.db.litellm_agentvmconfig = MagicMock()
    prisma.db.litellm_agentvmconfig.find_unique = AsyncMock(return_value=row)

    with pytest.raises(InvalidCredentialsError):
        await get_team_vm_config("team-empty", prisma_client=prisma)


@pytest.mark.asyncio
async def test_iam_role_method_returns_none_creds_falls_back_env(monkeypatch):
    """When `aws_auth_method == 'iam_role'`, the resolver does NOT pull keys
    from the DB — it leaves cred resolution to boto3's default chain. With no
    env vars set this raises InvalidCredentialsError."""
    _set_master_key(monkeypatch)
    monkeypatch.delenv("LITELLM_AGENT_AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("LITELLM_AGENT_AWS_SECRET_ACCESS_KEY", raising=False)

    from litellm.managed_agents.vms.base import (
        InvalidCredentialsError,
    )
    from litellm.managed_agents.vms.team_config import (
        get_team_vm_config,
    )

    row = MagicMock()
    row.provider = "ec2"
    row.aws_auth_method = "iam_role"
    row.aws_access_key_id_enc = "anything"
    row.aws_secret_access_key_enc = "anything"
    row.aws_role_arn_enc = "anything"
    row.aws_region = "us-west-2"
    row.subnet_id = None
    row.security_group_id = None
    row.iam_instance_profile = None
    row.instance_type = None
    row.use_spot = True
    row.ami_id = None

    prisma = MagicMock()
    prisma.db = MagicMock()
    prisma.db.litellm_agentvmconfig = MagicMock()
    prisma.db.litellm_agentvmconfig.find_unique = AsyncMock(return_value=row)

    with pytest.raises(InvalidCredentialsError):
        await get_team_vm_config("team-iam-role", prisma_client=prisma)

"""
Team-scoped BYOC AWS config + creds resolver.

Reads `LiteLLM_AgentVMConfig` (table owned by Epic G / LIT-2891) and decrypts
the per-field-encrypted AWS credentials using the proxy's salt key. Falls back
to env vars `LITELLM_AGENT_AWS_ACCESS_KEY_ID` / `..._SECRET_ACCESS_KEY` /
`..._SESSION_TOKEN` so local dev works without a DB row.

Epic G's schema stores creds as **separate encrypted columns** (not a single
JSON blob). Columns:

    provider                  -- "ec2" | "self_hosted" | "disabled"
    aws_auth_method           -- "access_keys" | "iam_role" | "instance_metadata"
    aws_access_key_id_enc     -- encrypt_value_helper(...)
    aws_secret_access_key_enc -- encrypt_value_helper(...)
    aws_role_arn_enc          -- encrypt_value_helper(...)  (cross-account mode)
    aws_region                -- plain
    ami_id, instance_type, subnet_id, security_group_id,
    iam_instance_profile, use_spot, ...

Reconciliation note: this module was originally written against Epic B's
single-column ``aws_creds_enc`` shape. The integration branch adopted Epic G's
column-per-field shape; this resolver was rewritten to match.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from litellm._logging import verbose_proxy_logger
from litellm.managed_agents.vms.base import (
    AwsCreds,
    Ec2Config,
    InvalidCredentialsError,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)


@dataclass
class TeamVMConfig:
    """Resolved team-scoped VM config (creds + EC2 overrides)."""

    aws_creds: AwsCreds
    ec2_config: Ec2Config


def encrypt_aws_field(value: Optional[str], field_name: str) -> Optional[str]:
    """Encrypt a single AWS credential field. Empty/None passes through."""
    if not value:
        return None
    return encrypt_value_helper(value)


def decrypt_aws_field(blob: Optional[str], field_name: str) -> Optional[str]:
    """Decrypt a single AWS credential field. Empty/None passes through.

    Returns ``None`` (not raises) if the blob is malformed — the caller is
    expected to validate that the required fields are present.
    """
    if not blob:
        return None
    try:
        return decrypt_value_helper(blob, key=field_name)
    except Exception as e:
        verbose_proxy_logger.warning(
            f"team_config: decrypt failed for {field_name}: {type(e).__name__}"
        )
        return None


def _ec2_config_from_row(row: Any, default_region: str) -> Ec2Config:
    """Build an `Ec2Config` from a `LiteLLM_AgentVMConfig` Prisma row.

    Reads G's column names: ``aws_region`` (not ``region``).
    """
    return Ec2Config(
        region=getattr(row, "aws_region", None) or default_region,
        subnet_id=getattr(row, "subnet_id", None),
        security_group_id=getattr(row, "security_group_id", None),
        iam_instance_profile=getattr(row, "iam_instance_profile", None),
        instance_type=getattr(row, "instance_type", None) or "t3.large",
        use_spot=bool(getattr(row, "use_spot", True)),
        ami_id=getattr(row, "ami_id", None),
    )


def _creds_from_env(default_region: str) -> Optional[AwsCreds]:
    """Build creds from `LITELLM_AGENT_AWS_*` env vars; None if unset."""
    access_key_id = os.getenv("LITELLM_AGENT_AWS_ACCESS_KEY_ID")
    secret_access_key = os.getenv("LITELLM_AGENT_AWS_SECRET_ACCESS_KEY")
    if not access_key_id or not secret_access_key:
        return None
    return AwsCreds(
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        session_token=os.getenv("LITELLM_AGENT_AWS_SESSION_TOKEN"),
        region=os.getenv("LITELLM_AGENT_AWS_REGION", default_region),
    )


def _creds_from_row(row: Any, default_region: str) -> Optional[AwsCreds]:
    """Decrypt G's per-field encrypted creds. Returns None if not configured.

    Only handles ``aws_auth_method == 'access_keys'``. Other modes
    (``iam_role`` / ``instance_metadata``) defer credential lookup to the
    boto3 default chain at provision time and don't surface raw keys here.
    """
    auth_method = getattr(row, "aws_auth_method", None)
    if auth_method and auth_method != "access_keys":
        # IAM role / instance metadata — caller should construct AwsCreds via
        # boto3's default chain instead of pulling from the DB.
        return None

    access_key_id = decrypt_aws_field(
        getattr(row, "aws_access_key_id_enc", None), "aws_access_key_id"
    )
    secret_access_key = decrypt_aws_field(
        getattr(row, "aws_secret_access_key_enc", None), "aws_secret_access_key"
    )
    if not access_key_id or not secret_access_key:
        return None

    return AwsCreds(
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        session_token=None,
        region=getattr(row, "aws_region", None) or default_region,
    )


async def get_team_vm_config(
    team_id: str,
    prisma_client: Any,
    default_region: str = "us-west-2",
) -> TeamVMConfig:
    """
    Resolve the team's BYOC AWS creds + EC2 overrides.

    Lookup order:
    1. `LiteLLM_AgentVMConfig` row keyed by `team_id` (G's per-field columns)
    2. `LITELLM_AGENT_AWS_*` env vars (local dev fallback)

    Raises `InvalidCredentialsError` if neither path yields creds.
    """
    row = None
    if prisma_client is not None:
        try:
            row = await prisma_client.db.litellm_agentvmconfig.find_unique(
                where={"team_id": team_id}
            )
        except Exception as e:
            # If the table doesn't exist yet (G1 hasn't run the migration),
            # silently fall through to env-var path so local dev is unblocked.
            verbose_proxy_logger.debug(
                f"LiteLLM_AgentVMConfig lookup failed for team={team_id}: "
                f"{type(e).__name__}; falling back to env vars."
            )

    if row is not None:
        ec2_config = _ec2_config_from_row(row, default_region=default_region)
        creds = _creds_from_row(row, default_region=default_region)
        if creds is None:
            raise InvalidCredentialsError(
                f"Team {team_id} has no usable AWS credentials configured. "
                "Add them under Settings → Cloud Agents."
            )
        # Override creds region with row.aws_region if explicitly set.
        if ec2_config.region:
            creds.region = ec2_config.region
        return TeamVMConfig(aws_creds=creds, ec2_config=ec2_config)

    env_creds = _creds_from_env(default_region=default_region)
    if env_creds is None:
        raise InvalidCredentialsError(
            f"Team {team_id} has no AWS credentials configured "
            "(no DB row, no LITELLM_AGENT_AWS_* env vars). "
            "Add them under Settings → Cloud Agents."
        )
    return TeamVMConfig(
        aws_creds=env_creds,
        ec2_config=Ec2Config(region=env_creds.region),
    )

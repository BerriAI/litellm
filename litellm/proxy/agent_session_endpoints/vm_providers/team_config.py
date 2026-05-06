"""
Team-scoped BYOC AWS config + creds resolver.

Reads `LiteLLM_AgentVMConfig` (table created by Epic G) and decrypts the
encrypted creds blob using the proxy's salt key. Falls back to env vars
`LITELLM_AGENT_AWS_ACCESS_KEY_ID` / `..._SECRET_ACCESS_KEY` /
`..._SESSION_TOKEN` so local dev works without a DB row.

The encrypted JSON shape stored in `aws_creds_enc`:

    {
      "access_key_id": "AKIA...",
      "secret_access_key": "...",
      "session_token": null
    }

Each field is `encrypt_value_helper`-encrypted individually so we never expose
the secret in DB dumps even when reading other columns.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy.agent_session_endpoints.vm_providers.base import (
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


def encrypt_aws_creds(creds: AwsCreds) -> str:
    """
    Encrypt an `AwsCreds` for storage in `LiteLLM_AgentVMConfig.aws_creds_enc`.

    Returns a JSON string with each field individually encrypted (so even a
    partial DB leak doesn't expose the secret).
    """
    payload = {
        "access_key_id": encrypt_value_helper(creds.access_key_id),
        "secret_access_key": encrypt_value_helper(creds.secret_access_key),
        "session_token": (
            encrypt_value_helper(creds.session_token) if creds.session_token else None
        ),
        "region": creds.region,
    }
    return json.dumps(payload)


def decrypt_aws_creds(blob: str) -> AwsCreds:
    """
    Decrypt the `aws_creds_enc` JSON blob into an `AwsCreds`.

    Raises `InvalidCredentialsError` if the blob is malformed or any required
    field fails to decrypt.
    """
    try:
        payload: Dict[str, Any] = json.loads(blob)
    except (ValueError, TypeError) as e:
        raise InvalidCredentialsError(
            f"aws_creds_enc is not valid JSON: {type(e).__name__}"
        )

    access_key_id = decrypt_value_helper(
        payload.get("access_key_id", ""), key="aws_access_key_id"
    )
    secret_access_key = decrypt_value_helper(
        payload.get("secret_access_key", ""), key="aws_secret_access_key"
    )
    session_token_enc = payload.get("session_token")
    session_token = (
        decrypt_value_helper(session_token_enc, key="aws_session_token")
        if session_token_enc
        else None
    )

    if not access_key_id or not secret_access_key:
        raise InvalidCredentialsError(
            "Decrypted AWS creds are empty — is LITELLM_SALT_KEY set correctly?"
        )

    return AwsCreds(
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        session_token=session_token,
        region=payload.get("region", "us-west-2"),
    )


def _ec2_config_from_row(row: Any, default_region: str) -> Ec2Config:
    """Build an `Ec2Config` from a `LiteLLM_AgentVMConfig` Prisma row."""
    return Ec2Config(
        region=getattr(row, "region", None) or default_region,
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


async def get_team_vm_config(
    team_id: str,
    prisma_client: Any,
    default_region: str = "us-west-2",
) -> TeamVMConfig:
    """
    Resolve the team's BYOC AWS creds + EC2 overrides.

    Lookup order:
    1. `LiteLLM_AgentVMConfig` row keyed by `team_id` (decrypt creds blob)
    2. `LITELLM_AGENT_AWS_*` env vars (local dev fallback)

    Raises `InvalidCredentialsError` if neither path yields creds. The DB
    table is owned by Epic G; this resolver also works before that table is
    populated by falling back to env vars.
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
        creds_enc = getattr(row, "aws_creds_enc", None)
        if not creds_enc:
            raise InvalidCredentialsError(
                f"Team {team_id} has no AWS credentials configured. "
                "Add them under Settings → Agent VM."
            )
        creds = decrypt_aws_creds(creds_enc)
        # Override creds region with row.region if explicitly set.
        if ec2_config.region:
            creds.region = ec2_config.region
        return TeamVMConfig(aws_creds=creds, ec2_config=ec2_config)

    env_creds = _creds_from_env(default_region=default_region)
    if env_creds is None:
        raise InvalidCredentialsError(
            f"Team {team_id} has no AWS credentials configured "
            "(no DB row, no LITELLM_AGENT_AWS_* env vars). "
            "Add them under Settings → Agent VM."
        )
    return TeamVMConfig(
        aws_creds=env_creds,
        ec2_config=Ec2Config(region=env_creds.region),
    )

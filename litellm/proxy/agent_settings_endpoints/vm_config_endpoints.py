"""
`/v2/agent-vm-config` endpoints (LIT-2891 / Screen 1, 2, 4).

Backs the Settings -> Cloud Agents -> Provider, Warm Pool, and Network Access
screens. The whole config is one row per team in `LiteLLM_AgentVMConfig`. The
GET response NEVER includes raw AWS creds — fields are returned as
`REDACTED_VALUE` if set, `None` if unset.

Test Connection currently MOCKS `sts:GetCallerIdentity` until B0 (LIT-2888)
closes — that ticket installs the real boto3 path. The mock is gated on the
`LITELLM_CLOUD_AGENT_MOCK_AWS` env var so we can flip to real once B0 lands
without changing this code.
"""

import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.agent_settings_endpoints.encryption import (
    decrypt_optional,
    encrypt_optional,
)
from litellm.proxy.agent_settings_endpoints.types import (
    REDACTED_VALUE,
    AgentVMConfigResponse,
    AgentVMConfigUpdateRequest,
    NetworkAccessConfig,
    TestConnectionResponse,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


_DEFAULT_NETWORK_ACCESS: Dict[str, Any] = {"mode": "allow_all", "allowlist": []}


def _resolve_team_id(user_api_key_dict: UserAPIKeyAuth) -> str:
    """Pick the team to scope this request to. Raise 400 if we can't.

    The settings UI is always opened in the context of a team (the user picks
    one from the team-switcher in the header). The dashboard sends the team
    ID via the auth context. If neither team_id nor team_alias is set we
    refuse — there is no sensible "default" config to return.
    """
    team_id = user_api_key_dict.team_id or (user_api_key_dict.metadata or {}).get(
        "team_id"
    )
    if not team_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Cloud Agent settings are scoped to a team. Pick a team from the "
                "header switcher and try again."
            ),
        )
    return team_id


def _redact(value: Optional[str]) -> Optional[str]:
    """Return REDACTED_VALUE if a non-empty ciphertext exists, else None."""
    return REDACTED_VALUE if value else None


def _row_to_response(row: Dict[str, Any]) -> AgentVMConfigResponse:
    """Map a Prisma row dict to the public response shape, redacting secrets."""
    network_access_raw = row.get("network_access") or _DEFAULT_NETWORK_ACCESS
    if isinstance(network_access_raw, str):
        # Prisma sometimes returns Json fields as already-parsed dicts but
        # under SQLite (test) it can hand back the raw string — be tolerant.
        import json as _json

        try:
            network_access_raw = _json.loads(network_access_raw)
        except Exception:
            network_access_raw = _DEFAULT_NETWORK_ACCESS
    network_access = NetworkAccessConfig(**network_access_raw)

    return AgentVMConfigResponse(
        team_id=row["team_id"],
        provider=row.get("provider") or "disabled",
        aws_auth_method=row.get("aws_auth_method"),
        aws_access_key_id=_redact(row.get("aws_access_key_id_enc")),
        aws_secret_access_key=_redact(row.get("aws_secret_access_key_enc")),
        aws_role_arn=_redact(row.get("aws_role_arn_enc")),
        aws_region=row.get("aws_region"),
        ami_id=row.get("ami_id"),
        instance_type=row.get("instance_type"),
        subnet_id=row.get("subnet_id"),
        security_group_id=row.get("security_group_id"),
        iam_instance_profile=row.get("iam_instance_profile"),
        use_spot=bool(row.get("use_spot", True)),
        max_session_minutes=int(row.get("max_session_minutes") or 120),
        warm_pool_enabled=bool(row.get("warm_pool_enabled", False)),
        warm_pool_size=int(row.get("warm_pool_size") or 0),
        max_idle_minutes=int(row.get("max_idle_minutes") or 30),
        hydrate_transport=row.get("hydrate_transport") or "auto",
        network_access=network_access,
        self_hosted_enabled=bool(row.get("self_hosted_enabled", False)),
    )


def _empty_row(team_id: str) -> Dict[str, Any]:
    """Synthesize a default row for teams that have never saved settings."""
    return {
        "team_id": team_id,
        "provider": "disabled",
        "aws_auth_method": None,
        "aws_access_key_id_enc": None,
        "aws_secret_access_key_enc": None,
        "aws_role_arn_enc": None,
        "aws_region": None,
        "ami_id": None,
        "instance_type": None,
        "subnet_id": None,
        "security_group_id": None,
        "iam_instance_profile": None,
        "use_spot": True,
        "max_session_minutes": 120,
        "warm_pool_enabled": False,
        "warm_pool_size": 0,
        "max_idle_minutes": 30,
        "hydrate_transport": "auto",
        "network_access": _DEFAULT_NETWORK_ACCESS,
        "self_hosted_enabled": False,
    }


def _build_update_payload(
    body: AgentVMConfigUpdateRequest,
) -> Dict[str, Any]:
    """Translate the request model into a dict suitable for Prisma upsert.

    AWS fields are encrypted in-place; sentinel `REDACTED_VALUE` from the UI
    means "leave the existing value alone" and is filtered out before write.
    Network access goes in as raw JSON (Prisma handles the encoding).
    """
    payload: Dict[str, Any] = {}

    plain_fields = (
        "provider",
        "aws_auth_method",
        "aws_region",
        "ami_id",
        "instance_type",
        "subnet_id",
        "security_group_id",
        "iam_instance_profile",
        "use_spot",
        "max_session_minutes",
        "warm_pool_enabled",
        "warm_pool_size",
        "max_idle_minutes",
        "hydrate_transport",
        "self_hosted_enabled",
    )
    for field in plain_fields:
        value = getattr(body, field, None)
        # Use `is not None` (not truthiness): `False`, `0`, and `""` are
        # all valid values that callers may legitimately want to write
        # (e.g. disabling warm pool with `warm_pool_enabled=False`,
        # zeroing `warm_pool_size`). A future contributor adding a field
        # here should keep this guard so those updates don't get dropped.
        if value is not None:
            payload[field] = value

    encrypted_fields = (
        ("aws_access_key_id", "aws_access_key_id_enc"),
        ("aws_secret_access_key", "aws_secret_access_key_enc"),
        ("aws_role_arn", "aws_role_arn_enc"),
    )
    for plain_field, db_field in encrypted_fields:
        value = getattr(body, plain_field, None)
        if value is None:
            # Field omitted entirely — leave existing value.
            continue
        if value == REDACTED_VALUE:
            # UI round-trip: don't overwrite with the redacted sentinel.
            continue
        payload[db_field] = encrypt_optional(value)

    if body.network_access is not None:
        payload["network_access"] = body.network_access.model_dump()

    return payload


async def _get_or_create_row(prisma_client: Any, team_id: str) -> Dict[str, Any]:
    """Fetch the row, returning a default if none exists."""
    row = await prisma_client.db.litellm_agentvmconfig.find_unique(
        where={"team_id": team_id}
    )
    if row is None:
        return _empty_row(team_id)
    return dict(row) if not isinstance(row, dict) else row


@router.get(
    "/v2/agent-vm-config",
    dependencies=[Depends(user_api_key_auth)],
    response_model=AgentVMConfigResponse,
    tags=["cloud agents"],
)
async def get_agent_vm_config(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AgentVMConfigResponse:
    """Return the team's VM config with AWS creds redacted."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    team_id = _resolve_team_id(user_api_key_dict)
    row = await _get_or_create_row(prisma_client, team_id)
    return _row_to_response(row)


@router.put(
    "/v2/agent-vm-config",
    dependencies=[Depends(user_api_key_auth)],
    response_model=AgentVMConfigResponse,
    tags=["cloud agents"],
)
async def update_agent_vm_config(
    request: Request,
    body: AgentVMConfigUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AgentVMConfigResponse:
    """Upsert the team's VM config. AWS creds are encrypted before write."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    team_id = _resolve_team_id(user_api_key_dict)
    payload = _build_update_payload(body)

    create_payload = {**_empty_row(team_id), **payload}
    update_payload = payload

    await prisma_client.db.litellm_agentvmconfig.upsert(
        where={"team_id": team_id},
        data={
            "create": create_payload,
            "update": update_payload,
        },
    )

    refreshed = await _get_or_create_row(prisma_client, team_id)
    return _row_to_response(refreshed)


async def _resolve_aws_creds(
    prisma_client: Any, team_id: str
) -> Dict[str, Optional[str]]:
    """Decrypt the team's stored AWS creds. Used by Test Connection + by B2.

    Exposed as a helper so the hydrate path (B2) can reuse it. Returns a dict
    with `access_key_id`, `secret_access_key`, `role_arn`, `region` — any of
    which may be None.
    """
    row = await prisma_client.db.litellm_agentvmconfig.find_unique(
        where={"team_id": team_id}
    )
    if row is None:
        return {
            "access_key_id": None,
            "secret_access_key": None,
            "role_arn": None,
            "region": None,
        }
    row_dict = dict(row) if not isinstance(row, dict) else row
    return {
        "access_key_id": decrypt_optional(
            row_dict.get("aws_access_key_id_enc"),
            key="aws_access_key_id",
        ),
        "secret_access_key": decrypt_optional(
            row_dict.get("aws_secret_access_key_enc"),
            key="aws_secret_access_key",
        ),
        "role_arn": decrypt_optional(
            row_dict.get("aws_role_arn_enc"),
            key="aws_role_arn",
        ),
        "region": row_dict.get("aws_region"),
    }


def _mock_caller_identity(creds: Dict[str, Optional[str]]) -> TestConnectionResponse:
    """Stand-in for boto3 sts:GetCallerIdentity until B0 ships the real call.

    Returns a deterministic ok/err response shaped like the real STS reply so
    the UI can be wired without depending on B0. The mock fails if no access
    key is configured — that mirrors the real failure mode and makes the
    "no creds yet" UX testable end-to-end.
    """
    access_key = creds.get("access_key_id")
    if not access_key:
        return TestConnectionResponse(
            ok=False,
            error=(
                "No AWS credentials configured for this team. Add an Access Key "
                "or IAM Role under Provider Settings and try again."
            ),
        )
    # The mock account ID is derived from the access key fingerprint so each
    # team gets a stable-but-distinct value during development.
    suffix = "".join(c for c in access_key if c.isdigit())[-12:].rjust(12, "0")
    region = creds.get("region") or "us-west-2"
    return TestConnectionResponse(
        ok=True,
        account_id=suffix,
        arn=f"arn:aws:iam::{suffix}:user/litellm-cloud-agents",
        region=region,
    )


def _aws_mock_enabled() -> bool:
    """Whether `test-connection` should return a synthetic mock response
    instead of calling the real `sts:GetCallerIdentity`.

    Defaults to OFF — a fresh production proxy must always validate AWS
    credentials against STS, never silently return success for invalid
    creds. Set `LITELLM_CLOUD_AGENT_MOCK_AWS=1` explicitly to opt into the
    mock path during local development / tests.
    """
    return os.getenv("LITELLM_CLOUD_AGENT_MOCK_AWS", "0") == "1"


@router.post(
    "/v2/agent-vm-config/test-connection",
    dependencies=[Depends(user_api_key_auth)],
    response_model=TestConnectionResponse,
    tags=["cloud agents"],
)
async def test_aws_connection(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> TestConnectionResponse:
    """Validate the team's stored AWS creds against `sts:GetCallerIdentity`.

    Phase 1 (current): mocked behind `LITELLM_CLOUD_AGENT_MOCK_AWS=1`.
    Phase 2 (post-B0): real boto3 client; same response shape.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    team_id = _resolve_team_id(user_api_key_dict)
    creds = await _resolve_aws_creds(prisma_client, team_id)

    if _aws_mock_enabled():
        return _mock_caller_identity(creds)

    # Real STS path — B0 will land this. Imported lazily to avoid pulling in
    # boto3 at module-load time for proxies that don't use cloud agents.
    try:
        import boto3  # type: ignore
        from botocore.exceptions import ClientError  # type: ignore
    except ImportError:
        return TestConnectionResponse(
            ok=False,
            error="boto3 not installed on this proxy — install litellm[proxy] extras.",
        )

    if not creds.get("access_key_id"):
        return TestConnectionResponse(
            ok=False,
            error="No AWS credentials configured for this team.",
        )

    try:
        client = boto3.client(
            "sts",
            aws_access_key_id=creds["access_key_id"],
            aws_secret_access_key=creds["secret_access_key"],
            region_name=creds.get("region") or "us-west-2",
        )
        identity = client.get_caller_identity()
    except ClientError as exc:  # pragma: no cover — exercised once B0 lands
        verbose_proxy_logger.warning(
            "agent-vm-config test-connection failed for team=%s: %s",
            team_id,
            exc,
        )
        return TestConnectionResponse(ok=False, error=str(exc))

    return TestConnectionResponse(
        ok=True,
        account_id=identity.get("Account"),
        arn=identity.get("Arn"),
        region=creds.get("region") or "us-west-2",
    )

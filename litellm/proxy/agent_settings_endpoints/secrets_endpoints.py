"""
`/v2/agent-secrets` endpoints (LIT-2891 / Screen 5).

Per-team encrypted secrets for cloud agent VMs. Two security invariants
worth calling out, since the rest of this file is built around them:

* **Write-only values.** `value` is accepted on POST/PUT and stored encrypted,
  but it is NEVER decrypted onto a GET response. The response schema
  (`AgentSecretResponse`) has no `value` field, so even an accidental
  `model_dump()` of the row can't leak the plaintext.
* **Per-team isolation.** Every query filters on `team_id` resolved from the
  caller's API key. Cross-team reads are not possible at this layer — the
  composite unique key `(team_id, name)` makes that explicit.

The session-create / hydrate path (B2) calls the lower-level
`partition_secrets_for_session` helper from `scope_filter.py` to figure out
which secrets to push into a given session. This module only handles the UI
CRUD surface.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.agent_settings_endpoints.encryption import encrypt_optional
from litellm.proxy.agent_settings_endpoints.types import (
    AgentSecretCreateRequest,
    AgentSecretListResponse,
    AgentSecretResponse,
    AgentSecretUpdateRequest,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


def _resolve_team_id(user_api_key_dict: UserAPIKeyAuth) -> str:
    """Pick the team to scope this request to. Raise 400 if missing.

    Same contract as the VM config endpoints — secrets are per-team and we
    refuse to silently fall back to a "default" scope.
    """
    team_id = user_api_key_dict.team_id or (user_api_key_dict.metadata or {}).get(
        "team_id"
    )
    if not team_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Cloud Agent secrets are scoped to a team. Pick a team from the "
                "header switcher and try again."
            ),
        )
    return team_id


def _row_to_response(row: Dict[str, Any]) -> AgentSecretResponse:
    """Map a Prisma row to the public response shape.

    Note: `value_enc` is intentionally NOT read here. The response model
    has no value field, so even if a future caller passed `**row` we
    couldn't accidentally surface the ciphertext.
    """
    scope = row.get("scope")
    if scope is None:
        scope = "all"
    if isinstance(scope, str) and scope not in ("all",):
        # SQLite stores Json columns as raw strings — tolerate both shapes.
        import json as _json

        try:
            parsed = _json.loads(scope)
        except Exception:
            parsed = "all"
        scope = parsed if parsed in ("all",) or isinstance(parsed, list) else "all"

    created_at = row.get("created_at")
    updated_at = row.get("updated_at")
    return AgentSecretResponse(
        name=row["name"],
        scope=scope,
        type=row.get("type") or "env",
        file_path=row.get("file_path"),
        created_at=str(created_at) if created_at is not None else "",
        updated_at=str(updated_at) if updated_at is not None else "",
    )


def _validate_secret_payload(
    *,
    type_: Optional[str],
    file_path: Optional[str],
    is_create: bool,
) -> None:
    """Reject `type=file` without a `file_path`. Mirrors the UI form validation
    so the rule lives in exactly one place server-side too."""
    if type_ == "file" and not file_path and is_create:
        raise HTTPException(
            status_code=400,
            detail="`file_path` is required when `type` is `file`.",
        )


def _build_create_payload(
    *,
    team_id: str,
    body: AgentSecretCreateRequest,
    created_by: Optional[str],
) -> Dict[str, Any]:
    """Build the dict for prisma.create — encrypts `value`, never stores raw."""
    encrypted = encrypt_optional(body.value)
    if encrypted is None:
        # Pydantic enforces min_length=1 already, so this is just a belt-and-
        # suspenders guard against future signature drift.
        raise HTTPException(status_code=400, detail="Secret `value` cannot be empty.")
    return {
        "team_id": team_id,
        "name": body.name,
        "value_enc": encrypted,
        "scope": body.scope,
        "type": body.type,
        "file_path": body.file_path,
        "created_by": created_by,
    }


def _build_update_payload(body: AgentSecretUpdateRequest) -> Dict[str, Any]:
    """Build the dict for prisma.update. Omits unset fields so partial PATCH-
    like updates from the UI don't clobber existing scope/type."""
    payload: Dict[str, Any] = {}
    if body.value is not None:
        encrypted = encrypt_optional(body.value)
        if encrypted is not None:
            payload["value_enc"] = encrypted
    if body.scope is not None:
        payload["scope"] = body.scope
    if body.type is not None:
        payload["type"] = body.type
    if body.file_path is not None:
        payload["file_path"] = body.file_path
    return payload


@router.get(
    "/v2/agent-secrets",
    dependencies=[Depends(user_api_key_auth)],
    response_model=AgentSecretListResponse,
    tags=["cloud agents"],
)
async def list_agent_secrets(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AgentSecretListResponse:
    """List secrets for the team. Returns metadata only — no values."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    team_id = _resolve_team_id(user_api_key_dict)
    rows = await prisma_client.db.litellm_agentsecret.find_many(
        where={"team_id": team_id},
        order={"name": "asc"},
    )
    secrets: List[AgentSecretResponse] = [
        _row_to_response(dict(r) if not isinstance(r, dict) else r) for r in rows
    ]
    return AgentSecretListResponse(secrets=secrets)


@router.post(
    "/v2/agent-secrets",
    dependencies=[Depends(user_api_key_auth)],
    response_model=AgentSecretResponse,
    tags=["cloud agents"],
)
async def create_agent_secret(
    request: Request,
    body: AgentSecretCreateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AgentSecretResponse:
    """Create a new secret. Returns metadata only."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    team_id = _resolve_team_id(user_api_key_dict)
    _validate_secret_payload(type_=body.type, file_path=body.file_path, is_create=True)

    # Conflict check — composite unique (team_id, name).
    existing = await prisma_client.db.litellm_agentsecret.find_unique(
        where={"team_id_name": {"team_id": team_id, "name": body.name}}
    )
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Secret `{body.name}` already exists for this team. Use PUT to update it.",
        )

    payload = _build_create_payload(
        team_id=team_id,
        body=body,
        created_by=user_api_key_dict.user_id,
    )

    try:
        created = await prisma_client.db.litellm_agentsecret.create(data=payload)
    except Exception as exc:
        verbose_proxy_logger.exception(
            "Failed to create agent secret name=%s team=%s: %s",
            body.name,
            team_id,
            exc,
        )
        raise HTTPException(status_code=500, detail="Failed to create secret.")

    row = dict(created) if not isinstance(created, dict) else created
    return _row_to_response(row)


@router.put(
    "/v2/agent-secrets/{name}",
    dependencies=[Depends(user_api_key_auth)],
    response_model=AgentSecretResponse,
    tags=["cloud agents"],
)
async def update_agent_secret(
    request: Request,
    name: str,
    body: AgentSecretUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AgentSecretResponse:
    """Update an existing secret by name."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    team_id = _resolve_team_id(user_api_key_dict)
    _validate_secret_payload(type_=body.type, file_path=body.file_path, is_create=False)

    payload = _build_update_payload(body)
    if not payload:
        raise HTTPException(
            status_code=400,
            detail="At least one field must be provided to update.",
        )

    existing = await prisma_client.db.litellm_agentsecret.find_unique(
        where={"team_id_name": {"team_id": team_id, "name": name}}
    )
    if existing is None:
        raise HTTPException(
            status_code=404, detail=f"Secret `{name}` not found for this team."
        )

    updated = await prisma_client.db.litellm_agentsecret.update(
        where={"team_id_name": {"team_id": team_id, "name": name}},
        data=payload,
    )
    row = dict(updated) if not isinstance(updated, dict) else updated
    return _row_to_response(row)


@router.delete(
    "/v2/agent-secrets/{name}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["cloud agents"],
)
async def delete_agent_secret(
    request: Request,
    name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    """Delete a secret by name. Idempotent — returns 404 if not present."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    team_id = _resolve_team_id(user_api_key_dict)

    existing = await prisma_client.db.litellm_agentsecret.find_unique(
        where={"team_id_name": {"team_id": team_id, "name": name}}
    )
    if existing is None:
        raise HTTPException(
            status_code=404, detail=f"Secret `{name}` not found for this team."
        )

    await prisma_client.db.litellm_agentsecret.delete(
        where={"team_id_name": {"team_id": team_id, "name": name}}
    )
    return {"deleted": True, "name": name}

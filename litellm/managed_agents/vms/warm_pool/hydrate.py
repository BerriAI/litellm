"""Hydrate payload builder (LIT-2890).

Combines per-session inputs (agent config, repos, env vars) with team-level
state (encrypted secrets from `LiteLLM_AgentSecret`, network policy from
`LiteLLM_AgentVMConfig.network_access`) into a single ``HydratePayload`` ready
to push at the warm VM.

Why a separate module: keeps secret-decryption logic out of the hot-path
endpoint and out of the transport. The same builder is used by both the SSM
transport and any future long-poll transport — they only differ in how the
payload reaches the daemon, not what's in it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.managed_agents.vms.warm_pool.types import (
    AgentConfig,
    HydratePayload,
    NetworkAccess,
    RepoSpec,
)
from litellm.proxy.agent_settings_endpoints.encryption import decrypt_optional
from litellm.proxy.agent_settings_endpoints.scope_filter import (
    partition_secrets_for_session,
    secret_in_scope,
)


_DEFAULT_NETWORK_ACCESS: Dict[str, Any] = {"mode": "allow_all"}


def _coerce_repos(raw: Any) -> List[RepoSpec]:
    """Normalize whatever the session row holds into ``List[RepoSpec]``."""
    if not isinstance(raw, list):
        return []
    out: List[RepoSpec] = []
    for entry in raw:
        if isinstance(entry, dict):
            url = entry.get("url")
            if not isinstance(url, str) or not url:
                continue
            ref = entry.get("ref") if isinstance(entry.get("ref"), str) else None
            out.append(RepoSpec(url=url, ref=ref))
        elif isinstance(entry, RepoSpec):
            out.append(entry)
    return out


def _coerce_env_vars(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if v is not None}


def _coerce_network_access(raw: Any) -> NetworkAccess:
    """Read `LiteLLM_AgentVMConfig.network_access` JSON into the typed model.

    Defaults to ``allow_all`` when the column is missing/malformed — the
    UI exposes a stricter default but we never want to crash the hydrate
    builder on a typo.
    """
    if not isinstance(raw, dict):
        raw = _DEFAULT_NETWORK_ACCESS
    mode = (
        raw.get("mode")
        if raw.get("mode") in ("allow_all", "allowlist")
        else "allow_all"
    )
    allowlist = raw.get("allowlist")
    if mode == "allowlist" and isinstance(allowlist, list):
        cleaned = [str(x) for x in allowlist if isinstance(x, str) and x]
        return NetworkAccess(mode="allowlist", allowlist=cleaned)
    return NetworkAccess(mode="allow_all")


async def _decrypt_in_scope_secrets(
    prisma_client: Any,
    team_id: str,
    session_repos: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Fetch all `LiteLLM_AgentSecret` rows for the team, scope-filter, decrypt.

    Returns ``{name: plaintext}`` for every secret in scope. Out-of-scope
    secret names are logged at info level (without values) so operators can
    audit which were applied to a session.
    """
    rows = await prisma_client.db.litellm_agentsecret.find_many(
        where={"team_id": team_id}
    )
    if not rows:
        return {}

    name_scope_pairs = [(r.name, r.scope) for r in rows]
    in_scope_names, out_of_scope_names = partition_secrets_for_session(
        name_scope_pairs, session_repos
    )
    if out_of_scope_names:
        verbose_proxy_logger.info(
            "warm_pool.hydrate: skipped out-of-scope secrets team_id=%s names=%s",
            team_id,
            sorted(out_of_scope_names),
        )

    out: Dict[str, str] = {}
    for row in rows:
        if not secret_in_scope(row.scope, session_repos):
            continue
        try:
            plaintext = decrypt_optional(row.value_enc, key=f"agent_secret:{row.name}")
        except Exception as exc:
            verbose_proxy_logger.error(
                "warm_pool.hydrate: failed to decrypt secret name=%s team=%s: %s",
                row.name,
                team_id,
                type(exc).__name__,
            )
            continue
        if plaintext is None:
            continue
        out[row.name] = plaintext
    return out


async def _load_vm_config_network_access(
    prisma_client: Any, team_id: str
) -> NetworkAccess:
    """Read network policy from ``LiteLLM_AgentVMConfig`` if present."""
    if not team_id:
        return NetworkAccess(mode="allow_all")
    row = await prisma_client.db.litellm_agentvmconfig.find_unique(
        where={"team_id": team_id}
    )
    if row is None:
        return NetworkAccess(mode="allow_all")
    return _coerce_network_access(row.network_access)


def _agent_config_from_row(agent_row: Any, model: Optional[str]) -> AgentConfig:
    """Build the daemon's runtime ``agent_config`` block."""
    return AgentConfig(
        model=model
        or (getattr(agent_row, "model", None) if agent_row is not None else None)
        or "gpt-4o-mini",
        system_prompt=(
            getattr(agent_row, "system_prompt", None) if agent_row is not None else None
        )
        or "",
        auto_create_pr=False,
    )


async def build_hydrate_payload(
    *,
    prisma_client: Any,
    session_id: str,
    agent_id: str,
    team_id: str,
    jwt: str,
    jwt_expires_at: datetime,
    repos: List[Dict[str, Any]],
    env_vars: Optional[Dict[str, str]],
    agent_row: Any = None,
) -> HydratePayload:
    """Assemble the complete ``HydratePayload`` for one session.

    Pure-async, no module-global state. Call from the session-create hot
    path AFTER the row is committed but BEFORE the SSM push so the JWT
    handed to the VM is the one durable in the DB.
    """
    secrets = await _decrypt_in_scope_secrets(prisma_client, team_id, repos)
    network_access = await _load_vm_config_network_access(prisma_client, team_id)

    expires_iso = jwt_expires_at.astimezone(timezone.utc).isoformat()

    return HydratePayload(
        session_id=session_id,
        agent_id=agent_id,
        jwt=jwt,
        jwt_expires_at=expires_iso,
        repos=_coerce_repos(repos),
        env_vars=_coerce_env_vars(env_vars),
        secrets=secrets,
        network_access=network_access,
        agent_config=_agent_config_from_row(agent_row, model=None),
    )

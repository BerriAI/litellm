"""
Agent — Python-side handle for one managed-agent definition.

This is the SDK-side counterpart to the public Agent CRUD HTTP endpoints
(``/v2/agents``). The proxy persists every agent to ``LiteLLM_Agent``;
``Agent`` is the typed, behaviour-rich accessor in-process code uses to
build sessions, list past sessions, and tear an agent down.

Boundary with the proxy
-----------------------
The proxy owns the HTTP surface and ownership/auth checks. The
``managed_agents`` layer (this file + ``Session``, ``Run``) is a pure
Python API on top of the same DB tables. They co-exist deliberately:
the HTTP endpoints are how external SDKs talk to the proxy, but
in-process callers (sub-agents, tests, internal tools) shouldn't have
to round-trip through HTTP just to spawn a session.

Lifecycle:
  * ``Agent.from_db_row(row, db, runtime, sandbox)`` — build from a
    Prisma row (the proxy wiring layer is responsible for fetching the
    row, applying ownership checks, then handing both off to us).
  * ``await agent.create_session(repos=..., env_vars=...)`` — INSERT a
    session row in ``ready`` status, mint a daemon JWT, and return a
    ``Session`` ready to ``send()`` prompts to.
  * ``await agent.get_session(session_id)`` — fetch one of the agent's
    existing sessions as a ``Session`` instance.
  * ``await agent.list_sessions()`` — list all of the agent's sessions.
  * ``await agent.delete()`` — hard-delete the agent row. Cascade rules
    on ``LiteLLM_Agent`` -> ``LiteLLM_AgentSession`` mean the DB will
    drop the sessions and runs too.

Why ``status=ready`` and not ``provisioning``?
-----------------------------------------------
The proxy's HTTP path uses ``provisioning`` because it kicks off a real
VM provider call in the background (EC2 cold boot can take ~60s). The
``managed_agents`` Python layer doesn't talk to a VM provider — the
sandbox is constructed in-process and is "ready" the moment it exists.
We mint the same daemon JWT for the same ``LITELLM_AGENT_JWT_SECRET``
auth surface so tests / internal callers can hit the existing daemon
endpoints if they want, but they don't have to.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import prisma

from litellm.managed_agents.agent_runtime.base import (
    AgentConfig,
    AgentRuntime,
)
from litellm.managed_agents.sandbox.base import Sandbox
from litellm.managed_agents.session import Session
from litellm.proxy.agent_session_endpoints.auth import (
    hash_daemon_token,
    mint_daemon_token,
)
from litellm.proxy.agent_session_endpoints.constants import (
    DEFAULT_MAX_SESSION_MINUTES,
    SESSION_STATUS_READY,
)
from litellm.proxy.agent_session_endpoints.ids import new_session_id


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Agent:
    """A live, in-process handle on one ``LiteLLM_Agent`` row.

    Holds the static agent definition (name, model, system_prompt,
    tools_config) plus the runtime + sandbox that any session it spawns
    will use. Tests can construct an ``Agent`` directly with a mock
    runtime and sandbox; production code goes through ``from_db_row``.

    The same ``runtime`` and ``sandbox`` instance get reused across all
    sessions this Agent spawns. That's a deliberate choice: most runtimes
    are stateless, and most sandboxes either are too (``LocalSandbox``
    creates its own per-instance tmpdir) or document explicitly when
    they're not. Callers that need per-session isolation should
    construct a fresh ``Agent`` per session.
    """

    id: str
    name: str
    model: str
    system_prompt: Optional[str] = None
    tools_config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    default_repos: List[Dict[str, Any]] = field(default_factory=list)
    default_env_vars: Dict[str, str] = field(default_factory=dict)
    user_api_key_hash: str = ""
    team_id: Optional[str] = None
    runtime: Optional[AgentRuntime] = None
    sandbox: Optional[Sandbox] = None
    db: Any = None

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    def to_runtime_config(self) -> AgentConfig:
        """Project this row into the smaller ``AgentConfig`` the runtime sees."""
        return AgentConfig(
            name=self.name,
            model=self.model,
            system_prompt=self.system_prompt,
            tools_config=self.tools_config,
            metadata=dict(self.metadata),
        )

    @classmethod
    async def from_db_row(
        cls,
        row: Any,
        db: Any,
        runtime: Optional[AgentRuntime] = None,
        sandbox: Optional[Sandbox] = None,
    ) -> "Agent":
        """Build an ``Agent`` from a Prisma ``LiteLLM_Agent`` row.

        ``runtime`` and ``sandbox`` are required for any session-spawning
        operation; pass ``None`` for read-only flows (e.g. ``list_sessions``
        / ``get_session`` from a CLI that just wants to inspect history).
        """
        return cls(
            id=getattr(row, "id"),
            name=getattr(row, "name"),
            model=getattr(row, "model"),
            system_prompt=getattr(row, "system_prompt", None),
            tools_config=_coerce_dict_or_none(getattr(row, "tools_config", None)),
            metadata=_coerce_dict(getattr(row, "metadata", None)),
            default_repos=_coerce_list_of_dict(getattr(row, "default_repos", None)),
            default_env_vars=_coerce_dict(getattr(row, "default_env_vars", None)),
            user_api_key_hash=getattr(row, "user_api_key_hash", "") or "",
            team_id=getattr(row, "team_id", None),
            runtime=runtime,
            sandbox=sandbox,
            db=db,
        )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def create_session(
        self,
        repos: Optional[List[Dict[str, Any]]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        max_session_minutes: int = DEFAULT_MAX_SESSION_MINUTES,
    ) -> Session:
        """Insert a session row and return a ``Session`` ready to send prompts.

        ``repos`` / ``env_vars`` overlay the agent defaults exactly the
        same way the HTTP endpoint does:
          * ``repos``: caller-provided wholly replaces the default list.
          * ``env_vars``: merged key by key, caller wins on collisions.

        Status starts at ``ready`` (not ``provisioning``) because the
        managed-agents layer owns the sandbox lifecycle directly — there
        is no remote VM to wait for. See module docstring.
        """
        self._require_db()
        self._require_runtime_and_sandbox(operation="create_session")

        resolved_repos = self._resolve_repos(repos)
        resolved_env_vars = self._resolve_env_vars(env_vars)

        session_id = new_session_id()
        expires_at = _now() + timedelta(minutes=max_session_minutes)
        daemon_token = mint_daemon_token(
            session_id=session_id,
            agent_id=self.id,
            expires_at_epoch=int(expires_at.timestamp()),
        )

        # Mirrors the HTTP path's payload shape so the same row format
        # works whether the session was created via /v2/sessions or via
        # this Python API. Json columns must be wrapped via prisma.Json,
        # relations must use {"connect": {"id": ...}}.
        payload: Dict[str, Any] = {
            "id": session_id,
            "agent": {"connect": {"id": self.id}},
            "user_api_key_hash": self.user_api_key_hash,
            "team_id": self.team_id,
            "repos": prisma.Json(resolved_repos or []),
            "status": SESSION_STATUS_READY,
            "daemon_token_hash": hash_daemon_token(daemon_token),
            "expires_at": expires_at,
            "updated_at": _now(),
        }
        if resolved_env_vars is not None:
            payload["env_vars"] = prisma.Json(resolved_env_vars)

        row = await self.db.litellm_agentsession.create(data=payload)

        return await Session.from_db_row(
            row=row,
            db=self.db,
            runtime=self.runtime,
            sandbox=self.sandbox,
            agent_config=self.to_runtime_config(),
            daemon_token=daemon_token,
        )

    async def get_session(self, session_id: str) -> Session:
        """Fetch one of this agent's sessions by id.

        Raises ``LookupError`` if the session doesn't exist or doesn't
        belong to this agent — callers that need richer auth handling
        should wrap and remap to HTTP errors.
        """
        self._require_db()
        row = await self.db.litellm_agentsession.find_unique(where={"id": session_id})
        if row is None or getattr(row, "agent_id", None) != self.id:
            raise LookupError(f"Session {session_id!r} not found for agent {self.id!r}")
        return await Session.from_db_row(
            row=row,
            db=self.db,
            runtime=self.runtime,
            sandbox=self.sandbox,
            agent_config=self.to_runtime_config(),
            daemon_token=None,
        )

    async def list_sessions(self) -> List[Session]:
        """Return every session this agent has spawned, newest first.

        No pagination here — call sites that expect lots of sessions
        should query ``self.db.litellm_agentsession`` directly with
        ``take``/``skip``. This helper is a convenience for tests and
        small-scale callers.
        """
        self._require_db()
        rows = await self.db.litellm_agentsession.find_many(
            where={"agent_id": self.id},
            order={"created_at": "desc"},
        )
        return [
            await Session.from_db_row(
                row=row,
                db=self.db,
                runtime=self.runtime,
                sandbox=self.sandbox,
                agent_config=self.to_runtime_config(),
                daemon_token=None,
            )
            for row in rows
        ]

    async def delete(self) -> None:
        """Hard-delete the agent row.

        Cascade rules in ``schema.prisma`` (``LiteLLM_AgentSession``
        ``onDelete: Cascade``, then ``LiteLLM_AgentRun`` and
        ``LiteLLM_AgentRunEvent`` cascading from there) drop everything
        downstream — no need to enumerate sessions here.
        """
        self._require_db()
        await self.db.litellm_agent.delete(where={"id": self.id})

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_repos(
        self,
        body_repos: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Caller-provided repos override agent defaults entirely (whole-list
        replace). Mirrors ``_resolve_repos`` in ``session_endpoints.py``.
        """
        if body_repos is not None:
            return [r for r in body_repos if isinstance(r, dict)]
        return list(self.default_repos)

    def _resolve_env_vars(
        self,
        body_env_vars: Optional[Dict[str, str]],
    ) -> Optional[Dict[str, str]]:
        """Merge: agent defaults first, caller overrides on top."""
        if body_env_vars is None and not self.default_env_vars:
            return None
        merged: Dict[str, str] = {}
        if self.default_env_vars:
            merged.update({str(k): str(v) for k, v in self.default_env_vars.items()})
        if body_env_vars:
            merged.update({str(k): str(v) for k, v in body_env_vars.items()})
        return merged or None

    def _require_db(self) -> None:
        if self.db is None:
            raise RuntimeError(
                "Agent operation requires a Prisma client; pass db=<client> "
                "via Agent.from_db_row(...) or set self.db before calling."
            )

    def _require_runtime_and_sandbox(self, operation: str) -> None:
        if self.runtime is None or self.sandbox is None:
            raise RuntimeError(
                f"Agent.{operation} requires both runtime and sandbox; "
                "construct the Agent via Agent.from_db_row(row, db, runtime, sandbox)."
            )


# ---------------------------------------------------------------------------
# Prisma JSON-column coercion helpers — same shape as ``serialization.py``
# but kept local so this module doesn't depend on the proxy serializer
# (which exists for the HTTP wire shape, not the Python API).
# ---------------------------------------------------------------------------


def _coerce_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _coerce_dict_or_none(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return value
    return None


def _coerce_list_of_dict(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    return []

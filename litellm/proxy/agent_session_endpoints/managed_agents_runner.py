"""
Drive a queued ``LiteLLM_AgentRun`` through the ``litellm.managed_agents``
runtime and persist every emitted event to ``LiteLLM_AgentRunEvent``.

This is the in-process alternative to the daemon-based execution path.
When ``agent_settings.managed_agents_enabled`` is true, ``POST
/v2/sessions/{sid}/runs`` schedules ``drive_run`` as a fire-and-forget
asyncio task right after inserting the row. The SSE stream at
``GET /v2/sessions/{sid}/runs/{rid}/events`` then surfaces the events
to clients exactly the same way it does for the daemon path.

We use the lower-level ``litellm.managed_agents`` primitives directly
(``AgentRuntime``, ``Sandbox``, ``AgentConfig``, ``SessionState``)
rather than the high-level ``Session.send`` API because the endpoint
already inserted the ``LiteLLM_AgentRun`` row before scheduling us —
``Session.send`` would insert a second row. The primitive path lets
us drive the existing row instead of creating a duplicate.

Imports of ``litellm.managed_agents`` are deferred to inside ``drive_run``
so the proxy keeps importing cleanly even if the managed_agents package
is mid-merge or partially shipped (TYPE_CHECKING block carries the
type info for static analysers).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

import prisma

from litellm._logging import verbose_proxy_logger
from litellm.proxy.agent_session_endpoints.constants import (
    RUN_STATUS_ERROR,
    RUN_STATUS_RUNNING,
    RUN_TERMINAL_EVENT_TYPES,
)
from litellm.proxy.agent_session_endpoints.session_status import (
    refresh_session_status_from_runs,
)

if TYPE_CHECKING:
    from litellm.managed_agents.agent_runtime.base import AgentRuntime
    from litellm.managed_agents.events import Event
    from litellm.managed_agents.sandbox.base import Sandbox


# ---------------------------------------------------------------------------
# Tiny helpers — kept out of `drive_run` so the orchestration there reads
# like a state machine instead of a wall of Prisma calls + ad-hoc dict
# munging.
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _agent_config_from_row(row: Any) -> "Any":
    """Snapshot a ``LiteLLM_Agent`` row into the runtime's static config.

    Imports ``AgentConfig`` lazily so this module imports cleanly even
    if ``litellm.managed_agents`` is missing pieces (e.g. mid-merge).
    """
    from litellm.managed_agents.agent_runtime.base import AgentConfig

    return AgentConfig(
        name=row.name,
        model=row.model,
        system_prompt=row.system_prompt,
        tools_config=row.tools_config if isinstance(row.tools_config, dict) else None,
        metadata=row.metadata if isinstance(row.metadata, dict) else {},
    )


def _session_state_from_row(row: Any, cwd: Optional[str]) -> "Any":
    """Snapshot a ``LiteLLM_AgentSession`` row into per-session runtime state."""
    from litellm.managed_agents.agent_runtime.base import SessionState

    env_vars = row.env_vars if isinstance(row.env_vars, dict) else {}
    repos = row.repos if isinstance(row.repos, list) else []
    return SessionState(
        session_id=row.id,
        cwd=cwd,
        env_vars=env_vars,
        repos=repos,
    )


def _extract_prompt_text(prompt: Any) -> str:
    """The wire ``RunCreate.prompt`` is a free-form dict; runtimes want a
    plain string. Pull ``text`` if present, otherwise stringify so the
    runtime still has something to work with.
    """
    if isinstance(prompt, dict):
        text = prompt.get("text")
        if isinstance(text, str):
            return text
        return str(prompt)
    if isinstance(prompt, str):
        return prompt
    return ""


async def _next_event_seq(prisma_client, run_id: str) -> int:
    last = await prisma_client.db.litellm_agentrunevent.find_first(
        where={"run_id": run_id},
        order={"seq": "desc"},
    )
    return (last.seq + 1) if last else 1


async def _persist_event(
    prisma_client, run_id: str, session_id: str, event: "Event"
) -> None:
    """Insert one event row and, if it's a terminal lifecycle event, flip
    the parent run + session status to match.

    Mirrors the daemon's ``events:append`` semantics so SSE consumers see
    identical wire shape regardless of which execution path drove the run.
    """
    seq = await _next_event_seq(prisma_client, run_id)
    payload = event.to_payload()
    try:
        await prisma_client.db.litellm_agentrunevent.create(
            data={
                "run": {"connect": {"id": run_id}},
                "seq": seq,
                "event_type": event.type,
                "payload": prisma.Json(payload),
            }
        )
    except Exception:
        # One retry on seq collision — under our single-task drive loop a
        # collision shouldn't happen, but a concurrent cancel could race.
        seq = await _next_event_seq(prisma_client, run_id)
        await prisma_client.db.litellm_agentrunevent.create(
            data={
                "run": {"connect": {"id": run_id}},
                "seq": seq,
                "event_type": event.type,
                "payload": prisma.Json(payload),
            }
        )

    new_status = RUN_TERMINAL_EVENT_TYPES.get(event.type)
    if new_status is not None:
        now = _now()
        result = payload.get("result") if isinstance(payload, dict) else None
        await prisma_client.db.litellm_agentrun.update(
            where={"id": run_id},
            data={
                "status": new_status,
                "terminated_at": now,
                "updated_at": now,
                "result": result if isinstance(result, str) else None,
            },
        )
        await refresh_session_status_from_runs(prisma_client, session_id)


async def _mark_run_running(prisma_client, run_id: str) -> None:
    """Flip queued -> running.

    The endpoint inserted the row in ``queued``; the daemon path normally
    runs ``_claim_next_queued_run`` to flip it. We do the equivalent here
    so the run status timeline looks identical between paths.
    """
    now = _now()
    try:
        await prisma_client.db.litellm_agentrun.update(
            where={"id": run_id},
            data={
                "status": RUN_STATUS_RUNNING,
                "started_at": now,
                "updated_at": now,
            },
        )
    except Exception as exc:
        verbose_proxy_logger.warning(
            "managed_agents_runner: mark_running failed run_id=%s: %s",
            run_id,
            exc,
        )


async def _mark_run_error(
    prisma_client, run_id: str, session_id: str, message: str
) -> None:
    """Failure path: persist a ``run_error`` event AND flip the row.

    Done as event-then-status so SSE clients see the error line before
    the terminal status — same ordering the daemon path produces via
    ``events:append`` -> terminal handler.
    """
    from litellm.managed_agents.events import EVENT_TYPE_RUN_ERROR, Event

    try:
        await _persist_event(
            prisma_client,
            run_id,
            session_id,
            Event(type=EVENT_TYPE_RUN_ERROR, data={"error": message}),
        )
    except Exception as exc:
        verbose_proxy_logger.exception(
            "managed_agents_runner: failed to persist run_error event " "run_id=%s: %s",
            run_id,
            exc,
        )
        # Last resort — even if event persistence broke, we still want
        # the row to reflect the failure so SSE clients eventually see
        # a terminal status (they tail until the run row is terminal).
        now = _now()
        try:
            await prisma_client.db.litellm_agentrun.update(
                where={"id": run_id},
                data={
                    "status": RUN_STATUS_ERROR,
                    "terminated_at": now,
                    "updated_at": now,
                    "result": message,
                },
            )
            await refresh_session_status_from_runs(prisma_client, session_id)
        except Exception as inner:
            verbose_proxy_logger.exception(
                "managed_agents_runner: failed to mark error run_id=%s: %s",
                run_id,
                inner,
            )


async def drive_run(
    *,
    run_id: str,
    session_id: str,
    runtime: Optional["AgentRuntime"] = None,
    sandbox: Optional["Sandbox"] = None,
) -> None:
    """Run the LLM tool loop for a queued run and persist every event.

    Designed to be scheduled with ``asyncio.create_task(...)`` from the
    ``POST /v2/sessions/{sid}/runs`` handler. Never raises — all failures
    are surfaced as ``run_error`` events on the run.

    ``runtime`` and ``sandbox`` are injectable for tests; production
    callers leave them ``None`` and we build the defaults
    (``ClaudeSDKAgentRuntime`` + ``LocalSandbox``).
    """
    # Late import to keep this module importable without litellm.managed_agents
    # being fully assembled. ``proxy_server`` is also late-imported to avoid a
    # circular: run_endpoints -> this module -> proxy_server -> run_router.
    from litellm.managed_agents.agent_runtime.claude_sdk import (
        ClaudeSDKAgentRuntime,
    )
    from litellm.managed_agents.events import EVENT_TYPE_RUN_STARTED, Event
    from litellm.managed_agents.sandbox.local import LocalSandbox
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        verbose_proxy_logger.warning(
            "managed_agents_runner: prisma_client is None; " "cannot drive run_id=%s",
            run_id,
        )
        return

    # Reload run + session + agent fresh — the row that was inserted by
    # the endpoint may have moved (cancelled) by the time this task runs.
    try:
        run_row = await prisma_client.db.litellm_agentrun.find_unique(
            where={"id": run_id}
        )
        if run_row is None or run_row.status != "queued":
            verbose_proxy_logger.info(
                "managed_agents_runner: run_id=%s not queued (status=%s); skipping",
                run_id,
                getattr(run_row, "status", None),
            )
            return
        session_row = await prisma_client.db.litellm_agentsession.find_unique(
            where={"id": session_id}
        )
        if session_row is None:
            await _mark_run_error(
                prisma_client, run_id, session_id, "session not found"
            )
            return
        agent_row = await prisma_client.db.litellm_agent.find_unique(
            where={"id": session_row.agent_id}
        )
        if agent_row is None:
            await _mark_run_error(
                prisma_client, run_id, session_id, "parent agent not found"
            )
            return
    except Exception as exc:
        verbose_proxy_logger.exception(
            "managed_agents_runner: bootstrap failed run_id=%s: %s", run_id, exc
        )
        await _mark_run_error(
            prisma_client, run_id, session_id, f"bootstrap failed: {exc}"
        )
        return

    sandbox_obj = sandbox if sandbox is not None else LocalSandbox()
    runtime_obj = (
        runtime if runtime is not None else ClaudeSDKAgentRuntime(model=agent_row.model)
    )

    agent_config = _agent_config_from_row(agent_row)
    session_state = _session_state_from_row(session_row, cwd=sandbox_obj.cwd)
    prompt_text = _extract_prompt_text(run_row.prompt)

    # Flip queued -> running and emit run_started before invoking the
    # runtime so SSE clients can render an immediate "I started" frame.
    await _mark_run_running(prisma_client, run_id)
    try:
        await _persist_event(
            prisma_client,
            run_id,
            session_id,
            Event(
                type=EVENT_TYPE_RUN_STARTED,
                data={"run_id": run_id, "session_id": session_id},
            ),
        )
    except Exception as exc:
        verbose_proxy_logger.warning(
            "managed_agents_runner: run_started persist failed run_id=%s: %s",
            run_id,
            exc,
        )

    try:
        async for event in runtime_obj.run(
            prompt=prompt_text,
            sandbox=sandbox_obj,
            session_state=session_state,
            agent_config=agent_config,
        ):
            await _persist_event(prisma_client, run_id, session_id, event)
    except Exception as exc:
        verbose_proxy_logger.exception(
            "managed_agents_runner: runtime failed run_id=%s: %s", run_id, exc
        )
        await _mark_run_error(prisma_client, run_id, session_id, str(exc))
    finally:
        # Best-effort sandbox teardown. LocalSandbox cleans up its tmpdir;
        # remote sandboxes release their VM. Swallow errors so a failing
        # teardown doesn't mask a successful run.
        try:
            await sandbox_obj.teardown()
        except Exception as exc:
            verbose_proxy_logger.warning(
                "managed_agents_runner: sandbox teardown failed " "run_id=%s: %s",
                run_id,
                exc,
            )


# Strong references to in-flight drive_run tasks. Without this set, the
# event loop would only weakly reference the task and could collect it
# mid-run. See https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task.
_BACKGROUND_TASKS: set = set()


def schedule_run(run_id: str, session_id: str) -> None:
    """Fire-and-forget scheduler called from the create_run endpoint.

    Wraps ``drive_run`` in ``asyncio.create_task`` so the HTTP handler
    can return immediately. Holds a strong reference to the task on the
    module so the GC doesn't collect it before it finishes.
    """
    task = asyncio.create_task(drive_run(run_id=run_id, session_id=session_id))
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


def is_managed_agents_enabled() -> bool:
    """Read ``agent_settings.managed_agents_enabled`` from the loaded
    proxy config.

    Defaults to ``True`` so the new path is on by default. Callers who
    want the legacy daemon-driven flow (e.g. existing unit tests pinned
    to ``NoopVMProvider`` semantics) can opt out with
    ``managed_agents_enabled: false`` under ``general_settings.agent_settings``,
    or by setting the ``LITELLM_DISABLE_MANAGED_AGENTS_RUN_DRIVER``
    env var to a truthy value.

    The env-var escape hatch is what the test suite uses — tests don't
    load a YAML config so the proxy's ``general_settings`` stays empty,
    and we don't want a single global flag mutation to cross-contaminate
    parallel pytest workers.
    """
    if os.environ.get("LITELLM_DISABLE_MANAGED_AGENTS_RUN_DRIVER", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return False
    try:
        from litellm.proxy import proxy_server

        general_settings = getattr(proxy_server, "general_settings", None) or {}
        agent_settings = general_settings.get("agent_settings") or {}
        # Also support a top-level ``agent_settings`` set on the module
        # directly (some test setups stash it there).
        if not agent_settings:
            agent_settings = getattr(proxy_server, "agent_settings", None) or {}
        return bool(agent_settings.get("managed_agents_enabled", True))
    except Exception:
        return True


__all__ = [
    "drive_run",
    "schedule_run",
    "is_managed_agents_enabled",
]

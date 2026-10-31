"""Session helpers for Vertex AI Agent Engine A2A."""

import asyncio
import hashlib
import json
from typing import Protocol
from weakref import WeakKeyDictionary, WeakValueDictionary


def session_user_id(calling_key_hash: str, context_id: str) -> str:
    """Create an isolated identity for one caller and conversation."""
    if not calling_key_hash:
        raise ValueError("Authenticated calling key hash is required")

    if not context_id:
        raise ValueError("A2A context ID is required")

    identity = json.dumps(
        [calling_key_hash, context_id],
        separators=(",", ":"),
        ensure_ascii=False,
    )

    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"litellm-a2a-{digest}"


class AgentEngineSessionBackend(Protocol):
    """Interface for Agent Engine session operations."""

    async def list_sessions(self, *, user_id: str) -> list[str]: ...

    async def create_session(self, *, user_id: str) -> str: ...


# Keep locks separate across event loops and release unused locks.
_SESSION_LOCKS: WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    WeakValueDictionary[str, asyncio.Lock],
] = WeakKeyDictionary()


def _session_lock(user_id: str) -> asyncio.Lock:
    """Return a shared lock for one conversation on this event loop."""
    loop = asyncio.get_running_loop()
    locks = _SESSION_LOCKS.get(loop)

    if locks is None:
        locks = WeakValueDictionary()
        _SESSION_LOCKS[loop] = locks

    lock = locks.get(user_id)

    if lock is None:
        lock = asyncio.Lock()
        locks[user_id] = lock

    return lock


async def resolve_session_id(
    backend: AgentEngineSessionBackend,
    *,
    calling_key_hash: str,
    context_id: str,
) -> str:
    """Find or create one session per caller and conversation."""

    user_id = session_user_id(calling_key_hash, context_id)

    # Serialize lookup and creation for this conversation.
    async with _session_lock(user_id):
        existing_sessions = await backend.list_sessions(user_id=user_id)

        if existing_sessions:
            return existing_sessions[0]

        return await backend.create_session(user_id=user_id)

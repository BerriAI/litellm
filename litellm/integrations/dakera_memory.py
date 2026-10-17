"""Persistent cross-session memory for LiteLLM via self-hosted Dakera.

Hooks into litellm's pre/post call lifecycle:
- async_pre_call_hook: recalls relevant prior memories and prepends them as a
  system message before the LLM sees the prompt
- async_log_success_event: persists the completed exchange to Dakera after success

Dakera is a self-hosted, decay-weighted vector memory server. Recall and storage
go through the official ``dakera`` Python SDK, so this logger always speaks the
same verified API as the rest of the Dakera ecosystem.

Install the SDK (optional dependency, only needed when this callback is used)::

    pip install dakera

Self-host Dakera with the public docker-compose (server + object store)::

    git clone https://github.com/dakera-ai/dakera-deploy && cd dakera-deploy
    docker compose up -d          # serves the API on http://localhost:3000

Usage::

    import litellm
    from litellm.integrations.dakera_memory import DakeraMemoryLogger

    litellm.callbacks = [
        DakeraMemoryLogger(
            base_url="http://localhost:3000",
            api_key="dk-your-key",
            top_k=5,
        )
    ]

    response = litellm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What did we discuss last time?"}],
        metadata={"session_id": "user-alice"},
    )
"""

import hashlib
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger

_DEFAULT_BASE_URL = "http://localhost:3000"


def _extract_text(content: Any) -> str:
    """Extract plain text from message content that may be str or list (multimodal)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text").strip()
    return str(content) if content else ""


class DakeraMemoryLogger(CustomLogger):
    """Persistent LLM memory via self-hosted Dakera.

    Dakera is a decay-weighted vector memory server you run on your own
    infrastructure. This logger gives every litellm call persistent cross-session
    memory without any cloud dependency, using the official ``dakera`` SDK.

    Args:
        base_url: Dakera server URL. Defaults to the ``DAKERA_API_URL`` env var,
            then ``http://localhost:3000``.
        api_key: Dakera API key. Defaults to the ``DAKERA_API_KEY`` env var.
        top_k: Number of memories to recall per LLM call.
        session_id_key: Key in litellm call metadata used to group memories
            by session. Defaults to ``"session_id"``.

    **Tenant isolation:** Always pass a unique ``session_id`` in call metadata to
    prevent cross-user memory leakage. When no ``session_id`` is provided the logger
    derives a namespace from the caller's API key hash, which isolates memory per
    litellm API key but still mixes sessions for the same key. For strict per-user
    isolation, supply ``metadata={"session_id": user_id}`` on every call.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        top_k: int = 5,
        session_id_key: str = "session_id",
    ) -> None:
        super().__init__()
        self.base_url = (base_url or os.getenv("DAKERA_API_URL") or _DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.getenv("DAKERA_API_KEY") or None
        self.top_k = top_k
        self.session_id_key = session_id_key
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily construct and cache the async Dakera SDK client.

        Raises a clear, actionable error if the optional ``dakera`` package is
        not installed, rather than failing silently at call time.
        """
        if self._client is None:
            try:
                from dakera import AsyncDakeraClient
            except ImportError as exc:
                raise ImportError(
                    "DakeraMemoryLogger requires the 'dakera' package. Install it with: pip install dakera"
                ) from exc
            self._client = AsyncDakeraClient(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def _session_id(
        self,
        metadata: Optional[Dict],
        user_api_key_dict: Any = None,
    ) -> str:
        if metadata:
            sid = metadata.get(self.session_id_key)
            if sid:
                return str(sid)
        # Fall back to a stable per-API-key namespace to prevent cross-tenant leakage.
        # Handles both UserAPIKeyAuth objects (has .api_key attribute) and serialized
        # dicts (kwargs["user_api_key_dict"] in async_log_success_event).
        caller_key: str = ""
        if user_api_key_dict is not None:
            caller_key = getattr(user_api_key_dict, "api_key", None)
            if caller_key is None and isinstance(user_api_key_dict, dict):
                caller_key = user_api_key_dict.get("api_key", "")
            caller_key = caller_key or ""
        if caller_key:
            return "key:" + hashlib.sha256(caller_key.encode()).hexdigest()[:16]
        return "default"

    # ------------------------------------------------------------------
    # Pre-call hook: inject recalled memories before the model sees the prompt
    # ------------------------------------------------------------------

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: Dict,
        call_type: str,
    ) -> Optional[Dict]:
        """Recall relevant prior memories and prepend them as a system message."""
        try:
            messages: List[Dict] = data.get("messages", [])
            if not messages:
                return data

            raw_content = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
                None,
            )
            if not raw_content:
                return data
            last_user = _extract_text(raw_content)
            if not last_user:
                return data

            session_id = self._session_id(data.get("metadata"), user_api_key_dict)

            resp = await self._get_client().recall(
                agent_id=session_id,
                query=last_user,
                top_k=self.top_k,
            )

            memories = getattr(resp, "memories", None) or []
            memory_lines = "\n".join(f"- {m.content}" for m in memories if getattr(m, "content", None))
            if not memory_lines:
                return data

            memory_msg = {
                "role": "system",
                "content": (
                    f"Relevant context from prior sessions (retrieved from persistent memory):\n{memory_lines}"
                ),
            }
            existing_system = [m for m in messages if m.get("role") == "system"]
            non_system = [m for m in messages if m.get("role") != "system"]
            data["messages"] = existing_system + [memory_msg] + non_system

            verbose_logger.debug(f"DakeraMemoryLogger: injected {len(memories)} memories for session={session_id}")
        except Exception as exc:  # noqa: BLE001
            verbose_logger.warning(f"DakeraMemoryLogger.async_pre_call_hook failed: {exc}")

        return data

    # ------------------------------------------------------------------
    # Post-call hook: persist the completed exchange after a successful call
    # ------------------------------------------------------------------

    async def async_log_success_event(
        self,
        kwargs: Dict,
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Store the completed LLM exchange in Dakera."""
        try:
            messages: List[Dict] = kwargs.get("messages", [])
            session_id = self._session_id(kwargs.get("metadata"), kwargs.get("user_api_key_dict"))

            raw_content = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
                None,
            )
            if not raw_content:
                return
            last_user = _extract_text(raw_content)
            if not last_user:
                return

            assistant_content: Optional[str] = None
            if hasattr(response_obj, "choices") and response_obj.choices:
                choice = response_obj.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    assistant_content = choice.message.content

            content = f"User: {last_user}"
            if assistant_content:
                content += f"\nAssistant: {assistant_content}"

            await self._get_client().store_memory(
                agent_id=session_id,
                content=content,
                session_id=session_id,
                metadata={
                    "model": kwargs.get("model", ""),
                    "call_type": kwargs.get("call_type", ""),
                },
            )
            verbose_logger.debug(f"DakeraMemoryLogger: stored exchange for session={session_id}")
        except Exception as exc:  # noqa: BLE001
            verbose_logger.warning(f"DakeraMemoryLogger.async_log_success_event failed: {exc}")

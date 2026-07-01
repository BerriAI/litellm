"""Persistent cross-session memory for LiteLLM via self-hosted Dakera.

Hooks into litellm's pre/post call lifecycle:
- async_pre_call_hook: recalls relevant prior memories and prepends them as a
  system message before the LLM sees the prompt
- async_log_success_event: persists the completed exchange to Dakera after success

Self-host Dakera:
    docker run -p 3300:3300 -e DAKERA_API_KEY=demo ghcr.io/dakera-ai/dakera:latest

Usage:
    import litellm
    from litellm.integrations.dakera_memory import DakeraMemoryLogger

    litellm.callbacks = [
        DakeraMemoryLogger(
            base_url="http://localhost:3300",
            api_key="dk_your_key",
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
from typing import Any, Dict, List, Optional, Union

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)


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
    memory without any cloud dependency.

    Args:
        base_url: Dakera server URL. Defaults to ``DAKERA_API_URL`` env var.
        api_key: Dakera API key. Defaults to ``DAKERA_API_KEY`` env var.
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
        self.base_url = (base_url or os.getenv("DAKERA_API_URL", "http://localhost:3300")).rstrip("/")
        self.api_key = api_key or os.getenv("DAKERA_API_KEY", "")
        self.top_k = top_k
        self.session_id_key = session_id_key
        self._http_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

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
        # This isolates memories per litellm API key when no explicit session is supplied.
        caller_key: str = ""
        if user_api_key_dict is not None:
            caller_key = getattr(user_api_key_dict, "api_key", "") or ""
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

            # Use the last user message as the semantic recall query.
            # _extract_text handles multimodal content (list of parts) gracefully.
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

            resp = await self._http_handler.post(
                url=f"{self.base_url}/v1/memories/search",
                headers=self._headers(),
                json={
                    "query": last_user,
                    "session_id": session_id,
                    "top_k": self.top_k,
                },
                timeout=5.0,
            )

            if resp.status_code != 200:
                return data

            results = resp.json().get("results", [])
            if not results:
                return data

            # Build memory context and inject before the first non-system message
            memory_lines = "\n".join(f"- {r.get('content', '')}" for r in results if r.get("content"))
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

            verbose_logger.debug(f"DakeraMemoryLogger: injected {len(results)} memories for session={session_id}")
        except Exception as exc:
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
            session_id = self._session_id(kwargs.get("metadata"))

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

            await self._http_handler.post(
                url=f"{self.base_url}/v1/memories",
                headers=self._headers(),
                json={
                    "content": content,
                    "session_id": session_id,
                    "metadata": {
                        "model": kwargs.get("model", ""),
                        "call_type": kwargs.get("call_type", ""),
                    },
                },
                timeout=5.0,
            )
            verbose_logger.debug(f"DakeraMemoryLogger: stored exchange for session={session_id}")
        except Exception as exc:
            verbose_logger.warning(f"DakeraMemoryLogger.async_log_success_event failed: {exc}")

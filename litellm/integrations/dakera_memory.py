# What is this?
## Persistent memory for LLM calls via Dakera
## Recalls relevant prior context before each LLM call, stores exchanges after.
##
## Usage:
##   import litellm
##   from litellm.integrations.dakera_memory import DakeraMemoryLogger
##   litellm.callbacks = [DakeraMemoryLogger(base_url="http://localhost:3000", api_key="dk_...")]
##
## Or as a named callback (add "dakera_memory" to litellm.success_callback):
##   litellm.success_callback = ["dakera_memory"]

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm._logging import verbose_logger


class DakeraMemoryLogger(CustomLogger):
    """Persistent LLM memory via self-hosted Dakera.

    Hooks into litellm's pre/post call lifecycle:
    - async_pre_call_hook: recalls relevant memories and injects them as a
      system message before the LLM call
    - async_log_success_event: persists the completed exchange to Dakera

    Self-host Dakera: docker run -p 3000:3000 dakera/dakera:latest

    Args:
        base_url: Dakera server URL. Defaults to DAKERA_API_URL env var.
        api_key: Dakera API key. Defaults to DAKERA_API_KEY env var.
        top_k: Number of memories to recall per call. Defaults to 5.
        session_id_key: Metadata key used to group memories by session.
            Reads from litellm call metadata dict. Defaults to "session_id".
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        top_k: int = 5,
        session_id_key: str = "session_id",
    ):
        self.base_url = (base_url or os.getenv("DAKERA_API_URL", "http://localhost:3000")).rstrip("/")
        self.api_key = api_key or os.getenv("DAKERA_API_KEY", "")
        self.top_k = top_k
        self.session_id_key = session_id_key

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _session_id(self, metadata: Optional[Dict]) -> str:
        if not metadata:
            return "default"
        return str(metadata.get(self.session_id_key, "default"))

    # ------------------------------------------------------------------
    # Pre-call hook: inject recalled memories before the LLM sees the prompt
    # ------------------------------------------------------------------

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: Dict,
        call_type: str,
    ) -> Optional[Dict]:
        """Recall relevant memories and prepend them as a system message."""
        try:
            messages: List[Dict] = data.get("messages", [])
            if not messages:
                return data

            # Extract the last user message as the recall query
            last_user = next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                None,
            )
            if not last_user:
                return data

            session_id = self._session_id(data.get("metadata"))

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/memories/search",
                    headers=self._headers(),
                    json={
                        "query": last_user if isinstance(last_user, str) else str(last_user),
                        "session_id": session_id,
                        "top_k": self.top_k,
                    },
                )
                if resp.status_code != 200:
                    return data
                results = resp.json().get("results", [])

            if not results:
                return data

            # Build memory context string
            memory_lines = "\n".join(f"- {r['content']}" for r in results)
            memory_msg = {
                "role": "system",
                "content": (
                    f"Relevant context from prior sessions (retrieved from persistent memory):\n"
                    f"{memory_lines}"
                ),
            }

            # Inject before the first non-system message
            existing_system = [m for m in messages if m.get("role") == "system"]
            non_system = [m for m in messages if m.get("role") != "system"]
            data["messages"] = existing_system + [memory_msg] + non_system

            verbose_logger.debug(
                f"DakeraMemoryLogger: injected {len(results)} memories for session={session_id}"
            )
        except Exception as e:
            verbose_logger.warning(f"DakeraMemoryLogger.async_pre_call_hook failed: {e}")

        return data

    # ------------------------------------------------------------------
    # Post-call hook: store the completed exchange to Dakera
    # ------------------------------------------------------------------

    async def async_log_success_event(
        self,
        kwargs: Dict,
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Persist the completed LLM exchange to Dakera after a successful call."""
        try:
            messages: List[Dict] = kwargs.get("messages", [])
            session_id = self._session_id(kwargs.get("metadata"))

            # Find last user message content to store
            last_user = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
                None,
            )
            if not last_user:
                return

            # Also grab the assistant response if available
            assistant_content: Optional[str] = None
            if hasattr(response_obj, "choices") and response_obj.choices:
                choice = response_obj.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    assistant_content = choice.message.content

            content = f"User: {last_user}"
            if assistant_content:
                content += f"\nAssistant: {assistant_content}"

            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{self.base_url}/v1/memories",
                    headers=self._headers(),
                    json={
                        "content": content,
                        "session_id": session_id,
                        "metadata": {
                            "model": kwargs.get("model", ""),
                            "call_type": kwargs.get("call_type", ""),
                        },
                    },
                )
            verbose_logger.debug(
                f"DakeraMemoryLogger: stored exchange for session={session_id}"
            )
        except Exception as e:
            verbose_logger.warning(f"DakeraMemoryLogger.async_log_success_event failed: {e}")

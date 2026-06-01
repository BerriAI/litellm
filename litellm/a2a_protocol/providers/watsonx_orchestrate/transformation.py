"""
Transformation layer for IBM watsonx Orchestrate (WXO) agent provider.

WXO uses a REST API (not A2A/JSON-RPC) with an async-poll execution model:
  POST /v1/orchestrate/runs       → submit run, get run_id
  GET  /v1/orchestrate/runs/{id}  → poll until terminal state
  POST /v1/orchestrate/runs/stream → native SSE streaming
"""

import asyncio
from typing import Any, AsyncIterator, Dict, Optional
from uuid import uuid4

from litellm._logging import verbose_logger


class WatsonxOrchestrateTransformation:
    """
    Handles request/response transformation between A2A and the WXO REST API.
    """

    TERMINAL_STATES = frozenset(
        {"completed", "succeeded", "failed", "error", "cancelled"}
    )
    SUCCESS_STATES = frozenset({"completed", "succeeded"})

    @staticmethod
    def get_api_base_url(cp4d_host: str, instance_id: str) -> str:
        """Build the WXO API base URL from host and instance ID."""
        return f"{cp4d_host.rstrip('/')}/orchestrate/cpd/instances/{instance_id}"

    @staticmethod
    def extract_text_from_a2a_params(params: Dict[str, Any]) -> str:
        """
        Extract user message text from A2A MessageSendParams.

        A2A format: params.message.parts[*] where part.kind == "text"
        """
        message = params.get("message", {})
        parts = message.get("parts", [])
        texts = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("kind") == "text" and part.get("text"):
                texts.append(part["text"])
            elif "text" in part and part["text"]:
                texts.append(part["text"])
        return " ".join(texts) or ""

    @staticmethod
    def build_wxo_run_body(
        wxo_agent_id: str,
        text: str,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the WXO POST /v1/orchestrate/runs request body."""
        body: Dict[str, Any] = {
            "agent_id": wxo_agent_id,
            "message": {
                "role": "user",
                "content": [
                    {
                        "response_type": "conversational_search",
                        "text": text,
                    }
                ],
            },
        }
        if thread_id:
            body["thread_id"] = thread_id
        return body

    @staticmethod
    def extract_text_from_wxo_result(result: Dict[str, Any]) -> str:
        """
        Extract response text from a WXO run result.

        WXO can return text in several locations; checks in priority order per the API spec.
        """
        # Primary: last_message.content[0].text
        try:
            text = result["last_message"]["content"][0]["text"]
            if text:
                return str(text)
        except (KeyError, IndexError, TypeError):
            pass

        # Secondary: result.data.message.content[0].text
        try:
            text = result["result"]["data"]["message"]["content"][0]["text"]
            if text:
                return str(text)
        except (KeyError, IndexError, TypeError):
            pass

        # Tertiary: results as a raw string
        results = result.get("results")
        if results and isinstance(results, str):
            return results

        return ""

    @staticmethod
    def build_a2a_message_response(request_id: str, text: str) -> Dict[str, Any]:
        """
        Build a standard A2A non-streaming SendMessageResponse (kind=message).
        """
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "kind": "message",
                "role": "agent",
                "parts": [{"kind": "text", "text": text}],
                "messageId": str(uuid4()),
            },
        }

    @staticmethod
    async def fake_streaming_from_text(
        text: str,
        request_id: str,
        chunk_size: int = 50,
        delay_ms: int = 10,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Emit standard A2A streaming events from a completed text response.

        Event sequence:
          1. task         (kind="task", state="submitted")
          2. status-update (kind="status-update", state="working")
          3. artifact-update chunks
          4. status-update (kind="status-update", state="completed", final=True)
        """
        task_id = str(uuid4())
        context_id = str(uuid4())
        artifact_id = str(uuid4())

        # 1. Task submitted
        yield {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "contextId": context_id,
                "id": task_id,
                "kind": "task",
                "status": {"state": "submitted"},
            },
        }

        # 2. Working
        yield {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "contextId": context_id,
                "final": False,
                "kind": "status-update",
                "status": {"state": "working"},
                "taskId": task_id,
            },
        }
        await asyncio.sleep(delay_ms / 1000.0)

        # 3. Artifact chunks (always emit at least one chunk, even for empty text)
        text_to_chunk = text or ""
        for i in range(0, max(len(text_to_chunk), 1), chunk_size):
            chunk_text = text_to_chunk[i : i + chunk_size]
            is_last = (i + chunk_size) >= max(len(text_to_chunk), 1)
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "contextId": context_id,
                    "kind": "artifact-update",
                    "taskId": task_id,
                    "artifact": {
                        "artifactId": artifact_id,
                        "parts": [{"kind": "text", "text": chunk_text}],
                    },
                },
            }
            if not is_last:
                await asyncio.sleep(delay_ms / 1000.0)

        # 4. Completed
        yield {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "contextId": context_id,
                "final": True,
                "kind": "status-update",
                "status": {"state": "completed"},
                "taskId": task_id,
            },
        }

        verbose_logger.debug(
            f"WXO: Fake streaming completed for request_id={request_id}"
        )

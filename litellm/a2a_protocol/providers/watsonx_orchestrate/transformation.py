"""
Transformation layer for IBM watsonx Orchestrate (WXO) agent provider.

WXO uses a REST API (not A2A/JSON-RPC) with an async-poll execution model:
  POST /v1/orchestrate/runs       → submit run, get run_id
  GET  /v1/orchestrate/runs/{id}  → poll until terminal state
  POST /v1/orchestrate/runs/stream → native SSE streaming
"""

import asyncio
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from itertools import chain
from typing import Final
from uuid import uuid4

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping, iter_items
from litellm.llms.openai.common_utils import decoded_json_object


def _a2a_text_part(part: object) -> Sequence[object]:
    part_fields: Final = as_str_mapping(part)
    if not isinstance(part_fields, dict):
        return ()
    if part_fields.get("kind") in (None, "", "text") and part_fields.get("text"):
        return (part_fields["text"],)
    return ()


def _join_with_spaces(texts: Sequence[object]) -> str:
    strings: Final = [text for text in texts if isinstance(text, str)]
    if len(strings) == len(texts):
        return " ".join(strings)
    first_bad: Final = next(position for position, text in enumerate(texts) if not isinstance(text, str))
    raise TypeError(f"sequence item {first_bad}: expected str instance, {type(texts[first_bad]).__name__} found")


class WatsonxOrchestrateTransformation:
    """
    Handles request/response transformation between A2A and the WXO REST API.
    """

    TERMINAL_STATES = frozenset({"completed", "succeeded", "failed", "error", "cancelled"})
    SUCCESS_STATES = frozenset({"completed", "succeeded"})

    @staticmethod
    def get_api_base_url(cp4d_host: str, instance_id: str) -> str:
        """Build the WXO API base URL from host and instance ID."""
        return f"{cp4d_host.rstrip('/')}/orchestrate/cpd/instances/{instance_id}"

    @staticmethod
    def extract_text_from_a2a_params(params: Mapping[str, object]) -> str:
        """
        Extract user message text from A2A MessageSendParams.

        A2A format: params.message.parts[*] where part.kind == "text"
        """
        message: Final = decoded_json_object(params.get("message", {}))
        parts: Final[Iterable[object]] = iter_items(message.get("parts", []))
        texts: Final = list(chain.from_iterable(map(_a2a_text_part, parts)))
        return _join_with_spaces(texts) or ""

    @staticmethod
    def build_wxo_run_body(
        wxo_agent_id: str,
        text: str,
        thread_id: str | None = None,
    ) -> dict[str, object]:
        """Build the WXO POST /v1/orchestrate/runs request body."""
        body: Final[dict[str, object]] = {
            "agent_id": wxo_agent_id,
            "message": {
                "role": "user",
                "content": [
                    {
                        "response_type": "text",
                        "text": text,
                    }
                ],
            },
        }
        if thread_id:
            body["thread_id"] = thread_id
        return body

    @staticmethod
    def extract_text_from_wxo_result(result: object) -> str:
        """
        Extract response text from a WXO run result.

        WXO can return text in several locations; checks in priority order per the API spec.
        """
        if not isinstance(result, dict):
            return ""

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
        results: Final = result.get("results")
        if results and isinstance(results, str):
            return results

        return ""

    @staticmethod
    def extract_text_from_a2a_message_response(a2a_response: Mapping[str, object]) -> str:
        result: Final = a2a_response.get("result")
        if not isinstance(result, dict):
            verbose_logger.warning("WXO: A2A response missing result object")
            return ""
        parts: Final = result.get("parts")
        if not isinstance(parts, list):
            verbose_logger.warning("WXO: A2A result has no parts list")
            return ""
        for part in parts:
            if isinstance(part, dict) and part.get("kind") == "text" and part.get("text"):
                return str(part["text"])
        verbose_logger.warning("WXO: A2A result parts contained no text")
        return ""

    @staticmethod
    def build_a2a_message_response(request_id: str, text: str) -> dict[str, object]:
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
    ) -> AsyncIterator[dict[str, object]]:
        """
        Emit standard A2A streaming events from a completed text response.

        Event sequence:
          1. task         (kind="task", state="submitted")
          2. status-update (kind="status-update", state="working")
          3. artifact-update chunks
          4. status-update (kind="status-update", state="completed", final=True)
        """
        task_id: Final = str(uuid4())
        context_id: Final = str(uuid4())
        artifact_id: Final = str(uuid4())

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
        text_to_chunk: Final = text or ""
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

        verbose_logger.debug("WXO: Fake streaming completed for request_id=%s", request_id)

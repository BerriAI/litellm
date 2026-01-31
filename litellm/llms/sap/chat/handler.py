from __future__ import annotations

import json
import time
from typing import AsyncIterator, Iterator, Optional

import httpx

from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.openai import OpenAIChatCompletionChunk

from ...custom_httpx.llm_http_handler import BaseLLMHTTPHandler


# -------------------------------
# Errors
# -------------------------------
class GenAIHubOrchestrationError(BaseLLMException):
    def __init__(self, status_code: int, message: str):
        super().__init__(status_code=status_code, message=message)
        self.status_code = status_code
        self.message = message


# -------------------------------
# Stream parsing helpers
# -------------------------------


def _now_ts() -> int:
    return int(time.time())


def _is_terminal_chunk(chunk: OpenAIChatCompletionChunk) -> bool:
    """OpenAI-shaped chunk is terminal if any choice has a non-None finish_reason."""
    try:
        for ch in chunk.choices or []:
            if ch.finish_reason is not None:
                return True
    except Exception:
        pass
    return False


class _StreamParser:
    """Normalize orchestration streaming events into OpenAI-like chunks."""

    @staticmethod
    def _from_orchestration_result(evt: dict) -> Optional[OpenAIChatCompletionChunk]:
        """
        Accepts orchestration_result shape and maps it to an OpenAI-like *chunk*.
        """
        orc = evt.get("orchestration_result") or {}
        if not orc:
            return None

        return OpenAIChatCompletionChunk.model_validate(
            {
                "id": orc.get("id") or evt.get("request_id") or "stream-chunk",
                "object": orc.get("object") or "chat.completion.chunk",
                "created": orc.get("created") or evt.get("created") or _now_ts(),
                "model": orc.get("model") or "unknown",
                "choices": [
                    {
                        "index": c.get("index", 0),
                        "delta": c.get("delta") or {},
                        "finish_reason": c.get("finish_reason"),
                    }
                    for c in (orc.get("choices") or [])
                ],
            }
        )

    @staticmethod
    def to_openai_chunk(event_obj: dict) -> Optional[OpenAIChatCompletionChunk]:
        """
        Accepts:
          - {"final_result": <openai-style CHUNK>}   (IMPORTANT: this is just another chunk, NOT terminal)
          - {"orchestration_result": {...}}          (map to chunk)
          - already-openai-shaped chunks
          - other events (ignored)
        Raises:
          - ValueError for in-stream error objects
        """
        # In-stream error per spec (surface as exception)
        if "code" in event_obj or "error" in event_obj:
            raise ValueError(json.dumps(event_obj))

        # FINAL RESULT IS *NOT* TERMINAL: treat it as the next chunk
        if "final_result" in event_obj:
            fr = event_obj["final_result"] or {}
            # ensure it looks like an OpenAI chunk
            if "object" not in fr:
                fr["object"] = "chat.completion.chunk"
            return OpenAIChatCompletionChunk.model_validate(fr)

        # Orchestration incremental delta
        if "orchestration_result" in event_obj:
            return _StreamParser._from_orchestration_result(event_obj)

        # Already an OpenAI-like chunk
        if "choices" in event_obj and "object" in event_obj:
            return OpenAIChatCompletionChunk.model_validate(event_obj)

        # Unknown / heartbeat / metrics
        return None


# -------------------------------
# Iterators
# -------------------------------
class SAPStreamIterator:
    """
    Sync iterator over an httpx streaming response that yields OpenAIChatCompletionChunk.
    Accepts both SSE `data: ...` and raw JSON lines. Closes on terminal chunk or [DONE].
    """

    def __init__(
        self,
        response: Iterator,
        event_prefix: str = "data: ",
        final_msg: str = "[DONE]",
    ):
        self._resp = response
        self._iter = response
        self._prefix = event_prefix
        self._final = final_msg
        self._done = False

    def __iter__(self) -> Iterator[OpenAIChatCompletionChunk]:
        return self

    def __next__(self) -> OpenAIChatCompletionChunk:
        if self._done:
            raise StopIteration

        for raw in self._iter:
            line = (raw or "").strip()
            if not line:
                continue

            payload = (
                line[len(self._prefix) :] if line.startswith(self._prefix) else line
            )
            if payload == self._final:
                self._safe_close()
                raise StopIteration

            try:
                obj = json.loads(payload)
            except Exception:
                continue

            try:
                chunk = _StreamParser.to_openai_chunk(obj)
            except ValueError as e:
                self._safe_close()
                raise e

            if chunk is None:
                continue

            # Close on terminal
            if _is_terminal_chunk(chunk):
                self._safe_close()

            return chunk

        self._safe_close()
        raise StopIteration

    def _safe_close(self) -> None:
        if self._done:
            return
        else:
            self._done = True


class AsyncSAPStreamIterator:
    sync_stream = False

    def __init__(
        self,
        response:AsyncIterator,
        event_prefix: str = "data: ",
        final_msg: str = "[DONE]",
    ):
        self._resp = response
        self._prefix = event_prefix
        self._final = final_msg
        self._line_iter = None
        self._done = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._done:
            raise StopAsyncIteration

        if self._line_iter is None:
            self._line_iter = self._resp

        while True:
            try:
                raw = await self._line_iter.__anext__()
            except (StopAsyncIteration, httpx.ReadError, OSError):
                await self._aclose()
                raise StopAsyncIteration

            line = (raw or "").strip()
            if not line:
                continue

            # now = lambda: int(time.time() * 1000)
            payload = (
                line[len(self._prefix) :] if line.startswith(self._prefix) else line
            )
            if payload == self._final:
                await self._aclose()
                raise StopAsyncIteration
            try:
                obj = json.loads(payload)
            except Exception:
                continue

            try:
                chunk = _StreamParser.to_openai_chunk(obj)
            except ValueError as e:
                await self._aclose()
                raise GenAIHubOrchestrationError(502, str(e))

            if chunk is None:
                continue

            # If terminal, close BEFORE returning. Next __anext__() will stop immediately.
            if any(c.finish_reason is not None for c in (chunk.choices or [])):
                await self._aclose()

            return chunk

    async def _aclose(self):
        if self._done:
            return
        else:
            self._done = True


# -------------------------------
# LLM handler
# -------------------------------
class GenAIHubOrchestration(BaseLLMHTTPHandler):
    def _add_stream_param_to_request_body(
            self,
            data: dict,
            provider_config: BaseConfig,
            fake_stream: bool
            ):
        if data.get("config", {}).get("stream", None) is not None:
            data["config"]["stream"]["enabled"] = True
        else:
            data["config"]["stream"] = {"enabled": True}
        return data

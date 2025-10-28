from __future__ import annotations

import json
import time
from functools import cached_property
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union, Callable

import httpx
from aiohttp import ClientSession

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObject
from litellm.types.utils import ModelResponse
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.llms.openai import OpenAIChatCompletionChunk

from ..credentials import get_token_creator
from ...base import BaseLLM


# -------------------------------
# Errors
# -------------------------------
class GenAIHubOrchestrationError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
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
        response: httpx.Response,
        event_prefix: str = "data: ",
        final_msg: str = "[DONE]",
    ):
        self._resp = response
        self._iter = response.iter_lines()
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
        try:
            self._resp.close()
        finally:
            self._done = True


class AsyncSAPStreamIterator:
    sync_stream = False

    def __init__(
        self,
        response: httpx.Response,
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
            self._line_iter = self._resp.aiter_lines()

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
        try:
            await self._resp.aclose()
        finally:
            self._done = True


# -------------------------------
# LLM handler
# -------------------------------
class GenAIHubOrchestration(BaseLLM):
    def __init__(self) -> None:
        super().__init__()
        self.token_creator = None
        self._base_url = None
        self._resource_group = None

    def run_env_setup(self) -> None:
        try:
            self.token_creator, self._base_url, self._resource_group = get_token_creator() # type: ignore
        except ValueError as err:
            raise GenAIHubOrchestrationError(status_code=400, message=err.args[0])


    @property
    def headers(self) -> Dict[str, str]:
        if self.token_creator is None:
            self.run_env_setup()
        access_token = self.token_creator() # type: ignore
        return {
            "Authorization": access_token,
            "AI-Resource-Group": self.resource_group,
            "Content-Type": "application/json",
        }

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            self.run_env_setup()
        return self._base_url # type: ignore


    @property
    def resource_group(self) -> str:
        if self._resource_group is None:
            self.run_env_setup()
        return self._resource_group # type: ignore

    @cached_property
    def deployment_url(self) -> str:
        # Keep a short, tight client lifecycle here to avoid fd leaks
        with httpx.Client(timeout=30) as client:
            deployments = client.get(
                f"{self.base_url}/lm/deployments", headers=self.headers
            ).json()
            valid: List[Tuple[str, str]] = []
            for dep in deployments.get("resources", []):
                if dep.get("scenarioId") == "orchestration":
                    cfg = client.get(
                        f'{self.base_url}/lm/configurations/{dep["configurationId"]}',
                        headers=self.headers,
                    ).json()
                    if cfg.get("executableId") == "orchestration":
                        valid.append((dep["deploymentUrl"], dep["createdAt"]))
            # newest first
            return sorted(valid, key=lambda x: x[1], reverse=True)[0][0]

    def validate_environment(
        self, endpoint_type: Literal["chat_completions", "embeddings"]
    ) -> Tuple[str, Dict[str, str]]:
        api_base = (
            f"{self.deployment_url}/v2/completion"
            if endpoint_type == "chat_completions"
            else f"{self.deployment_url}/v2/embeddings"
        )
        return api_base, self.headers

    # ---------- Async paths ----------
    async def _async_streaming(
        self,
        config: Dict[str, Any],
        model: str,
        api_base: str,
        headers: Dict[str, str],
        model_response: ModelResponse,
        print_verbose: Callable,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        optional_params: Dict[str, Any],
        extra_headers: Optional[Dict[str, str]] = None,
        shared_session: Optional[ClientSession] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> CustomStreamWrapper:
        try:
            # Ensure the server returns SSE:
            hdrs = dict(headers)
            hdrs.setdefault("Accept", "text/event-stream")
            hdrs.setdefault("Cache-Control", "no-cache")
            hdrs.setdefault("Connection", "keep-alive")
            hdrs.setdefault("Accept-Encoding", "identity")

            client = litellm.AsyncHTTPHandler(shared_session=shared_session)
            resp = await client.post(
                url=api_base, headers=hdrs, json=config, timeout=timeout
            )
            resp.raise_for_status()
            completion_stream = AsyncSAPStreamIterator(resp)

        except httpx.HTTPStatusError as err:
            raise GenAIHubOrchestrationError(
                err.response.status_code, err.response.text
            )
        except httpx.TimeoutException:
            raise GenAIHubOrchestrationError(408, "Timeout error occurred.")
        except ValueError as in_stream_err:
            raise GenAIHubOrchestrationError(502, str(in_stream_err))

        return CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            logging_obj=logging_obj,
            stream_options={},
            make_call=None,
        )

    async def _async_completion(
        self,
        config: Dict[str, Any],
        model: str,
        api_base: str,
        headers: Dict[str, str],
        model_response: ModelResponse,
        print_verbose: Callable,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        optional_params: Dict[str, Any],
        extra_headers: Optional[Dict[str, str]] = None,
        client: Optional[AsyncHTTPHandler] = None,
        shared_session: Optional[ClientSession] = None,
    ) -> ModelResponse:
        try:
            if client is None or not isinstance(client, AsyncHTTPHandler):
                client = litellm.AsyncHTTPHandler(shared_session=shared_session)
            resp = await client.post(
                url=api_base, headers=headers, json=config, timeout=timeout
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as err:
            raise GenAIHubOrchestrationError(
                err.response.status_code, err.response.text
            )
        except httpx.TimeoutException:
            raise GenAIHubOrchestrationError(408, "Timeout error occurred.")

        return litellm.GenAIHubOrchestrationConfig()._transform_response(response=resp)

    # ---------- Sync paths ----------
    def _streaming(
        self,
        config: Dict[str, Any],
        model: str,
        api_base: str,
        headers: Dict[str, str],
        model_response: ModelResponse,
        print_verbose: Callable,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        optional_params: Dict[str, Any],
        extra_headers: Optional[Dict[str, str]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> CustomStreamWrapper:
        try:
            if client is None or not isinstance(client, HTTPHandler):
                client = litellm.module_level_client

            resp = client.post(
                url=api_base,
                headers=headers,
                json=config,
                stream=True,
                timeout=timeout,
            )
            resp.raise_for_status()

            completion_stream = SAPStreamIterator(resp)

        except httpx.HTTPStatusError as err:
            raise GenAIHubOrchestrationError(
                err.response.status_code, err.response.text
            )
        except httpx.TimeoutException:
            raise GenAIHubOrchestrationError(408, "Timeout error occurred.")
        except ValueError as in_stream_err:
            raise GenAIHubOrchestrationError(502, str(in_stream_err))

        return CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            logging_obj=logging_obj,
            stream_options={},
            make_call=None,
        )

    def _complete(
        self,
        config: Dict[str, Any],
        model: str,
        api_base: str,
        headers: Dict[str, str],
        model_response: ModelResponse,
        print_verbose: Callable,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        optional_params: Dict[str, Any],
        extra_headers: Optional[Dict[str, str]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> ModelResponse:
        try:
            if client is None or not isinstance(client, HTTPHandler):
                client = litellm.module_level_client
            resp = client.post(
                url=api_base, headers=headers, json=config, timeout=timeout
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as err:
            raise GenAIHubOrchestrationError(
                err.response.status_code, err.response.text
            )
        except httpx.TimeoutException:
            raise GenAIHubOrchestrationError(408, "Timeout error occurred.")

        return litellm.GenAIHubOrchestrationConfig()._transform_response(response=resp)

    # ---------- entrypoint ----------
    def completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        logging_obj: LiteLLMLoggingObject,
        optional_params: Dict[str, Any],
        acompletion: bool,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        litellm_params: Optional[Dict[str, Any]] = None,
        logger_fn=None,
        extra_headers: Optional[Dict[str, str]] = None,
        shared_session: Optional[ClientSession] = None,
        client: Optional[Union[AsyncHTTPHandler, HTTPHandler]] = None,
        **kwargs,
    ):
        stream = optional_params.get("stream", None)
        api_base, hdrs = self.validate_environment("chat_completions")

        config = litellm.GenAIHubOrchestrationConfig()._transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
        )

        # logging
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": config,
                "api_base": api_base,
                "headers": hdrs,
            },
        )

        if acompletion:
            if stream:
                return self._async_streaming(
                    config=config,
                    model=model,
                    api_base=api_base,
                    headers=hdrs,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    timeout=timeout,
                    encoding=encoding,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    extra_headers=extra_headers,
                    shared_session=shared_session,
                    client=client if isinstance(client, AsyncHTTPHandler) else None,
                )
            return self._async_completion(
                config=config,
                model=model,
                api_base=api_base,
                headers=hdrs,
                model_response=model_response,
                print_verbose=print_verbose,
                timeout=timeout,
                encoding=encoding,
                logging_obj=logging_obj,
                optional_params=optional_params,
                extra_headers=extra_headers,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
                shared_session=shared_session,
            )

        if stream:
            return self._streaming(
                config=config,
                model=model,
                api_base=api_base,
                headers=hdrs,
                model_response=model_response,
                print_verbose=print_verbose,
                timeout=timeout,
                encoding=encoding,
                logging_obj=logging_obj,
                optional_params=optional_params,
                extra_headers=extra_headers,
                client=client if isinstance(client, HTTPHandler) else None,
            )

        return self._complete(
            config=config,
            model=model,
            api_base=api_base,
            headers=hdrs,
            model_response=model_response,
            print_verbose=print_verbose,
            timeout=timeout,
            encoding=encoding,
            logging_obj=logging_obj,
            optional_params=optional_params,
            extra_headers=extra_headers,
            client=client if isinstance(client, HTTPHandler) else None,
        )

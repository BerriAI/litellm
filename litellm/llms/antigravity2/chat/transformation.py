import asyncio
import os
import queue
import threading
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator, List, Optional, Tuple, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.utils import (
    AllMessageValues,
    GenericStreamingChunk,
    ModelResponse,
    Usage,
)


@dataclass
class Antigravity2SDK:
    Agent: Any
    LocalAgentConfig: Any
    types: Any


class Antigravity2Config(BaseConfig):
    """LiteLLM provider for the official Antigravity 2.0 Python SDK.

    Antigravity 2.0 is exposed by Google as a local agent runtime. Authentication
    is owned by that runtime (system keyring / Google sign-in) or by Gemini API / Vertex
    settings configured on the server, not by forwarding end-user API keys.
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "antigravity2"

    def get_supported_openai_params(self, model: str) -> list:
        return ["stream", "reasoning_effort"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        if "stream" in non_default_params:
            optional_params["stream"] = non_default_params["stream"]
        if "reasoning_effort" in non_default_params:
            optional_params["reasoning_effort"] = non_default_params["reasoning_effort"]
        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return headers

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        return {"model": model, "messages": messages, **optional_params}

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: Any,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def _load_sdk(self) -> Antigravity2SDK:
        try:
            from google import antigravity as google_antigravity
            from google.antigravity import types as ag_types
        except ImportError as exc:
            raise ImportError(
                "The antigravity2 provider requires Google's official Antigravity 2.0 SDK. "
                "Install it on the LiteLLM server with `pip install google-antigravity`."
            ) from exc
        return Antigravity2SDK(
            Agent=google_antigravity.Agent,
            LocalAgentConfig=google_antigravity.LocalAgentConfig,
            types=ag_types,
        )

    def _message_content_to_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        return str(content)

    def _split_system_and_prompt(self, messages: List[AllMessageValues]) -> Tuple[Optional[str], str]:
        system_parts: List[str] = []
        transcript_parts: List[str] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = self._message_content_to_text(message.get("content"))
            if role in {"system", "developer"}:
                if content:
                    system_parts.append(content)
                continue
            transcript_parts.append(f"{role}: {content}")
        prompt = "\n\n".join(transcript_parts).strip()
        return ("\n\n".join(system_parts) or None, prompt)

    def _thinking_level(self, sdk: Antigravity2SDK, optional_params: dict) -> Optional[Any]:
        value = optional_params.get("reasoning_effort")
        if value is None:
            return None
        mapping = {
            "minimal": sdk.types.ThinkingLevel.MINIMAL,
            "low": sdk.types.ThinkingLevel.LOW,
            "medium": sdk.types.ThinkingLevel.MEDIUM,
            "high": sdk.types.ThinkingLevel.HIGH,
        }
        return mapping.get(str(value).lower())

    def _build_local_agent_config(
        self,
        sdk: Antigravity2SDK,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        api_key: Optional[str],
    ) -> Tuple[Any, str]:
        system_instructions, prompt = self._split_system_and_prompt(messages)
        thinking_level = self._thinking_level(sdk, optional_params)
        generation = sdk.types.GenerationConfig(thinking_level=thinking_level)
        model_entry = sdk.types.ModelEntry(name=model, api_key=api_key, generation=generation)
        gemini_config = sdk.types.GeminiConfig(
            api_key=api_key or os.getenv("ANTIGRAVITY2_API_KEY"),
            vertex=os.getenv("ANTIGRAVITY2_VERTEX", "").lower() in {"1", "true", "yes"},
            project=os.getenv("ANTIGRAVITY2_PROJECT"),
            location=os.getenv("ANTIGRAVITY2_LOCATION"),
            models=sdk.types.ModelConfig(default=model_entry),
        )
        capabilities = sdk.types.CapabilitiesConfig(
            enabled_tools=sdk.types.BuiltinTools.none(),
            enable_subagents=False,
        )
        kwargs = {
            "system_instructions": system_instructions,
            "capabilities": capabilities,
            "policies": [],
            "workspaces": [],
            "gemini_config": gemini_config,
            "model": model,
            "api_key": api_key or os.getenv("ANTIGRAVITY2_API_KEY"),
        }
        app_data_dir = os.getenv("ANTIGRAVITY2_APP_DATA_DIR")
        if app_data_dir:
            kwargs["app_data_dir"] = app_data_dir
        return sdk.LocalAgentConfig(**kwargs), prompt

    def _usage_from_metadata(self, usage_metadata: Any) -> Usage:
        if usage_metadata is None:
            return Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        prompt_tokens = int(getattr(usage_metadata, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage_metadata, "candidates_token_count", 0) or 0)
        total_tokens = int(getattr(usage_metadata, "total_token_count", 0) or (prompt_tokens + completion_tokens))
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    async def _achat_text(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        api_key: Optional[str],
    ) -> Tuple[str, Usage]:
        sdk = self._load_sdk()
        config, prompt = self._build_local_agent_config(sdk, model, messages, optional_params, api_key)
        async with sdk.Agent(config) as agent:
            response = await agent.chat(prompt)
            text = await response.text()
            return text, self._usage_from_metadata(getattr(response, "usage_metadata", None))

    async def _achat_stream(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        api_key: Optional[str],
    ) -> AsyncIterator[GenericStreamingChunk]:
        sdk = self._load_sdk()
        config, prompt = self._build_local_agent_config(sdk, model, messages, optional_params, api_key)
        async with sdk.Agent(config) as agent:
            response = await agent.chat(prompt)
            async for token in response:
                yield {
                    "text": token,
                    "is_finished": False,
                    "finish_reason": "",
                    "usage": None,
                }
            yield {
                "text": "",
                "is_finished": True,
                "finish_reason": "stop",
                "usage": self._usage_from_metadata(getattr(response, "usage_metadata", None)),
            }

    def completion(
        self,
        model: str,
        messages: List[AllMessageValues],
        model_response: ModelResponse,
        optional_params: dict,
        api_key: Optional[str],
        logging_obj: Any,
        custom_llm_provider: str,
        acompletion: bool = False,
    ) -> Any:
        from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

        if optional_params.get("stream") is True:
            stream = self._achat_stream(model, messages, optional_params, api_key)
            if acompletion:
                return CustomStreamWrapper(
                    stream, model=model, custom_llm_provider=custom_llm_provider, logging_obj=logging_obj
                )
            return CustomStreamWrapper(
                self._sync_from_async_stream(stream),
                model=model,
                custom_llm_provider=custom_llm_provider,
                logging_obj=logging_obj,
            )

        if acompletion:
            return self._acompletion(model, messages, model_response, optional_params, api_key)
        text, usage = self._run_sync(self._achat_text(model, messages, optional_params, api_key))
        model_response.choices[0].message.content = text
        model_response.usage = usage
        return model_response

    async def _acompletion(
        self,
        model: str,
        messages: List[AllMessageValues],
        model_response: ModelResponse,
        optional_params: dict,
        api_key: Optional[str],
    ) -> ModelResponse:
        text, usage = await self._achat_text(model, messages, optional_params, api_key)
        model_response.choices[0].message.content = text
        model_response.usage = usage
        return model_response

    def _run_sync(self, coro: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError("Use litellm.acompletion() for antigravity2 calls from an active event loop")

    def _sync_from_async_stream(
        self, async_stream: AsyncIterator[GenericStreamingChunk]
    ) -> Iterator[GenericStreamingChunk]:
        items: "queue.Queue[Any]" = queue.Queue()
        sentinel = object()

        async def produce() -> None:
            try:
                async for chunk in async_stream:
                    items.put(chunk)
            except BaseException as exc:
                items.put(exc)
            finally:
                items.put(sentinel)

        thread = threading.Thread(target=lambda: asyncio.run(produce()), daemon=True)
        thread.start()
        while True:
            item = items.get()
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            yield item

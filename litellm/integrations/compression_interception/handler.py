"""
Compression Interception Handler for /v1/messages.

CustomLogger that applies prompt compression in async_pre_request_hook and
executes litellm_content_retrieve tool calls in the Anthropic agentic loop.
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import litellm
from litellm._logging import verbose_logger
from litellm.anthropic_interface import messages as anthropic_messages
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.integrations.compression_interception import (
    CompressionInterceptionConfig,
)
from litellm.types.llms.anthropic import AllAnthropicMessageValues
from litellm.types.utils import LlmProviders

_RETRIEVAL_TOOL_NAME = "litellm_content_retrieve"
_COMPRESSION_CACHE_KEY = "_compression_interception_cache"


def _is_retrieval_tool(tool: Dict[str, Any]) -> bool:
    if not isinstance(tool, dict):
        return False
    if tool.get("name") == _RETRIEVAL_TOOL_NAME:
        return True
    if tool.get("type") == "function":
        function_obj = tool.get("function", {})
        if isinstance(function_obj, dict):
            return function_obj.get("name") == _RETRIEVAL_TOOL_NAME
    return False


def _merge_tools(
    existing_tools: Optional[List[Dict[str, Any]]],
    new_tools: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen_names = set()

    for tool in (existing_tools or []) + (new_tools or []):
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not name and tool.get("type") == "function":
            function_obj = tool.get("function", {})
            if isinstance(function_obj, dict):
                name = function_obj.get("name")
        dedupe_key = name or str(tool)
        if dedupe_key in seen_names:
            continue
        seen_names.add(dedupe_key)
        merged.append(tool)

    return merged


def _get_cache_from_kwargs(kwargs: Dict[str, Any]) -> Dict[str, str]:
    cache = kwargs.get(_COMPRESSION_CACHE_KEY)
    if isinstance(cache, dict):
        return cast(Dict[str, str], cache)

    litellm_params = kwargs.get("litellm_params")
    if isinstance(litellm_params, dict):
        nested_cache = litellm_params.get(_COMPRESSION_CACHE_KEY)
        if isinstance(nested_cache, dict):
            return cast(Dict[str, str], nested_cache)
    return {}


def _extract_tool_call_key(tool_call: Dict[str, Any]) -> str:
    if "input" in tool_call and isinstance(tool_call["input"], dict):
        return str(tool_call["input"].get("key", ""))

    function_obj = tool_call.get("function", {})
    if isinstance(function_obj, dict):
        arguments = function_obj.get("arguments", {})
        if isinstance(arguments, dict):
            return str(arguments.get("key", ""))
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                if isinstance(parsed, dict):
                    return str(parsed.get("key", ""))
            except json.JSONDecodeError:
                return ""
    return ""


class CompressionInterceptionLogger(CustomLogger):
    def __init__(
        self,
        enabled_providers: Optional[List[Union[LlmProviders, str]]] = None,
        compression_trigger: int = 200_000,
        compression_target: Optional[int] = None,
        embedding_model: Optional[str] = None,
        embedding_model_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.enabled_providers = (
            [p.value if isinstance(p, LlmProviders) else p for p in enabled_providers]
            if enabled_providers
            else None
        )
        self.compression_trigger = compression_trigger
        self.compression_target = compression_target
        self.embedding_model = embedding_model
        self.embedding_model_params = embedding_model_params

    def _resolve_provider(self, kwargs: Dict[str, Any], model: str) -> str:
        provider = kwargs.get("custom_llm_provider", "") or kwargs.get(
            "litellm_params", {}
        ).get("custom_llm_provider", "")
        if provider:
            return str(provider)
        try:
            _, provider, _, _ = litellm.get_llm_provider(model=model)
            return str(provider)
        except Exception:
            return ""

    def _provider_enabled(self, provider: str) -> bool:
        if self.enabled_providers is None:
            return True
        return provider in self.enabled_providers

    def _apply_compression(
        self, model: str, messages: List[Dict[str, Any]], kwargs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        result = litellm.compress(
            messages=cast(List[AllAnthropicMessageValues], messages),
            model=model,
            input_type="anthropic_messages",
            compression_trigger=self.compression_trigger,
            compression_target=self.compression_target,
            embedding_model=self.embedding_model,
            embedding_model_params=self.embedding_model_params,
        )

        if result["compression_ratio"] <= 0 and not result["cache"]:
            return None

        modified_kwargs = dict(kwargs)
        modified_kwargs["messages"] = result["messages"]
        modified_kwargs["tools"] = _merge_tools(
            cast(Optional[List[Dict[str, Any]]], kwargs.get("tools")),
            cast(Optional[List[Dict[str, Any]]], result.get("tools")),
        )
        modified_kwargs[_COMPRESSION_CACHE_KEY] = result["cache"]

        litellm_params = modified_kwargs.get("litellm_params")
        if isinstance(litellm_params, dict):
            litellm_params = dict(litellm_params)
        else:
            litellm_params = {}
        litellm_params[_COMPRESSION_CACHE_KEY] = result["cache"]
        modified_kwargs["litellm_params"] = litellm_params

        # Reuse existing fake-stream conversion path used by agentic callbacks.
        if modified_kwargs.get("stream"):
            modified_kwargs["stream"] = False
            modified_kwargs["_websearch_interception_converted_stream"] = True

        verbose_logger.debug(
            "CompressionInterception: compressed request "
            "original_tokens=%s compressed_tokens=%s ratio=%s",
            result["original_tokens"],
            result["compressed_tokens"],
            result["compression_ratio"],
        )
        return modified_kwargs

    async def async_pre_request_hook(
        self, model: str, messages: List[Dict], kwargs: Dict
    ) -> Optional[Dict]:
        provider = self._resolve_provider(kwargs, model=model)
        if not self._provider_enabled(provider):
            return None

        return self._apply_compression(
            model=model,
            messages=cast(List[Dict[str, Any]], messages),
            kwargs=kwargs,
        )

    def _extract_anthropic_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        if isinstance(response, dict):
            content = response.get("content", []) or []
        else:
            content = getattr(response, "content", []) or []

        tool_calls: List[Dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            if block.get("name") != _RETRIEVAL_TOOL_NAME:
                continue
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": block.get("input", {}),
                }
            )
        return tool_calls

    async def async_should_run_agentic_loop(
        self,
        response: Any,
        model: str,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        stream: bool,
        custom_llm_provider: str,
        kwargs: Dict,
    ) -> Tuple[bool, Dict]:
        if not self._provider_enabled(custom_llm_provider):
            return False, {}
        if not any(_is_retrieval_tool(t) for t in (tools or [])):
            return False, {}
        cache = _get_cache_from_kwargs(kwargs)
        if not cache:
            return False, {}

        tool_calls = self._extract_anthropic_tool_calls(response)
        if not tool_calls:
            return False, {}
        return True, {"tool_calls": tool_calls, "cache": cache}

    async def async_run_agentic_loop(
        self,
        tools: Dict,
        model: str,
        messages: List[Dict],
        response: Any,
        anthropic_messages_provider_config: Any,
        anthropic_messages_optional_request_params: Dict,
        logging_obj: Any,
        stream: bool,
        kwargs: Dict,
    ) -> Any:
        tool_calls = cast(List[Dict[str, Any]], tools.get("tool_calls", []))
        cache = cast(Dict[str, str], tools.get("cache", {}))

        assistant_blocks: List[Dict[str, Any]] = []
        tool_result_blocks: List[Dict[str, Any]] = []

        for tool_call in tool_calls:
            key = _extract_tool_call_key(tool_call)
            tool_call_id = str(tool_call.get("id", ""))
            assistant_blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_call_id,
                    "name": _RETRIEVAL_TOOL_NAME,
                    "input": {"key": key},
                }
            )
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": cache.get(key, f"[key {key!r} not found in cache]"),
                }
            )

        follow_up_messages = messages + [
            {"role": "assistant", "content": assistant_blocks},
            {"role": "user", "content": tool_result_blocks},
        ]

        kwargs_for_followup = {
            k: v
            for k, v in kwargs.items()
            if not k.startswith("_compression_interception")
            and not k.startswith("_websearch_interception")
            and k != "litellm_logging_obj"
        }
        optional_params_without_max_tokens = {
            k: v
            for k, v in anthropic_messages_optional_request_params.items()
            if k != "max_tokens"
        }

        max_tokens = anthropic_messages_optional_request_params.get(
            "max_tokens", kwargs.get("max_tokens", 1024)
        )
        full_model_name = model
        if logging_obj is not None:
            agentic_params = logging_obj.model_call_details.get(
                "agentic_loop_params", {}
            )
            full_model_name = agentic_params.get("model", model)

        return await anthropic_messages.acreate(
            max_tokens=max_tokens,
            messages=follow_up_messages,
            model=full_model_name,
            **optional_params_without_max_tokens,
            **kwargs_for_followup,
        )

    @classmethod
    def from_config_yaml(
        cls, config: CompressionInterceptionConfig
    ) -> "CompressionInterceptionLogger":
        enabled_providers = config.get("enabled_providers")
        return cls(
            enabled_providers=enabled_providers,
            compression_trigger=config.get("compression_trigger", 200_000),
            compression_target=config.get("compression_target"),
            embedding_model=config.get("embedding_model"),
            embedding_model_params=config.get("embedding_model_params"),
        )

    @staticmethod
    def initialize_from_proxy_config(
        litellm_settings: Dict[str, Any],
        callback_specific_params: Dict[str, Any],
    ) -> "CompressionInterceptionLogger":
        compression_params: CompressionInterceptionConfig = {}
        if "compression_interception_params" in litellm_settings:
            compression_params = litellm_settings["compression_interception_params"]
        elif "compression_interception" in callback_specific_params:
            compression_params = callback_specific_params["compression_interception"]

        return CompressionInterceptionLogger.from_config_yaml(compression_params)

"""
Compression Interception Handler

CustomLogger that compresses inbound Anthropic Messages requests and fulfills
litellm_content_retrieve tool calls server-side via the typed agentic loop plan.
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, cast

from litellm._logging import verbose_logger
from litellm.compression import compress
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.integrations.compression_interception import (
    CompressionInterceptionConfig,
)
from litellm.types.integrations.custom_logger import (
    AgenticLoopPlan,
    AgenticLoopRequestPatch,
)
from litellm.types.utils import CallTypes

LITELLM_CONTENT_RETRIEVE_TOOL_NAME = "litellm_content_retrieve"
_CACHE_TTL_SECONDS = 15 * 60


class CompressionInterceptionLogger(CustomLogger):
    """
    CustomLogger that implements transparent prompt compression + retrieval loops.

    Flow:
    1. Compress inbound /v1/messages requests in pre-call hook.
    2. Inject litellm_content_retrieve tool and persist compressed cache by call_id.
    3. Detect retrieval tool_use blocks in first model response.
    4. Build typed rerun plan with tool_result blocks from the compressed cache.
    """

    def __init__(
        self,
        enabled: bool = True,
        compression_trigger: int = 200_000,
        compression_target: Optional[int] = None,
        embedding_model: Optional[str] = None,
        embedding_model_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.enabled = enabled
        self.compression_trigger = compression_trigger
        self.compression_target = compression_target
        self.embedding_model = embedding_model
        self.embedding_model_params = embedding_model_params
        self._compression_cache_by_call_id: Dict[str, Tuple[Dict[str, str], float]] = {}

    @classmethod
    def from_config_yaml(
        cls, config: CompressionInterceptionConfig
    ) -> "CompressionInterceptionLogger":
        return cls(
            enabled=bool(config.get("enabled", True)),
            compression_trigger=int(config.get("compression_trigger", 200_000)),
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

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Optional[CallTypes]
    ) -> Optional[dict]:
        if not self.enabled:
            return None
        if call_type is not None and call_type != CallTypes.anthropic_messages:
            return None
        if int(kwargs.get("_agentic_loop_depth", 0) or 0) > 0:
            return None

        messages = kwargs.get("messages")
        model = kwargs.get("model")
        if not isinstance(messages, list) or not isinstance(model, str):
            return None

        if self._has_retrieval_tool(kwargs.get("tools")):
            return None

        self._prune_expired_cache()

        compressed = compress(  # type: ignore
            messages=messages,
            model=model,
            call_type=CallTypes.anthropic_messages,
            compression_trigger=self.compression_trigger,
            compression_target=self.compression_target,
            embedding_model=self.embedding_model,
            embedding_model_params=self.embedding_model_params,
        )

        cache = cast(Dict[str, str], compressed.get("cache", {}))
        skip_reason = cast(Optional[str], compressed.get("compression_skipped_reason"))
        compressed_tools = cast(List[Dict[str, Any]], compressed.get("tools", []))

        # Only mutate kwargs when compression actually produced a result.
        # If compression was a no-op (below trigger, invalid tool sequence, etc.),
        # leave ``messages`` and ``tools`` untouched — injecting an empty
        # ``tools: []`` onto a request that originally had no tools breaks
        # Anthropic Messages requests.
        if cache:
            kwargs["messages"] = compressed["messages"]
            if compressed_tools:
                kwargs["tools"] = self._merge_tools(
                    existing_tools=cast(
                        Optional[List[Dict[str, Any]]], kwargs.get("tools")
                    ),
                    compressed_tools=compressed_tools,
                )
            call_id = cast(Optional[str], kwargs.get("litellm_call_id"))
            if not call_id:
                call_id = str(uuid.uuid4())
                kwargs["litellm_call_id"] = call_id
            self._compression_cache_by_call_id[call_id] = (cache, time.time())
            verbose_logger.debug(
                "CompressionInterception: compressed request [call_id=%s original=%d compressed=%d cached_keys=%d]",
                call_id,
                compressed.get("original_tokens"),
                compressed.get("compressed_tokens"),
                len(cache),
            )
        elif skip_reason is not None:
            verbose_logger.debug(
                "CompressionInterception: compression skipped [reason=%s original=%d compressed=%d]",
                skip_reason,
                compressed.get("original_tokens"),
                compressed.get("compressed_tokens"),
            )

        return kwargs

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
        if not self.enabled:
            return False, {}
        if not self._has_retrieval_tool(tools):
            return False, {}

        tool_calls, thinking_blocks = self._extract_retrieval_tool_calls(
            response=response
        )
        if not tool_calls:
            return False, {}

        return True, {
            "tool_calls": tool_calls,
            "thinking_blocks": thinking_blocks,
            "tool_type": "compression_retrieval",
        }

    async def async_build_agentic_loop_plan(
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
    ) -> AgenticLoopPlan:
        self._prune_expired_cache()
        tool_calls = cast(List[Dict[str, Any]], tools.get("tool_calls", []))
        thinking_blocks = cast(List[Dict[str, Any]], tools.get("thinking_blocks", []))

        call_id = self._resolve_call_id(logging_obj=logging_obj, kwargs=kwargs)
        cache = self._get_cache(call_id=call_id)
        retrieval_results = [
            self._resolve_retrieval_content(tc, cache) for tc in tool_calls
        ]

        assistant_message = {
            "role": "assistant",
            "content": thinking_blocks
            + [
                {
                    "type": "tool_use",
                    "id": tc.get("id"),
                    "name": tc.get("name", LITELLM_CONTENT_RETRIEVE_TOOL_NAME),
                    "input": tc.get("input", {}),
                }
                for tc in tool_calls
            ],
        }
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_calls[i].get("id"),
                    "content": retrieval_results[i],
                }
                for i in range(len(tool_calls))
            ],
        }
        follow_up_messages = messages + [assistant_message, user_message]

        max_tokens = cast(
            Optional[int],
            anthropic_messages_optional_request_params.get("max_tokens")
            or kwargs.get("max_tokens"),
        )
        optional_params_without_max_tokens = {
            k: v
            for k, v in anthropic_messages_optional_request_params.items()
            if k != "max_tokens"
        }

        full_model_name = model
        if logging_obj is not None:
            agentic_params = logging_obj.model_call_details.get(
                "agentic_loop_params", {}
            )
            full_model_name = cast(str, agentic_params.get("model", model))

        request_patch = AgenticLoopRequestPatch(
            model=full_model_name,
            messages=follow_up_messages,
            max_tokens=max_tokens,
            optional_params=optional_params_without_max_tokens,
            kwargs=self._prepare_followup_kwargs(kwargs=kwargs),
        )

        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=request_patch,
            metadata={"tool_type": "compression_retrieval", "call_id": call_id or ""},
        )

    def _prune_expired_cache(self) -> None:
        now = time.time()
        self._compression_cache_by_call_id = {
            call_id: (cache, created_at)
            for call_id, (
                cache,
                created_at,
            ) in self._compression_cache_by_call_id.items()
            if now - created_at <= _CACHE_TTL_SECONDS
        }

    def _get_cache(self, call_id: Optional[str]) -> Dict[str, str]:
        if not call_id:
            return {}
        cache_entry = self._compression_cache_by_call_id.get(call_id)
        if cache_entry is None:
            return {}
        return cache_entry[0]

    def _resolve_call_id(
        self, logging_obj: Any, kwargs: Dict[str, Any]
    ) -> Optional[str]:
        if logging_obj is not None:
            logging_call_id = getattr(logging_obj, "litellm_call_id", None)
            if isinstance(logging_call_id, str) and logging_call_id:
                return logging_call_id
        kwargs_call_id = kwargs.get("litellm_call_id")
        return cast(
            Optional[str], kwargs_call_id if isinstance(kwargs_call_id, str) else None
        )

    def _resolve_retrieval_content(
        self, tool_call: Dict[str, Any], cache: Dict[str, str]
    ) -> str:
        raw_input = tool_call.get("input", {})
        key = ""
        if isinstance(raw_input, dict):
            key = str(raw_input.get("key", "") or "")
        if not key:
            return "No retrieval key provided."
        if key in cache:
            return cache[key]
        return f"[compressed content key '{key}' not found]"

    def _extract_retrieval_tool_calls(
        self, response: Any
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if isinstance(response, dict):
            content = response.get("content", [])
        else:
            content = getattr(response, "content", []) or []

        if not isinstance(content, list):
            return [], []

        tool_calls: List[Dict[str, Any]] = []
        thinking_blocks: List[Dict[str, Any]] = []

        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                block_name = block.get("name")
                if block_type in ("thinking", "redacted_thinking"):
                    thinking_blocks.append(block)
                if (
                    block_type == "tool_use"
                    and block_name == LITELLM_CONTENT_RETRIEVE_TOOL_NAME
                ):
                    tool_calls.append(
                        {
                            "id": block.get("id"),
                            "type": "tool_use",
                            "name": block_name,
                            "input": block.get("input", {}),
                        }
                    )
            else:
                block_type = getattr(block, "type", None)
                block_name = getattr(block, "name", None)
                if block_type == "thinking":
                    thinking_blocks.append(
                        {
                            "type": "thinking",
                            "thinking": getattr(block, "thinking", ""),
                            "signature": getattr(block, "signature", ""),
                        }
                    )
                elif block_type == "redacted_thinking":
                    thinking_blocks.append(
                        {
                            "type": "redacted_thinking",
                            "data": getattr(block, "data", ""),
                        }
                    )
                if (
                    block_type == "tool_use"
                    and block_name == LITELLM_CONTENT_RETRIEVE_TOOL_NAME
                ):
                    tool_calls.append(
                        {
                            "id": getattr(block, "id", None),
                            "type": "tool_use",
                            "name": block_name,
                            "input": getattr(block, "input", {}) or {},
                        }
                    )

        return tool_calls, thinking_blocks

    def _prepare_followup_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        internal_keys = {"litellm_logging_obj"}
        return {
            k: v
            for k, v in kwargs.items()
            if not k.startswith("_compression_interception") and k not in internal_keys
        }

    def _has_retrieval_tool(self, tools: Any) -> bool:
        if not isinstance(tools, list):
            return False
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if tool.get("type") == "function" and isinstance(function, dict):
                if function.get("name") == LITELLM_CONTENT_RETRIEVE_TOOL_NAME:
                    return True
            if (
                tool.get("type") == "custom"
                and tool.get("name") == LITELLM_CONTENT_RETRIEVE_TOOL_NAME
            ):
                return True
        return False

    def _merge_tools(
        self,
        existing_tools: Optional[List[Dict[str, Any]]],
        compressed_tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged = list(existing_tools or [])
        if self._has_retrieval_tool(merged):
            return merged
        merged.extend(compressed_tools)
        return merged

"""
Tool Policy Guardrail

Reads call_policy from LiteLLM_ToolTable and enforces it on LLM requests/responses.

Policy values:
  "trusted"   - allow through (no action)
  "untrusted" - allow through (no action; default for newly discovered tools)
  "blocked"   - raise HTTPException, preventing the tool call
  "dual_llm"  - (Phase 3) send to second LLM for verification; currently treated as allowed

Configuration in proxy config YAML:
  guardrails:
    - guardrail_name: "tool_policy"
      litellm_params:
        guardrail: tool_policy
        mode: post_call

or both pre and post call:
    - guardrail_name: "tool_policy"
      litellm_params:
        guardrail: tool_policy
        mode: during_call  # runs before LLM and on response
"""

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.constants import TOOL_POLICY_CACHE_TTL_SECONDS
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

GUARDRAIL_NAME = "tool_policy"


class ToolPolicyGuardrail(CustomGuardrail):
    """
    Guardrail that enforces per-tool call policies stored in LiteLLM_ToolTable.

    Tools with call_policy="blocked" are rejected before/after the LLM call.
    Tools with call_policy="trusted" or "untrusted" pass through unchanged.
    """

    def __init__(self, **kwargs: Any) -> None:
        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.during_call,
            ]
        super().__init__(**kwargs)
        self._policy_cache: DualCache = DualCache()

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Enforce tool policies on both request tools and response tool_calls.

        - input_type="request":  check inputs["tools"] (tool definitions in the LLM request)
        - input_type="response": check inputs["tool_calls"] (tool_calls in the LLM response)

        Raises HTTPException (400) if any tool is "blocked".
        """
        if input_type == "request":
            tools = inputs.get("tools") or []
            tool_names = [
                t["function"]["name"]
                for t in tools
                if isinstance(t, dict)
                and isinstance(t.get("function"), dict)
                and t["function"].get("name")
            ]
        else:  # response
            tool_calls = inputs.get("tool_calls") or []
            tool_names = []
            for tc in tool_calls:
                fn = None
                if isinstance(tc, dict):
                    fn = (tc.get("function") or {}).get("name")
                elif hasattr(tc, "function"):
                    fn = getattr(tc.function, "name", None)
                if fn:
                    tool_names.append(fn)

        if not tool_names:
            return inputs

        policy_map = await self._get_policies_cached(tool_names)

        blocked = [name for name in tool_names if policy_map.get(name) == "blocked"]
        if blocked:
            verbose_proxy_logger.warning(
                "ToolPolicyGuardrail: blocking tool(s) %s (policy=blocked)", blocked
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Violated tool policy",
                    "blocked_tools": blocked,
                    "message": f"Tool(s) {blocked} are blocked by policy.",
                },
            )

        return inputs

    async def _get_policies_cached(self, tool_names: List[str]) -> Dict[str, str]:
        """
        Batch-fetch call_policy for the given tool names.

        Caches per individual tool name (not per combination) so that adding
        a new tool to a request doesn't invalidate the cached policies for all
        the other tools already in the cache.
        """
        from litellm.proxy.db.tool_registry_writer import get_tools_by_names
        from litellm.proxy.proxy_server import prisma_client

        if not tool_names or prisma_client is None:
            return {}

        result: Dict[str, str] = {}
        cache_misses: List[str] = []

        for name in tool_names:
            cached = await self._policy_cache.async_get_cache(f"tool_policy:{name}")
            if cached is not None and isinstance(cached, str):
                result[name] = cached
            else:
                cache_misses.append(name)

        if cache_misses:
            fetched = await get_tools_by_names(
                prisma_client=prisma_client, tool_names=cache_misses
            )
            for name, policy in fetched.items():
                result[name] = policy
                await self._policy_cache.async_set_cache(
                    key=f"tool_policy:{name}",
                    value=policy,
                    ttl=TOOL_POLICY_CACHE_TTL_SECONDS,
                )
            verbose_proxy_logger.debug(
                "ToolPolicyGuardrail: fetched %d policies from DB (cache hits: %d)",
                len(cache_misses),
                len(tool_names) - len(cache_misses),
            )

        return result

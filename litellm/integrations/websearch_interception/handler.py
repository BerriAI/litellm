"""
WebSearch Interception Handler

CustomLogger that intercepts WebSearch tool calls for models that don't
natively support web search (e.g., Bedrock/Claude) and executes them
server-side using litellm router's search tools.
"""

import asyncio
import json
import math
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import litellm
from litellm._logging import verbose_logger
from litellm.anthropic_interface import messages as anthropic_messages
from litellm.constants import LITELLM_WEB_SEARCH_TOOL_NAME
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.websearch_interception.tools import (
    get_litellm_web_search_tool,
    get_litellm_web_search_tool_openai,
    get_litellm_web_search_tool_responses_api,
    is_anthropic_native_web_search_tool,
    is_web_search_tool,
    is_web_search_tool_chat_completion,
    is_web_search_tool_responses_api,
)
from litellm.integrations.websearch_interception.transformation import (
    WebSearchTransformation,
)
from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.types.integrations.websearch_interception import (
    WebSearchInterceptionConfig,
)
from litellm.types.integrations.custom_logger import (
    CHAT_COMPLETION_AGENTIC_SURFACE,
    AgenticLoopPlan,
    AgenticLoopRequestPatch,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

# Key used to flag, on per-request kwargs, that the originating client sent
# an Anthropic-native ``web_search_*`` tool — meaning the final response
# should include ``web_search_tool_result`` content blocks so the client
# (e.g. Claude Desktop's citations panel) can render sources.
WEBSEARCH_EMIT_NATIVE_BLOCKS_KEY = "_websearch_interception_emit_native_blocks"

# Key on ``AgenticLoopPlan.metadata`` carrying the list of pre-built
# ``web_search_tool_result`` blocks to inject into the final response.
WEBSEARCH_NATIVE_BLOCKS_METADATA_KEY = "websearch_native_blocks"


class WebSearchInterceptionLogger(CustomLogger):
    """
    CustomLogger that intercepts WebSearch tool calls for models that don't
    natively support web search.

    Implements agentic loop:
    1. Detects WebSearch tool_use in model response
    2. Executes litellm.asearch() for each query using router's search tools
    3. Makes follow-up request with search results
    4. Returns final response
    """

    def __init__(
        self,
        enabled_providers: Optional[List[Union[LlmProviders, str]]] = None,
        search_tool_name: Optional[str] = None,
    ):
        """
        Args:
            enabled_providers: List of LLM providers to enable interception for.
                              Use LlmProviders enum values (e.g., [LlmProviders.BEDROCK])
                              If None or empty list, enables for ALL providers.
                              Default: None (all providers enabled)
            search_tool_name: Name of search tool configured in router's search_tools.
                             If None, will attempt to use first available search tool.
        """
        super().__init__()
        # Convert enum values to strings for comparison
        if enabled_providers is None:
            self.enabled_providers = [LlmProviders.BEDROCK.value]
        else:
            self.enabled_providers = [p.value if isinstance(p, LlmProviders) else p for p in enabled_providers]
        self.search_tool_name = search_tool_name
        self._request_has_websearch = False  # Track if current request has web search

    async def try_short_circuit_search(
        self,
        model: str,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        custom_llm_provider: Optional[str],
        kwargs: Optional[dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Short-circuit web-search-only requests by executing the search directly.

        Claude Code sends web search as a separate, standalone /v1/messages
        request with a simple prompt and only web_search tool(s). For providers
        that don't natively support web search (e.g. github_copilot), there is
        no need to route this through the backend LLM — we can detect the
        pattern, execute the search via Tavily/Perplexity, and return a
        synthetic Anthropic response immediately.

        Args:
            model: Model name from the request
            messages: Messages list from the request
            tools: Tools list from the request
            custom_llm_provider: Provider name

        Returns:
            An AnthropicMessagesResponse dict if short-circuited, or None to
            continue normal processing.
        """
        if not tools:
            return None

        # Check if provider is in enabled list
        provider_str = custom_llm_provider or ""
        if self.enabled_providers is not None and provider_str not in self.enabled_providers:
            return None

        # Only short-circuit for providers whose Anthropic Messages agentic loop
        # does not run web_search itself. Providers that have a
        # BaseAnthropicMessagesConfig which handles web search natively (bedrock,
        # vertex_ai, azure_ai, anthropic) already perform the search plus a
        # follow-up LLM synthesis step; short-circuiting those would skip that
        # synthesis and return raw search text — a regression for existing users.
        #
        # github_copilot has a BaseAnthropicMessagesConfig (added for thinking
        # passthrough) but does not handle web_search natively, so its config
        # returns handles_web_search_natively() == False and we still short-circuit
        # web-search-only requests against it.
        try:
            provider_enum = LlmProviders(provider_str)
            anthropic_config = ProviderConfigManager.get_provider_anthropic_messages_config(
                model=model, provider=provider_enum
            )
            if anthropic_config is not None and anthropic_config.handles_web_search_natively():
                verbose_logger.debug(
                    f"WebSearchInterception: Skipping short-circuit for {provider_str} "
                    "(provider handles web search natively via the agentic loop)"
                )
                return None
        except (ValueError, Exception):
            pass  # unknown provider enum → safe to short-circuit

        # All tools must be web search tools
        if not all(is_web_search_tool(t) for t in tools):
            return None

        # Extract search query from the last user message
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            get_last_user_message,
        )

        query = get_last_user_message(cast(List[AllMessageValues], messages))
        if not query:
            return None

        verbose_logger.debug(
            f"WebSearchInterception: Short-circuit search detected (provider={provider_str}, query='{query}')"
        )

        # Native clients (Claude Desktop / Cowork / Anthropic SDK) make a
        # standalone /v1/messages sub-request just for the search, and they
        # expect the response in native shape with server_tool_use +
        # web_search_tool_result content blocks so the citations panel can
        # render. The agentic-loop post-hook never fires on this path because
        # there is no model call — emit the native blocks here instead.
        native_tool = next(
            (t for t in tools if is_anthropic_native_web_search_tool(t)),
            None,
        )

        # Execute search — keep the structured SearchResponse so the native
        # block can carry per-result url/title/page_age.
        try:
            if kwargs is None:
                search_result_text, structured = await self._execute_search(query)
            else:
                search_result_text, structured = await self._execute_search(query, kwargs=kwargs)
        except Exception as e:
            verbose_logger.error(f"WebSearchInterception: Short-circuit search failed: {e}")
            search_result_text, structured = f"Search failed: {e}", None

        content: List[Dict[str, Any]] = []
        if native_tool is not None:
            tool_use_id = f"srvtoolu_{uuid.uuid4().hex}"
            tool_name = native_tool.get("name") or "web_search"
            content.append(
                {
                    "type": "server_tool_use",
                    "id": tool_use_id,
                    "name": tool_name,
                    "input": {"query": query},
                }
            )
            content.append(
                WebSearchTransformation.build_web_search_tool_result_block(
                    tool_use_id=tool_use_id,
                    search_response=structured,
                )
            )
        # Keep the text block so non-native short-circuit callers (Claude Code,
        # github_copilot, etc.) see the same payload they always have.
        content.append({"type": "text", "text": search_result_text})

        response: Dict[str, Any] = {
            "id": f"msg_{str(uuid.uuid4())}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

        verbose_logger.debug(
            "WebSearchInterception: Short-circuit search completed, "
            f"returning synthetic response ({len(search_result_text)} chars, "
            f"native_blocks={native_tool is not None})"
        )
        return response

    async def async_pre_call_deployment_hook(self, kwargs: Dict[str, Any], call_type: Optional[Any]) -> Optional[dict]:
        """
        Pre-call hook to convert native Anthropic web_search tools to regular tools.

        This prevents Bedrock from trying to execute web search server-side (which fails).
        Instead, we convert it to a regular tool so the model returns tool_use blocks
        that we can intercept and execute ourselves.
        """
        # Check if this is for an enabled provider
        # Try top-level kwargs first, then nested litellm_params, then derive from model name
        custom_llm_provider = kwargs.get("custom_llm_provider", "") or kwargs.get("litellm_params", {}).get(
            "custom_llm_provider", ""
        )
        if not custom_llm_provider:
            try:
                _, custom_llm_provider, _, _ = litellm.get_llm_provider(model=kwargs.get("model", ""))
            except Exception:
                custom_llm_provider = ""
        if custom_llm_provider not in self.enabled_providers:
            return None

        # Check if request has tools with native web_search
        tools = kwargs.get("tools")
        if not tools:
            return None

        # Branch on call surface. Chat Completions tools use the OpenAI-nested
        # ``{"type": "function", "function": {...}}`` shape. Responses-API tools
        # are flat (``{"type": "function", "name": "..."}``) and the server-hosted
        # web search variant is ``{"type": "web_search"}`` — neither of which
        # ``is_web_search_tool`` matches.
        if call_type is None:
            call_type_str = ""
        else:
            call_type_str = getattr(call_type, "value", None) or str(call_type)
        is_responses_api_call = call_type_str == "aresponses"

        if is_responses_api_call:
            tool_predicate = is_web_search_tool_responses_api
            standard_tool_factory = get_litellm_web_search_tool_responses_api
        else:
            tool_predicate = is_web_search_tool
            standard_tool_factory = get_litellm_web_search_tool_openai

        has_websearch = any(tool_predicate(t) for t in tools)

        if not has_websearch:
            return None

        verbose_logger.debug(
            "WebSearchInterception: Converting native web_search tools to LiteLLM standard "
            f"(call_type={call_type_str or 'unknown'})"
        )

        # If the client sent an Anthropic-native web_search_* tool, mark the
        # request so the agentic loop emits native web_search_tool_result
        # blocks in the final response (matches async_pre_request_hook). This
        # deployment hook fires before async_pre_request_hook on some paths,
        # so flagging here ensures the signal isn't lost regardless of order.
        if any(is_anthropic_native_web_search_tool(t) for t in tools):
            kwargs[WEBSEARCH_EMIT_NATIVE_BLOCKS_KEY] = True

        # Convert native/custom web_search tools to LiteLLM standard
        converted_tools = []
        for tool in tools:
            if tool_predicate(tool):
                converted_tool = standard_tool_factory()
                converted_tools.append(converted_tool)
                verbose_logger.debug(
                    f"WebSearchInterception: Converted {tool.get('name', 'unknown')} "
                    f"(type={tool.get('type', 'none')}) to {LITELLM_WEB_SEARCH_TOOL_NAME}"
                )
            else:
                # Keep other tools as-is
                converted_tools.append(tool)

        kwargs["tools"] = converted_tools

        if kwargs.get("stream"):
            verbose_logger.debug("WebSearchInterception: deployment hook converting stream=True to stream=False")
            kwargs["stream"] = False
            kwargs["_websearch_interception_converted_stream"] = True

        return kwargs

    @classmethod
    def from_config_yaml(cls, config: WebSearchInterceptionConfig) -> "WebSearchInterceptionLogger":
        """
        Initialize WebSearchInterceptionLogger from proxy config.yaml parameters.

        Args:
            config: Configuration dictionary from litellm_settings.websearch_interception_params

        Returns:
            Configured WebSearchInterceptionLogger instance

        Example:
            From proxy_config.yaml:
                litellm_settings:
                  websearch_interception_params:
                    enabled_providers: ["bedrock"]
                    search_tool_name: "my-perplexity-search"

            Usage:
                config = litellm_settings.get("websearch_interception_params", {})
                logger = WebSearchInterceptionLogger.from_config_yaml(config)
        """
        # Extract parameters from config
        enabled_providers_str = config.get("enabled_providers", None)
        search_tool_name = config.get("search_tool_name", None)

        # Convert string provider names to LlmProviders enum values
        enabled_providers: Optional[List[Union[LlmProviders, str]]] = None
        if enabled_providers_str is not None:
            enabled_providers = []
            for provider in enabled_providers_str:
                try:
                    # Try to convert string to LlmProviders enum
                    provider_enum = LlmProviders(provider)
                    enabled_providers.append(provider_enum)
                except ValueError:
                    # If conversion fails, keep as string
                    enabled_providers.append(provider)

        return cls(
            enabled_providers=enabled_providers,
            search_tool_name=search_tool_name,
        )

    @staticmethod
    def _tool_name(tool: dict[str, Any]) -> Optional[str]:
        """Effective tool name, handling OpenAI ``function`` wrapper shape."""
        fn = tool.get("function")
        if tool.get("type") == "function" and isinstance(fn, dict):
            return fn.get("name")
        return tool.get("name")

    @classmethod
    def _sync_forced_tool_choice(cls, tool_choice: Any, converted_tools: list[dict[str, Any]]) -> Any:
        """Repoint a forced ``tool_choice`` at ``litellm_web_search`` when it
        names a web-search tool that was just converted away.

        Native clients (e.g. Claude Code) force the search tool via
        ``tool_choice={"type": "tool", "name": "web_search"}``. Since the tool
        definition gets renamed to ``litellm_web_search``, an unrewritten
        ``tool_choice`` points at a tool that no longer exists, which Anthropic
        rejects with "Tool 'web_search' not found in provided tools".
        """
        if not isinstance(tool_choice, dict) or tool_choice.get("type") != "tool":
            return tool_choice
        converted_names = {cls._tool_name(t) for t in converted_tools}
        if tool_choice.get("name") in converted_names:
            return tool_choice
        return {**tool_choice, "name": LITELLM_WEB_SEARCH_TOOL_NAME}

    async def async_pre_request_hook(self, model: str, messages: List[Dict], kwargs: Dict) -> Optional[Dict]:
        """
        Pre-request hook to convert native web search tools to LiteLLM standard.

        This hook is called before the API request is made, allowing us to:
        1. Detect native web search tools (web_search_20250305, etc.)
        2. Convert them to LiteLLM standard format (litellm_web_search)
        3. Convert stream=True to stream=False for interception

        This prevents providers like Bedrock from trying to execute web search
        natively (which fails), and ensures our agentic loop can intercept tool_use.

        Returns:
            Modified kwargs dict with converted tools, or None if no modifications needed
        """
        # Check if this request is for an enabled provider
        custom_llm_provider = kwargs.get("litellm_params", {}).get("custom_llm_provider", "")

        verbose_logger.debug(
            f"WebSearchInterception: Pre-request hook called"
            f" - custom_llm_provider={custom_llm_provider}"
            f" - enabled_providers={self.enabled_providers or 'ALL'}"
        )

        if self.enabled_providers is not None and custom_llm_provider not in self.enabled_providers:
            verbose_logger.debug(
                f"WebSearchInterception: Skipping - provider {custom_llm_provider} not in {self.enabled_providers}"
            )
            return None

        # Check if request has tools
        tools = kwargs.get("tools")
        if not tools:
            return None

        # Check if any tool is a web search tool
        has_websearch = any(is_web_search_tool(t) for t in tools)
        if not has_websearch:
            return None

        verbose_logger.debug(f"WebSearchInterception: Pre-request hook triggered for provider={custom_llm_provider}")

        # If the client sent an Anthropic-native web_search_* tool, mark the
        # request so the agentic loop emits native web_search_tool_result
        # blocks in the final response (for citations panels, etc.). The flag
        # is read by async_build_agentic_loop_plan; the leading underscore
        # prefix ensures it is stripped before the follow-up call kwargs.
        if any(is_anthropic_native_web_search_tool(t) for t in tools):
            kwargs[WEBSEARCH_EMIT_NATIVE_BLOCKS_KEY] = True

        # Convert native web search tools to LiteLLM standard
        converted_tools = []
        for tool in tools:
            if is_web_search_tool(tool):
                standard_tool = get_litellm_web_search_tool()
                converted_tools.append(standard_tool)
                verbose_logger.debug(
                    f"WebSearchInterception: Converted {tool.get('name', 'unknown')} "
                    f"(type={tool.get('type', 'none')}) to {LITELLM_WEB_SEARCH_TOOL_NAME}"
                )
            else:
                converted_tools.append(tool)

        kwargs["tools"] = converted_tools
        verbose_logger.debug(
            f"WebSearchInterception: Tools after conversion: {[t.get('name') for t in converted_tools]}"
        )

        if "tool_choice" in kwargs:
            kwargs["tool_choice"] = self._sync_forced_tool_choice(kwargs.get("tool_choice"), converted_tools)

        # Also convert here for direct callers that bypass the deployment hook.
        if kwargs.get("stream"):
            verbose_logger.debug("WebSearchInterception: Converting stream=True to stream=False")
            kwargs["stream"] = False
            kwargs["_websearch_interception_converted_stream"] = True

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
        if kwargs.get("_agentic_loop_api_surface") == CHAT_COMPLETION_AGENTIC_SURFACE:
            return await self.async_should_run_chat_completion_agentic_loop(
                response=response,
                model=model,
                messages=messages,
                tools=tools,
                stream=stream,
                custom_llm_provider=custom_llm_provider,
                kwargs=kwargs,
            )

        verbose_logger.debug(f"WebSearchInterception: Hook called! provider={custom_llm_provider}, stream={stream}")
        verbose_logger.debug(f"WebSearchInterception: Response type: {type(response)}")

        # Check if provider should be intercepted
        # Note: custom_llm_provider is already normalized by get_llm_provider()
        # (e.g., "bedrock/invoke/..." -> "bedrock")
        if self.enabled_providers is not None and custom_llm_provider not in self.enabled_providers:
            verbose_logger.debug(
                f"WebSearchInterception: Skipping provider {custom_llm_provider} (not in enabled list: {self.enabled_providers})"
            )
            return False, {}

        # Check if tools include any web search tool (LiteLLM standard or native)
        has_websearch_tool = any(is_web_search_tool(t) for t in (tools or []))
        if not has_websearch_tool:
            verbose_logger.debug("WebSearchInterception: No web search tool in request")
            return False, {}

        # Detect WebSearch tool_use in response (Anthropic format)
        should_intercept, tool_calls = WebSearchTransformation.transform_request(
            response=response,
            stream=stream,
            response_format="anthropic",
        )

        if not should_intercept:
            verbose_logger.debug("WebSearchInterception: No WebSearch tool_use detected in response")
            return False, {}

        verbose_logger.debug(
            f"WebSearchInterception: Detected {len(tool_calls)} WebSearch tool call(s), executing agentic loop"
        )

        # Extract thinking blocks from response content.
        # When extended thinking is enabled, the model response includes
        # thinking/redacted_thinking blocks that must be preserved and
        # prepended to the follow-up assistant message.
        thinking_blocks: List[Dict] = []
        if isinstance(response, dict):
            content = response.get("content", [])
        else:
            content = getattr(response, "content", []) or []

        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
            else:
                block_type = getattr(block, "type", None)

            if block_type in ("thinking", "redacted_thinking"):
                if isinstance(block, dict):
                    thinking_blocks.append(block)
                else:
                    # Convert object to dict using getattr, matching the
                    # pattern in _detect_from_non_streaming_response
                    thinking_block_dict: Dict = {"type": block_type}
                    if block_type == "thinking":
                        thinking_block_dict["thinking"] = getattr(block, "thinking", "")
                        thinking_block_dict["signature"] = getattr(block, "signature", "")
                    else:  # redacted_thinking
                        thinking_block_dict["data"] = getattr(block, "data", "")
                    thinking_blocks.append(thinking_block_dict)

        if thinking_blocks:
            verbose_logger.debug(
                f"WebSearchInterception: Extracted {len(thinking_blocks)} thinking block(s) from response"
            )

        # Return tools dict with tool calls and thinking blocks
        tools_dict = {
            "tool_calls": tool_calls,
            "tool_type": "websearch",
            "provider": custom_llm_provider,
            "response_format": "anthropic",
            "thinking_blocks": thinking_blocks,
        }
        return True, tools_dict

    async def async_should_run_chat_completion_agentic_loop(
        self,
        response: Any,
        model: str,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        stream: bool,
        custom_llm_provider: str,
        kwargs: Dict,
    ) -> Tuple[bool, Dict]:
        """
        Check if WebSearch tool interception is needed for Chat Completions API.

        Similar to async_should_run_agentic_loop but for OpenAI-style chat completions.
        """

        verbose_logger.debug(
            f"WebSearchInterception: Chat completion hook called! provider={custom_llm_provider}, stream={stream}"
        )
        verbose_logger.debug(f"WebSearchInterception: Response type: {type(response)}")

        # Check if provider should be intercepted
        if self.enabled_providers is not None and custom_llm_provider not in self.enabled_providers:
            verbose_logger.debug(
                f"WebSearchInterception: Skipping provider {custom_llm_provider} (not in enabled list: {self.enabled_providers})"
            )
            return False, {}

        # Check if tools include any web search tool (strict check for chat completions)
        has_websearch_tool = any(is_web_search_tool_chat_completion(t) for t in (tools or []))
        if not has_websearch_tool:
            verbose_logger.debug("WebSearchInterception: No litellm_web_search tool in request")
            return False, {}

        # Detect WebSearch tool_calls in response (OpenAI format)
        should_intercept, tool_calls = WebSearchTransformation.transform_request(
            response=response,
            stream=stream,
            response_format="openai",
        )

        if not should_intercept:
            verbose_logger.debug("WebSearchInterception: No WebSearch tool_calls detected in response")
            return False, {}

        verbose_logger.debug(
            f"WebSearchInterception: Detected {len(tool_calls)} WebSearch tool call(s), executing agentic loop"
        )

        # Return tools dict with tool calls
        tools_dict = {
            "tool_calls": tool_calls,
            "tool_type": "websearch",
            "provider": custom_llm_provider,
            "response_format": "openai",
        }
        return True, tools_dict

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
        """
        Execute agentic loop with WebSearch execution for Anthropic Messages API.

        This is the legacy method for Anthropic-style responses.
        """

        tool_calls = tools["tool_calls"]
        thinking_blocks = tools.get("thinking_blocks", [])

        verbose_logger.debug(f"WebSearchInterception: Executing agentic loop for {len(tool_calls)} search(es)")

        return await self._execute_agentic_loop(
            model=model,
            messages=messages,
            tool_calls=tool_calls,
            thinking_blocks=thinking_blocks,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            logging_obj=logging_obj,
            stream=stream,
            kwargs=kwargs,
        )

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
        if kwargs.get("_agentic_loop_api_surface") == CHAT_COMPLETION_AGENTIC_SURFACE:
            return await self.async_build_chat_completion_agentic_loop_plan(
                tools=tools,
                model=model,
                messages=messages,
                response=response,
                optional_params=anthropic_messages_optional_request_params,
                logging_obj=logging_obj,
                stream=stream,
                kwargs=kwargs,
            )

        tool_calls = tools["tool_calls"]
        thinking_blocks = tools.get("thinking_blocks", [])
        request_patch, structured_results = await self._build_anthropic_request_patch(
            model=model,
            messages=messages,
            tool_calls=tool_calls,
            thinking_blocks=thinking_blocks,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            logging_obj=logging_obj,
            kwargs=kwargs,
        )

        metadata: Dict[str, Any] = {
            "tool_type": "websearch",
            "response_format": "anthropic",
        }

        # If the client request originally carried a native web_search_* tool,
        # pre-build the Anthropic-native ``web_search_tool_result`` blocks now
        # (while we still have the structured SearchResponse list) and stash
        # them on plan metadata for the post-hook to inject.
        if kwargs.get(WEBSEARCH_EMIT_NATIVE_BLOCKS_KEY):
            metadata[WEBSEARCH_NATIVE_BLOCKS_METADATA_KEY] = self._build_native_result_blocks(
                tool_calls=tool_calls,
                structured_results=structured_results,
            )

        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=request_patch,
            metadata=metadata,
        )

    async def async_post_agentic_loop_response_hook(
        self,
        response: Any,
        plan: AgenticLoopPlan,
        kwargs: Dict,
    ) -> Any:
        """
        Inject Anthropic-native ``web_search_tool_result`` blocks into the
        final response when the originating client used a native
        ``web_search_*`` tool.

        See ``WebSearchTransformation.build_web_search_tool_result_block`` for
        the block shape. The blocks are prepended to ``response.content`` so
        Anthropic-native clients (Claude Desktop, the Anthropic SDK) can
        render citations / sources alongside the model's textual reply.
        """
        native_blocks = plan.metadata.get(WEBSEARCH_NATIVE_BLOCKS_METADATA_KEY)
        if not native_blocks:
            return response
        return self._inject_native_blocks(response, native_blocks)

    @staticmethod
    def _build_native_result_blocks(
        tool_calls: List[Dict],
        structured_results: List[Optional[SearchResponse]],
    ) -> List[Dict[str, Any]]:
        """Build one ``web_search_tool_result`` block per tool_call."""
        blocks: List[Dict[str, Any]] = []
        for i, tool_call in enumerate(tool_calls):
            tool_use_id = tool_call.get("id") or ""
            structured = structured_results[i] if i < len(structured_results) else None
            blocks.append(
                WebSearchTransformation.build_web_search_tool_result_block(
                    tool_use_id=tool_use_id,
                    search_response=structured,
                )
            )
        return blocks

    @staticmethod
    def _inject_native_blocks(response: Any, native_blocks: List[Dict[str, Any]]) -> Any:
        """Prepend native blocks to response content, dict or object form."""
        if not native_blocks:
            return response
        if isinstance(response, dict):
            existing = response.get("content") or []
            response["content"] = list(native_blocks) + list(existing)
            return response
        existing = getattr(response, "content", None) or []
        try:
            response.content = list(native_blocks) + list(existing)
        except (AttributeError, TypeError):
            # Object refused write — fall through and leave the response
            # untouched rather than crash the request.
            verbose_logger.debug(
                f"WebSearchInterception: could not inject native blocks into response of type {type(response).__name__}"
            )
        return response

    async def async_run_chat_completion_agentic_loop(
        self,
        tools: Dict,
        model: str,
        messages: List[Dict],
        response: Any,
        optional_params: Dict,
        logging_obj: Any,
        stream: bool,
        kwargs: Dict,
    ) -> Any:
        """
        Execute agentic loop with WebSearch execution for Chat Completions API.

        Similar to async_run_agentic_loop but for OpenAI-style chat completions.
        """

        tool_calls = tools["tool_calls"]
        response_format = tools.get("response_format", "openai")

        verbose_logger.debug(
            f"WebSearchInterception: Executing chat completion agentic loop for {len(tool_calls)} search(es)"
        )

        return await self._execute_chat_completion_agentic_loop(
            model=model,
            messages=messages,
            tool_calls=tool_calls,
            optional_params=optional_params,
            logging_obj=logging_obj,
            stream=stream,
            kwargs=kwargs,
            response_format=response_format,
        )

    async def async_build_chat_completion_agentic_loop_plan(
        self,
        tools: Dict,
        model: str,
        messages: List[Dict],
        response: Any,
        optional_params: Dict,
        logging_obj: Any,
        stream: bool,
        kwargs: Dict,
    ) -> AgenticLoopPlan:
        tool_calls = tools["tool_calls"]
        response_format = tools.get("response_format", "openai")
        request_patch = await self._build_chat_completion_request_patch(
            model=model,
            messages=messages,
            tool_calls=tool_calls,
            optional_params=optional_params,
            kwargs=kwargs,
            response_format=response_format,
        )
        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=request_patch,
            metadata={"tool_type": "websearch", "response_format": response_format},
        )

    async def async_should_run_responses_api_agentic_loop(
        self,
        response: Any,
        model: str,
        input: Any,
        tools: Optional[List[Dict]],
        stream: bool,
        custom_llm_provider: str,
        kwargs: Dict,
        original_stream: Optional[bool] = None,
    ) -> Tuple[bool, Dict]:
        """Detect ``litellm_web_search`` ``function_call`` items in a Responses-API response."""
        verbose_logger.debug(
            f"WebSearchInterception: Responses-API hook called! provider={custom_llm_provider}, stream={stream}"
        )

        if (
            self.enabled_providers is not None
            and custom_llm_provider not in self.enabled_providers
        ):
            verbose_logger.debug(
                f"WebSearchInterception: Skipping provider {custom_llm_provider} "
                f"(not in enabled list: {self.enabled_providers})"
            )
            return False, {}

        has_websearch_tool = any(
            is_web_search_tool_responses_api(t) for t in (tools or [])
        )
        if not has_websearch_tool:
            verbose_logger.debug(
                "WebSearchInterception: No web_search tool in Responses-API request"
            )
            return False, {}

        should_intercept, tool_calls = WebSearchTransformation.transform_request(
            response=response,
            stream=stream,
            response_format="responses",
        )

        if not should_intercept:
            verbose_logger.debug(
                "WebSearchInterception: No litellm_web_search function_call in Responses-API output"
            )
            return False, {}

        verbose_logger.debug(
            f"WebSearchInterception: Detected {len(tool_calls)} Responses-API function_call(s), "
            "executing agentic loop"
        )
        return True, {
            "tool_calls": tool_calls,
            "tool_type": "websearch",
            "provider": custom_llm_provider,
            "response_format": "responses",
        }

    async def async_run_responses_api_agentic_loop(  # noqa: PLR0915
        self,
        tools: Dict,
        model: str,
        input: Any,
        response: Any,
        response_api_optional_request_params: Dict,
        litellm_params: Dict,
        logging_obj: Any,
        stream: bool,
        kwargs: Dict,
        original_stream: Optional[bool] = None,
    ) -> Any:
        """Execute searches and re-run the Responses-API call with ``function_call_output`` items."""
        tool_calls = tools["tool_calls"]

        # Check depth + fingerprint cycle BEFORE issuing any work. Otherwise a
        # client that drives the model into emitting ``litellm_web_search``
        # calls past the cap still gets ``len(tool_calls)`` parallel Tavily
        # requests per iteration before the loop aborts. Mirrors
        # ``_check_agentic_loop_safety`` in the chat-completion path, which
        # also runs before any rerun work.
        depth = int(kwargs.get("_agentic_loop_depth", 0) or 0)
        max_loops = max(int(kwargs.get("max_agentic_loops", 3) or 3), 1)
        fingerprints = list(kwargs.get("_agentic_loop_fingerprints", []) or [])
        try:
            fingerprint = json.dumps(tool_calls, sort_keys=True, default=str)
        except (TypeError, ValueError):
            fingerprint = str(tool_calls)
        if fingerprint in fingerprints:
            raise ValueError(
                "Responses-API agentic loop detected repeated tool-call "
                "fingerprint; aborting rerun"
            )
        if depth >= max_loops:
            raise ValueError(
                f"Responses-API agentic loop exceeded max_agentic_loops={max_loops} for model={model}"
            )

        verbose_logger.debug(
            f"WebSearchInterception: Executing Responses-API agentic loop for {len(tool_calls)} search(es)"
        )

        # Run searches in parallel.
        search_tasks = []
        for tool_call in tool_calls:
            query = (tool_call.get("input") or {}).get("query")
            if query:
                search_tasks.append(self._execute_search(query))
            else:
                search_tasks.append(self._create_empty_search_result())
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        result_texts: List[str] = []
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                verbose_logger.error(
                    f"WebSearchInterception: Search {i} failed: {str(result)}"
                )
                result_texts.append(f"Search failed: {str(result)}")
            elif isinstance(result, tuple) and len(result) == 2:
                text_value, _ = result
                result_texts.append(
                    cast(str, text_value)
                    if isinstance(text_value, str)
                    else str(text_value)
                )
            else:
                result_texts.append(str(result))

        # Build follow-up ``input`` chain. Responses-API ``input`` accepts
        # mixed message + function_call + function_call_output items. Strict
        # providers (OpenAI-native) validate that every ``function_call_output``
        # is preceded in the conversation by its assistant ``function_call``,
        # and reject the follow-up otherwise. Forward the first response's
        # full ``output`` (which already includes the function_call items
        # alongside reasoning / text / message blocks) so the assistant turn
        # stays intact, then append our paired ``function_call_output`` items.
        if isinstance(input, str):
            follow_up_input: List[Dict[str, Any]] = [
                {"role": "user", "content": input},
            ]
        elif isinstance(input, list):
            follow_up_input = list(input)
        else:
            follow_up_input = []

        first_response_output = (
            response.get("output", []) or []
            if isinstance(response, dict)
            else (getattr(response, "output", None) or [])
        )
        for item in first_response_output:
            follow_up_input.append(
                item if isinstance(item, dict) else self._dump_output_item(item)
            )

        for tool_call, result_text in zip(tool_calls, result_texts):
            call_id = tool_call.get("call_id") or tool_call.get("id") or ""
            follow_up_input.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result_text,
                }
            )

        # Re-run the Responses-API call. ``previous_response_id`` would be a
        # one-line shortcut, but only works when ``store=True`` and isn't
        # supported by all backends — rebuilding the input chain works
        # universally and matches how the chat-completion path handles this.
        followup_kwargs = self._prepare_followup_kwargs(kwargs)
        # Strip Responses-API-only flags and the converted-stream marker.
        followup_kwargs.pop("response_id", None)

        followup_params = dict(response_api_optional_request_params or {})
        followup_params.pop("stream", None)
        followup_params["tools"] = followup_kwargs.pop(
            "tools", followup_params.get("tools")
        )

        # Reconstruct ``provider/model`` for the follow-up call. ``model`` here
        # is the bare backend ID (e.g. ``openai.gpt-5.5``) because the
        # litellm.aresponses dispatcher already stripped the ``bedrock_mantle/``
        # prefix before reaching the HTTP handler. Without the prefix,
        # ``get_llm_provider`` raises ``LLM Provider NOT provided``.
        custom_llm_provider = (
            (litellm_params or {}).get("custom_llm_provider")
            or kwargs.get("custom_llm_provider")
            or ""
        )
        full_model_name = model
        if (
            custom_llm_provider
            and "/" not in model
            and not model.startswith(f"{custom_llm_provider}/")
        ):
            full_model_name = f"{custom_llm_provider}/{model}"

        # Forward provider-specific params from the original ``litellm_params``
        # (e.g. ``aws_region_name``, ``api_base``) so the follow-up call lands
        # on the same backend region as the initial call. Without this, a
        # bedrock_mantle request whose model registration sets
        # ``aws_region_name=us-east-2`` falls back to whatever the global
        # default points at (``BEDROCK_MANTLE_REGION`` env, otherwise
        # us-east-1) and 404s on the follow-up.
        _PROVIDER_PARAM_KEYS = (
            "aws_region_name",
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "aws_role_name",
            "aws_session_name",
            "aws_profile_name",
            "aws_web_identity_token",
            "aws_sts_endpoint",
            "aws_bedrock_runtime_endpoint",
            "api_base",
            "api_key",
            "api_version",
        )
        for k in _PROVIDER_PARAM_KEYS:
            v = (litellm_params or {}).get(k)
            if v is not None and k not in followup_params:
                followup_params[k] = v

        # Forward caller-context fields the proxy uses for budget / spend
        # attribution. Without ``metadata`` / ``litellm_metadata`` / ``user``,
        # the internal follow-up call is logged against an empty key/team and
        # bypasses budgets configured on the original API key.
        # ``litellm_logging_obj`` is intentionally excluded — see
        # ``_prepare_followup_kwargs`` — so the follow-up creates its own.
        _ATTRIBUTION_KEYS = (
            "metadata",
            "litellm_metadata",
            "user",
            "user_api_key",
            "user_api_key_alias",
            "user_api_key_user_id",
            "user_api_key_team_id",
            "user_api_key_team_alias",
            "user_api_key_org_id",
            "proxy_server_request",
        )
        for k in _ATTRIBUTION_KEYS:
            v = (litellm_params or {}).get(k)
            if v is not None and k not in followup_kwargs:
                followup_kwargs[k] = v

        # Loop-state propagation. ``max_agentic_loops`` must travel with the
        # follow-up call so the cap a deployment configured up-front isn't
        # silently reset to the default on each hop.
        followup_kwargs["_agentic_loop_depth"] = depth + 1
        followup_kwargs["_agentic_loop_fingerprints"] = fingerprints + [fingerprint]
        followup_kwargs["max_agentic_loops"] = max_loops

        verbose_logger.debug(
            "WebSearchInterception: Responses-API follow-up call "
            f"[items={len(follow_up_input)} model={full_model_name} "
            f"depth={depth + 1}/{max_loops}]"
        )

        return await litellm.aresponses(
            model=full_model_name,
            input=follow_up_input,
            **followup_params,
            **followup_kwargs,
        )

    @staticmethod
    def _dump_output_item(item: Any) -> Dict[str, Any]:
        """Best-effort conversion of a Responses-API output item to a dict."""
        if hasattr(item, "model_dump"):
            return cast(Dict[str, Any], item.model_dump())
        if hasattr(item, "dict"):
            try:
                return cast(Dict[str, Any], item.dict())
            except Exception:
                pass
        return {k: getattr(item, k) for k in dir(item) if not k.startswith("_")}

    @staticmethod
    def _resolve_max_tokens(
        optional_params: Dict,
        kwargs: Dict,
    ) -> int:
        """Extract max_tokens and validate against thinking.budget_tokens.

        Anthropic API requires ``max_tokens > thinking.budget_tokens``.
        If the constraint is violated, auto-adjust to ``budget_tokens + 1024``.
        """
        max_tokens: int = optional_params.get(
            "max_tokens",
            kwargs.get("max_tokens", 1024),
        )
        thinking_param = optional_params.get("thinking")
        if thinking_param and isinstance(thinking_param, dict):
            budget_tokens = thinking_param.get("budget_tokens")
            if (
                budget_tokens is not None
                and isinstance(budget_tokens, (int, float))
                and math.isfinite(budget_tokens)
                and budget_tokens > 0
            ):
                if max_tokens <= budget_tokens:
                    adjusted = math.ceil(budget_tokens) + 1024
                    verbose_logger.debug(
                        "WebSearchInterception: max_tokens=%s <= thinking.budget_tokens=%s, "
                        "adjusting to %s to satisfy Anthropic API constraint",
                        max_tokens,
                        budget_tokens,
                        adjusted,
                    )
                    max_tokens = adjusted
        return max_tokens

    @staticmethod
    def _prepare_followup_kwargs(kwargs: Dict) -> Dict:
        """Build kwargs for the follow-up call, excluding internal keys.

        ``litellm_logging_obj`` MUST be excluded so the follow-up call creates
        its own ``Logging`` instance via ``function_setup``.  Reusing the
        initial call's logging object triggers the dedup flag
        (``has_logged_async_success``) which silently prevents the initial
        call's spend from being recorded — the root cause of the
        SpendLog / AWS billing mismatch.
        """
        _internal_keys = {"litellm_logging_obj"}
        return {
            k: v for k, v in kwargs.items() if not k.startswith("_websearch_interception") and k not in _internal_keys
        }

    async def _execute_agentic_loop(
        self,
        model: str,
        messages: List[Dict],
        tool_calls: List[Dict],
        thinking_blocks: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        logging_obj: Any,
        stream: bool,
        kwargs: Dict,
    ) -> Any:
        """Legacy path: execute search + build patch + run follow-up call."""
        request_patch, structured_results = await self._build_anthropic_request_patch(
            model=model,
            messages=messages,
            tool_calls=tool_calls,
            thinking_blocks=thinking_blocks,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            logging_obj=logging_obj,
            kwargs=kwargs,
        )
        if request_patch.messages is None:
            raise ValueError("WebSearchInterception: missing follow-up messages")

        optional_params = dict(anthropic_messages_optional_request_params)
        optional_params.update(request_patch.optional_params)
        max_tokens = request_patch.max_tokens
        if max_tokens is None:
            max_tokens = cast(Optional[int], optional_params.pop("max_tokens", None))
        else:
            optional_params.pop("max_tokens", None)
        if max_tokens is None:
            max_tokens = cast(int, kwargs.get("max_tokens", 1024))

        response = await anthropic_messages.acreate(
            max_tokens=max_tokens,
            messages=request_patch.messages,
            model=request_patch.model or model,
            **optional_params,
            **request_patch.kwargs,
        )

        # Legacy path: the new path goes through the typed plan + core
        # dispatcher which runs the post-hook automatically. Mirror the
        # native-block injection here so both paths behave identically.
        if kwargs.get(WEBSEARCH_EMIT_NATIVE_BLOCKS_KEY):
            native_blocks = self._build_native_result_blocks(
                tool_calls=tool_calls,
                structured_results=structured_results,
            )
            response = self._inject_native_blocks(response, native_blocks)

        return response

    async def _build_anthropic_request_patch(
        self,
        model: str,
        messages: List[Dict],
        tool_calls: List[Dict],
        thinking_blocks: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        logging_obj: Any,
        kwargs: Dict,
    ) -> Tuple[AgenticLoopRequestPatch, List[Optional[SearchResponse]]]:
        """
        Execute litellm.search() and build follow-up request patch.

        Returns the patch alongside the parallel list of structured
        ``SearchResponse`` objects (one per tool_call, ``None`` when the
        search failed or the tool_call had no query). The caller uses these
        to optionally build Anthropic-native ``web_search_tool_result``
        content blocks for the final response.
        """

        # Extract search queries from tool_use blocks
        search_tasks = []
        for tool_call in tool_calls:
            query = tool_call["input"].get("query")
            if query:
                verbose_logger.debug(f"WebSearchInterception: Queuing search for query='{query}'")
                search_tasks.append(self._execute_search(query, kwargs=kwargs))
            else:
                verbose_logger.debug(f"WebSearchInterception: Tool call {tool_call['id']} has no query")
                # Add empty result for tools without query
                search_tasks.append(self._create_empty_search_result())

        # Execute searches in parallel
        verbose_logger.debug(f"WebSearchInterception: Executing {len(search_tasks)} search(es) in parallel")
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Split the gathered (text, structured) tuples into two parallel lists.
        # The text list feeds the follow-up model call; the structured list
        # is returned to the caller for native-block emission.
        final_search_results: List[str] = []
        structured_results: List[Optional[SearchResponse]] = []
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                verbose_logger.error(f"WebSearchInterception: Search {i} failed with error: {str(result)}")
                final_search_results.append(f"Search failed: {str(result)}")
                structured_results.append(None)
            elif isinstance(result, tuple) and len(result) == 2:
                text_value, structured_value = result
                final_search_results.append(cast(str, text_value) if isinstance(text_value, str) else str(text_value))
                structured_results.append(structured_value if isinstance(structured_value, SearchResponse) else None)
            else:
                # Defensive: legacy callers / unexpected shape — preserve text,
                # drop structure.
                verbose_logger.debug(f"WebSearchInterception: Unexpected result type {type(result)} at index {i}")
                final_search_results.append(str(result))
                structured_results.append(None)

        # Build assistant and user messages using transformation
        assistant_message, user_message = WebSearchTransformation.transform_response(
            tool_calls=tool_calls,
            search_results=final_search_results,
            thinking_blocks=thinking_blocks,
        )

        follow_up_messages = messages + [assistant_message, cast(Dict, user_message)]

        # Correlation context for structured logging
        _call_id = getattr(logging_obj, "litellm_call_id", None) or kwargs.get("litellm_call_id", "unknown")

        full_model_name = model  # safe default before try block

        max_tokens = self._resolve_max_tokens(anthropic_messages_optional_request_params, kwargs)

        verbose_logger.debug(f"WebSearchInterception: Using max_tokens={max_tokens} for follow-up request")

        optional_params_without_max_tokens = {
            k: v for k, v in anthropic_messages_optional_request_params.items() if k != "max_tokens"
        }
        kwargs_for_followup = self._prepare_followup_kwargs(kwargs)

        if logging_obj is not None:
            agentic_params = logging_obj.model_call_details.get("agentic_loop_params", {})
            full_model_name = agentic_params.get("model", model)
        verbose_logger.debug(
            "WebSearchInterception: Built anthropic request patch [call_id=%s model=%s messages=%d searches=%d]",
            _call_id,
            full_model_name,
            len(follow_up_messages),
            len(final_search_results),
        )
        patch = AgenticLoopRequestPatch(
            model=full_model_name,
            messages=follow_up_messages,
            max_tokens=max_tokens,
            optional_params=optional_params_without_max_tokens,
            kwargs=kwargs_for_followup,
        )
        return patch, structured_results

    async def _execute_search(
        self, query: str, kwargs: Optional[dict[str, Any]] = None
    ) -> Tuple[str, Optional[SearchResponse]]:
        """
        Execute a single web search using router's search tools.

        Returns both the formatted text (fed back to the model in the follow-up
        call) and the structured ``SearchResponse`` (preserved so callers can
        build Anthropic-native ``web_search_tool_result`` blocks for clients
        that requested a native ``web_search_*`` tool). The structured value
        is None on the failure path so callers can still emit an empty result
        block rather than dropping the search entirely.
        """
        try:
            # Import router from proxy_server
            try:
                from litellm.proxy.proxy_server import llm_router
            except ImportError:
                verbose_logger.debug(
                    "WebSearchInterception: Could not import llm_router from proxy_server, "
                    "falling back to direct litellm.asearch() with perplexity"
                )
                llm_router = None

            search_tool = self._select_search_tool_from_router(llm_router=llm_router)
            search_provider: Optional[str] = None
            search_litellm_params: dict[str, Any] = {}
            if search_tool is not None:
                await self._authorize_search_tool(search_tool=search_tool, kwargs=kwargs)
                search_litellm_params = dict(search_tool.get("litellm_params", {}) or {})
                search_provider = search_litellm_params.get("search_provider")

            # Fallback to perplexity if no router or no search tools configured
            if not search_provider:
                search_provider = "perplexity"
                verbose_logger.debug(
                    "WebSearchInterception: No search tools configured in router, "
                    f"using default provider '{search_provider}'"
                )

            verbose_logger.debug(
                f"WebSearchInterception: Executing search for '{query}' using provider '{search_provider}'"
            )
            search_kwargs = {
                key: value
                for key, value in search_litellm_params.items()
                if key != "search_provider" and value is not None
            }
            result = await litellm.asearch(query=query, search_provider=search_provider, **search_kwargs)

            # Format using transformation function
            search_result_text = WebSearchTransformation.format_search_response(result)

            verbose_logger.debug(
                f"WebSearchInterception: Search completed for '{query}', got {len(search_result_text)} chars"
            )
            return search_result_text, result
        except Exception as e:
            verbose_logger.error(f"WebSearchInterception: Search failed for '{query}': {str(e)}")
            raise

    async def _authorize_search_tool(
        self,
        search_tool: dict[str, Any],
        kwargs: Optional[dict[str, Any]],
    ) -> None:
        search_tool_name = search_tool.get("search_tool_name")
        if not isinstance(search_tool_name, str) or not search_tool_name:
            return

        user_api_key_auth = self._get_user_api_key_auth_from_kwargs(kwargs)
        if user_api_key_auth is None:
            return

        from litellm.proxy.auth.auth_checks import (
            can_key_call_search_tool,
            can_team_call_search_tool,
            get_team_object,
        )

        await can_key_call_search_tool(
            search_tool_name=search_tool_name,
            valid_token=user_api_key_auth,
        )

        team_id = getattr(user_api_key_auth, "team_id", None)
        if team_id:
            from litellm.proxy.proxy_server import (
                prisma_client,
                proxy_logging_obj,
                user_api_key_cache,
            )

            team_object = await get_team_object(
                team_id=team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=getattr(user_api_key_auth, "parent_otel_span", None),
                proxy_logging_obj=proxy_logging_obj,
            )
            await can_team_call_search_tool(
                search_tool_name=search_tool_name,
                team_object=team_object,
            )

    @staticmethod
    def _get_user_api_key_auth_from_kwargs(kwargs: Optional[dict[str, Any]]) -> Any:
        if not kwargs:
            return None

        for metadata_key in ("metadata", "litellm_metadata"):
            metadata = kwargs.get(metadata_key)
            if isinstance(metadata, dict) and metadata.get("user_api_key_auth") is not None:
                return metadata["user_api_key_auth"]

        litellm_params = kwargs.get("litellm_params")
        if not isinstance(litellm_params, dict):
            return None

        for metadata_key in ("metadata", "litellm_metadata"):
            metadata = litellm_params.get(metadata_key)
            if isinstance(metadata, dict) and metadata.get("user_api_key_auth") is not None:
                return metadata["user_api_key_auth"]

        return None

    def _select_search_tool_from_router(self, llm_router: Any) -> Optional[dict[str, Any]]:
        if llm_router is None or not hasattr(llm_router, "search_tools"):
            return None
        search_tools = list(getattr(llm_router, "search_tools") or [])
        return self._select_search_tool_from_list(search_tools=search_tools, source="router")

    def _select_search_tool_from_list(
        self,
        search_tools: list[dict[str, Any]],
        source: str,
    ) -> Optional[dict[str, Any]]:
        if self.search_tool_name:
            matching_tools = [tool for tool in search_tools if tool.get("search_tool_name") == self.search_tool_name]
            if matching_tools:
                search_provider = (matching_tools[0].get("litellm_params", {}) or {}).get("search_provider")
                verbose_logger.debug(
                    f"WebSearchInterception: Found search tool '{self.search_tool_name}' "
                    f"from {source} with provider '{search_provider}'"
                )
                return matching_tools[0]
            verbose_logger.debug(
                f"WebSearchInterception: Search tool '{self.search_tool_name}' not found in {source}, "
                "falling back to first available or perplexity"
            )

        if search_tools:
            first_tool = search_tools[0]
            search_provider = (first_tool.get("litellm_params", {}) or {}).get("search_provider")
            verbose_logger.debug(
                f"WebSearchInterception: Using first available search tool from {source} "
                f"with provider '{search_provider}'"
            )
            return first_tool

        return None

    async def _execute_chat_completion_agentic_loop(
        self,
        model: str,
        messages: List[Dict],
        tool_calls: List[Dict],
        optional_params: Dict,
        logging_obj: Any,
        stream: bool,
        kwargs: Dict,
        response_format: str = "openai",
    ) -> Any:
        """Legacy path: execute search + build patch + run follow-up call."""
        request_patch = await self._build_chat_completion_request_patch(
            model=model,
            messages=messages,
            tool_calls=tool_calls,
            optional_params=optional_params,
            kwargs=kwargs,
            response_format=response_format,
        )
        if request_patch.messages is None:
            raise ValueError("WebSearchInterception: missing follow-up messages")
        params = dict(optional_params)
        params.update(request_patch.optional_params)
        params.pop("tool_choice", None)
        return await litellm.acompletion(
            model=request_patch.model or model,
            messages=request_patch.messages,
            **params,
            **request_patch.kwargs,
        )

    async def _build_chat_completion_request_patch(
        self,
        model: str,
        messages: List[Dict],
        tool_calls: List[Dict],
        optional_params: Dict,
        kwargs: Dict,
        response_format: str = "openai",
    ) -> AgenticLoopRequestPatch:
        """Execute litellm.search() and build chat-completion rerun patch."""

        # Extract search queries from tool_calls
        search_tasks = []
        for tool_call in tool_calls:
            # Handle both Anthropic-style input and OpenAI-style function.arguments
            query = None
            if "input" in tool_call and isinstance(tool_call["input"], dict):
                query = tool_call["input"].get("query")
            elif "function" in tool_call:
                func = tool_call["function"]
                if isinstance(func, dict):
                    args = func.get("arguments", {})
                    if isinstance(args, dict):
                        query = args.get("query")

            if query:
                verbose_logger.debug(f"WebSearchInterception: Queuing search for query='{query}'")
                search_tasks.append(self._execute_search(query, kwargs=kwargs))
            else:
                verbose_logger.debug(f"WebSearchInterception: Tool call {tool_call.get('id')} has no query")
                # Add empty result for tools without query
                search_tasks.append(self._create_empty_search_result())

        # Execute searches in parallel
        verbose_logger.debug(f"WebSearchInterception: Executing {len(search_tasks)} search(es) in parallel")
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Chat-completion path only needs text — OpenAI tool_result format
        # has no equivalent of Anthropic's web_search_tool_result block.
        final_search_results: List[str] = []
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                verbose_logger.error(f"WebSearchInterception: Search {i} failed with error: {str(result)}")
                final_search_results.append(f"Search failed: {str(result)}")
            elif isinstance(result, tuple) and len(result) == 2:
                text_value, _ = result
                final_search_results.append(cast(str, text_value) if isinstance(text_value, str) else str(text_value))
            else:
                verbose_logger.debug(f"WebSearchInterception: Unexpected result type {type(result)} at index {i}")
                final_search_results.append(str(result))

        # Build assistant and tool messages using transformation
        (
            assistant_message,
            tool_messages_or_user,
        ) = WebSearchTransformation.transform_response(
            tool_calls=tool_calls,
            search_results=final_search_results,
            response_format=response_format,
        )

        # Make follow-up request with search results
        # For OpenAI format, tool_messages_or_user is a list of tool messages
        if response_format == "openai":
            follow_up_messages = messages + [assistant_message] + cast(List[Dict], tool_messages_or_user)
        else:
            # For Anthropic format (shouldn't happen in this method, but handle it)
            follow_up_messages = messages + [
                assistant_message,
                cast(Dict, tool_messages_or_user),
            ]

        verbose_logger.debug("WebSearchInterception: Making follow-up chat completion request with search results")
        verbose_logger.debug(f"WebSearchInterception: Follow-up messages count: {len(follow_up_messages)}")

        # Remove internal parameters that shouldn't be passed to follow-up request
        internal_params = {
            "_websearch_interception",
            "acompletion",
            "litellm_logging_obj",
            "custom_llm_provider",
            "model_alias_map",
            "stream_response",
            "custom_prompt_dict",
        }
        kwargs_for_followup = {
            k: v for k, v in kwargs.items() if not k.startswith("_websearch_interception") and k not in internal_params
        }

        full_model_name = model
        if "custom_llm_provider" in kwargs:
            custom_llm_provider = kwargs["custom_llm_provider"]
            if not model.startswith(custom_llm_provider) and "/" not in model:
                full_model_name = f"{custom_llm_provider}/{model}"

        verbose_logger.debug(
            "WebSearchInterception: Built chat completion request patch model=%s messages=%d",
            full_model_name,
            len(follow_up_messages),
        )

        tools_param = optional_params.get("tools")
        optional_params_clean = {
            k: v
            for k, v in optional_params.items()
            if k
            not in {
                "tools",
                "tool_choice",
                "extra_body",
                "model_alias_map",
                "stream_response",
                "custom_prompt_dict",
            }
        }
        if tools_param is not None:
            optional_params_clean["tools"] = tools_param

        return AgenticLoopRequestPatch(
            model=full_model_name,
            messages=follow_up_messages,
            optional_params=optional_params_clean,
            kwargs=kwargs_for_followup,
        )

    async def _create_empty_search_result(
        self,
    ) -> Tuple[str, Optional[SearchResponse]]:
        """Create an empty search result for tool calls without queries"""
        return "No search query provided", None

    @staticmethod
    def initialize_from_proxy_config(
        litellm_settings: Dict[str, Any],
        callback_specific_params: Dict[str, Any],
    ) -> "WebSearchInterceptionLogger":
        """
        Static method to initialize WebSearchInterceptionLogger from proxy config.

        Used in callback_utils.py to simplify initialization logic.

        Args:
            litellm_settings: Dictionary containing litellm_settings from proxy_config.yaml
            callback_specific_params: Dictionary containing callback-specific parameters

        Returns:
            Configured WebSearchInterceptionLogger instance

        Example:
            From callback_utils.py:
                websearch_obj = WebSearchInterceptionLogger.initialize_from_proxy_config(
                    litellm_settings=litellm_settings,
                    callback_specific_params=callback_specific_params
                )
        """
        # Get websearch_interception_params from litellm_settings or callback_specific_params
        websearch_params: WebSearchInterceptionConfig = {}
        if "websearch_interception_params" in litellm_settings:
            websearch_params = litellm_settings["websearch_interception_params"]
        elif "websearch_interception" in callback_specific_params and isinstance(
            callback_specific_params["websearch_interception"], dict
        ):
            websearch_params = cast(
                WebSearchInterceptionConfig,
                callback_specific_params["websearch_interception"],
            )

        # Use classmethod to initialize from config
        return WebSearchInterceptionLogger.from_config_yaml(websearch_params)

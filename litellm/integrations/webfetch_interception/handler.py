"""
WebFetch Interception Handler

CustomLogger that intercepts WebFetch tool calls for models that don't
natively support web fetch (e.g., Bedrock/Claude) and executes them
server-side using litellm router's fetch tools.
"""

import asyncio
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import litellm
from litellm._logging import verbose_logger
from litellm.anthropic_interface import messages as anthropic_messages
from litellm.constants import LITELLM_WEB_FETCH_TOOL_NAME
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.webfetch_interception.tools import (
    get_litellm_web_fetch_tool,
    get_litellm_web_fetch_tool_openai,
    is_web_fetch_tool,
    is_web_fetch_tool_chat_completion,
)
from litellm.integrations.webfetch_interception.transformation import (
    WebFetchTransformation,
)
from litellm.llms.base_llm.fetch.transformation import BaseFetchConfig
from litellm.types.integrations.custom_logger import (
    AgenticLoopPlan,
    AgenticLoopRequestPatch,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


class WebFetchInterceptionLogger(CustomLogger):
    """
    CustomLogger that intercepts WebFetch tool calls for models that don't
    natively support web fetch.

    Implements agentic loop:
    1. Detects WebFetch tool_use in model response
    2. Executes fetch using router's fetch tools
    3. Makes follow-up request with fetch results
    4. Returns final response
    """

    def __init__(
        self,
        enabled_providers: Optional[List[Union[LlmProviders, str]]] = None,
        fetch_tool_name: Optional[str] = None,
    ):
        """
        Args:
            enabled_providers: List of LLM providers to enable interception for.
                              Use LlmProviders enum values (e.g., [LlmProviders.BEDROCK])
                              If None or empty list, enables for ALL providers.
                              Default: None (all providers enabled)
            fetch_tool_name: Name of fetch tool configured in router's fetch_tools.
                            If None, will attempt to use first available fetch tool.
        """
        super().__init__()
        # Convert enum values to strings for comparison
        if enabled_providers is None:
            self.enabled_providers = [LlmProviders.BEDROCK.value]
        else:
            self.enabled_providers = [
                p.value if isinstance(p, LlmProviders) else p for p in enabled_providers
            ]
        self.fetch_tool_name = fetch_tool_name
        self._request_has_webfetch = False  # Track if current request has web fetch

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Optional[Any]
    ) -> Optional[dict]:
        """
        Pre-call hook to convert native Anthropic web_fetch tools to regular tools.

        This prevents Bedrock from trying to execute web_fetch server-side (which fails).
        Instead, we convert it to a regular tool so the model returns tool_use blocks
        that we can intercept and execute ourselves.
        """
        # Check if this is for an enabled provider
        custom_llm_provider = kwargs.get("custom_llm_provider", "") or kwargs.get(
            "litellm_params", {}
        ).get("custom_llm_provider", "")
        if not custom_llm_provider:
            try:
                _, custom_llm_provider, _, _ = litellm.get_llm_provider(
                    model=kwargs.get("model", "")
                )
            except Exception:
                custom_llm_provider = ""
        if custom_llm_provider not in self.enabled_providers:
            return None

        # Check if request has tools with native web_fetch
        tools = kwargs.get("tools")
        if not tools:
            return None

        # Check if any tool is a web fetch tool (native or already LiteLLM standard)
        has_webfetch = any(is_web_fetch_tool(t) for t in tools)

        if not has_webfetch:
            return None

        verbose_logger.debug(
            "WebFetchInterception: Converting native web_fetch tools to LiteLLM standard"
        )

        # Convert native/custom web_fetch tools to LiteLLM standard
        converted_tools = []
        for tool in tools:
            if is_web_fetch_tool(tool):
                # Convert to LiteLLM standard web fetch tool
                converted_tool = get_litellm_web_fetch_tool_openai()
                converted_tools.append(converted_tool)
                verbose_logger.debug(
                    f"WebFetchInterception: Converted {tool.get('name', 'unknown')} "
                    f"(type={tool.get('type', 'none')}) to {LITELLM_WEB_FETCH_TOOL_NAME}"
                )
            else:
                # Keep other tools as-is
                converted_tools.append(tool)

        kwargs["tools"] = converted_tools

        if kwargs.get("stream"):
            verbose_logger.debug(
                "WebFetchInterception: deployment hook converting stream=True to stream=False"
            )
            kwargs["stream"] = False
            kwargs["_webfetch_interception_converted_stream"] = True

        return kwargs

    @classmethod
    def from_config_yaml(cls, config: Dict[str, Any]) -> "WebFetchInterceptionLogger":
        """
        Initialize WebFetchInterceptionLogger from proxy config.yaml parameters.

        Args:
            config: Configuration dictionary from litellm_settings.webfetch_interception_params

        Returns:
            Configured WebFetchInterceptionLogger instance

        Example:
            From proxy_config.yaml:
                litellm_settings:
                  webfetch_interception_params:
                    enabled_providers: ["bedrock"]
                    fetch_tool_name: "my-firecrawl-fetch"

            Usage:
                config = litellm_settings.get("webfetch_interception_params", {})
                logger = WebFetchInterceptionLogger.from_config_yaml(config)
        """
        # Extract parameters from config
        enabled_providers_str = config.get("enabled_providers", None)
        fetch_tool_name = config.get("fetch_tool_name", None)

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
            fetch_tool_name=fetch_tool_name,
        )

    async def async_pre_request_hook(
        self, model: str, messages: List[Dict], kwargs: Dict
    ) -> Optional[Dict]:
        """
        Pre-request hook to convert native web fetch tools to LiteLLM standard.

        This hook is called before the API request is made, allowing us to:
        1. Detect native web fetch tools (web_fetch_20250305, etc.)
        2. Convert them to LiteLLM standard format (litellm_web_fetch)
        3. Convert stream=True to stream=False for interception

        This prevents providers like Bedrock from trying to execute web_fetch
        natively (which fails), and ensures our agentic loop can intercept tool_use.

        Returns:
            Modified kwargs dict with converted tools, or None if no modifications needed
        """
        # Check if this request is for an enabled provider
        custom_llm_provider = kwargs.get("litellm_params", {}).get(
            "custom_llm_provider", ""
        )

        verbose_logger.debug(
            f"WebFetchInterception: Pre-request hook called"
            f" - custom_llm_provider={custom_llm_provider}"
            f" - enabled_providers={self.enabled_providers or 'ALL'}"
        )

        if (
            self.enabled_providers is not None
            and custom_llm_provider not in self.enabled_providers
        ):
            verbose_logger.debug(
                f"WebFetchInterception: Skipping - provider {custom_llm_provider} not in {self.enabled_providers}"
            )
            return None

        # Check if request has tools
        tools = kwargs.get("tools")
        if not tools:
            return None

        # Check if any tool is a web fetch tool
        has_webfetch = any(is_web_fetch_tool(t) for t in tools)
        if not has_webfetch:
            return None

        verbose_logger.debug(
            f"WebFetchInterception: Pre-request hook triggered for provider={custom_llm_provider}"
        )

        # Convert native web fetch tools to LiteLLM standard
        converted_tools = []
        for tool in tools:
            if is_web_fetch_tool(tool):
                standard_tool = get_litellm_web_fetch_tool()
                converted_tools.append(standard_tool)
                verbose_logger.debug(
                    f"WebFetchInterception: Converted {tool.get('name', 'unknown')} "
                    f"(type={tool.get('type', 'none')}) to {LITELLM_WEB_FETCH_TOOL_NAME}"
                )
            else:
                converted_tools.append(tool)

        kwargs["tools"] = converted_tools
        verbose_logger.debug(
            f"WebFetchInterception: Tools after conversion: {[t.get('name') for t in converted_tools]}"
        )

        # Also convert here for direct callers that bypass the deployment hook.
        if kwargs.get("stream"):
            verbose_logger.debug(
                "WebFetchInterception: Converting stream=True to stream=False"
            )
            kwargs["stream"] = False
            kwargs["_webfetch_interception_converted_stream"] = True

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
        """
        Check if WebFetch tool interception is needed for Anthropic Messages API.
        """

        verbose_logger.debug(
            f"WebFetchInterception: Hook called! provider={custom_llm_provider}, stream={stream}"
        )
        verbose_logger.debug(f"WebFetchInterception: Response type: {type(response)}")

        # Check if provider should be intercepted
        if (
            self.enabled_providers is not None
            and custom_llm_provider not in self.enabled_providers
        ):
            verbose_logger.debug(
                f"WebFetchInterception: Skipping provider {custom_llm_provider} (not in enabled list: {self.enabled_providers})"
            )
            return False, {}

        # Check if tools include any web fetch tool (LiteLLM standard or native)
        has_webfetch_tool = any(is_web_fetch_tool(t) for t in (tools or []))
        if not has_webfetch_tool:
            verbose_logger.debug("WebFetchInterception: No web fetch tool in request")
            return False, {}

        # Detect WebFetch tool_use in response (Anthropic format)
        should_intercept, tool_calls = WebFetchTransformation.transform_request(
            response=response,
            stream=stream,
            response_format="anthropic",
        )

        if not should_intercept:
            verbose_logger.debug(
                "WebFetchInterception: No WebFetch tool_use detected in response"
            )
            return False, {}

        verbose_logger.debug(
            f"WebFetchInterception: Detected {len(tool_calls)} WebFetch tool call(s), executing agentic loop"
        )

        # Extract thinking blocks from response content.
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
                    thinking_block_dict: Dict = {"type": block_type}
                    if block_type == "thinking":
                        thinking_block_dict["thinking"] = getattr(block, "thinking", "")
                        thinking_block_dict["signature"] = getattr(
                            block, "signature", ""
                        )
                    else:  # redacted_thinking
                        thinking_block_dict["data"] = getattr(block, "data", "")
                    thinking_blocks.append(thinking_block_dict)

        if thinking_blocks:
            verbose_logger.debug(
                f"WebFetchInterception: Extracted {len(thinking_blocks)} thinking block(s) from response"
            )

        # Return tools dict with tool calls and thinking blocks
        tools_dict = {
            "tool_calls": tool_calls,
            "tool_type": "webfetch",
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
        Check if WebFetch tool interception is needed for Chat Completions API.
        """

        verbose_logger.debug(
            f"WebFetchInterception: Chat completion hook called! provider={custom_llm_provider}, stream={stream}"
        )
        verbose_logger.debug(f"WebFetchInterception: Response type: {type(response)}")

        # Check if provider should be intercepted
        if (
            self.enabled_providers is not None
            and custom_llm_provider not in self.enabled_providers
        ):
            verbose_logger.debug(
                f"WebFetchInterception: Skipping provider {custom_llm_provider} (not in enabled list: {self.enabled_providers})"
            )
            return False, {}

        # Check if tools include any web fetch tool (strict check for chat completions)
        has_webfetch_tool = any(
            is_web_fetch_tool_chat_completion(t) for t in (tools or [])
        )
        if not has_webfetch_tool:
            verbose_logger.debug(
                "WebFetchInterception: No litellm_web_fetch tool in request"
            )
            return False, {}

        # Detect WebFetch tool_calls in response (OpenAI format)
        should_intercept, tool_calls = WebFetchTransformation.transform_request(
            response=response,
            stream=stream,
            response_format="openai",
        )

        if not should_intercept:
            verbose_logger.debug(
                "WebFetchInterception: No WebFetch tool_calls detected in response"
            )
            return False, {}

        verbose_logger.debug(
            f"WebFetchInterception: Detected {len(tool_calls)} WebFetch tool call(s), executing agentic loop"
        )

        # Return tools dict with tool calls
        tools_dict = {
            "tool_calls": tool_calls,
            "tool_type": "webfetch",
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
        Execute agentic loop with WebFetch execution for Anthropic Messages API.
        """

        tool_calls = tools["tool_calls"]
        thinking_blocks = tools.get("thinking_blocks", [])

        verbose_logger.debug(
            f"WebFetchInterception: Executing agentic loop for {len(tool_calls)} fetch(es)"
        )

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
        tool_calls = tools["tool_calls"]
        thinking_blocks = tools.get("thinking_blocks", [])
        request_patch = await self._build_anthropic_request_patch(
            model=model,
            messages=messages,
            tool_calls=tool_calls,
            thinking_blocks=thinking_blocks,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            logging_obj=logging_obj,
            kwargs=kwargs,
        )
        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=request_patch,
            metadata={"tool_type": "webfetch", "response_format": "anthropic"},
        )

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
        Execute agentic loop with WebFetch execution for Chat Completions API.
        """

        tool_calls = tools["tool_calls"]
        response_format = tools.get("response_format", "openai")

        verbose_logger.debug(
            f"WebFetchInterception: Executing chat completion agentic loop for {len(tool_calls)} fetch(es)"
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
            metadata={"tool_type": "webfetch", "response_format": response_format},
        )

    async def _get_fetch_config(self) -> BaseFetchConfig:
        """Get the fetch config for the configured fetch tool."""
        # Import router from proxy_server
        try:
            from litellm.proxy.proxy_server import llm_router
        except ImportError:
            verbose_logger.debug(
                "WebFetchInterception: Could not import llm_router from proxy_server, "
                "falling back to direct fetch"
            )
            llm_router = None

        # Determine fetch provider from router's fetch_tools
        fetch_provider: Optional[str] = None
        api_key: Optional[str] = None
        api_base: Optional[str] = None

        if llm_router is not None and hasattr(llm_router, "fetch_tools"):
            if self.fetch_tool_name:
                # Find specific fetch tool by name
                matching_tools = [
                    tool
                    for tool in llm_router.fetch_tools
                    if tool.get("fetch_tool_name") == self.fetch_tool_name
                ]
                if matching_tools:
                    fetch_tool = matching_tools[0]
                    litellm_params = fetch_tool.get("litellm_params", {})
                    fetch_provider = litellm_params.get("fetch_provider")
                    api_key = litellm_params.get("api_key")
                    api_base = litellm_params.get("api_base")
                    verbose_logger.debug(
                        f"WebFetchInterception: Found fetch tool '{self.fetch_tool_name}' "
                        f"with provider '{fetch_provider}'"
                    )
                else:
                    verbose_logger.debug(
                        f"WebFetchInterception: Fetch tool '{self.fetch_tool_name}' not found in router"
                    )

            # If no specific tool or not found, use first available
            if not fetch_provider and llm_router.fetch_tools:
                first_tool = llm_router.fetch_tools[0]
                litellm_params = first_tool.get("litellm_params", {})
                fetch_provider = litellm_params.get("fetch_provider")
                api_key = litellm_params.get("api_key")
                api_base = litellm_params.get("api_base")
                verbose_logger.debug(
                    f"WebFetchInterception: Using first available fetch tool with provider '{fetch_provider}'"
                )

        # Fallback to firecrawl if no router or no fetch tools configured
        if not fetch_provider:
            fetch_provider = "firecrawl"
            verbose_logger.debug(
                f"WebFetchInterception: No fetch tools configured in router, "
                f"using default provider '{fetch_provider}'"
            )

        # Get the fetch config
        fetch_config = ProviderConfigManager.get_provider_fetch_config(fetch_provider)
        if fetch_config is None:
            raise ValueError(f"Unknown fetch provider: {fetch_provider}")

        # Set up headers with API key
        headers = {}
        if api_key:
            headers["api_key"] = api_key
        if api_base:
            headers["api_base"] = api_base

        fetch_config.validate_environment(
            headers=headers, api_key=api_key, api_base=api_base
        )

        return fetch_config

    async def _execute_fetch(self, url: str) -> str:
        """Execute a single web fetch using router's fetch tools"""
        try:
            # Get the fetch config
            fetch_config = await self._get_fetch_config()

            verbose_logger.debug(
                f"WebFetchInterception: Executing fetch for '{url}' using provider '{fetch_config.ui_friendly_name()}'"
            )

            # Execute fetch
            fetch_response = await fetch_config.afetch_url(
                url=url,
                headers={},  # Headers already set up in _get_fetch_config
                optional_params={},
            )

            # Format using transformation function
            fetch_result_text = WebFetchTransformation.format_fetch_response(
                fetch_response
            )

            verbose_logger.debug(
                f"WebFetchInterception: Fetch completed for '{url}', got {len(fetch_result_text)} chars"
            )
            return fetch_result_text
        except Exception as e:
            verbose_logger.error(
                f"WebFetchInterception: Fetch failed for '{url}': {str(e)}"
            )
            raise

    @staticmethod
    def _resolve_max_tokens(
        optional_params: Dict,
        kwargs: Dict,
    ) -> int:
        """Extract max_tokens and validate against thinking.budget_tokens."""
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
                        "WebFetchInterception: max_tokens=%s <= thinking.budget_tokens=%s, "
                        "adjusting to %s to satisfy Anthropic API constraint",
                        max_tokens,
                        budget_tokens,
                        adjusted,
                    )
                    max_tokens = adjusted
        return max_tokens

    @staticmethod
    def _prepare_followup_kwargs(kwargs: Dict) -> Dict:
        """Build kwargs for the follow-up call, excluding internal keys."""
        _internal_keys = {"litellm_logging_obj"}
        return {
            k: v
            for k, v in kwargs.items()
            if not k.startswith("_webfetch_interception") and k not in _internal_keys
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
        """Legacy path: execute fetch + build patch + run follow-up call."""
        request_patch = await self._build_anthropic_request_patch(
            model=model,
            messages=messages,
            tool_calls=tool_calls,
            thinking_blocks=thinking_blocks,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            logging_obj=logging_obj,
            kwargs=kwargs,
        )
        if request_patch.messages is None:
            raise ValueError("WebFetchInterception: missing follow-up messages")

        optional_params = dict(anthropic_messages_optional_request_params)
        optional_params.update(request_patch.optional_params)
        max_tokens = request_patch.max_tokens
        if max_tokens is None:
            max_tokens = cast(Optional[int], optional_params.pop("max_tokens", None))
        else:
            optional_params.pop("max_tokens", None)
        if max_tokens is None:
            max_tokens = cast(int, kwargs.get("max_tokens", 1024))

        return await anthropic_messages.acreate(
            max_tokens=max_tokens,
            messages=request_patch.messages,
            model=request_patch.model or model,
            **optional_params,
            **request_patch.kwargs,
        )

    async def _build_anthropic_request_patch(
        self,
        model: str,
        messages: List[Dict],
        tool_calls: List[Dict],
        thinking_blocks: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        logging_obj: Any,
        kwargs: Dict,
    ) -> AgenticLoopRequestPatch:
        """Execute fetch and build follow-up request patch."""

        # Extract URLs from tool_use blocks
        fetch_tasks = []
        for tool_call in tool_calls:
            url = tool_call["input"].get("url")
            if url:
                verbose_logger.debug(
                    f"WebFetchInterception: Queuing fetch for url='{url}'"
                )
                fetch_tasks.append(self._execute_fetch(url))
            else:
                verbose_logger.debug(
                    f"WebFetchInterception: Tool call {tool_call['id']} has no URL"
                )
                # Add empty result for tools without URL
                fetch_tasks.append(self._create_empty_fetch_result())

        # Execute fetches in parallel
        verbose_logger.debug(
            f"WebFetchInterception: Executing {len(fetch_tasks)} fetch(es) in parallel"
        )
        fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        # Handle any exceptions in fetch results
        final_fetch_results: List[str] = []
        for i, result in enumerate(fetch_results):
            if isinstance(result, Exception):
                verbose_logger.error(
                    f"WebFetchInterception: Fetch {i} failed with error: {str(result)}"
                )
                final_fetch_results.append(f"Fetch failed: {str(result)}")
            elif isinstance(result, str):
                final_fetch_results.append(cast(str, result))
            else:
                verbose_logger.debug(
                    f"WebFetchInterception: Unexpected result type {type(result)} at index {i}"
                )
                final_fetch_results.append(str(result))

        # Build assistant and user messages using transformation
        assistant_message, user_message = WebFetchTransformation.transform_response(
            tool_calls=tool_calls,
            fetch_results=final_fetch_results,
            thinking_blocks=thinking_blocks,
        )

        follow_up_messages = messages + [assistant_message, cast(Dict, user_message)]

        # Correlation context for structured logging
        _call_id = getattr(logging_obj, "litellm_call_id", None) or kwargs.get(
            "litellm_call_id", "unknown"
        )

        full_model_name = model  # safe default before try block

        max_tokens = self._resolve_max_tokens(
            anthropic_messages_optional_request_params, kwargs
        )

        verbose_logger.debug(
            f"WebFetchInterception: Using max_tokens={max_tokens} for follow-up request"
        )

        optional_params_without_max_tokens = {
            k: v
            for k, v in anthropic_messages_optional_request_params.items()
            if k != "max_tokens"
        }
        kwargs_for_followup = self._prepare_followup_kwargs(kwargs)

        if logging_obj is not None:
            agentic_params = logging_obj.model_call_details.get(
                "agentic_loop_params", {}
            )
            full_model_name = agentic_params.get("model", model)
        verbose_logger.debug(
            "WebFetchInterception: Built anthropic request patch "
            "[call_id=%s model=%s messages=%d fetches=%d]",
            _call_id,
            full_model_name,
            len(follow_up_messages),
            len(final_fetch_results),
        )
        return AgenticLoopRequestPatch(
            model=full_model_name,
            messages=follow_up_messages,
            max_tokens=max_tokens,
            optional_params=optional_params_without_max_tokens,
            kwargs=kwargs_for_followup,
        )

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
        """Legacy path: execute fetch + build patch + run follow-up call."""
        request_patch = await self._build_chat_completion_request_patch(
            model=model,
            messages=messages,
            tool_calls=tool_calls,
            optional_params=optional_params,
            kwargs=kwargs,
            response_format=response_format,
        )
        if request_patch.messages is None:
            raise ValueError("WebFetchInterception: missing follow-up messages")
        params = dict(optional_params)
        params.update(request_patch.optional_params)
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
        """Execute fetch and build chat-completion rerun patch."""

        fetch_results = await self._execute_tool_call_fetches(tool_calls)

        # Build assistant and tool messages using transformation
        (
            assistant_message,
            tool_messages_or_user,
        ) = WebFetchTransformation.transform_response(
            tool_calls=tool_calls,
            fetch_results=fetch_results,
            response_format=response_format,
        )

        follow_up_messages = self._build_follow_up_messages(
            messages, assistant_message, tool_messages_or_user, response_format
        )

        kwargs_for_followup = self._build_kwargs_for_followup(kwargs)
        full_model_name = self._resolve_full_model_name(model, kwargs)
        optional_params_clean = self._clean_optional_params(optional_params)

        return AgenticLoopRequestPatch(
            model=full_model_name,
            messages=follow_up_messages,
            optional_params=optional_params_clean,
            kwargs=kwargs_for_followup,
        )

    async def _execute_tool_call_fetches(self, tool_calls: List[Dict]) -> List[str]:
        """Execute fetches for tool calls in parallel and return results."""
        fetch_tasks: List[Any] = []
        for tool_call in tool_calls:
            url = self._extract_url_from_tool_call(tool_call)
            if url:
                verbose_logger.debug(
                    f"WebFetchInterception: Queuing fetch for url='{url}'"
                )
                fetch_tasks.append(self._execute_fetch(url))
            else:
                verbose_logger.debug(
                    f"WebFetchInterception: Tool call {tool_call.get('id')} has no URL"
                )
                fetch_tasks.append(self._create_empty_fetch_result())

        verbose_logger.debug(
            f"WebFetchInterception: Executing {len(fetch_tasks)} fetch(es) in parallel"
        )
        fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        final_fetch_results: List[str] = []
        for i, result in enumerate(fetch_results):
            if isinstance(result, Exception):
                verbose_logger.error(
                    f"WebFetchInterception: Fetch {i} failed with error: {str(result)}"
                )
                final_fetch_results.append(f"Fetch failed: {str(result)}")
            elif isinstance(result, str):
                final_fetch_results.append(result)
            else:
                verbose_logger.debug(
                    f"WebFetchInterception: Unexpected result type {type(result)} at index {i}"
                )
                final_fetch_results.append(str(result))

        return final_fetch_results

    def _extract_url_from_tool_call(self, tool_call: Dict) -> Optional[str]:
        """Extract URL from tool call arguments."""
        if "input" in tool_call and isinstance(tool_call["input"], dict):
            return cast(Optional[str], tool_call["input"].get("url"))
        if "function" in tool_call:
            func = tool_call["function"]
            if isinstance(func, dict):
                args = func.get("arguments", {})
                if isinstance(args, dict):
                    return cast(Optional[str], args.get("url"))
                elif isinstance(args, str):
                    try:
                        args_dict = json.loads(args)
                        return cast(Optional[str], args_dict.get("url"))
                    except (json.JSONDecodeError, TypeError):
                        pass
        return None

    def _build_follow_up_messages(
        self,
        messages: List[Dict],
        assistant_message: Dict,
        tool_messages_or_user: Any,
        response_format: str,
    ) -> List[Dict]:
        """Build follow-up messages with fetch results."""
        if response_format == "openai":
            return (
                messages + [assistant_message] + cast(List[Dict], tool_messages_or_user)
            )
        return messages + [assistant_message, cast(Dict, tool_messages_or_user)]

    def _build_kwargs_for_followup(self, kwargs: Dict) -> Dict:
        """Remove internal parameters from kwargs for follow-up request."""
        internal_params = {
            "_webfetch_interception",
            "acompletion",
            "litellm_logging_obj",
            "custom_llm_provider",
            "model_alias_map",
            "stream_response",
            "custom_prompt_dict",
        }
        return {
            k: v
            for k, v in kwargs.items()
            if not k.startswith("_webfetch_interception") and k not in internal_params
        }

    def _resolve_full_model_name(self, model: str, kwargs: Dict) -> str:
        """Resolve full model name including provider prefix."""
        if "custom_llm_provider" in kwargs:
            custom_llm_provider = kwargs["custom_llm_provider"]
            if not model.startswith(custom_llm_provider) and "/" not in model:
                return f"{custom_llm_provider}/{model}"
        return model

    def _clean_optional_params(self, optional_params: Dict) -> Dict:
        """Remove internal parameters from optional_params while preserving tools."""
        tools_param = optional_params.get("tools")
        optional_params_clean = {
            k: v
            for k, v in optional_params.items()
            if k
            not in {
                "tools",
                "extra_body",
                "model_alias_map",
                "stream_response",
                "custom_prompt_dict",
            }
        }
        if tools_param is not None:
            optional_params_clean["tools"] = tools_param
        return optional_params_clean

    async def _create_empty_fetch_result(self) -> str:
        """Create an empty fetch result for tool calls without URLs"""
        return "No URL provided for fetch"

    @staticmethod
    def initialize_from_proxy_config(
        litellm_settings: Dict[str, Any],
        callback_specific_params: Dict[str, Any],
    ) -> "WebFetchInterceptionLogger":
        """
        Static method to initialize WebFetchInterceptionLogger from proxy config.

        Used in callback_utils.py to simplify initialization logic.
        """
        # Get webfetch_interception_params from litellm_settings or callback_specific_params
        webfetch_params: Dict[str, Any] = {}
        if "webfetch_interception_params" in litellm_settings:
            webfetch_params = litellm_settings["webfetch_interception_params"]
        elif "webfetch_interception" in callback_specific_params:
            webfetch_params = callback_specific_params["webfetch_interception"]

        # Use classmethod to initialize from config
        return WebFetchInterceptionLogger.from_config_yaml(webfetch_params)

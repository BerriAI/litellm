"""
Vector Store Pre-Call Hook

This hook is called before making an LLM request when a vector store is configured.

For chat completions:
  Searches the vector store for relevant context and appends it to the messages.

For Responses API (file_search tool with vector_store_ids):
  - Native providers (OpenAI, Azure): pass file_search through; map unified VS IDs to
    provider-specific IDs.
  - Non-native providers (Claude, Gemini, etc.): intercept, run concurrent VS searches
    with timeout, inject context into input, strip file_search from tools.

Data flow for Responses API RAG injection:
  tools=[{type:file_search, vector_store_ids:[...]}]
         │
  extract VS IDs ──── none? ──► passthrough unchanged
         │
  check supports_native_file_search(provider)
         │
    ┌────┴─────────┐
    │  native?     │──► keep file_search in tools, passthrough
    │  (OAI/Azure) │
    └──────────────┘
    │  non-native? │──► asyncio.gather(asearch per VS, timeout=5s)
    │              │    inject context into data["input"]
    │              │    strip file_search from data["tools"]
    │              │    store results in data["vs_search_results"]
    └──────────────┘
"""

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import litellm
import litellm.vector_stores
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.llms.openai import AllMessageValues, ChatCompletionUserMessage
from litellm.types.prompts.init_prompts import PromptSpec
from litellm.types.utils import CallTypes, StandardCallbackDynamicParams
from litellm.types.vector_stores import (
    LiteLLM_ManagedVectorStore,
    VectorStoreResultContent,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy.utils import DualCache, PrismaClient
    from litellm.proxy._types import UserAPIKeyAuth
else:
    LiteLLMLoggingObj = Any
    DualCache = Any
    PrismaClient = Any
    UserAPIKeyAuth = Any

# Lazy module-level prisma_client reference — avoids inline imports inside methods.
# Set to None at import time; populated on first access if the proxy is running.
_prisma_client: Optional[Any] = None


def _get_prisma_client() -> Optional[Any]:
    """Return the proxy's prisma_client without an inline import."""
    global _prisma_client
    if _prisma_client is not None:
        return _prisma_client
    try:
        from litellm.proxy.proxy_server import (  # noqa: PLC0415
            prisma_client as _pc,
        )
        _prisma_client = _pc
        return _prisma_client
    except ImportError:
        return None


# Default timeout (seconds) for each vector store search call.
# Prevents a slow VS from blocking the LLM request indefinitely.
_VS_SEARCH_TIMEOUT_SECONDS = 5.0


class VectorStorePreCallHook(CustomLogger):
    CONTENT_PREFIX_STRING = "Context:\n\n"
    """
    Custom logger that handles vector store searches before LLM calls.

    Chat completions path  → async_get_chat_completion_prompt
    Responses API path     → async_pre_call_hook (CallTypes.aresponses)
    """

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # CHAT COMPLETIONS PATH
    # ------------------------------------------------------------------

    async def async_get_chat_completion_prompt(
        self,
        model: str,
        messages: List[AllMessageValues],
        non_default_params: dict,
        prompt_id: Optional[str],
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
        litellm_logging_obj: LiteLLMLoggingObj,
        prompt_spec: Optional[PromptSpec] = None,
        tools: Optional[List[Dict]] = None,
        prompt_label: Optional[str] = None,
        prompt_version: Optional[int] = None,
        ignore_prompt_manager_model: Optional[bool] = False,
        ignore_prompt_manager_optional_params: Optional[bool] = False,
    ) -> Tuple[str, List[AllMessageValues], dict]:
        """
        Perform vector store search and append results as context to messages.

        Returns:
            Tuple of (model, modified_messages, non_default_params)
        """
        try:
            if litellm.vector_store_registry is None:
                return model, messages, non_default_params

            vector_stores_to_run: List[
                LiteLLM_ManagedVectorStore
            ] = await litellm.vector_store_registry.pop_vector_stores_to_run_with_db_fallback(
                non_default_params=non_default_params,
                tools=tools,
                prisma_client=_get_prisma_client(),
            )

            if not vector_stores_to_run:
                return model, messages, non_default_params

            query = self._extract_query_from_messages(messages)
            if not query:
                verbose_logger.debug(
                    "No query found in messages for vector store search"
                )
                return model, messages, non_default_params

            all_search_results = await self._search_vector_stores_concurrent(
                vector_stores=vector_stores_to_run, query=query
            )

            modified_messages: List[AllMessageValues] = messages.copy()
            for search_response in all_search_results:
                modified_messages = self._append_search_results_to_messages(
                    messages=modified_messages, search_response=search_response
                )
                num_results = len(search_response.get("data", []) or [])
                verbose_logger.debug(
                    f"Vector store search completed. Added context from {num_results} results"
                )

            if litellm_logging_obj and all_search_results:
                litellm_logging_obj.model_call_details[
                    "search_results"
                ] = all_search_results

            return model, modified_messages, non_default_params

        except Exception as e:
            verbose_logger.exception(f"Error in VectorStorePreCallHook: {str(e)}")
            return model, messages, non_default_params

    # ------------------------------------------------------------------
    # RESPONSES API PATH
    # ------------------------------------------------------------------

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: Dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, Dict]]:
        """
        Intercept Responses API calls that include a file_search tool.

        For native providers (OpenAI, Azure): passthrough unchanged.
        For non-native providers: RAG-inject context, strip file_search from tools.
        """
        if call_type not in (
            CallTypes.aresponses.value,
            CallTypes.responses.value,
        ):
            return data

        tools: Optional[List[Dict]] = data.get("tools")
        vs_ids = self._get_vs_ids_from_file_search_tools(tools)
        if not vs_ids:
            return data

        model: str = data.get("model", "")
        is_native = self._is_native_file_search_provider(model)

        if is_native:
            verbose_logger.debug(
                f"VectorStorePreCallHook: native file_search provider for model={model}, "
                "passing through as-is"
            )
            return data

        # Non-native: RAG injection mode
        query = self._extract_query_from_responses_input(data.get("input"))
        if not query:
            verbose_logger.debug(
                "VectorStorePreCallHook: no query found in responses input, "
                "skipping VS search"
            )
            # Still strip file_search — non-native providers will error on it
            data["tools"] = self._strip_file_search_from_tools(tools)
            return data

        # Resolve VS configs from registry / DB
        vector_stores_to_run = await self._resolve_vector_stores(vs_ids)
        if not vector_stores_to_run:
            verbose_logger.debug(
                f"VectorStorePreCallHook: no VS configs found for ids={vs_ids}, "
                "stripping file_search and proceeding without context"
            )
            data["tools"] = self._strip_file_search_from_tools(tools)
            return data

        verbose_logger.info(
            f"VectorStorePreCallHook: searching {len(vector_stores_to_run)} vector store(s) "
            f"for model={model}"
        )

        all_search_results = await self._search_vector_stores_concurrent(
            vector_stores=vector_stores_to_run, query=query
        )

        if all_search_results:
            data["input"] = self._inject_context_into_responses_input(
                input_value=data.get("input"), search_results=all_search_results
            )
            # Store for post-call hook (responses API path stores in data, not logging_obj)
            data["vs_search_results"] = all_search_results

            total_results = sum(
                len(r.get("data", []) or []) for r in all_search_results
            )
            verbose_logger.info(
                f"VectorStorePreCallHook: injected context from {total_results} VS result(s) "
                f"across {len(all_search_results)} store(s)"
            )

        # Strip file_search — non-native provider must not receive it
        data["tools"] = self._strip_file_search_from_tools(tools)

        return data

    # ------------------------------------------------------------------
    # SHARED: CONCURRENT SEARCH WITH TIMEOUT
    # ------------------------------------------------------------------

    async def _search_vector_stores_concurrent(
        self,
        vector_stores: List[LiteLLM_ManagedVectorStore],
        query: str,
    ) -> List[VectorStoreSearchResponse]:
        """
        Search multiple vector stores concurrently via asyncio.gather.

        Each search has a hard _VS_SEARCH_TIMEOUT_SECONDS timeout. Failures and
        timeouts are logged and skipped — they never fail the LLM request.

        Returns:
            List of successful VectorStoreSearchResponse objects (failures excluded).
        """

        async def _search_one(
            vs: LiteLLM_ManagedVectorStore,
        ) -> Optional[VectorStoreSearchResponse]:
            vs_id = vs.get("vector_store_id", "")
            provider = vs.get("custom_llm_provider")
            extra_params = vs.get("litellm_params", {}) or {}
            try:
                result = await asyncio.wait_for(
                    litellm.vector_stores.asearch(
                        vector_store_id=vs_id,
                        query=query,
                        custom_llm_provider=provider,
                        **extra_params,
                    ),
                    timeout=_VS_SEARCH_TIMEOUT_SECONDS,
                )
                return result
            except asyncio.TimeoutError:
                verbose_logger.warning(
                    f"VectorStorePreCallHook: search timed out after "
                    f"{_VS_SEARCH_TIMEOUT_SECONDS}s for vs_id={vs_id}"
                )
                return None
            except Exception as exc:
                verbose_logger.warning(
                    f"VectorStorePreCallHook: search failed for vs_id={vs_id}: {exc}"
                )
                return None

        raw_results = await asyncio.gather(
            *[_search_one(vs) for vs in vector_stores],
            return_exceptions=False,  # _search_one catches all exceptions itself
        )
        return [r for r in raw_results if r is not None]

    # ------------------------------------------------------------------
    # HELPERS: QUERY EXTRACTION
    # ------------------------------------------------------------------

    def _extract_query_from_messages(
        self, messages: List[AllMessageValues]
    ) -> Optional[str]:
        """Extract query from the last user message (chat completions format)."""
        if not messages:
            return None

        last_message = messages[-1]
        if not isinstance(last_message, dict) or "content" not in last_message:
            return None

        content = last_message["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                    and "text" in item
                ):
                    return item["text"]
        return None

    def _extract_query_from_responses_input(
        self, input_value: Optional[Any]
    ) -> Optional[str]:
        """
        Extract a query string from a Responses API input parameter.

        Handles:
          - str → returned directly
          - list of input items → text from the last text item
          - None / empty → None
        """
        if not input_value:
            return None
        if isinstance(input_value, str):
            return input_value or None
        if isinstance(input_value, list):
            # Walk backwards to find the last text content
            for item in reversed(input_value):
                if not isinstance(item, dict):
                    continue
                # Direct text item: {type: "message", content: [{type: "input_text", text: ...}]}
                if item.get("type") == "message":
                    content = item.get("content")
                    if isinstance(content, str):
                        return content or None
                    if isinstance(content, list):
                        for part in reversed(content):
                            if isinstance(part, dict) and part.get("type") in (
                                "input_text",
                                "text",
                            ):
                                text = part.get("text")
                                if text:
                                    return text
                # Simple text item: {type: "input_text", text: ...}
                if item.get("type") in ("input_text", "text"):
                    text = item.get("text")
                    if text:
                        return text
        return None

    # ------------------------------------------------------------------
    # HELPERS: CONTEXT INJECTION (RESPONSES API)
    # ------------------------------------------------------------------

    def _inject_context_into_responses_input(
        self,
        input_value: Optional[Any],
        search_results: List[VectorStoreSearchResponse],
    ) -> Any:
        """
        Inject vector store search results as context into a Responses API input.

        For str inputs: prepend "Context:\\n\\n<results>\\n\\n<original input>"
        For list inputs: insert a context message item before the last message item.
        """
        context_text = self._build_context_text(search_results)
        if not context_text:
            return input_value

        if isinstance(input_value, str):
            return f"{self.CONTENT_PREFIX_STRING}{context_text}\n\n{input_value}"

        if isinstance(input_value, list):
            context_item = {
                "type": "message",
                "role": "user",
                "content": f"{self.CONTENT_PREFIX_STRING}{context_text}",
            }
            modified = list(input_value)
            # Insert before the last item (preserves original last user message)
            insert_pos = max(len(modified) - 1, 0)
            modified.insert(insert_pos, context_item)
            return modified

        return input_value

    def _build_context_text(
        self, search_results: List[VectorStoreSearchResponse]
    ) -> str:
        """Concatenate text content from all search results into a single string."""
        parts: List[str] = []
        for result_set in search_results:
            for result in result_set.get("data", []) or []:
                result_content: Optional[List[VectorStoreResultContent]] = result.get(
                    "content"
                )
                if result_content:
                    for content_item in result_content:
                        text: Optional[str] = content_item.get("text")
                        if text:
                            parts.append(text)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # HELPERS: TOOLS MANIPULATION
    # ------------------------------------------------------------------

    def _get_vs_ids_from_file_search_tools(
        self, tools: Optional[List[Dict]]
    ) -> List[str]:
        """Extract all vector_store_ids from file_search tools in the tools list."""
        if not tools or not isinstance(tools, list):
            return []
        ids: List[str] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "file_search":
                vs_ids = tool.get("vector_store_ids")
                if isinstance(vs_ids, list):
                    ids.extend(v for v in vs_ids if isinstance(v, str) and v)
        return list(dict.fromkeys(ids))  # deduplicate, preserve order

    def _strip_file_search_from_tools(
        self, tools: Optional[List[Dict]]
    ) -> Optional[List[Dict]]:
        """
        Return a copy of the tools list with all file_search entries removed.
        Returns None if the result would be empty and the input was non-None,
        to avoid sending an empty tools array to some providers.
        """
        if tools is None:
            return None
        filtered = [t for t in tools if not (isinstance(t, dict) and t.get("type") == "file_search")]
        return filtered if filtered else None

    # ------------------------------------------------------------------
    # HELPERS: PROVIDER CAPABILITY CHECK
    # ------------------------------------------------------------------

    def _is_native_file_search_provider(self, model: str) -> bool:
        """
        Return True if the model's provider natively supports file_search with
        vector_store_ids in the Responses API.

        Uses BaseResponsesAPIConfig.supports_native_file_search() via
        ProviderConfigManager. Defaults to False (RAG mode) on any failure.
        """
        try:
            from litellm.utils import ProviderConfigManager  # noqa: PLC0415

            _, provider, _, _ = litellm.get_llm_provider(model)
            config = ProviderConfigManager.get_provider_responses_api_config(
                model=model, provider=provider
            )
            if config is not None:
                return config.supports_native_file_search()
        except Exception as exc:
            verbose_logger.debug(
                f"VectorStorePreCallHook: could not determine provider for model={model!r}, "
                f"defaulting to RAG mode: {exc}"
            )
        return False

    # ------------------------------------------------------------------
    # HELPERS: VS RESOLUTION
    # ------------------------------------------------------------------

    async def _resolve_vector_stores(
        self, vs_ids: List[str]
    ) -> List[LiteLLM_ManagedVectorStore]:
        """Look up VS configs from registry + DB for the given IDs."""
        if litellm.vector_store_registry is None:
            return []
        results: List[LiteLLM_ManagedVectorStore] = []
        for vs_id in vs_ids:
            vs = await litellm.vector_store_registry.get_litellm_managed_vector_store_from_registry_or_db(
                vector_store_id=vs_id,
                prisma_client=_get_prisma_client(),
            )
            if vs is not None:
                results.append(vs)
            else:
                verbose_logger.debug(
                    f"VectorStorePreCallHook: vs_id={vs_id!r} not found in registry or DB"
                )
        return results

    # ------------------------------------------------------------------
    # POST-CALL HOOKS
    # ------------------------------------------------------------------

    def _append_search_results_to_messages(
        self,
        messages: List[AllMessageValues],
        search_response: VectorStoreSearchResponse,
    ) -> List[AllMessageValues]:
        """Append search results as context to the messages (chat completions)."""
        search_response_data: Optional[
            List[VectorStoreSearchResult]
        ] = search_response.get("data")
        if not search_response_data:
            return messages

        context_content = self.CONTENT_PREFIX_STRING

        for result in search_response_data:
            result_content: Optional[List[VectorStoreResultContent]] = result.get(
                "content"
            )
            if result_content:
                for content_item in result_content:
                    content_text: Optional[str] = content_item.get("text")
                    if content_text:
                        context_content += content_text + "\n\n"

        if context_content != "Context:\n\n":
            modified_messages = messages.copy()
            context_message: ChatCompletionUserMessage = {
                "role": "user",
                "content": context_content,
            }
            modified_messages.insert(-1, cast(AllMessageValues, context_message))
            return modified_messages

        return messages

    async def async_post_call_success_deployment_hook(
        self,
        request_data: dict,
        response: Any,
        call_type: Optional[Any],
    ) -> Optional[Any]:
        """
        Add search results to the response after a successful LLM call.

        Handles two storage paths:
        - Chat completions: results in litellm_logging_obj.model_call_details
        - Responses API: results in request_data["vs_search_results"]
        """
        try:
            verbose_logger.debug(
                "VectorStorePreCallHook.async_post_call_success_deployment_hook called"
            )

            # Responses API path: results stored in request_data by async_pre_call_hook
            search_results: Optional[List[VectorStoreSearchResponse]] = (
                request_data.get("vs_search_results")
            )

            # Chat completions path: results stored in litellm_logging_obj
            if not search_results:
                litellm_logging_obj = request_data.get("litellm_logging_obj")
                if litellm_logging_obj:
                    search_results = litellm_logging_obj.model_call_details.get(
                        "search_results"
                    )

            if not search_results:
                verbose_logger.debug(
                    "VectorStorePreCallHook: no search results to attach to response"
                )
                return None

            if hasattr(response, "choices") and response.choices:
                for choice in response.choices:
                    if hasattr(choice, "message") and choice.message:
                        provider_fields = (
                            getattr(choice.message, "provider_specific_fields", None)
                            or {}
                        )
                        provider_fields["search_results"] = search_results
                        setattr(
                            choice.message, "provider_specific_fields", provider_fields
                        )

            verbose_logger.debug(
                f"VectorStorePreCallHook: attached {len(search_results)} search result(s) to response"
            )
            return response

        except Exception as e:
            verbose_logger.exception(
                f"Error adding search results to response: {str(e)}"
            )
            return None

    async def async_post_call_streaming_deployment_hook(
        self,
        request_data: dict,
        response_chunk: Any,
        call_type: Optional[Any],
    ) -> Optional[Any]:
        """Add search results to the final streaming chunk."""
        try:
            verbose_logger.debug(
                "VectorStorePreCallHook.async_post_call_streaming_deployment_hook called"
            )

            search_results: Optional[List[VectorStoreSearchResponse]] = (
                request_data.get("vs_search_results")
                or request_data.get("search_results")
            )

            if not search_results:
                return response_chunk

            if hasattr(response_chunk, "choices") and response_chunk.choices:
                for choice in response_chunk.choices:
                    if hasattr(choice, "delta") and choice.delta:
                        provider_fields = (
                            getattr(choice.delta, "provider_specific_fields", None)
                            or {}
                        )
                        provider_fields["search_results"] = search_results
                        choice.delta.provider_specific_fields = provider_fields

            verbose_logger.debug(
                f"VectorStorePreCallHook: attached {len(search_results)} search result(s) to streaming chunk"
            )
            return response_chunk

        except Exception as e:
            verbose_logger.exception(
                f"Error adding search results to streaming chunk: {str(e)}"
            )
            return response_chunk

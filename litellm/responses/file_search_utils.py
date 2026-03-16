"""
Utilities for handling file_search tools in the Responses API for non-native providers.

For providers that do not natively support file_search (e.g. Claude, Gemini), LiteLLM
intercepts the request, searches the specified vector stores via litellm.vector_stores,
injects the retrieved context into the input, and strips the file_search tool before
forwarding to the provider.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import litellm
import litellm.vector_stores
from litellm._logging import verbose_logger
from litellm.types.llms.openai import ResponseInputParam
from litellm.types.vector_stores import (
    VectorStoreResultContent,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)

CONTEXT_PREFIX = "Context:\n\n"


class FileSearchResponsesAPIUtils:
    """Utilities for file_search tool interception in the Responses API."""

    @staticmethod
    def has_file_search_tools(tools: List[Dict[str, Any]]) -> bool:
        """Return True if any tool has type 'file_search'."""
        return any(
            isinstance(t, dict) and t.get("type") == "file_search" for t in tools
        )

    @staticmethod
    def split_file_search_tools(
        tools: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split tools into file_search tools and all other tools.

        Returns:
            (file_search_tools, other_tools)
        """
        file_search_tools: List[Dict[str, Any]] = []
        other_tools: List[Dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, dict) and tool.get("type") == "file_search":
                file_search_tools.append(tool)
            else:
                other_tools.append(tool)
        return file_search_tools, other_tools

    @staticmethod
    def extract_query_from_responses_input(
        input: Union[str, ResponseInputParam],
    ) -> Optional[str]:
        """
        Extract the query text from a Responses API input value.

        Handles both plain string inputs and list-of-input-item inputs.
        For list inputs the text of the last user message is returned.
        """
        if isinstance(input, str):
            return input

        if not isinstance(input, list) or len(input) == 0:
            return None

        # Walk backwards to find the last user message with text content
        for item in reversed(input):
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            if role != "user":
                continue
            content = item.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "text"
                        and part.get("text")
                    ):
                        return str(part["text"])
        return None

    @staticmethod
    def inject_context_into_responses_input(
        input: Union[str, ResponseInputParam],
        context: str,
    ) -> Union[str, ResponseInputParam]:
        """
        Inject a context message into a Responses API input.

        The context is inserted as a user message immediately before the last item
        in the input list.  If input is a plain string it is first wrapped in a
        single-element list so that injection is consistent.
        """
        context_item: Dict[str, Any] = {
            "role": "user",
            "content": context,
        }

        if isinstance(input, str):
            return [context_item, {"role": "user", "content": input}]

        if not isinstance(input, list):
            return input

        modified = list(input)
        if len(modified) == 0:
            modified.append(context_item)
        else:
            modified.insert(len(modified) - 1, context_item)
        return modified

    @staticmethod
    def _build_context_from_search_results(
        results: List[VectorStoreSearchResponse],
    ) -> str:
        """Build a plain-text context string from a list of vector store search responses."""
        context = CONTEXT_PREFIX
        for search_response in results:
            data: Optional[List[VectorStoreSearchResult]] = search_response.get("data")
            if not data:
                continue
            for result in data:
                content_items: Optional[List[VectorStoreResultContent]] = result.get("content")
                if not content_items:
                    continue
                for content_item in content_items:
                    text: Optional[str] = content_item.get("text")
                    if text:
                        context += text + "\n\n"
        return context

    @staticmethod
    async def asearch_and_inject_context(
        input: Union[str, ResponseInputParam],
        tools: List[Dict[str, Any]],
        litellm_logging_obj: Optional[Any] = None,
    ) -> Tuple[Union[str, ResponseInputParam], List[Dict[str, Any]]]:
        """
        Intercept file_search tools for a non-native provider.

        Steps:
        1. Split file_search tools from remaining tools.
        2. Extract the query from the input.
        3. For every vector_store_id found across all file_search tools, run
           litellm.vector_stores.asearch().
        4. Build a context string from all results.
        5. Inject the context into the input.
        6. Return (modified_input, other_tools) — file_search tools are dropped.

        If no query can be extracted, or no vector_store_ids are found, the
        original (input, tools) is returned unchanged.
        """
        file_search_tools, other_tools = FileSearchResponsesAPIUtils.split_file_search_tools(tools)

        if not file_search_tools:
            return input, tools

        query = FileSearchResponsesAPIUtils.extract_query_from_responses_input(input)
        if not query:
            verbose_logger.debug(
                "FileSearchResponsesAPIUtils: no query found in input; skipping vector store search"
            )
            return input, other_tools

        # Collect all vector_store_ids from all file_search tools
        vector_store_ids: List[str] = []
        for fs_tool in file_search_tools:
            ids = fs_tool.get("vector_store_ids") or []
            vector_store_ids.extend(ids)

        if not vector_store_ids:
            verbose_logger.debug(
                "FileSearchResponsesAPIUtils: file_search tool has no vector_store_ids; stripping tool only"
            )
            return input, other_tools

        all_search_results: List[VectorStoreSearchResponse] = []
        for vector_store_id in vector_store_ids:
            try:
                search_response: VectorStoreSearchResponse = (
                    await litellm.vector_stores.asearch(
                        vector_store_id=vector_store_id,
                        query=query,
                    )
                )
                verbose_logger.debug(
                    "FileSearchResponsesAPIUtils: searched vector_store_id=%s, got %d results",
                    vector_store_id,
                    len(search_response.get("data") or []),
                )
                all_search_results.append(search_response)
            except Exception as e:
                verbose_logger.exception(
                    "FileSearchResponsesAPIUtils: error searching vector_store_id=%s: %s",
                    vector_store_id,
                    str(e),
                )

        if not all_search_results:
            return input, other_tools

        context = FileSearchResponsesAPIUtils._build_context_from_search_results(all_search_results)

        # Only inject if we found actual content beyond the prefix
        if context == CONTEXT_PREFIX:
            return input, other_tools

        modified_input = FileSearchResponsesAPIUtils.inject_context_into_responses_input(
            input=input, context=context
        )

        # Store results in logging object for downstream hooks / observability
        if litellm_logging_obj is not None:
            try:
                litellm_logging_obj.model_call_details["search_results"] = all_search_results
            except Exception:
                pass

        return modified_input, other_tools

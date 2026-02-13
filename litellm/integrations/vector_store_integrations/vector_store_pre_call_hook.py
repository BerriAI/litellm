"""
Vector Store Pre-Call Hook

This hook is called before making an LLM request when a vector store is configured.
It searches the vector store for relevant context and appends it to the messages.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import litellm
import litellm.vector_stores
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.llms.openai import AllMessageValues, ChatCompletionUserMessage
from litellm.types.prompts.init_prompts import PromptSpec
from litellm.types.utils import StandardCallbackDynamicParams
from litellm.types.vector_stores import (
    LiteLLM_ManagedVectorStore,
    VectorStoreResultContent,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class VectorStorePreCallHook(CustomLogger):
    CONTENT_PREFIX_STRING = "Context:\n\n"
    """
    Custom logger that handles vector store searches before LLM calls.
    
    When a vector store is configured, this hook:
    1. Extracts the query from the last user message
    2. Calls litellm.vector_stores.search() to get relevant context
    3. Appends the search results as context to the messages
    """

    def __init__(self):
        super().__init__()

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

        Args:
            model: The model name
            messages: List of messages
            non_default_params: Non-default parameters
            prompt_id: Optional prompt ID
            prompt_variables: Optional prompt variables
            dynamic_callback_params: Optional dynamic callback parameters
            prompt_label: Optional prompt label
            prompt_version: Optional prompt version

        Returns:
            Tuple of (model, modified_messages, non_default_params)
        """
        try:
            # Check if vector store is configured
            if litellm.vector_store_registry is None:
                return model, messages, non_default_params

            # Get prisma_client for database fallback
            prisma_client = None
            try:
                from litellm.proxy.proxy_server import prisma_client as _prisma_client
                prisma_client = _prisma_client
            except ImportError:
                pass

            # Use database fallback to ensure synchronization across instances
            vector_stores_to_run: List[LiteLLM_ManagedVectorStore] = (
                await litellm.vector_store_registry.pop_vector_stores_to_run_with_db_fallback(
                    non_default_params=non_default_params, 
                    tools=tools,
                    prisma_client=prisma_client
                )
            )

            if not vector_stores_to_run:
                return model, messages, non_default_params

            # Extract the query from the last user message
            query = self._extract_query_from_messages(messages)

            if not query:
                verbose_logger.debug(
                    "No query found in messages for vector store search"
                )
                return model, messages, non_default_params

            modified_messages: List[AllMessageValues] = messages.copy()
            all_search_results: List[VectorStoreSearchResponse] = []

            for vector_store_to_run in vector_stores_to_run:

                # Get vector store id from the vector store config
                vector_store_id = vector_store_to_run.get("vector_store_id", "")
                custom_llm_provider = vector_store_to_run.get("custom_llm_provider")
                litellm_params_for_vector_store = (
                    vector_store_to_run.get("litellm_params", {}) or {}
                )
                # Call litellm.vector_stores.search() with the required parameters
                search_response = await litellm.vector_stores.asearch(
                    **{
                        "vector_store_id": vector_store_id,
                        "query": query,
                        "custom_llm_provider": custom_llm_provider,
                        **litellm_params_for_vector_store,
                    },
                )

                verbose_logger.debug(f"search_response: {search_response}")

                # Store search results for later use in citations
                all_search_results.append(search_response)

                # Process search results and append as context
                modified_messages = self._append_search_results_to_messages(
                    messages=messages, search_response=search_response
                )

                # Get the number of results for logging
                num_results = 0
                num_results = len(search_response.get("data", []) or [])
                verbose_logger.debug(
                    f"Vector store search completed. Added context from {num_results} results"
                )

            # Store search results as-is (already in OpenAI-compatible format)
            if litellm_logging_obj and all_search_results:
                litellm_logging_obj.model_call_details["search_results"] = (
                    all_search_results
                )

            return model, modified_messages, non_default_params

        except Exception as e:
            verbose_logger.exception(f"Error in VectorStorePreCallHook: {str(e)}")
            # Return original parameters on error
            return model, messages, non_default_params

    def _extract_query_from_messages(
        self, messages: List[AllMessageValues]
    ) -> Optional[str]:
        """
        Extract the query from the last user message.

        Args:
            messages: List of messages

        Returns:
            The extracted query string or None if not found
        """
        if not messages or len(messages) == 0:
            return None

        last_message = messages[-1]
        if not isinstance(last_message, dict) or "content" not in last_message:
            return None

        content = last_message["content"]

        if isinstance(content, str):
            return content
        elif isinstance(content, list) and len(content) > 0:
            # Handle list of content items, extract text from first text item
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                    and "text" in item
                ):
                    return item["text"]

        return None

    def _append_search_results_to_messages(
        self,
        messages: List[AllMessageValues],
        search_response: VectorStoreSearchResponse,
    ) -> List[AllMessageValues]:
        """
        Append search results as context to the messages.

        Args:
            messages: Original list of messages
            search_response: Response from vector store search

        Returns:
            Modified list of messages with context appended
        """
        search_response_data: Optional[List[VectorStoreSearchResult]] = (
            search_response.get("data")
        )
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

        # Only add context if we found any content
        if context_content != "Context:\n\n":
            # Create a copy of messages to avoid modifying the original
            modified_messages = messages.copy()
            # Add context as a new message before the last user message
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
        Add search results to the response after successful LLM call.

        This hook adds the vector store search results (already in OpenAI-compatible format)
        to the response's provider_specific_fields.
        """
        try:
            verbose_logger.debug(
                "VectorStorePreCallHook.async_post_call_success_deployment_hook called"
            )

            # Get logging object from request_data
            litellm_logging_obj = request_data.get("litellm_logging_obj")
            if not litellm_logging_obj:
                verbose_logger.debug("No litellm_logging_obj in request_data")
                return None

            verbose_logger.debug(
                f"model_call_details keys: {list(litellm_logging_obj.model_call_details.keys())}"
            )

            # Get search results from model_call_details (already in OpenAI format)
            search_results: Optional[List[VectorStoreSearchResponse]] = (
                litellm_logging_obj.model_call_details.get("search_results")
            )

            verbose_logger.debug(f"Search results found: {search_results is not None}")

            if not search_results:
                verbose_logger.debug("No search results found")
                return None

            # Add search results to response object
            if hasattr(response, "choices") and response.choices:
                for choice in response.choices:
                    if hasattr(choice, "message") and choice.message:
                        # Get existing provider_specific_fields or create new dict
                        provider_fields = (
                            getattr(choice.message, "provider_specific_fields", None)
                            or {}
                        )

                        # Add search results (already in OpenAI-compatible format)
                        provider_fields["search_results"] = search_results

                        # Set the provider_specific_fields
                        setattr(
                            choice.message, "provider_specific_fields", provider_fields
                        )

            verbose_logger.debug(
                f"Added {len(search_results)} search results to response"
            )

            # Return modified response
            return response

        except Exception as e:
            verbose_logger.exception(
                f"Error adding search results to response: {str(e)}"
            )
            # Don't fail the request if search results fail to be added
            return None

    async def async_post_call_streaming_deployment_hook(
        self,
        request_data: dict,
        response_chunk: Any,
        call_type: Optional[Any],
    ) -> Optional[Any]:
        """
        Add search results to the final streaming chunk.

        This hook is called for the final streaming chunk, allowing us to add
        search results to the stream before it's returned to the user.
        """
        try:
            verbose_logger.debug(
                "VectorStorePreCallHook.async_post_call_streaming_deployment_hook called"
            )

            # Get search results from model_call_details (already in OpenAI format)
            search_results: Optional[List[VectorStoreSearchResponse]] = (
                request_data.get("search_results")
            )

            verbose_logger.debug(
                f"Search results found for streaming chunk: {search_results is not None}"
            )

            if not search_results:
                verbose_logger.debug("No search results found for streaming chunk")
                return response_chunk

            # Add search results to streaming chunk
            if hasattr(response_chunk, "choices") and response_chunk.choices:
                for choice in response_chunk.choices:
                    if hasattr(choice, "delta") and choice.delta:
                        # Get existing provider_specific_fields or create new dict
                        provider_fields = (
                            getattr(choice.delta, "provider_specific_fields", None)
                            or {}
                        )

                        # Add search results (already in OpenAI-compatible format)
                        provider_fields["search_results"] = search_results

                        # Set the provider_specific_fields
                        choice.delta.provider_specific_fields = provider_fields

            verbose_logger.debug(
                f"Added {len(search_results)} search results to streaming chunk"
            )

            # Return modified chunk
            return response_chunk

        except Exception as e:
            verbose_logger.exception(
                f"Error adding search results to streaming chunk: {str(e)}"
            )
            # Don't fail the request if search results fail to be added
            return response_chunk

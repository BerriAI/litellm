"""
Vector Store Pre-Call Hook

This hook is called before making an LLM request when a vector store is configured.
It searches the vector store for relevant context and appends it to the messages.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, cast, get_args

from pydantic import TypeAdapter, ValidationError
from typing_extensions import assert_never

import litellm
import litellm.vector_stores
from litellm._logging import verbose_logger
from litellm.exceptions import VectorStoreSearchError
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.llms.openai import AllMessageValues, ChatCompletionUserMessage
from litellm.types.prompts.init_prompts import PromptSpec
from litellm.types.utils import CallTypes, StandardCallbackDynamicParams
from litellm.types.vector_stores import (
    LiteLLM_ManagedVectorStore,
    VectorStoreSearchFailure,
    VectorStoreSearchFailureMode,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy.utils import PrismaClient
    from litellm.router import Router
else:
    LiteLLMLoggingObj = Any

SEARCH_FAILURES_FIELD: Final = "vector_store_search_failures"
_DEFAULT_FAILURE_MODE: Final[VectorStoreSearchFailureMode] = "annotate"
_FAILURE_MODE_ADAPTER: Final = TypeAdapter(VectorStoreSearchFailureMode)


class ProxyRuntime(Protocol):
    def llm_router(self) -> "Router | None": ...

    def prisma_client(self) -> "PrismaClient | None": ...


@dataclass(frozen=True, slots=True)
class ProxyServerRuntime:
    def llm_router(self) -> "Router | None":
        try:
            from litellm.proxy.proxy_server import llm_router
        except ImportError:
            return None
        return llm_router

    def prisma_client(self) -> "PrismaClient | None":
        try:
            from litellm.proxy.proxy_server import prisma_client
        except ImportError:
            return None
        return prisma_client


@dataclass(frozen=True, slots=True)
class SearchSucceeded:
    response: VectorStoreSearchResponse


@dataclass(frozen=True, slots=True)
class SearchFailed:
    failure: VectorStoreSearchFailure


SearchOutcome = SearchSucceeded | SearchFailed


@dataclass(frozen=True, slots=True)
class VectorStoreAugmentation:
    messages: tuple[AllMessageValues, ...]
    search_results: tuple[VectorStoreSearchResponse, ...]
    failures: tuple[VectorStoreSearchFailure, ...]


class VectorStorePreCallHook(CustomLogger):
    CONTENT_PREFIX_STRING = "Context:\n\n"
    """
    Custom logger that handles vector store searches before LLM calls.

    When a vector store is configured, this hook:
    1. Extracts the query from the last user message
    2. Calls litellm.vector_stores.search() to get relevant context
    3. Appends the search results as context to the messages
    """

    def __init__(self, proxy_runtime: ProxyRuntime | None = None):
        super().__init__()
        self.proxy_runtime: Final[ProxyRuntime] = proxy_runtime or ProxyServerRuntime()

    async def async_get_chat_completion_prompt(
        self,
        model: str,
        messages: list[AllMessageValues],
        non_default_params: dict,
        prompt_id: str | None,
        prompt_variables: dict | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
        litellm_logging_obj: LiteLLMLoggingObj,
        prompt_spec: PromptSpec | None = None,
        tools: list[dict] | None = None,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
        ignore_prompt_manager_model: bool | None = False,
        ignore_prompt_manager_optional_params: bool | None = False,
    ) -> tuple[str, list[AllMessageValues], dict]:
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
        requested_vector_store_ids: Final = _requested_vector_store_ids(non_default_params)
        try:
            augmentation: VectorStoreAugmentation | None = await self._augment_messages(
                messages=messages,
                non_default_params=non_default_params,
                tools=tools,
                litellm_logging_obj=litellm_logging_obj,
            )
        except Exception as e:
            verbose_logger.exception(
                "Error in VectorStorePreCallHook for vector_store_ids=%s: %s",
                requested_vector_store_ids,
                e,
            )
            return model, messages, non_default_params

        if augmentation is None:
            return model, messages, non_default_params

        for detail, value in (
            ("search_results", list(augmentation.search_results)),
            (SEARCH_FAILURES_FIELD, augmentation.failures),
        ):
            if value:
                litellm_logging_obj.model_call_details[detail] = value

        if augmentation.failures:
            match _configured_failure_mode():
                case "error":
                    raise VectorStoreSearchError(failures=augmentation.failures, model=model)
                case "annotate":
                    pass
                case unreachable:
                    assert_never(unreachable)

        return model, list(augmentation.messages), non_default_params

    async def _augment_messages(
        self,
        messages: Sequence[AllMessageValues],
        non_default_params: dict,
        tools: list[dict] | None,
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> VectorStoreAugmentation | None:
        if litellm.vector_store_registry is None:
            return None

        prisma_client: Final = self.proxy_runtime.prisma_client()
        llm_router: Final = self.proxy_runtime.llm_router()

        # Use database fallback to ensure synchronization across instances
        vector_stores_to_run: Final[
            Sequence[LiteLLM_ManagedVectorStore]
        ] = await litellm.vector_store_registry.pop_vector_stores_to_run_with_db_fallback(
            non_default_params=non_default_params,
            tools=tools,
            prisma_client=prisma_client,
        )

        if not vector_stores_to_run:
            return None

        query: Final = self._extract_query_from_messages(messages)

        if not query:
            verbose_logger.debug("No query found in messages for vector store search")
            return None

        request_litellm_params: Final = litellm_logging_obj.model_call_details.get("litellm_params", {})
        request_metadata: Final = (
            request_litellm_params.get("metadata", {}) if isinstance(request_litellm_params, dict) else {}
        )
        search_function: Final = (
            cast(  # cast-ok: normalize router search callable
                Callable[..., Awaitable[VectorStoreSearchResponse]],
                llm_router.avector_store_search,
            )
            if llm_router is not None
            else cast(  # cast-ok: normalize SDK search callable
                Callable[..., Awaitable[VectorStoreSearchResponse]],
                litellm.vector_stores.asearch,
            )
        )

        outcomes: Final = tuple(
            [
                await self._search_one(
                    vector_store=vector_store_to_run,
                    query=query,
                    request_metadata=request_metadata,
                    search_function=search_function,
                )
                for vector_store_to_run in vector_stores_to_run
            ]
        )
        search_results: Final = tuple(outcome.response for outcome in outcomes if isinstance(outcome, SearchSucceeded))
        failures: Final = tuple(outcome.failure for outcome in outcomes if isinstance(outcome, SearchFailed))

        return VectorStoreAugmentation(
            messages=self._messages_with_context(messages=messages, search_results=search_results),
            search_results=search_results,
            failures=failures,
        )

    async def _search_one(
        self,
        vector_store: LiteLLM_ManagedVectorStore,
        query: str,
        request_metadata: Mapping[str, object],
        search_function: Callable[..., Awaitable[VectorStoreSearchResponse]],
    ) -> SearchOutcome:
        vector_store_id: Final = vector_store.get("vector_store_id", "")
        custom_llm_provider: Final = vector_store.get("custom_llm_provider")
        litellm_params_for_vector_store: Final = vector_store.get("litellm_params", {}) or {}
        try:
            search_response: Final = await search_function(
                **{
                    "vector_store_id": vector_store_id,
                    "query": query,
                    "custom_llm_provider": custom_llm_provider,
                    "metadata": request_metadata,
                    **litellm_params_for_vector_store,
                },
            )
        except Exception as search_error:
            verbose_logger.warning(
                "Vector store search failed for vector_store_id=%s, continuing without its context: %s",
                vector_store_id,
                search_error,
            )
            return SearchFailed(
                failure=VectorStoreSearchFailure(
                    vector_store_id=vector_store_id,
                    custom_llm_provider=custom_llm_provider,
                    error=str(search_error),
                )
            )

        verbose_logger.debug(
            "Vector store search completed for vector_store_id=%s. Added context from %s results",
            vector_store_id,
            len(search_response.get("data", []) or []),
        )
        return SearchSucceeded(response=search_response)

    def _extract_query_from_messages(self, messages: Sequence[AllMessageValues]) -> str | None:
        """
        Extract the query from the last user message.

        Args:
            messages: List of messages

        Returns:
            The extracted query string or None if not found
        """
        if not messages or len(messages) == 0:
            return None

        last_message: Final = messages[-1]
        if not isinstance(last_message, dict) or "content" not in last_message:
            return None

        content: Final = last_message["content"]

        if isinstance(content, str):
            return content
        elif isinstance(content, list) and len(content) > 0:
            # Handle list of content items, extract text from first text item
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                    return item["text"]

        return None

    def _messages_with_context(
        self,
        messages: Sequence[AllMessageValues],
        search_results: Sequence[VectorStoreSearchResponse],
    ) -> tuple[AllMessageValues, ...]:
        context_messages: Final = tuple(
            context_message
            for search_response in search_results
            if (context_message := self._context_message(search_response)) is not None
        )
        if not context_messages:
            return tuple(messages)
        return (*messages[:-1], *context_messages, *messages[-1:])

    def _context_message(self, search_response: VectorStoreSearchResponse) -> AllMessageValues | None:
        """Build the context message for one vector store's results, or None when it returned nothing usable."""
        search_response_data: Final[Sequence[VectorStoreSearchResult] | None] = search_response.get("data")
        if not search_response_data:
            return None

        context_texts: Final = tuple(
            content_text
            for result in search_response_data
            for content_item in (result.get("content") or ())
            if (content_text := content_item.get("text"))
        )
        if not context_texts:
            return None

        context_message: Final[ChatCompletionUserMessage] = {
            "role": "user",
            "content": self.CONTENT_PREFIX_STRING + "".join(f"{text}\n\n" for text in context_texts),
        }
        return cast(AllMessageValues, context_message)

    async def async_post_call_success_deployment_hook(
        self,
        request_data: dict,
        response: Any,
        call_type: CallTypes | None,
    ) -> Any | None:
        """
        Add search results to the response after successful LLM call.

        This hook adds the vector store search results (already in OpenAI-compatible format)
        to the response's provider_specific_fields.
        """
        try:
            verbose_logger.debug("VectorStorePreCallHook.async_post_call_success_deployment_hook called")

            # Get logging object from request_data
            litellm_logging_obj: Final = request_data.get("litellm_logging_obj")
            if not litellm_logging_obj:
                verbose_logger.debug("No litellm_logging_obj in request_data")
                return None

            # Get search results from model_call_details (already in OpenAI format)
            search_results: Final[Sequence[VectorStoreSearchResponse] | None] = (
                litellm_logging_obj.model_call_details.get("search_results")
            )
            search_failures: Final[Sequence[VectorStoreSearchFailure] | None] = (
                litellm_logging_obj.model_call_details.get(SEARCH_FAILURES_FIELD)
            )

            if not search_results and not search_failures:
                verbose_logger.debug("No search results or search failures found")
                return None

            # Add search results to response object
            if hasattr(response, "choices") and response.choices:
                for choice in response.choices:
                    if hasattr(choice, "message") and choice.message:
                        provider_fields = getattr(choice.message, "provider_specific_fields", None) or {}
                        if search_results:
                            provider_fields["search_results"] = search_results
                        if search_failures:
                            provider_fields[SEARCH_FAILURES_FIELD] = search_failures
                        setattr(choice.message, "provider_specific_fields", provider_fields)

            # Return modified response
            return response

        except Exception as e:
            verbose_logger.exception("Error adding search results to response: %s", e)
            # Don't fail the request if search results fail to be added
            return None

    async def async_post_call_streaming_deployment_hook(
        self,
        request_data: dict,
        response_chunk: Any,
        call_type: CallTypes | None,
    ) -> Any | None:
        """
        Add search results to the final streaming chunk.

        This hook is called for the final streaming chunk, allowing us to add
        search results to the stream before it's returned to the user.
        """
        try:
            verbose_logger.debug("VectorStorePreCallHook.async_post_call_streaming_deployment_hook called")

            # Get search results from model_call_details (already in OpenAI format)
            search_results: Final[Sequence[VectorStoreSearchResponse] | None] = request_data.get("search_results")
            search_failures: Final[Sequence[VectorStoreSearchFailure] | None] = request_data.get(SEARCH_FAILURES_FIELD)

            if not search_results and not search_failures:
                verbose_logger.debug("No search results or search failures found for streaming chunk")
                return response_chunk

            # Add search results to streaming chunk
            if hasattr(response_chunk, "choices") and response_chunk.choices:
                for choice in response_chunk.choices:
                    if hasattr(choice, "delta") and choice.delta:
                        provider_fields = getattr(choice.delta, "provider_specific_fields", None) or {}
                        if search_results:
                            provider_fields["search_results"] = search_results
                        if search_failures:
                            provider_fields[SEARCH_FAILURES_FIELD] = search_failures
                        choice.delta.provider_specific_fields = provider_fields

            # Return modified chunk
            return response_chunk

        except Exception as e:
            verbose_logger.exception("Error adding search results to streaming chunk: %s", e)
            # Don't fail the request if search results fail to be added
            return response_chunk


def _requested_vector_store_ids(non_default_params: Mapping[str, object]) -> tuple[str, ...]:
    requested: Final = non_default_params.get("vector_store_ids")
    if not isinstance(requested, (list, tuple)):
        return ()
    return tuple(str(vector_store_id) for vector_store_id in requested)


def _configured_failure_mode() -> VectorStoreSearchFailureMode:
    try:
        return _FAILURE_MODE_ADAPTER.validate_python(litellm.vector_store_search_failure_mode)
    except ValidationError:
        verbose_logger.warning(
            "Unsupported vector_store_search_failure_mode=%r, falling back to %r. Supported modes: %s",
            litellm.vector_store_search_failure_mode,
            _DEFAULT_FAILURE_MODE,
            ", ".join(get_args(VectorStoreSearchFailureMode)),
        )
        return _DEFAULT_FAILURE_MODE

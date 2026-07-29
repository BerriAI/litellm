"""
RAG Ingest API for LiteLLM.

Provides an all-in-one API for document ingestion:
Upload -> (OCR) -> Chunk -> Embed -> Vector Store
"""

from __future__ import annotations

__all__ = ["ingest", "aingest", "query", "aquery"]

import asyncio
import contextvars
from contextlib import contextmanager
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Coroutine,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import httpx

import litellm
from litellm._internal_context import is_internal_call
from litellm.cost_calculator import vector_store_search_cost
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.rag.ingestion.base_ingestion import BaseRAGIngestion
from litellm.rag.ingestion.bedrock_ingestion import BedrockRAGIngestion
from litellm.rag.ingestion.gemini_ingestion import GeminiRAGIngestion
from litellm.rag.ingestion.openai_ingestion import OpenAIRAGIngestion
from litellm.rag.ingestion.s3_vectors_ingestion import S3VectorsRAGIngestion
from litellm.rag.ingestion.vertex_ai_ingestion import VertexAIRAGIngestion
from litellm.rag.rag_query import RAGQuery
from litellm.types.rag import (
    RAGIngestOptions,
    RAGIngestResponse,
)
from litellm.types.utils import ModelResponse
from litellm.utils import client

if TYPE_CHECKING:
    from litellm import Router


# Registry of provider-specific ingestion classes
INGESTION_REGISTRY: Dict[str, Type[BaseRAGIngestion]] = {
    "openai": OpenAIRAGIngestion,
    "bedrock": BedrockRAGIngestion,
    "gemini": GeminiRAGIngestion,
    "s3_vectors": S3VectorsRAGIngestion,
    "vertex_ai": VertexAIRAGIngestion,
}


def get_ingestion_class(provider: str) -> Type[BaseRAGIngestion]:
    """
    Get the ingestion class for a given provider.

    Args:
        provider: The vector store provider name (e.g., 'openai')

    Returns:
        The ingestion class for the provider

    Raises:
        ValueError: If provider is not supported
    """
    ingestion_class = INGESTION_REGISTRY.get(provider)
    if ingestion_class is None:
        supported = ", ".join(INGESTION_REGISTRY.keys())
        raise ValueError(f"Provider '{provider}' is not supported for RAG ingestion. Supported providers: {supported}")
    return ingestion_class


async def _execute_ingest_pipeline(
    ingest_options: RAGIngestOptions,
    file_data: Optional[Tuple[str, bytes, str]] = None,
    file_url: Optional[str] = None,
    file_id: Optional[str] = None,
    router: Optional["Router"] = None,
) -> RAGIngestResponse:
    """
    Execute the RAG ingest pipeline using provider-specific implementation.

    Args:
        ingest_options: Configuration for the ingest pipeline
        file_data: Tuple of (filename, content_bytes, content_type)
        file_url: URL to fetch file from
        file_id: Existing file ID to use
        router: Optional LiteLLM router for load balancing

    Returns:
        RAGIngestResponse with status and IDs
    """
    # Get provider from vector store config
    vector_store_config = ingest_options.get("vector_store") or {}
    provider = vector_store_config.get("custom_llm_provider", "openai")

    # Get provider-specific ingestion class
    ingestion_class = get_ingestion_class(provider)

    # Create ingestion instance
    ingestion = ingestion_class(
        ingest_options=ingest_options,
        router=router,
    )

    # Execute ingestion pipeline
    return await ingestion.ingest(
        file_data=file_data,
        file_url=file_url,
        file_id=file_id,
    )


####### PUBLIC API ###################


@client
async def aingest(
    ingest_options: Dict[str, Any],
    file_data: Optional[Tuple[str, bytes, str]] = None,
    file: Optional[Dict[str, str]] = None,
    file_url: Optional[str] = None,
    file_id: Optional[str] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    **kwargs,
) -> RAGIngestResponse:
    """
    Async: Ingest a document into a vector store.

    Args:
        ingest_options: Configuration for the ingest pipeline
        file_data: Tuple of (filename, content_bytes, content_type)
        file: Dict with {filename, content (base64), content_type} - for JSON API
        file_url: URL to fetch file from
        file_id: Existing file ID to use

    Example:
        ```python
        response = await litellm.aingest(
            ingest_options={
                "vector_store": {
                    "custom_llm_provider": "openai",
                    "litellm_credential_name": "my-openai-creds",  # optional
                }
            },
            file_url="https://example.com/doc.pdf",
        )
        ```
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["aingest"] = True

        func = partial(
            ingest,
            ingest_options=ingest_options,
            file_data=file_data,
            file=file,
            file_url=file_url,
            file_id=file_id,
            timeout=timeout,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=ingest_options.get("vector_store", {}).get("custom_llm_provider"),
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@contextmanager
def _suppressed_sub_call_billing() -> Iterator[None]:
    """
    Suppress a sub-call's own billing event so the parent aquery event bills it.

    Every suppressed sub-call's cost must be folded into the parent event:
    into the response's hidden response_cost on the non-streaming path, or via
    the logging object's additional_response_cost on the streaming path (the
    streamed cost is computed from assembled chunks after this pipeline
    returns, so there is no response object to fold into here).
    """
    previous = is_internal_call.get()
    is_internal_call.set(True)
    try:
        yield
    finally:
        is_internal_call.set(previous)


async def _execute_query_pipeline(
    model: str,
    messages: List[Any],
    retrieval_config: Dict[str, Any],
    rerank: Optional[Dict[str, Any]] = None,
    stream: bool = False,
    **kwargs,
) -> ModelResponse:
    """
    Execute the RAG query pipeline.
    """
    # Extract router from kwargs - use it for completion if available
    # to properly resolve virtual model names
    router: Optional["Router"] = kwargs.pop("router", None)

    # 1. Extract query from last user message
    query_text = RAGQuery.extract_query_from_messages(messages)
    if not query_text:
        raise ValueError("No query found in messages for RAG query")

    # 2. Search vector store
    with _suppressed_sub_call_billing():
        search_response = await litellm.vector_stores.asearch(
            vector_store_id=retrieval_config["vector_store_id"],
            query=query_text,
            max_num_results=retrieval_config.get("top_k", 10),
            custom_llm_provider=retrieval_config.get("custom_llm_provider", "openai"),
            **kwargs,
        )

    search_provider = retrieval_config.get("custom_llm_provider", "openai")
    try:
        search_cost = sum(
            vector_store_search_cost(
                model=search_provider if "/" in search_provider else None,
                custom_llm_provider=search_provider,
                response=search_response,
            )
        )
    except Exception:  # noqa: BLE001 - cost accounting must never break the query path
        search_cost = 0.0

    rerank_response = None
    rerank_cost = 0.0
    context_chunks = search_response.get("data", [])

    # 3. Optional rerank
    if rerank and rerank.get("enabled"):
        documents = RAGQuery.extract_documents_from_search(search_response)
        if documents:
            with _suppressed_sub_call_billing():
                rerank_response = await litellm.arerank(
                    model=rerank["model"],
                    query=query_text,
                    documents=documents,
                    top_n=rerank.get("top_n", 5),
                )
            rerank_hidden_params = getattr(rerank_response, "_hidden_params", None)
            if isinstance(rerank_hidden_params, dict):
                rerank_response_cost: float | None = rerank_hidden_params.get("response_cost")
                rerank_cost = rerank_response_cost or 0.0
            context_chunks = RAGQuery.get_top_chunks_from_rerank(search_response, rerank_response)

    # 4. Build context message and call completion
    context_message = RAGQuery.build_context_message(context_chunks)
    modified_messages = messages[:-1] + [context_message] + [messages[-1]]

    # Use router if available to properly resolve virtual model names
    with _suppressed_sub_call_billing():
        if router is not None:
            response = await router.acompletion(
                model=model,
                messages=modified_messages,
                stream=stream,
                **kwargs,
            )
        else:
            response = await litellm.acompletion(
                model=model,
                messages=modified_messages,
                stream=stream,
                **kwargs,
            )

    # 5. Attach search results to response
    sub_call_cost = search_cost + rerank_cost
    if not stream and isinstance(response, ModelResponse):
        response = RAGQuery.add_search_results_to_response(
            response=response,
            search_results=search_response,
            rerank_results=rerank_response,
        )
        if sub_call_cost > 0:
            hidden_params = getattr(response, "_hidden_params", None)
            if isinstance(hidden_params, dict):
                completion_response_cost: float | None = hidden_params.get("response_cost")
                if completion_response_cost is not None:
                    hidden_params["response_cost"] = completion_response_cost + sub_call_cost
    elif sub_call_cost > 0:
        logging_obj: object = kwargs.get("litellm_logging_obj")
        if isinstance(logging_obj, LiteLLMLoggingObj):
            logging_obj.model_call_details["additional_response_cost"] = sub_call_cost

    return response  # type: ignore[return-value]


@client
async def aquery(
    model: str,
    messages: List[Any],
    retrieval_config: Dict[str, Any],
    rerank: Optional[Dict[str, Any]] = None,
    stream: bool = False,
    **kwargs,
) -> ModelResponse:
    """
    Async: Query a RAG pipeline.
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["aquery"] = True

        func = partial(
            query,
            model=model,
            messages=messages,
            retrieval_config=retrieval_config,
            rerank=rerank,
            stream=stream,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=retrieval_config.get("custom_llm_provider"),
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def query(
    model: str,
    messages: List[Any],
    retrieval_config: Dict[str, Any],
    rerank: Optional[Dict[str, Any]] = None,
    stream: bool = False,
    **kwargs,
) -> Union[ModelResponse, Coroutine[Any, Any, ModelResponse]]:
    """
    Query a RAG pipeline.
    """
    local_vars = locals()
    try:
        _is_async = kwargs.pop("aquery", False) is True

        if _is_async:
            return _execute_query_pipeline(
                model=model,
                messages=messages,
                retrieval_config=retrieval_config,
                rerank=rerank,
                stream=stream,
                **kwargs,
            )
        else:
            return asyncio.get_event_loop().run_until_complete(
                _execute_query_pipeline(
                    model=model,
                    messages=messages,
                    retrieval_config=retrieval_config,
                    rerank=rerank,
                    stream=stream,
                    **kwargs,
                )
            )
    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=retrieval_config.get("custom_llm_provider"),
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def ingest(
    ingest_options: Dict[str, Any],
    file_data: Optional[Tuple[str, bytes, str]] = None,
    file: Optional[Dict[str, str]] = None,
    file_url: Optional[str] = None,
    file_id: Optional[str] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    **kwargs,
) -> Union[RAGIngestResponse, Coroutine[Any, Any, RAGIngestResponse]]:
    """
    Ingest a document into a vector store.

    Args:
        ingest_options: Configuration for the ingest pipeline
        file_data: Tuple of (filename, content_bytes, content_type)
        file: Dict with {filename, content (base64), content_type} - for JSON API
        file_url: URL to fetch file from
        file_id: Existing file ID to use

    Example:
        ```python
        response = litellm.ingest(
            ingest_options={
                "vector_store": {
                    "custom_llm_provider": "openai",
                    "litellm_credential_name": "my-openai-creds",  # optional
                }
            },
            file_data=("doc.txt", b"Hello world", "text/plain"),
        )
        ```
    """
    import base64

    local_vars = locals()
    try:
        _is_async = kwargs.pop("aingest", False) is True
        router: Optional["Router"] = kwargs.get("router")

        # Convert file dict to file_data tuple if provided
        if file is not None and file_data is None:
            filename = file.get("filename", "document")
            content_b64 = file.get("content", "")
            content_type = file.get("content_type", "application/octet-stream")
            content_bytes = base64.b64decode(content_b64)
            file_data = (filename, content_bytes, content_type)

        if _is_async:
            return _execute_ingest_pipeline(
                ingest_options=ingest_options,  # type: ignore
                file_data=file_data,
                file_url=file_url,
                file_id=file_id,
                router=router,
            )
        else:
            return asyncio.get_event_loop().run_until_complete(
                _execute_ingest_pipeline(
                    ingest_options=ingest_options,  # type: ignore
                    file_data=file_data,
                    file_url=file_url,
                    file_id=file_id,
                    router=router,
                )
            )
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=ingest_options.get("vector_store", {}).get("custom_llm_provider"),
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )

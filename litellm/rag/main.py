"""
RAG Ingest API for LiteLLM.

Provides an all-in-one API for document ingestion:
Upload -> (OCR) -> Chunk -> Embed -> Vector Store
"""

import asyncio
import base64
import contextvars
from functools import partial
from typing import TYPE_CHECKING, Any, Coroutine, Dict, List, Optional, Tuple, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm._uuid import uuid4
from litellm.types.rag import (
    RAGIngestOptions,
    RAGIngestResponse,
)
from litellm.utils import client
from litellm.vector_store_files.main import acreate as vector_store_file_acreate
from litellm.vector_stores.main import acreate as vector_store_acreate

if TYPE_CHECKING:
    from litellm import Router


####### STAGE HELPERS ###################


async def _stage_upload(
    file_data: Optional[Tuple[str, bytes, str]],
    file_url: Optional[str],
    file_id: Optional[str],
) -> Tuple[Optional[str], Optional[bytes], Optional[str], Optional[str]]:
    """
    Stage 1: Upload / Prepare file.

    Returns:
        Tuple of (filename, file_content, content_type, existing_file_id)
    """
    if file_data:
        filename, file_content, content_type = file_data
        return filename, file_content, content_type, None

    if file_url:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(file_url)
            response.raise_for_status()
            file_content = response.content
            filename = file_url.split("/")[-1] or "document"
            content_type = response.headers.get("content-type", "application/octet-stream")
        return filename, file_content, content_type, None

    if file_id:
        return None, None, None, file_id

    raise ValueError("Must provide file_data, file_url, or file_id")


async def _stage_ocr(
    ocr_config: Any,
    file_content: Optional[bytes],
    content_type: Optional[str],
    router: Optional["Router"],
) -> Optional[str]:
    """
    Stage 2: OCR (optional) - Extract text from images/PDFs.

    Returns:
        Extracted text or None if OCR not configured/failed
    """
    if not ocr_config or not file_content:
        return None

    ocr_model = ocr_config.get("model", "mistral/mistral-ocr-latest")

    # Determine document type
    if content_type and "image" in content_type:
        doc_type, url_key = "image_url", "image_url"
    else:
        doc_type, url_key = "document_url", "document_url"

    # Encode as base64 data URL
    b64_content = base64.b64encode(file_content).decode("utf-8")
    data_url = f"data:{content_type};base64,{b64_content}"

    # Use router if available
    if router is not None:
        ocr_response = await router.aocr(
            model=ocr_model,
            document={"type": doc_type, url_key: data_url},
        )
    else:
        ocr_response = await litellm.aocr(
            model=ocr_model,
            document={"type": doc_type, url_key: data_url},
        )

    # Extract text from pages
    if hasattr(ocr_response, "pages") and ocr_response.pages:  # type: ignore
        return "\n\n".join(
            page.markdown for page in ocr_response.pages if hasattr(page, "markdown")  # type: ignore
        )

    return None


def _stage_chunking(
    text: Optional[str],
    file_content: Optional[bytes],
    ocr_was_used: bool,
    chunking_strategy: Any,
) -> List[str]:
    """
    Stage 3: Chunking - Split text into chunks using RecursiveCharacterTextSplitter.

    Args:
        text: Text from OCR (if used)
        file_content: Raw file content bytes
        ocr_was_used: Whether OCR was performed
        chunking_strategy: Dict with RecursiveCharacterTextSplitter args:
            - chunk_size: int (default 1000)
            - chunk_overlap: int (default 200)
            - separators: List[str] (optional)

    Returns:
        List of text chunks
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Get text to chunk
    text_to_chunk: Optional[str] = None
    if text:
        text_to_chunk = text
    elif file_content and not ocr_was_used:
        try:
            text_to_chunk = file_content.decode("utf-8")
        except UnicodeDecodeError:
            verbose_logger.debug("Binary file detected, skipping text chunking")
            return []

    if not text_to_chunk:
        return []

    # Extract RecursiveCharacterTextSplitter args from chunking_strategy
    splitter_args = chunking_strategy or {}
    chunk_size = splitter_args.get("chunk_size", 1000)
    chunk_overlap = splitter_args.get("chunk_overlap", 200)
    separators = splitter_args.get("separators", None)

    # Build splitter kwargs
    splitter_kwargs: Dict[str, Any] = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    if separators:
        splitter_kwargs["separators"] = separators

    text_splitter = RecursiveCharacterTextSplitter(**splitter_kwargs)
    chunks = text_splitter.split_text(text_to_chunk)

    return chunks


async def _stage_embedding(
    embedding_config: Any,
    chunks: List[str],
    router: Optional["Router"],
) -> Optional[List[List[float]]]:
    """
    Stage 4: Embedding (optional) - Generate embeddings.

    Returns:
        List of embeddings or None
    """
    if not embedding_config or not chunks:
        return None

    embedding_model = embedding_config.get("model", "text-embedding-3-small")

    if router is not None:
        response = await router.aembedding(model=embedding_model, input=chunks)
    else:
        response = await litellm.aembedding(model=embedding_model, input=chunks)

    return [item["embedding"] for item in response.data]


async def _stage_vector_store(
    vector_store_config: Any,
    ingest_name: Optional[str],
    file_content: Optional[bytes],
    filename: Optional[str],
    content_type: Optional[str],
    chunking_strategy: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Stage 5: Vector Store - Create/use vector store and upload file.

    Returns:
        Tuple of (vector_store_id, file_id)
    """
    custom_llm_provider = vector_store_config.get("custom_llm_provider", "openai")
    vector_store_id = vector_store_config.get("vector_store_id")
    ttl_days = vector_store_config.get("ttl_days")

    # Create vector store if not provided
    if not vector_store_id:
        expires_after = {"anchor": "last_active_at", "days": ttl_days} if ttl_days else None
        create_response = await vector_store_acreate(
            name=ingest_name or "litellm-rag-ingest",
            custom_llm_provider=custom_llm_provider,
            expires_after=expires_after,
        )
        vector_store_id = create_response.get("id")

    # Upload file to vector store
    result_file_id = None
    if file_content and filename and vector_store_id:
        file_response = await litellm.acreate_file(
            file=(filename, file_content, content_type or "application/octet-stream"),
            purpose="assistants",
            custom_llm_provider=custom_llm_provider,
        )
        result_file_id = file_response.id

        # Attach file to vector store
        await vector_store_file_acreate(
            vector_store_id=vector_store_id,
            file_id=result_file_id,
            custom_llm_provider=custom_llm_provider,
            chunking_strategy=chunking_strategy,
        )

    return vector_store_id, result_file_id


####### MAIN PIPELINE ###################


async def _execute_ingest_pipeline(
    ingest_options: RAGIngestOptions,
    file_data: Optional[Tuple[str, bytes, str]] = None,
    file_url: Optional[str] = None,
    file_id: Optional[str] = None,
    router: Optional["Router"] = None,
) -> RAGIngestResponse:
    """Execute the RAG ingest pipeline."""
    ingest_id = f"ingest_{uuid4()}"

    try:
        # Stage 1: Upload
        filename, file_content, content_type, existing_file_id = await _stage_upload(
            file_data=file_data, file_url=file_url, file_id=file_id
        )

        # Stage 2: OCR (optional)
        ocr_config = ingest_options.get("ocr")
        extracted_text = await _stage_ocr(
            ocr_config=ocr_config,
            file_content=file_content,
            content_type=content_type,
            router=router,
        )

        # Stage 3: Chunking
        chunking_strategy = ingest_options.get("chunking_strategy", {"type": "auto"})
        chunks = _stage_chunking(
            text=extracted_text,
            file_content=file_content,
            ocr_was_used=ocr_config is not None,
            chunking_strategy=chunking_strategy,
        )

        # Stage 4: Embedding (optional)
        embedding_config = ingest_options.get("embedding")
        await _stage_embedding(
            embedding_config=embedding_config,
            chunks=chunks,
            router=router,
        )

        # Stage 5: Vector Store
        vector_store_config = ingest_options.get("vector_store", {})
        vector_store_id, result_file_id = await _stage_vector_store(
            vector_store_config=vector_store_config,
            ingest_name=ingest_options.get("name"),
            file_content=file_content,
            filename=filename,
            content_type=content_type,
            chunking_strategy=chunking_strategy,
        )

        return RAGIngestResponse(
            id=ingest_id,
            status="completed",
            vector_store_id=vector_store_id or "",
            file_id=result_file_id or existing_file_id,
        )

    except Exception as e:
        verbose_logger.exception(f"RAG Pipeline failed: {e}")
        return RAGIngestResponse(
            id=ingest_id,
            status="failed",
            vector_store_id="",
            file_id=None,
        )


####### PUBLIC API ###################


@client
async def aingest(
    ingest_options: Dict[str, Any],
    file_data: Optional[Tuple[str, bytes, str]] = None,
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
        file_url: URL to fetch file from
        file_id: Existing file ID to use

    Example:
        ```python
        response = await litellm.rag.aingest(
            ingest_options={
                "ocr": {"model": "mistral/mistral-ocr-latest"},
                "vector_store": {"custom_llm_provider": "openai"}
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


@client
def ingest(
    ingest_options: Dict[str, Any],
    file_data: Optional[Tuple[str, bytes, str]] = None,
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
        file_url: URL to fetch file from
        file_id: Existing file ID to use

    Example:
        ```python
        response = litellm.rag.ingest(
            ingest_options={
                "vector_store": {"custom_llm_provider": "openai"}
            },
            file_data=("doc.txt", b"Hello world", "text/plain"),
        )
        ```
    """
    local_vars = locals()
    try:
        _is_async = kwargs.pop("aingest", False) is True
        router: Optional["Router"] = kwargs.get("router")

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

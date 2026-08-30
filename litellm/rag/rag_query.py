from typing import Any, Final

from litellm.types.llms.openai import AllMessageValues, ChatCompletionUserMessage
from litellm.types.utils import ModelResponse
from litellm.types.vector_stores import (
    VectorStoreResultContent,
    VectorStoreSearchResponse,
)


class RAGQuery:
    CONTENT_PREFIX_STRING = "Context:\n\n"

    @staticmethod
    def extract_query_from_messages(messages: list[AllMessageValues]) -> str | None:
        """
        Extract the query from the last user message.
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

    @staticmethod
    def build_context_message(context_chunks: list[Any]) -> ChatCompletionUserMessage:
        """
        Process search results and build a context message.
        """
        context_content = RAGQuery.CONTENT_PREFIX_STRING

        for chunk in context_chunks:
            if isinstance(chunk, dict):
                result_content: list[VectorStoreResultContent] | None = chunk.get("content")
                if result_content:
                    for content_item in result_content:
                        content_text: str | None = content_item.get("text")
                        if content_text:
                            context_content += content_text + "\n\n"
                elif "text" in chunk:  # Fallback for simple dict with text
                    context_content += chunk["text"] + "\n\n"
            elif isinstance(chunk, str):
                context_content += chunk + "\n\n"

        return {
            "role": "user",
            "content": context_content,
        }

    @staticmethod
    def add_search_results_to_response(
        response: ModelResponse,
        search_results: VectorStoreSearchResponse,
        rerank_results: Any | None = None,
    ) -> ModelResponse:
        """
        Add search results to the response choices.
        """
        if hasattr(response, "choices") and response.choices:
            for choice in response.choices:
                message = getattr(choice, "message", None)
                if message is not None:
                    # Get existing provider_specific_fields or create new dict
                    provider_fields = getattr(message, "provider_specific_fields", None) or {}

                    # Add search results
                    provider_fields["search_results"] = search_results
                    if rerank_results:
                        provider_fields["rerank_results"] = rerank_results

                    # Set the provider_specific_fields
                    setattr(message, "provider_specific_fields", provider_fields)
        return response

    @staticmethod
    def extract_documents_from_search(
        search_response: Any,
    ) -> list[str | dict[str, Any]]:
        """Extract text documents from vector store search response."""
        documents: Final[list[str | dict[str, Any]]] = []
        for result in search_response.get("data", []):
            content_list = result.get("content", [])
            for content in content_list:
                if content.get("type") == "text" and content.get("text"):
                    documents.append(content["text"])
        return documents

    @staticmethod
    def get_top_chunks_from_rerank(search_response: Any, rerank_response: Any) -> list[Any]:
        """Get the original search results corresponding to the top reranked results."""
        top_chunks: Final = []
        original_results: Final = search_response.get("data", [])
        for result in rerank_response.get("results", []):
            index = result.get("index")
            if index is not None and index < len(original_results):
                top_chunks.append(original_results[index])
        return top_chunks

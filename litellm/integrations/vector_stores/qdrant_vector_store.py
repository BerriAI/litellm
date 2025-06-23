import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.vector_stores.base_vector_store import BaseVectorStore
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.llms.openai import AllMessageValues, ChatCompletionUserMessage
from litellm.types.utils import StandardLoggingVectorStoreRequest
from litellm.types.vector_stores import (
    VectorStoreResultContent,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.litellm_logging import StandardCallbackDynamicParams
else:
    LiteLLMLoggingObj = Any
    StandardCallbackDynamicParams = Any


class QdrantVectorStore(BaseVectorStore):
    CONTENT_PREFIX_STRING = "Context: \n\n"
    CUSTOM_LLM_PROVIDER = "qdrant"

    def __init__(
        self,
        qdrant_api_base: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        collection_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        self.qdrant_api_base = (
            qdrant_api_base or os.getenv("QDRANT_URL") or os.getenv("QDRANT_API_BASE")
        )
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.embedding_model = (
            embedding_model
            or os.getenv("QDRANT_VECTOR_STORE_EMBEDDING_MODEL")
            or "text-embedding-ada-002"
        )
        self.collection_name = collection_name or kwargs.get("collection_name")
        self.optional_params = kwargs
        super().__init__(**kwargs)

    async def async_get_chat_completion_prompt(
        self,
        model: str,
        messages: List[AllMessageValues],
        non_default_params: dict,
        prompt_id: Optional[str],
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
        litellm_logging_obj: LiteLLMLoggingObj,
        tools: Optional[List[Dict]] = None,
        prompt_label: Optional[str] = None,
    ) -> Tuple[str, List[AllMessageValues], dict]:
        if litellm.vector_store_registry is None:
            return model, messages, non_default_params

        vector_store_ids = litellm.vector_store_registry.pop_vector_store_ids_to_run(
            non_default_params=non_default_params, tools=tools
        )
        vector_store_request_metadata: List[StandardLoggingVectorStoreRequest] = []
        if vector_store_ids:
            for vector_store_id in vector_store_ids:
                start_time = datetime.now()
                query = self._get_query_from_messages(messages)
                qdrant_response = await self.make_qdrant_query_request(
                    collection_name=vector_store_id,
                    query=query,
                    non_default_params=non_default_params,
                )
                (
                    context_message,
                    context_string,
                ) = self.get_chat_completion_message_from_qdrant_response(
                    qdrant_response
                )
                if context_message is not None:
                    messages.append(context_message)

                vector_store_search_response: VectorStoreSearchResponse = (
                    self.transform_qdrant_response_to_vector_store_search_response(
                        qdrant_response=qdrant_response, query=query
                    )
                )
                vector_store_request_metadata.append(
                    StandardLoggingVectorStoreRequest(
                        vector_store_id=vector_store_id,
                        query=query,
                        vector_store_search_response=vector_store_search_response,
                        custom_llm_provider=self.CUSTOM_LLM_PROVIDER,
                        start_time=start_time.timestamp(),
                        end_time=datetime.now().timestamp(),
                    )
                )
            litellm_logging_obj.model_call_details["vector_store_request_metadata"] = (
                vector_store_request_metadata
            )

        return model, messages, non_default_params

    def transform_qdrant_response_to_vector_store_search_response(
        self, qdrant_response: Dict[str, Any], query: str
    ) -> VectorStoreSearchResponse:
        results: List[Dict[str, Any]] = qdrant_response.get("result", [])
        vector_store_search_response: VectorStoreSearchResponse = (
            VectorStoreSearchResponse(search_query=query, data=[])
        )
        vector_search_response_data: List[VectorStoreSearchResult] = []
        for res in results:
            payload = res.get("payload", {})
            text_content = None
            if isinstance(payload, dict):
                text_content = payload.get("text")
            if text_content is None:
                continue
            vector_store_search_result: VectorStoreSearchResult = (
                VectorStoreSearchResult(
                    score=res.get("score"),
                    content=[VectorStoreResultContent(text=text_content, type="text")],
                )
            )
            vector_search_response_data.append(vector_store_search_result)
        vector_store_search_response["data"] = vector_search_response_data
        return vector_store_search_response

    def _get_query_from_messages(self, messages: List[AllMessageValues]) -> str:
        if len(messages) == 0:
            return ""
        last_message = messages[-1]
        last_message_content = last_message.get("content", None)
        if last_message_content is None:
            return ""
        if isinstance(last_message_content, str):
            return last_message_content
        elif isinstance(last_message_content, list):
            return "\n".join([item.get("text", "") for item in last_message_content])
        return ""

    async def make_qdrant_query_request(
        self,
        collection_name: str,
        query: Any,
        vector_dimension: Optional[int] = None,
        non_default_params: Optional[dict] = None,
    ) -> Dict[str, Any]:
        from fastapi import HTTPException

        non_default_params = non_default_params or {}
        credentials_dict: Dict[str, Any] = {}
        if litellm.vector_store_registry is not None:
            credentials_dict = (
                litellm.vector_store_registry.get_credentials_for_vector_store(
                    collection_name
                )
            )

        api_base = credentials_dict.get("qdrant_api_base", self.qdrant_api_base)
        api_key = credentials_dict.get("qdrant_api_key", self.qdrant_api_key)
        _collection = (
            credentials_dict.get("collection_name", self.collection_name)
            or collection_name
        )

        if api_base is None:
            raise ValueError("qdrant_api_base must be provided")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["api-key"] = api_key

        url = f"{api_base}/collections/{_collection}/points/query"
        body = {"query": query}
        if isinstance(query, str):
            embedding_response = await litellm.aembedding(
                model=self.embedding_model,
                input=query,
                dimensions=vector_dimension,
                cache={"no-store": True, "no-cache": True},
            )
            query_vector = embedding_response["data"][0]["embedding"]
            body["query"] = query_vector
        body.setdefault("with_payload", True)
        if "limit" in non_default_params:
            body["limit"] = non_default_params["limit"]

        verbose_proxy_logger.debug(
            "Qdrant query request body: %s, url %s, headers: %s",
            body,
            url,
            headers,
        )
        response = await self.async_handler.post(url=url, headers=headers, json=body)
        verbose_proxy_logger.debug("Qdrant response: %s", response.text)
        if response.status_code == 200:
            return response.json()
        raise HTTPException(
            status_code=response.status_code,
            detail={"error": "Error calling Qdrant", "response": response.text},
        )

    @staticmethod
    def get_initialized_custom_logger() -> Optional[CustomLogger]:
        from litellm.litellm_core_utils.litellm_logging import (
            _init_custom_logger_compatible_class,
        )

        return _init_custom_logger_compatible_class(
            logging_integration="qdrant_vector_store",
            internal_usage_cache=None,
            llm_router=None,
        )

    @staticmethod
    def get_chat_completion_message_from_qdrant_response(
        response: Dict[str, Any],
    ) -> Tuple[Optional[ChatCompletionUserMessage], str]:
        results: List[Dict[str, Any]] = response.get("result", [])
        if not results:
            return None, ""
        context_string: str = QdrantVectorStore.CONTENT_PREFIX_STRING
        for res in results:
            payload = res.get("payload", {})
            text_content = None
            if isinstance(payload, dict):
                text_content = payload.get("text")
            if text_content is None:
                continue
            context_string += text_content
        message = ChatCompletionUserMessage(role="user", content=context_string)
        return message, context_string

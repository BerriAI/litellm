# +-------------------------------------------------------------+
#
#           Add Bedrock Knowledge Base Context to your LLM calls
#
# +-------------------------------------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import litellm
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.vector_stores.base_vector_store import BaseVectorStore
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.rag.bedrock_knowledgebase import (
    BedrockKBContent,
    BedrockKBGuardrailConfiguration,
    BedrockKBRequest,
    BedrockKBResponse,
    BedrockKBRetrievalConfiguration,
    BedrockKBRetrievalQuery,
    BedrockKBRetrievalResult,
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
else:
    LiteLLMLoggingObj = Any

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import StandardCallbackDynamicParams
else:
    StandardCallbackDynamicParams = Any


class BedrockVectorStore(BaseVectorStore, BaseAWSLLM):
    CONTENT_PREFIX_STRING = "Context: \n\n"
    CUSTOM_LLM_PROVIDER = "bedrock"

    def __init__(
        self,
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )

        # store kwargs as optional_params
        self.optional_params = kwargs

        super().__init__(**kwargs)
        BaseAWSLLM.__init__(self)

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
        """
        Retrieves the context from the Bedrock Knowledge Base and appends it to the messages.
        """
        if litellm.vector_store_registry is None:
            return model, messages, non_default_params

        vector_store_ids = litellm.vector_store_registry.pop_vector_store_ids_to_run(
            non_default_params=non_default_params, tools=tools
        )
        vector_store_request_metadata: List[StandardLoggingVectorStoreRequest] = []
        if vector_store_ids:
            for vector_store_id in vector_store_ids:
                start_time = datetime.now()
                query = self._get_kb_query_from_messages(messages)
                bedrock_kb_response = await self.make_bedrock_kb_retrieve_request(
                    knowledge_base_id=vector_store_id,
                    query=query,
                    non_default_params=non_default_params,
                )
                verbose_logger.debug(
                    f"Bedrock Knowledge Base Response: {bedrock_kb_response}"
                )

                (
                    context_message,
                    context_string,
                ) = self.get_chat_completion_message_from_bedrock_kb_response(
                    bedrock_kb_response
                )
                if context_message is not None:
                    messages.append(context_message)

                #################################################################################################
                ########## LOGGING for Standard Logging Payload, Langfuse, s3, LiteLLM DB etc. ##################
                #################################################################################################
                vector_store_search_response: VectorStoreSearchResponse = (
                    self.transform_bedrock_kb_response_to_vector_store_search_response(
                        bedrock_kb_response=bedrock_kb_response, query=query
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

            litellm_logging_obj.model_call_details[
                "vector_store_request_metadata"
            ] = vector_store_request_metadata

        return model, messages, non_default_params

    def transform_bedrock_kb_response_to_vector_store_search_response(
        self,
        bedrock_kb_response: BedrockKBResponse,
        query: str,
    ) -> VectorStoreSearchResponse:
        """
        Transform a BedrockKBResponse to a VectorStoreSearchResponse
        """
        retrieval_results: Optional[
            List[BedrockKBRetrievalResult]
        ] = bedrock_kb_response.get("retrievalResults", None)
        vector_store_search_response: VectorStoreSearchResponse = (
            VectorStoreSearchResponse(search_query=query, data=[])
        )
        if retrieval_results is None:
            return vector_store_search_response

        vector_search_response_data: List[VectorStoreSearchResult] = []
        for retrieval_result in retrieval_results:
            content: Optional[BedrockKBContent] = retrieval_result.get("content", None)
            if content is None:
                continue
            content_text: Optional[str] = content.get("text", None)
            if content_text is None:
                continue
            vector_store_search_result: VectorStoreSearchResult = (
                VectorStoreSearchResult(
                    score=retrieval_result.get("score", None),
                    content=[VectorStoreResultContent(text=content_text, type="text")],
                )
            )
            vector_search_response_data.append(vector_store_search_result)
        vector_store_search_response["data"] = vector_search_response_data
        return vector_store_search_response

    def _get_kb_query_from_messages(self, messages: List[AllMessageValues]) -> str:
        """
        Uses the text `content` field of the last message in the list of messages
        """
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

    def _prepare_request(
        self,
        credentials: Any,
        data: BedrockKBRequest,
        optional_params: dict,
        aws_region_name: str,
        api_base: str,
        extra_headers: Optional[dict] = None,
    ) -> Any:
        """
        Prepare a signed AWS request.

        Args:
            credentials: AWS credentials
            data: Request data
            optional_params: Additional parameters
            aws_region_name: AWS region name
            api_base: Base API URL
            extra_headers: Additional headers

        Returns:
            AWSRequest: A signed AWS request
        """
        try:
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        sigv4 = SigV4Auth(credentials, "bedrock", aws_region_name)

        encoded_data = json.dumps(data).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if extra_headers is not None:
            headers = {"Content-Type": "application/json", **extra_headers}

        request = AWSRequest(
            method="POST", url=api_base, data=encoded_data, headers=headers
        )
        sigv4.add_auth(request)
        if extra_headers is not None and "Authorization" in extra_headers:
            # prevent sigv4 from overwriting the auth header
            request.headers["Authorization"] = extra_headers["Authorization"]

        return request.prepare()

    async def make_bedrock_kb_retrieve_request(
        self,
        knowledge_base_id: str,
        query: str,
        guardrail_id: Optional[str] = None,
        guardrail_version: Optional[str] = None,
        next_token: Optional[str] = None,
        retrieval_configuration: Optional[BedrockKBRetrievalConfiguration] = None,
        non_default_params: Optional[dict] = None,
    ) -> BedrockKBResponse:
        """
        Make a Bedrock Knowledge Base retrieve request.

        Args:
            knowledge_base_id (str): The unique identifier of the knowledge base to query
            query (str): The query text to search for
            guardrail_id (Optional[str]): The guardrail ID to apply
            guardrail_version (Optional[str]): The version of the guardrail to apply
            next_token (Optional[str]): Token for pagination
            retrieval_configuration (Optional[BedrockKBRetrievalConfiguration]): Configuration for the retrieval process

        Returns:
            BedrockKBRetrievalResponse: A typed response object containing the retrieval results
        """
        from fastapi import HTTPException

        non_default_params = non_default_params or {}
        credentials_dict: Dict[str, Any] = {}
        if litellm.vector_store_registry is not None:
            credentials_dict = (
                litellm.vector_store_registry.get_credentials_for_vector_store(
                    knowledge_base_id
                )
            )

        credentials = self.get_credentials(
            aws_access_key_id=credentials_dict.get(
                "aws_access_key_id", non_default_params.get("aws_access_key_id", None)
            ),
            aws_secret_access_key=credentials_dict.get(
                "aws_secret_access_key",
                non_default_params.get("aws_secret_access_key", None),
            ),
            aws_session_token=credentials_dict.get(
                "aws_session_token", non_default_params.get("aws_session_token", None)
            ),
            aws_region_name=credentials_dict.get(
                "aws_region_name", non_default_params.get("aws_region_name", None)
            ),
            aws_session_name=credentials_dict.get(
                "aws_session_name", non_default_params.get("aws_session_name", None)
            ),
            aws_profile_name=credentials_dict.get(
                "aws_profile_name", non_default_params.get("aws_profile_name", None)
            ),
            aws_role_name=credentials_dict.get(
                "aws_role_name", non_default_params.get("aws_role_name", None)
            ),
            aws_web_identity_token=credentials_dict.get(
                "aws_web_identity_token",
                non_default_params.get("aws_web_identity_token", None),
            ),
            aws_sts_endpoint=credentials_dict.get(
                "aws_sts_endpoint", non_default_params.get("aws_sts_endpoint", None)
            ),
        )
        aws_region_name = self.get_aws_region_name_for_non_llm_api_calls(
            aws_region_name=credentials_dict.get(
                "aws_region_name", non_default_params.get("aws_region_name", None)
            ),
        )

        # Prepare request data
        request_data: BedrockKBRequest = BedrockKBRequest(
            retrievalQuery=BedrockKBRetrievalQuery(text=query),
        )
        if next_token:
            request_data["nextToken"] = next_token
        if retrieval_configuration:
            request_data["retrievalConfiguration"] = retrieval_configuration
        if guardrail_id and guardrail_version:
            request_data["guardrailConfiguration"] = BedrockKBGuardrailConfiguration(
                guardrailId=guardrail_id, guardrailVersion=guardrail_version
            )
        verbose_logger.debug(
            f"Request Data: {json.dumps(request_data, indent=4, default=str)}"
        )

        # Prepare the request
        api_base = f"https://bedrock-agent-runtime.{aws_region_name}.amazonaws.com/knowledgebases/{knowledge_base_id}/retrieve"

        prepared_request = self._prepare_request(
            credentials=credentials,
            data=request_data,
            optional_params=self.optional_params,
            aws_region_name=aws_region_name,
            api_base=api_base,
        )

        verbose_proxy_logger.debug(
            "Bedrock Knowledge Base request body: %s, url %s, headers: %s",
            request_data,
            prepared_request.url,
            prepared_request.headers,
        )

        response = await self.async_handler.post(
            url=prepared_request.url,
            data=prepared_request.body,  # type: ignore
            headers=prepared_request.headers,  # type: ignore
        )

        verbose_proxy_logger.debug("Bedrock Knowledge Base response: %s", response.text)

        if response.status_code == 200:
            response_data = response.json()
            return BedrockKBResponse(**response_data)
        else:
            verbose_proxy_logger.error(
                "Bedrock Knowledge Base: error in response. Status code: %s, response: %s",
                response.status_code,
                response.text,
            )
            raise HTTPException(
                status_code=response.status_code,
                detail={
                    "error": "Error calling Bedrock Knowledge Base",
                    "response": response.text,
                },
            )

    @staticmethod
    def get_initialized_custom_logger() -> Optional[CustomLogger]:
        from litellm.litellm_core_utils.litellm_logging import (
            _init_custom_logger_compatible_class,
        )

        return _init_custom_logger_compatible_class(
            logging_integration="bedrock_vector_store",
            internal_usage_cache=None,
            llm_router=None,
        )

    @staticmethod
    def get_chat_completion_message_from_bedrock_kb_response(
        response: BedrockKBResponse,
    ) -> Tuple[Optional[ChatCompletionUserMessage], str]:
        """
        Retrieves the context from the Bedrock Knowledge Base response and returns a ChatCompletionUserMessage object.
        """
        retrieval_results: Optional[List[BedrockKBRetrievalResult]] = response.get(
            "retrievalResults", None
        )
        if retrieval_results is None:
            return None, ""

        # string to combine the context from the knowledge base
        context_string: str = BedrockVectorStore.CONTENT_PREFIX_STRING
        for retrieval_result in retrieval_results:
            retrieval_result_content: Optional[BedrockKBContent] = (
                retrieval_result.get("content", None) or {}
            )
            if retrieval_result_content is None:
                continue
            retrieval_result_text: Optional[str] = retrieval_result_content.get(
                "text", None
            )
            if retrieval_result_text is None:
                continue
            context_string += retrieval_result_text
        message = ChatCompletionUserMessage(
            role="user",
            content=context_string,
        )
        return message, context_string

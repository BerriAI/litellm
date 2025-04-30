# +-------------------------------------------------------------+
#
#           Add Bedrock Knowledge Base Context to your LLM calls
#
# +-------------------------------------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_prompt_management import CustomPromptManagement
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

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import StandardCallbackDynamicParams
else:
    StandardCallbackDynamicParams = Any


class BedrockKnowledgeBaseHook(CustomPromptManagement, BaseAWSLLM):
    CONTENT_PREFIX_STRING = "Context: \n\n"

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
    ) -> Tuple[str, List[AllMessageValues], dict]:
        """
        Retrieves the context from the Bedrock Knowledge Base and appends it to the messages.
        """
        knowledge_bases = non_default_params.pop("knowledge_bases", None)
        if knowledge_bases:
            for knowledge_base in knowledge_bases:
                response = await self.make_bedrock_kb_retrieve_request(
                    knowledge_base_id=knowledge_base,
                    query=self._get_kb_query_from_messages(messages),
                )
                verbose_logger.debug(f"Bedrock Knowledge Base Response: {response}")

                context_message = (
                    self.get_chat_completion_message_from_bedrock_kb_response(response)
                )
                if context_message is not None:
                    messages.append(context_message)
        return model, messages, non_default_params

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

        credentials = self.get_credentials()
        aws_region_name = self._get_aws_region_name(
            optional_params=self.optional_params
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
    def should_use_prompt_management_hook(non_default_params: Dict) -> bool:
        if non_default_params.get("knowledge_bases", None):
            return True
        return False

    @staticmethod
    def get_initialized_custom_logger(
        non_default_params: Dict,
    ) -> Optional[CustomLogger]:
        from litellm.litellm_core_utils.litellm_logging import (
            _init_custom_logger_compatible_class,
        )

        if BedrockKnowledgeBaseHook.should_use_prompt_management_hook(
            non_default_params
        ):
            return _init_custom_logger_compatible_class(
                logging_integration="bedrock_knowledgebase_hook",
                internal_usage_cache=None,
                llm_router=None,
            )
        return None

    @staticmethod
    def get_chat_completion_message_from_bedrock_kb_response(
        response: BedrockKBResponse,
    ) -> Optional[ChatCompletionUserMessage]:
        """
        Retrieves the context from the Bedrock Knowledge Base response and returns a ChatCompletionUserMessage object.
        """
        retrieval_results: Optional[List[BedrockKBRetrievalResult]] = response.get(
            "retrievalResults", None
        )
        if retrieval_results is None:
            return None

        # string to combine the context from the knowledge base
        context_string: str = BedrockKnowledgeBaseHook.CONTENT_PREFIX_STRING
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
        return message

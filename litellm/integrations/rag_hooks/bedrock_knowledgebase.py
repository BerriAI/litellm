# +-------------------------------------------------------------+
#
#           Use Bedrock Guardrails for your LLM calls
#
# +-------------------------------------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import json
import sys
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from fastapi import HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.integrations.rag.bedrock_knowledgebase import (
    BedrockKBGuardrailConfiguration,
    BedrockKBRequest,
    BedrockKBResponse,
    BedrockKBRetrievalConfiguration,
    BedrockKBRetrievalQuery,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

GUARDRAIL_NAME = "bedrock"


class BedrockKnowledgeBaseHook(CustomLogger, BaseAWSLLM):
    def __init__(
        self,
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        # store kwargs as optional_params
        self.optional_params = kwargs

        super().__init__(**kwargs)
        BaseAWSLLM.__init__(self)

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
        credentials = self.get_credentials()
        aws_region_name = self._get_aws_region_name(
            optional_params=self.optional_params
        )

        # Prepare request data
        request_data: BedrockKBRequest = BedrockKBRequest(
            retrievalQuery=BedrockKBRetrievalQuery(text=query),
            nextToken=next_token,
            retrievalConfiguration=retrieval_configuration,
            guardrailConfiguration=BedrockKBGuardrailConfiguration(
                guardrailId=guardrail_id, guardrailVersion=guardrail_version
            ),
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
            return BedrockKBResponse(response_data)
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

"""
Bedrock Batches API Handler
"""

import asyncio
from typing import Any, Coroutine, Dict, Optional, Union

import httpx

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.llms.openai import (
    Batch,
    CancelBatchRequest,
    CreateBatchRequest,
    RetrieveBatchRequest,
)
from litellm.types.utils import LiteLLMBatch

from ..base_aws_llm import BaseAWSLLM
from .transformation import BedrockBatchesConfig


class BedrockBatchesAPI(BaseAWSLLM):
    """
    Bedrock methods to support for batches
    - create_batch()
    - retrieve_batch()
    - cancel_batch()
    - list_batch()
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = BedrockBatchesConfig()

    async def acancel_batch(
        self,
        cancel_batch_data: CancelBatchRequest,
        api_key: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        optional_params: dict,
        litellm_params: dict,
        logging_obj: Any,
        model: str,
    ) -> Batch:
        """
        Async cancel batch method for Bedrock
        """
        # Transform cancel batch request
        transformed_request = self.config.transform_cancel_batch_request(
            batch_id=cancel_batch_data["batch_id"],
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        # Make HTTP request using the transformed data
        response = await AsyncHTTPHandler().post(
            url=transformed_request["url"],
            headers=transformed_request["headers"],
            data=transformed_request["data"],
            timeout=timeout,
        )

        # Transform response back to LiteLLM format
        return self.config.transform_cancel_batch_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
            litellm_params=litellm_params,
        )

    def cancel_batch(
        self,
        _is_async: bool,
        cancel_batch_data: CancelBatchRequest,
        api_key: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        optional_params: dict,
        litellm_params: dict,
        logging_obj: Any,
        model: str,
    ) -> Union[Batch, Coroutine[Any, Any, Batch]]:
        """
        Cancel batch method for Bedrock
        """
        if _is_async is True:
            return self.acancel_batch(
                cancel_batch_data=cancel_batch_data,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                model=model,
            )

        # Transform cancel batch request
        transformed_request = self.config.transform_cancel_batch_request(
            batch_id=cancel_batch_data["batch_id"],
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        # Make HTTP request using the transformed data
        response = HTTPHandler().post(
            url=transformed_request["url"],
            headers=transformed_request["headers"],
            data=transformed_request["data"],
            timeout=timeout,
        )

        # Transform response back to LiteLLM format
        return self.config.transform_cancel_batch_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
            litellm_params=litellm_params,
        )

    async def acreate_batch(
        self,
        create_batch_data: CreateBatchRequest,
        api_key: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        optional_params: dict,
        litellm_params: dict,
        logging_obj: Any,
        model: str,
    ) -> LiteLLMBatch:
        """
        Async create batch method for Bedrock
        """
        # Transform create batch request
        transformed_request = self.config.transform_create_batch_request(
            model=model,
            create_batch_data=create_batch_data,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        # Make HTTP request using the transformed data
        response = await AsyncHTTPHandler().post(
            url=transformed_request["url"],
            headers=transformed_request["headers"],
            data=transformed_request["data"],
            timeout=timeout,
        )

        # Transform response back to LiteLLM format
        return self.config.transform_create_batch_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
            litellm_params=litellm_params,
        )

    def create_batch(
        self,
        _is_async: bool,
        create_batch_data: CreateBatchRequest,
        api_key: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        optional_params: dict,
        litellm_params: dict,
        logging_obj: Any,
        model: str,
    ) -> Union[LiteLLMBatch, Coroutine[Any, Any, LiteLLMBatch]]:
        """
        Create batch method for Bedrock
        """
        if _is_async is True:
            return self.acreate_batch(
                create_batch_data=create_batch_data,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                model=model,
            )

        # Transform create batch request
        transformed_request = self.config.transform_create_batch_request(
            model=model,
            create_batch_data=create_batch_data,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        # Make HTTP request using the transformed data
        response = HTTPHandler().post(
            url=transformed_request["url"],
            headers=transformed_request["headers"],
            data=transformed_request["data"],
            timeout=timeout,
        )

        # Transform response back to LiteLLM format
        return self.config.transform_create_batch_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
            litellm_params=litellm_params,
        )

    async def aretrieve_batch(
        self,
        retrieve_batch_data: RetrieveBatchRequest,
        api_key: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        optional_params: dict,
        litellm_params: dict,
        logging_obj: Any,
        model: str,
    ) -> LiteLLMBatch:
        """
        Async retrieve batch method for Bedrock
        """
        # Transform retrieve batch request
        transformed_request = self.config.transform_retrieve_batch_request(
            batch_id=retrieve_batch_data["batch_id"],
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        # Make HTTP request using the transformed data
        response = await AsyncHTTPHandler().get(
            url=transformed_request["url"],
            headers=transformed_request["headers"],
            timeout=timeout,
        )

        # Transform response back to LiteLLM format
        return self.config.transform_retrieve_batch_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
            litellm_params=litellm_params,
        )

    def retrieve_batch(
        self,
        _is_async: bool,
        retrieve_batch_data: RetrieveBatchRequest,
        api_key: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        optional_params: dict,
        litellm_params: dict,
        logging_obj: Any,
        model: str,
    ) -> Union[LiteLLMBatch, Coroutine[Any, Any, LiteLLMBatch]]:
        """
        Retrieve batch method for Bedrock
        """
        if _is_async is True:
            return self.aretrieve_batch(
                retrieve_batch_data=retrieve_batch_data,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                model=model,
            )

        # Transform retrieve batch request
        transformed_request = self.config.transform_retrieve_batch_request(
            batch_id=retrieve_batch_data["batch_id"],
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        # Make HTTP request using the transformed data
        response = HTTPHandler().get(
            url=transformed_request["url"],
            headers=transformed_request["headers"],
            timeout=timeout,
        )

        # Transform response back to LiteLLM format
        return self.config.transform_retrieve_batch_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
            litellm_params=litellm_params,
        )
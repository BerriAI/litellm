"""
Bedrock Image Edit Handler

Handles image edit requests for Bedrock stability models.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional, Union

import httpx
from pydantic import BaseModel

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.llms.bedrock.image_edit.stability_transformation import (
    BedrockStabilityImageEditConfig,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.utils import ImageResponse

from ..base_aws_llm import BaseAWSLLM
from ..common_utils import BedrockError

if TYPE_CHECKING:
    from botocore.awsrequest import AWSPreparedRequest
else:
    AWSPreparedRequest = Any


class BedrockImageEditPreparedRequest(BaseModel):
    """
    Internal/Helper class for preparing the request for bedrock image edit
    """

    endpoint_url: str
    prepped: AWSPreparedRequest
    body: bytes
    data: dict


class BedrockImageEdit(BaseAWSLLM):
    """
    Bedrock Image Edit handler
    """

    @classmethod
    def get_config_class(cls, model: str | None):
        if BedrockStabilityImageEditConfig._is_stability_edit_model(model):
            return BedrockStabilityImageEditConfig
        else:
            raise ValueError(f"Unsupported model for bedrock image edit: {model}")

    def image_edit(
        self,
        model: str,
        image: list,
        prompt: Optional[str],
        model_response: ImageResponse,
        optional_params: dict,
        logging_obj: LitellmLogging,
        timeout: Optional[Union[float, httpx.Timeout]],
        aimage_edit: bool = False,
        api_base: Optional[str] = None,
        extra_headers: Optional[dict] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        api_key: Optional[str] = None,
    ):
        prepared_request = self._prepare_request(
            model=model,
            image=image,
            prompt=prompt,
            optional_params=optional_params,
            api_base=api_base,
            extra_headers=extra_headers,
            logging_obj=logging_obj,
            api_key=api_key,
        )

        if aimage_edit is True:
            return self.async_image_edit(
                prepared_request=prepared_request,
                timeout=timeout,
                model=model,
                logging_obj=logging_obj,
                prompt=prompt,
                model_response=model_response,
                client=(
                    client
                    if client is not None and isinstance(client, AsyncHTTPHandler)
                    else None
                ),
            )

        if client is None or not isinstance(client, HTTPHandler):
            client = _get_httpx_client()
        try:
            response = client.post(url=prepared_request.endpoint_url, headers=prepared_request.prepped.headers, data=prepared_request.body)  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise BedrockError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
            raise BedrockError(status_code=408, message="Timeout error occurred.")

        ### FORMAT RESPONSE TO OPENAI FORMAT ###
        model_response = self._transform_response_dict_to_openai_response(
            model_response=model_response,
            model=model,
            logging_obj=logging_obj,
            prompt=prompt,
            response=response,
            data=prepared_request.data,
        )
        return model_response

    async def async_image_edit(
        self,
        prepared_request: BedrockImageEditPreparedRequest,
        timeout: Optional[Union[float, httpx.Timeout]],
        model: str,
        logging_obj: LitellmLogging,
        prompt: Optional[str],
        model_response: ImageResponse,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> ImageResponse:
        """
        Asynchronous handler for bedrock image edit
        """
        async_client = client or get_async_httpx_client(
            llm_provider=litellm.LlmProviders.BEDROCK,
            params={"timeout": timeout},
        )

        try:
            response = await async_client.post(url=prepared_request.endpoint_url, headers=prepared_request.prepped.headers, data=prepared_request.body)  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise BedrockError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
            raise BedrockError(status_code=408, message="Timeout error occurred.")

        ### FORMAT RESPONSE TO OPENAI FORMAT ###
        model_response = self._transform_response_dict_to_openai_response(
            model=model,
            logging_obj=logging_obj,
            prompt=prompt,
            response=response,
            data=prepared_request.data,
            model_response=model_response,
        )
        return model_response

    def _prepare_request(
        self,
        model: str,
        image: list,
        prompt: Optional[str],
        optional_params: dict,
        api_base: Optional[str],
        extra_headers: Optional[dict],
        logging_obj: LitellmLogging,
        api_key: Optional[str],
    ) -> BedrockImageEditPreparedRequest:
        """
        Prepare the request body, headers, and endpoint URL for the Bedrock Image Edit API

        Args:
            model (str): The model to use for the image edit
            image (list): The images to edit
            prompt (Optional[str]): The prompt for the edit
            optional_params (dict): The optional parameters for the image edit
            api_base (Optional[str]): The base URL for the Bedrock API
            extra_headers (Optional[dict]): The extra headers to include in the request
            logging_obj (LitellmLogging): The logging object to use for logging
            api_key (Optional[str]): The API key to use

        Returns:
            BedrockImageEditPreparedRequest: The prepared request object
        """
        boto3_credentials_info = self._get_boto_credentials_from_optional_params(
            optional_params, model
        )

        # Use the existing ARN-aware provider detection method
        bedrock_provider = self.get_bedrock_invoke_provider(model)
        ### SET RUNTIME ENDPOINT ###
        modelId = self.get_bedrock_model_id(
            model=model,
            provider=bedrock_provider,
            optional_params=optional_params,
        )
        _, proxy_endpoint_url = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=boto3_credentials_info.aws_bedrock_runtime_endpoint,
            aws_region_name=boto3_credentials_info.aws_region_name,
        )
        proxy_endpoint_url = f"{proxy_endpoint_url}/model/{modelId}/invoke"
        data = self._get_request_body(
            model=model,
            image=image,
            prompt=prompt,
            optional_params=optional_params,
        )

        # Make POST Request
        body = json.dumps(data).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if extra_headers is not None:
            headers = {"Content-Type": "application/json", **extra_headers}

        prepped = self.get_request_headers(
            credentials=boto3_credentials_info.credentials,
            aws_region_name=boto3_credentials_info.aws_region_name,
            extra_headers=extra_headers,
            endpoint_url=proxy_endpoint_url,
            data=body,
            headers=headers,
            api_key=api_key,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": proxy_endpoint_url,
                "headers": prepped.headers,
            },
        )
        return BedrockImageEditPreparedRequest(
            endpoint_url=proxy_endpoint_url,
            prepped=prepped,
            body=body,
            data=data,
        )

    def _get_request_body(
        self,
        model: str,
        image: list,
        prompt: Optional[str],
        optional_params: dict,
    ) -> dict:
        """
        Get the request body for the Bedrock Image Edit API

        Checks the model/provider and transforms the request body accordingly

        Returns:
            dict: The request body to use for the Bedrock Image Edit API
        """
        config_class = self.get_config_class(model=model)
        config_instance = config_class()
        request_body, _ = config_instance.transform_image_edit_request(
            model=model,
            prompt=prompt,
            image=image[0] if image else None,
            image_edit_optional_request_params=optional_params,
            litellm_params={},
            headers={},
        )
        return dict(request_body)

    def _transform_response_dict_to_openai_response(
        self,
        model_response: ImageResponse,
        model: str,
        logging_obj: LitellmLogging,
        prompt: Optional[str],
        response: httpx.Response,
        data: dict,
    ) -> ImageResponse:
        """
        Transforms the Image Edit response from Bedrock to OpenAI format
        """

        ## LOGGING
        if logging_obj is not None:
            logging_obj.post_call(
                input=prompt,
                api_key="",
                original_response=response.text,
                additional_args={"complete_input_dict": data},
            )
        verbose_logger.debug("raw model_response: %s", response.text)
        response_dict = response.json()
        if response_dict is None:
            raise ValueError("Error in response object format, got None")

        config_class = self.get_config_class(model=model)
        config_instance = config_class()

        model_response = config_instance.transform_image_edit_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
        )

        return model_response


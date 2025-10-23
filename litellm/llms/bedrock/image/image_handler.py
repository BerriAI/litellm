import copy
import json
import os
from typing import TYPE_CHECKING, Any, Optional, Union

import httpx
from pydantic import BaseModel

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
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


class BedrockImagePreparedRequest(BaseModel):
    """
    Internal/Helper class for preparing the request for bedrock image generation
    """

    endpoint_url: str
    prepped: AWSPreparedRequest
    body: bytes
    data: dict


class BedrockImageGeneration(BaseAWSLLM):
    """
    Bedrock Image Generation handler
    """

    def image_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        optional_params: dict,
        logging_obj: LitellmLogging,
        timeout: Optional[Union[float, httpx.Timeout]],
        aimg_generation: bool = False,
        api_base: Optional[str] = None,
        extra_headers: Optional[dict] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        api_key: Optional[str] = None,
    ):
        prepared_request = self._prepare_request(
            model=model,
            optional_params=optional_params,
            api_base=api_base,
            extra_headers=extra_headers,
            logging_obj=logging_obj,
            prompt=prompt,
            api_key=api_key
        )

        if aimg_generation is True:
            return self.async_image_generation(
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

    async def async_image_generation(
        self,
        prepared_request: BedrockImagePreparedRequest,
        timeout: Optional[Union[float, httpx.Timeout]],
        model: str,
        logging_obj: LitellmLogging,
        prompt: str,
        model_response: ImageResponse,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> ImageResponse:
        """
        Asynchronous handler for bedrock image generation

        Awaits the response from the bedrock image generation endpoint
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
        optional_params: dict,
        api_base: Optional[str],
        extra_headers: Optional[dict],
        logging_obj: LitellmLogging,
        prompt: str,
        api_key: Optional[str],
    ) -> BedrockImagePreparedRequest:
        """
        Prepare the request body, headers, and endpoint URL for the Bedrock Image Generation API

        Args:
            model (str): The model to use for the image generation
            optional_params (dict): The optional parameters for the image generation
            api_base (Optional[str]): The base URL for the Bedrock API
            extra_headers (Optional[dict]): The extra headers to include in the request
            logging_obj (LitellmLogging): The logging object to use for logging
            prompt (str): The prompt to use for the image generation
        Returns:
            BedrockImagePreparedRequest: The prepared request object

        The BedrockImagePreparedRequest contains:
            endpoint_url (str): The endpoint URL for the Bedrock Image Generation API
            prepped (httpx.Request): The prepared request object
            body (bytes): The request body
        """
        boto3_credentials_info = self._get_boto_credentials_from_optional_params(
            optional_params, model
        )

        ### SET RUNTIME ENDPOINT ###
        modelId = model
        _, proxy_endpoint_url = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=boto3_credentials_info.aws_bedrock_runtime_endpoint,
            aws_region_name=boto3_credentials_info.aws_region_name,
        )
        proxy_endpoint_url = f"{proxy_endpoint_url}/model/{modelId}/invoke"
        data = self._get_request_body(
            model=model, prompt=prompt, optional_params=optional_params
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
        return BedrockImagePreparedRequest(
            endpoint_url=proxy_endpoint_url,
            prepped=prepped,
            body=body,
            data=data,
        )

    def _get_request_body(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
    ) -> dict:
        """
        Get the request body for the Bedrock Image Generation API

        Checks the model/provider and transforms the request body accordingly

        Returns:
            dict: The request body to use for the Bedrock Image Generation API
        """
        # Use the existing ARN-aware provider detection method
        bedrock_provider = self.get_bedrock_invoke_provider(model)

        if bedrock_provider == "amazon" or bedrock_provider == "nova":
            # Handle Amazon Nova Canvas models
            provider = "amazon"
        elif bedrock_provider == "stability":
            provider = "stability"
        else:
            # Fallback to original logic for backward compatibility
            provider = model.split(".")[0]
        inference_params = copy.deepcopy(optional_params)
        inference_params.pop(
            "user", None
        )  # make sure user is not passed in for bedrock call
        data = {}
        if provider == "stability":
            if litellm.AmazonStability3Config._is_stability_3_model(model):
                request_body = litellm.AmazonStability3Config.transform_request_body(
                    prompt=prompt, optional_params=optional_params
                )
                return dict(request_body)
            else:
                prompt = prompt.replace(os.linesep, " ")
                ## LOAD CONFIG
                config = litellm.AmazonStabilityConfig.get_config()
                for k, v in config.items():
                    if (
                        k not in inference_params
                    ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                        inference_params[k] = v
                data = {
                    "text_prompts": [{"text": prompt, "weight": 1}],
                    **inference_params,
                }
        elif provider == "amazon":
            return dict(
                litellm.AmazonNovaCanvasConfig.transform_request_body(
                    text=prompt, optional_params=optional_params
                )
            )
        else:
            raise BedrockError(
                status_code=422, message=f"Unsupported model={model}, passed in"
            )
        return data

    def _transform_response_dict_to_openai_response(
        self,
        model_response: ImageResponse,
        model: str,
        logging_obj: LitellmLogging,
        prompt: str,
        response: httpx.Response,
        data: dict,
    ) -> ImageResponse:
        """
        Transforms the Image Generation response from Bedrock to OpenAI format
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

        config_class = (
            litellm.AmazonStability3Config
            if litellm.AmazonStability3Config._is_stability_3_model(model=model)
            else (
                litellm.AmazonNovaCanvasConfig
                if litellm.AmazonNovaCanvasConfig._is_nova_model(model=model)
                else litellm.AmazonStabilityConfig
            )
        )
        config_class.transform_response_dict_to_openai_response(
            model_response=model_response,
            response_dict=response_dict,
        )

        return model_response

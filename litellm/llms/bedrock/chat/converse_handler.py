import json
import urllib
from typing import Any, Optional, Union

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObject
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper

from ..common_utils import BedrockError
from ..sap_aws_llm import SAPAWSLLM, SAPOAuthToken
from .invoke_handler import AWSEventStreamDecoder, MockResponseIterator, make_call


def make_sync_call(
        client: Optional[HTTPHandler],
        api_base: str,
        headers: dict,
        data: str,
        model: str,
        messages: list,
        logging_obj: LiteLLMLoggingObject,
        json_mode: Optional[bool] = False,
        fake_stream: bool = False,
):
    if client is None:
        client = _get_httpx_client()  # Create a new client if none provided

    response = client.post(
        api_base,
        headers=headers,
        data=data,
        stream=not fake_stream,
        logging_obj=logging_obj,
    )

    if response.status_code != 200:
        raise BedrockError(
            status_code=response.status_code, message=str(response.read())
        )

    if fake_stream:
        model_response: (
            ModelResponse
        ) = litellm.AmazonConverseConfig()._transform_response(
            model=model,
            response=response,
            model_response=litellm.ModelResponse(),
            stream=True,
            logging_obj=logging_obj,
            optional_params={},
            api_key="",
            data=data,
            messages=messages,
            encoding=litellm.encoding,
        )  # type: ignore
        completion_stream: Any = MockResponseIterator(
            model_response=model_response, json_mode=json_mode
        )
    else:
        decoder = AWSEventStreamDecoder(model=model)
        completion_stream = decoder.iter_bytes(response.iter_bytes(chunk_size=1024))

    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response="first stream response received",
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


class BedrockConverseLLM(SAPAWSLLM):
    """
    SAP AI Core Converse LLM implementation that uses SAP's OAuth authentication
    instead of AWS SigV4.
    """

    def __init__(self) -> None:
        super().__init__()

    def encode_model_id(self, model_id: str) -> str:
        """
        For SAP, we don't need to encode the model ID.
        Just return it as-is or extract deployment ID.
        """
        return self._extract_deployment_id_from_model(model_id) or model_id

    async def async_streaming(
            self,
            model: str,
            messages: list,
            api_base: str,
            model_response: ModelResponse,
            timeout: Optional[Union[float, httpx.Timeout]],
            encoding,
            logging_obj,
            stream,
            optional_params: dict,
            litellm_params: dict,
            credentials: SAPOAuthToken,
            logger_fn=None,
            headers={},
            client: Optional[AsyncHTTPHandler] = None,
            fake_stream: bool = False,
            json_mode: Optional[bool] = False,
    ) -> CustomStreamWrapper:
        request_data = await litellm.AmazonConverseConfig()._async_transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )
        data = json.dumps(request_data)

        # For SAP, we use Bearer token authentication
        request_headers = {
            "Content-Type": "application/json",
            "Authorization": f"{credentials.token_type} {credentials.access_token}",
            "AI-Resource-Group": "default"
        }
        request_headers.update(headers)

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": request_headers,
            },
        )

        completion_stream = await make_call(
            client=client,
            api_base=api_base,
            headers=request_headers,
            data=data,
            model=model,
            messages=messages,
            logging_obj=logging_obj,
            fake_stream=fake_stream,
            json_mode=json_mode,
        )
        streaming_response = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="sap-bedrock",
            logging_obj=logging_obj,
        )
        return streaming_response

    async def async_completion(
            self,
            model: str,
            messages: list,
            api_base: str,
            model_response: ModelResponse,
            timeout: Optional[Union[float, httpx.Timeout]],
            encoding,
            logging_obj: LiteLLMLoggingObject,
            stream,
            optional_params: dict,
            litellm_params: dict,
            credentials: SAPOAuthToken,
            logger_fn=None,
            headers: dict = {},
            client: Optional[AsyncHTTPHandler] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        request_data = await litellm.AmazonConverseConfig()._async_transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )
        data = json.dumps(request_data)

        # For SAP, we use Bearer token authentication
        request_headers = {
            "Content-Type": "application/json",
            "Authorization": f"{credentials.token_type} {credentials.access_token}",
            "AI-Resource-Group": "default"
        }
        request_headers.update(headers)

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": request_headers,
            },
        )

        if client is None or not isinstance(client, AsyncHTTPHandler):
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            client = get_async_httpx_client(
                params=_params, llm_provider=litellm.LlmProviders.BEDROCK
            )
        else:
            client = client  # type: ignore

        try:
            response = await client.post(
                url=api_base,
                headers=request_headers,
                data=data,
                logging_obj=logging_obj,
            )  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise BedrockError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
            raise BedrockError(status_code=408, message="Timeout error occurred.")

        return litellm.AmazonConverseConfig()._transform_response(
            model=model,
            response=response,
            model_response=model_response,
            stream=stream if isinstance(stream, bool) else False,
            logging_obj=logging_obj,
            api_key="",
            data=data,
            messages=messages,
            optional_params=optional_params,
            encoding=encoding,
        )

    def completion(  # noqa: PLR0915
            self,
            model: str,
            messages: list,
            api_base: Optional[str],
            custom_prompt_dict: dict,
            model_response: ModelResponse,
            encoding,
            logging_obj: LiteLLMLoggingObject,
            optional_params: dict,
            acompletion: bool,
            timeout: Optional[Union[float, httpx.Timeout]],
            litellm_params: dict,
            logger_fn=None,
            extra_headers: Optional[dict] = None,
            client: Optional[Union[AsyncHTTPHandler, HTTPHandler]] = None,
    ):
        ## SETUP ##
        stream = optional_params.pop("stream", None)
        deployment_id = optional_params.pop("sap_deployment_id", None)
        fake_stream = optional_params.pop("fake_stream", False)
        json_mode = optional_params.get("json_mode", False)

        # Extract deployment ID from model if not provided
        if deployment_id is None:
            deployment_id = self._extract_deployment_id_from_model(model)

        ### GET SAP CREDENTIALS ###
        sap_client_id = optional_params.pop("sap_client_id", None)
        sap_client_secret = optional_params.pop("sap_client_secret", None)
        sap_xsuaa_url = optional_params.pop("sap_xsuaa_url", None)
        sap_ai_core_base_url = optional_params.pop("sap_ai_core_base_url", None)

        # Get credentials (OAuth token)
        credentials: SAPOAuthToken = self.get_credentials(
            sap_client_id=sap_client_id,
            sap_client_secret=sap_client_secret,
            sap_xsuaa_url=sap_xsuaa_url,
        )

        ### SET RUNTIME ENDPOINT ###
        endpoint_url, proxy_endpoint_url = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=sap_ai_core_base_url,
            aws_region_name="sap-ai-core",  # Dummy value
            deployment_id=deployment_id,
            stream=(stream is not None and stream is True) and not fake_stream,
        )

        ## COMPLETION CALL
        headers = {"Content-Type": "application/json"}
        if extra_headers is not None:
            headers.update(extra_headers)

        ### ROUTING (ASYNC, STREAMING, SYNC)
        if acompletion:
            if isinstance(client, HTTPHandler):
                client = None
            if stream is True:
                return self.async_streaming(
                    model=model,
                    messages=messages,
                    api_base=proxy_endpoint_url,
                    model_response=model_response,
                    encoding=encoding,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    stream=True,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    headers=headers,
                    timeout=timeout,
                    client=client,
                    json_mode=json_mode,
                    fake_stream=fake_stream,
                    credentials=credentials,
                )  # type: ignore
            ### ASYNC COMPLETION
            return self.async_completion(
                model=model,
                messages=messages,
                api_base=proxy_endpoint_url,
                model_response=model_response,
                encoding=encoding,
                logging_obj=logging_obj,
                optional_params=optional_params,
                stream=stream,  # type: ignore
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                headers=headers,
                timeout=timeout,
                client=client,
                credentials=credentials,
            )  # type: ignore

        ## TRANSFORMATION ##
        _data = litellm.AmazonConverseConfig()._transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )
        data = json.dumps(_data)

        # Prepare headers with Bearer token
        request_headers = {
            "Content-Type": "application/json",
            "Authorization": f"{credentials.token_type} {credentials.access_token}",
            "AI-Resource-Group": "default"
        }
        if extra_headers:
            request_headers.update(extra_headers)

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": proxy_endpoint_url,
                "headers": request_headers,
            },
        )

        if client is None or isinstance(client, AsyncHTTPHandler):
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            client = _get_httpx_client(_params)  # type: ignore
        else:
            client = client

        if stream is not None and stream is True:
            completion_stream = make_sync_call(
                client=(
                    client
                    if client is not None and isinstance(client, HTTPHandler)
                    else None
                ),
                api_base=proxy_endpoint_url,
                headers=request_headers,
                data=data,
                model=model,
                messages=messages,
                logging_obj=logging_obj,
                json_mode=json_mode,
                fake_stream=fake_stream,
            )
            streaming_response = CustomStreamWrapper(
                completion_stream=completion_stream,
                model=model,
                custom_llm_provider="sap-bedrock",
                logging_obj=logging_obj,
            )

            return streaming_response

        ### COMPLETION
        try:
            response = client.post(
                url=proxy_endpoint_url,
                headers=request_headers,
                data=data,
                logging_obj=logging_obj,
            )  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise BedrockError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
            raise BedrockError(status_code=408, message="Timeout error occurred.")

        return litellm.AmazonConverseConfig()._transform_response(
            model=model,
            response=response,
            model_response=model_response,
            stream=stream if isinstance(stream, bool) else False,
            logging_obj=logging_obj,
            api_key="",
            data=data,
            messages=messages,
            optional_params=optional_params,
            encoding=encoding,
        )
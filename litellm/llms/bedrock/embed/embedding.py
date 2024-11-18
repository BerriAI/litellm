"""
Handles embedding calls to Bedrock's `/invoke` endpoint 
"""

import copy
import json
import os
from copy import deepcopy
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import httpx

import litellm
from litellm.llms.cohere.embed.handler import embedding as cohere_embedding
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.secret_managers.main import get_secret
from litellm.types.llms.bedrock import AmazonEmbeddingRequest, CohereEmbeddingRequest
from litellm.types.utils import Embedding, EmbeddingResponse, Usage

from ...base_aws_llm import BaseAWSLLM
from ..common_utils import BedrockError
from .amazon_titan_g1_transformation import AmazonTitanG1Config
from .amazon_titan_multimodal_transformation import (
    AmazonTitanMultimodalEmbeddingG1Config,
)
from .amazon_titan_v2_transformation import AmazonTitanV2Config
from .cohere_transformation import BedrockCohereEmbeddingConfig


class BedrockEmbedding(BaseAWSLLM):
    def _load_credentials(
        self,
        optional_params: dict,
    ) -> Tuple[Any, str]:
        try:
            from botocore.credentials import Credentials
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")
        ## CREDENTIALS ##
        # pop aws_secret_access_key, aws_access_key_id, aws_session_token, aws_region_name from kwargs, since completion calls fail with them
        aws_secret_access_key = optional_params.pop("aws_secret_access_key", None)
        aws_access_key_id = optional_params.pop("aws_access_key_id", None)
        aws_session_token = optional_params.pop("aws_session_token", None)
        aws_region_name = optional_params.pop("aws_region_name", None)
        aws_role_name = optional_params.pop("aws_role_name", None)
        aws_session_name = optional_params.pop("aws_session_name", None)
        aws_profile_name = optional_params.pop("aws_profile_name", None)
        aws_web_identity_token = optional_params.pop("aws_web_identity_token", None)
        aws_sts_endpoint = optional_params.pop("aws_sts_endpoint", None)

        ### SET REGION NAME ###
        if aws_region_name is None:
            # check env #
            litellm_aws_region_name = get_secret("AWS_REGION_NAME", None)

            if litellm_aws_region_name is not None and isinstance(
                litellm_aws_region_name, str
            ):
                aws_region_name = litellm_aws_region_name

            standard_aws_region_name = get_secret("AWS_REGION", None)
            if standard_aws_region_name is not None and isinstance(
                standard_aws_region_name, str
            ):
                aws_region_name = standard_aws_region_name

            if aws_region_name is None:
                aws_region_name = "us-west-2"

        credentials: Credentials = self.get_credentials(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            aws_region_name=aws_region_name,
            aws_session_name=aws_session_name,
            aws_profile_name=aws_profile_name,
            aws_role_name=aws_role_name,
            aws_web_identity_token=aws_web_identity_token,
            aws_sts_endpoint=aws_sts_endpoint,
        )
        return credentials, aws_region_name

    async def async_embeddings(self):
        pass

    def _make_sync_call(
        self,
        client: Optional[HTTPHandler],
        timeout: Optional[Union[float, httpx.Timeout]],
        api_base: str,
        headers: dict,
        data: dict,
    ) -> dict:
        if client is None or not isinstance(client, HTTPHandler):
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            client = _get_httpx_client(_params)  # type: ignore
        else:
            client = client
        try:
            response = client.post(url=api_base, headers=headers, data=json.dumps(data))  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise BedrockError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
            raise BedrockError(status_code=408, message="Timeout error occurred.")

        return response.json()

    async def _make_async_call(
        self,
        client: Optional[AsyncHTTPHandler],
        timeout: Optional[Union[float, httpx.Timeout]],
        api_base: str,
        headers: dict,
        data: dict,
    ) -> dict:
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
            client = client

        try:
            response = await client.post(url=api_base, headers=headers, data=json.dumps(data))  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise BedrockError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
            raise BedrockError(status_code=408, message="Timeout error occurred.")

        return response.json()

    def _single_func_embeddings(
        self,
        client: Optional[HTTPHandler],
        timeout: Optional[Union[float, httpx.Timeout]],
        batch_data: List[dict],
        credentials: Any,
        extra_headers: Optional[dict],
        endpoint_url: str,
        aws_region_name: str,
        model: str,
        logging_obj: Any,
    ):
        try:
            import boto3
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            from botocore.credentials import Credentials
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        responses: List[dict] = []
        for data in batch_data:
            sigv4 = SigV4Auth(credentials, "bedrock", aws_region_name)
            headers = {"Content-Type": "application/json"}
            if extra_headers is not None:
                headers = {"Content-Type": "application/json", **extra_headers}
            request = AWSRequest(
                method="POST", url=endpoint_url, data=json.dumps(data), headers=headers
            )
            sigv4.add_auth(request)
            if (
                extra_headers is not None and "Authorization" in extra_headers
            ):  # prevent sigv4 from overwriting the auth header
                request.headers["Authorization"] = extra_headers["Authorization"]
            prepped = request.prepare()

            ## LOGGING
            logging_obj.pre_call(
                input=data,
                api_key="",
                additional_args={
                    "complete_input_dict": data,
                    "api_base": prepped.url,
                    "headers": prepped.headers,
                },
            )
            response = self._make_sync_call(
                client=client,
                timeout=timeout,
                api_base=prepped.url,
                headers=prepped.headers,  # type: ignore
                data=data,
            )

            ## LOGGING
            logging_obj.post_call(
                input=data,
                api_key="",
                original_response=response,
                additional_args={"complete_input_dict": data},
            )

            responses.append(response)

        returned_response: Optional[EmbeddingResponse] = None

        ## TRANSFORM RESPONSE ##
        if model == "amazon.titan-embed-image-v1":
            returned_response = (
                AmazonTitanMultimodalEmbeddingG1Config()._transform_response(
                    response_list=responses, model=model
                )
            )
        elif model == "amazon.titan-embed-text-v1":
            returned_response = AmazonTitanG1Config()._transform_response(
                response_list=responses, model=model
            )
        elif model == "amazon.titan-embed-text-v2:0":
            returned_response = AmazonTitanV2Config()._transform_response(
                response_list=responses, model=model
            )

        if returned_response is None:
            raise Exception(
                "Unable to map model response to known provider format. model={}".format(
                    model
                )
            )

        return returned_response

    async def _async_single_func_embeddings(
        self,
        client: Optional[AsyncHTTPHandler],
        timeout: Optional[Union[float, httpx.Timeout]],
        batch_data: List[dict],
        credentials: Any,
        extra_headers: Optional[dict],
        endpoint_url: str,
        aws_region_name: str,
        model: str,
        logging_obj: Any,
    ):
        try:
            import boto3
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            from botocore.credentials import Credentials
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        responses: List[dict] = []
        for data in batch_data:
            sigv4 = SigV4Auth(credentials, "bedrock", aws_region_name)
            headers = {"Content-Type": "application/json"}
            if extra_headers is not None:
                headers = {"Content-Type": "application/json", **extra_headers}
            request = AWSRequest(
                method="POST", url=endpoint_url, data=json.dumps(data), headers=headers
            )
            sigv4.add_auth(request)
            if (
                extra_headers is not None and "Authorization" in extra_headers
            ):  # prevent sigv4 from overwriting the auth header
                request.headers["Authorization"] = extra_headers["Authorization"]
            prepped = request.prepare()

            ## LOGGING
            logging_obj.pre_call(
                input=data,
                api_key="",
                additional_args={
                    "complete_input_dict": data,
                    "api_base": prepped.url,
                    "headers": prepped.headers,
                },
            )
            response = await self._make_async_call(
                client=client,
                timeout=timeout,
                api_base=prepped.url,
                headers=prepped.headers,  # type: ignore
                data=data,
            )

            ## LOGGING
            logging_obj.post_call(
                input=data,
                api_key="",
                original_response=response,
                additional_args={"complete_input_dict": data},
            )

            responses.append(response)

        returned_response: Optional[EmbeddingResponse] = None

        ## TRANSFORM RESPONSE ##
        if model == "amazon.titan-embed-image-v1":
            returned_response = (
                AmazonTitanMultimodalEmbeddingG1Config()._transform_response(
                    response_list=responses, model=model
                )
            )
        elif model == "amazon.titan-embed-text-v1":
            returned_response = AmazonTitanG1Config()._transform_response(
                response_list=responses, model=model
            )
        elif model == "amazon.titan-embed-text-v2:0":
            returned_response = AmazonTitanV2Config()._transform_response(
                response_list=responses, model=model
            )

        if returned_response is None:
            raise Exception(
                "Unable to map model response to known provider format. model={}".format(
                    model
                )
            )

        return returned_response

    def embeddings(
        self,
        model: str,
        input: List[str],
        api_base: Optional[str],
        model_response: EmbeddingResponse,
        print_verbose: Callable,
        encoding,
        logging_obj,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]],
        timeout: Optional[Union[float, httpx.Timeout]],
        aembedding: Optional[bool],
        extra_headers: Optional[dict],
        optional_params: dict,
        litellm_params: dict,
    ) -> EmbeddingResponse:
        try:
            import boto3
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            from botocore.credentials import Credentials
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        credentials, aws_region_name = self._load_credentials(optional_params)

        ### TRANSFORMATION ###
        provider = model.split(".")[0]
        inference_params = copy.deepcopy(optional_params)
        inference_params.pop(
            "user", None
        )  # make sure user is not passed in for bedrock call
        modelId = (
            optional_params.pop("model_id", None) or model
        )  # default to model if not passed

        data: Optional[CohereEmbeddingRequest] = None
        batch_data: Optional[List] = None
        if provider == "cohere":
            data = BedrockCohereEmbeddingConfig()._transform_request(
                model=model, input=input, inference_params=inference_params
            )
        elif provider == "amazon" and model in [
            "amazon.titan-embed-image-v1",
            "amazon.titan-embed-text-v1",
            "amazon.titan-embed-text-v2:0",
        ]:
            batch_data = []
            for i in input:
                if model == "amazon.titan-embed-image-v1":
                    transformed_request: (
                        AmazonEmbeddingRequest
                    ) = AmazonTitanMultimodalEmbeddingG1Config()._transform_request(
                        input=i, inference_params=inference_params
                    )
                elif model == "amazon.titan-embed-text-v1":
                    transformed_request = AmazonTitanG1Config()._transform_request(
                        input=i, inference_params=inference_params
                    )
                elif model == "amazon.titan-embed-text-v2:0":
                    transformed_request = AmazonTitanV2Config()._transform_request(
                        input=i, inference_params=inference_params
                    )
                else:
                    raise Exception(
                        "Unmapped model. Received={}. Expected={}".format(
                            model,
                            [
                                "amazon.titan-embed-image-v1",
                                "amazon.titan-embed-text-v1",
                                "amazon.titan-embed-text-v2:0",
                            ],
                        )
                    )
                batch_data.append(transformed_request)

        ### SET RUNTIME ENDPOINT ###
        endpoint_url, proxy_endpoint_url = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=optional_params.pop(
                "aws_bedrock_runtime_endpoint", None
            ),
            aws_region_name=aws_region_name,
        )
        endpoint_url = f"{endpoint_url}/model/{modelId}/invoke"

        if batch_data is not None:
            if aembedding:
                return self._async_single_func_embeddings(  # type: ignore
                    client=(
                        client
                        if client is not None and isinstance(client, AsyncHTTPHandler)
                        else None
                    ),
                    timeout=timeout,
                    batch_data=batch_data,
                    credentials=credentials,
                    extra_headers=extra_headers,
                    endpoint_url=endpoint_url,
                    aws_region_name=aws_region_name,
                    model=model,
                    logging_obj=logging_obj,
                )
            return self._single_func_embeddings(
                client=(
                    client
                    if client is not None and isinstance(client, HTTPHandler)
                    else None
                ),
                timeout=timeout,
                batch_data=batch_data,
                credentials=credentials,
                extra_headers=extra_headers,
                endpoint_url=endpoint_url,
                aws_region_name=aws_region_name,
                model=model,
                logging_obj=logging_obj,
            )
        elif data is None:
            raise Exception("Unable to map request to provider")

        sigv4 = SigV4Auth(credentials, "bedrock", aws_region_name)
        headers = {"Content-Type": "application/json"}
        if extra_headers is not None:
            headers = {"Content-Type": "application/json", **extra_headers}

        request = AWSRequest(
            method="POST", url=endpoint_url, data=json.dumps(data), headers=headers
        )
        sigv4.add_auth(request)
        if (
            extra_headers is not None and "Authorization" in extra_headers
        ):  # prevent sigv4 from overwriting the auth header
            request.headers["Authorization"] = extra_headers["Authorization"]
        prepped = request.prepare()

        ## ROUTING ##
        return cohere_embedding(
            model=model,
            input=input,
            model_response=model_response,
            logging_obj=logging_obj,
            optional_params=optional_params,
            encoding=encoding,
            data=data,  # type: ignore
            complete_api_base=prepped.url,
            api_key=None,
            aembedding=aembedding,
            timeout=timeout,
            client=client,
            headers=prepped.headers,  # type: ignore
        )

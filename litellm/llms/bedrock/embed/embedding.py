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
from litellm.llms.cohere.embed import embedding as cohere_embedding
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_async_httpx_client,
    _get_httpx_client,
)
from litellm.secret_managers.main import get_secret
from litellm.types.llms.bedrock import AmazonEmbeddingRequest, CohereEmbeddingRequest
from litellm.types.utils import Embedding, EmbeddingResponse, Usage

from ...base_aws_llm import BaseAWSLLM
from ..common_utils import BedrockError, get_runtime_endpoint
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
        except ImportError as e:
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
            raise BedrockError(status_code=error_code, message=response.text)
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
            client = _get_async_httpx_client(_params)  # type: ignore
        else:
            client = client

        try:
            response = await client.post(url=api_base, headers=headers, data=json.dumps(data))  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise BedrockError(status_code=error_code, message=response.text)
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
                headers=prepped.headers,
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
                headers=prepped.headers,
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
        optional_params=None,
        litellm_params=None,
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
                input=input, inference_params=inference_params
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
                batch_data.append(transformed_request)

        ### SET RUNTIME ENDPOINT ###
        endpoint_url = get_runtime_endpoint(
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
            headers=prepped.headers,
        )

    # def _embedding_func_single(
    #     model: str,
    #     input: str,
    #     client: Any,
    #     optional_params=None,
    #     encoding=None,
    #     logging_obj=None,
    # ):
    #     if isinstance(input, str) is False:
    #         raise BedrockError(
    #             message="Bedrock Embedding API input must be type str | List[str]",
    #             status_code=400,
    #         )
    #     # logic for parsing in - calling - parsing out model embedding calls
    #     ## FORMAT EMBEDDING INPUT ##
    #     provider = model.split(".")[0]
    #     inference_params = copy.deepcopy(optional_params)
    #     inference_params.pop(
    #         "user", None
    #     )  # make sure user is not passed in for bedrock call
    #     modelId = (
    #         optional_params.pop("model_id", None) or model
    #     )  # default to model if not passed
    #     if provider == "amazon":
    #         input = input.replace(os.linesep, " ")
    #         data = {"inputText": input, **inference_params}
    #         # data = json.dumps(data)
    #     elif provider == "cohere":
    #         inference_params["input_type"] = inference_params.get(
    #             "input_type", "search_document"
    #         )  # aws bedrock example default - https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/providers?model=cohere.embed-english-v3
    #         data = {"texts": [input], **inference_params}  # type: ignore
    #     body = json.dumps(data).encode("utf-8")  # type: ignore
    #     ## LOGGING
    #     request_str = f"""
    #     response = client.invoke_model(
    #         body={body},
    #         modelId={modelId},
    #         accept="*/*",
    #         contentType="application/json",
    #     )"""  # type: ignore
    #     logging_obj.pre_call(
    #         input=input,
    #         api_key="",  # boto3 is used for init.
    #         additional_args={
    #             "complete_input_dict": {"model": modelId, "texts": input},
    #             "request_str": request_str,
    #         },
    #     )
    #     try:
    #         response = client.invoke_model(
    #             body=body,
    #             modelId=modelId,
    #             accept="*/*",
    #             contentType="application/json",
    #         )
    #         response_body = json.loads(response.get("body").read())
    #         ## LOGGING
    #         logging_obj.post_call(
    #             input=input,
    #             api_key="",
    #             additional_args={"complete_input_dict": data},
    #             original_response=json.dumps(response_body),
    #         )
    #         if provider == "cohere":
    #             response = response_body.get("embeddings")
    #             # flatten list
    #             response = [item for sublist in response for item in sublist]
    #             return response
    #         elif provider == "amazon":
    #             return response_body.get("embedding")
    #     except Exception as e:
    #         raise BedrockError(
    #             message=f"Embedding Error with model {model}: {e}", status_code=500
    #         )

    # def embedding(
    #     model: str,
    #     input: Union[list, str],
    #     model_response: litellm.EmbeddingResponse,
    #     api_key: Optional[str] = None,
    #     logging_obj=None,
    #     optional_params=None,
    #     encoding=None,
    # ):
    #     ### BOTO3 INIT ###
    #     # pop aws_secret_access_key, aws_access_key_id, aws_region_name from kwargs, since completion calls fail with them
    #     aws_secret_access_key = optional_params.pop("aws_secret_access_key", None)
    #     aws_access_key_id = optional_params.pop("aws_access_key_id", None)
    #     aws_region_name = optional_params.pop("aws_region_name", None)
    #     aws_role_name = optional_params.pop("aws_role_name", None)
    #     aws_session_name = optional_params.pop("aws_session_name", None)
    #     aws_bedrock_runtime_endpoint = optional_params.pop(
    #         "aws_bedrock_runtime_endpoint", None
    #     )
    #     aws_web_identity_token = optional_params.pop("aws_web_identity_token", None)

    #     # use passed in BedrockRuntime.Client if provided, otherwise create a new one
    #     client = init_bedrock_client(
    #         aws_access_key_id=aws_access_key_id,
    #         aws_secret_access_key=aws_secret_access_key,
    #         aws_region_name=aws_region_name,
    #         aws_bedrock_runtime_endpoint=aws_bedrock_runtime_endpoint,
    #         aws_web_identity_token=aws_web_identity_token,
    #         aws_role_name=aws_role_name,
    #         aws_session_name=aws_session_name,
    #     )
    #     if isinstance(input, str):
    #         ## Embedding Call
    #         embeddings = [
    #             _embedding_func_single(
    #                 model,
    #                 input,
    #                 optional_params=optional_params,
    #                 client=client,
    #                 logging_obj=logging_obj,
    #             )
    #         ]
    #     elif isinstance(input, list):
    #         ## Embedding Call - assuming this is a List[str]
    #         embeddings = [
    #             _embedding_func_single(
    #                 model,
    #                 i,
    #                 optional_params=optional_params,
    #                 client=client,
    #                 logging_obj=logging_obj,
    #             )
    #             for i in input
    #         ]  # [TODO]: make these parallel calls
    #     else:
    #         # enters this branch if input = int, ex. input=2
    #         raise BedrockError(
    #             message="Bedrock Embedding API input must be type str | List[str]",
    #             status_code=400,
    #         )

    #     ## Populate OpenAI compliant dictionary
    #     embedding_response = []
    #     for idx, embedding in enumerate(embeddings):
    #         embedding_response.append(
    #             {
    #                 "object": "embedding",
    #                 "index": idx,
    #                 "embedding": embedding,
    #             }
    #         )
    #     model_response.object = "list"
    #     model_response.data = embedding_response
    #     model_response.model = model
    #     input_tokens = 0

    #     input_str = "".join(input)

    #     input_tokens += len(encoding.encode(input_str))

    #     usage = Usage(
    #         prompt_tokens=input_tokens,
    #         completion_tokens=0,
    #         total_tokens=input_tokens + 0,
    #     )
    #     model_response.usage = usage

    #     return model_response

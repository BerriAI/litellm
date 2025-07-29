import json
from copy import deepcopy
from typing import Any, Callable, List, Optional, Union, cast

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import (
    CustomStreamWrapper,
    EmbeddingResponse,
    ModelResponse,
    Usage,
    get_secret,
)

from ..common_utils import AWSEventStreamDecoder, SagemakerError
from .transformation import SagemakerConfig

sagemaker_config = SagemakerConfig()

"""
SAGEMAKER AUTH Keys/Vars
os.environ['AWS_ACCESS_KEY_ID'] = ""
os.environ['AWS_SECRET_ACCESS_KEY'] = ""
"""


# set os.environ['AWS_REGION_NAME'] = <your-region_name>
class SagemakerLLM(BaseAWSLLM):
    def _load_credentials(
        self,
        optional_params: dict,
    ):
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
        optional_params.pop(
            "aws_bedrock_runtime_endpoint", None
        )  # https://bedrock-runtime.{region_name}.amazonaws.com
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

    def _prepare_request(
        self,
        credentials,
        model: str,
        data: dict,
        messages: List[AllMessageValues],
        litellm_params: dict,
        optional_params: dict,
        aws_region_name: str,
        extra_headers: Optional[dict] = None,
    ):
        try:
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        sigv4 = SigV4Auth(credentials, "sagemaker", aws_region_name)
        if optional_params.get("stream") is True:
            api_base = f"https://runtime.sagemaker.{aws_region_name}.amazonaws.com/endpoints/{model}/invocations-response-stream"
        else:
            api_base = f"https://runtime.sagemaker.{aws_region_name}.amazonaws.com/endpoints/{model}/invocations"

        sagemaker_base_url = optional_params.get("sagemaker_base_url", None)
        if sagemaker_base_url is not None:
            api_base = sagemaker_base_url

        encoded_data = json.dumps(data).encode("utf-8")
        headers = sagemaker_config.validate_environment(
            headers=extra_headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )
        request = AWSRequest(
            method="POST", url=api_base, data=encoded_data, headers=headers
        )
        sigv4.add_auth(request)
        if (
            extra_headers is not None and "Authorization" in extra_headers
        ):  # prevent sigv4 from overwriting the auth header
            request.headers["Authorization"] = extra_headers["Authorization"]

        prepped_request = request.prepare()

        return prepped_request

    def completion(  # noqa: PLR0915
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        logging_obj,
        optional_params: dict,
        litellm_params: dict,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        custom_prompt_dict={},
        hf_model_name=None,
        logger_fn=None,
        acompletion: bool = False,
        headers: dict = {},
    ):
        # pop streaming if it's in the optional params as 'stream' raises an error with sagemaker
        credentials, aws_region_name = self._load_credentials(optional_params)
        inference_params = deepcopy(optional_params)
        stream = inference_params.pop("stream", None)
        model_id = optional_params.get("model_id", None)

        ## Load Config
        config = litellm.SagemakerConfig.get_config()
        for k, v in config.items():
            if (
                k not in inference_params
            ):  # completion(top_k=3) > sagemaker_config(top_k=3) <- allows for dynamic variables to be passed in
                inference_params[k] = v

        if stream is True:
            if acompletion is True:
                response = self.async_streaming(
                    messages=messages,
                    model=model,
                    custom_prompt_dict=custom_prompt_dict,
                    hf_model_name=hf_model_name,
                    optional_params=optional_params,
                    encoding=encoding,
                    model_response=model_response,
                    logging_obj=logging_obj,
                    model_id=model_id,
                    aws_region_name=aws_region_name,
                    credentials=credentials,
                    headers=headers,
                    litellm_params=litellm_params,
                )
                return response
            else:
                data = sagemaker_config.transform_request(
                    model=model,
                    messages=messages,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    headers=headers,
                )
                prepared_request = self._prepare_request(
                    model=model,
                    data=data,
                    messages=messages,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    credentials=credentials,
                    aws_region_name=aws_region_name,
                )
                if model_id is not None:
                    # Add model_id as InferenceComponentName header
                    # boto3 doc: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html
                    prepared_request.headers.update(
                        {"X-Amzn-SageMaker-Inference-Component": model_id}
                    )
                sync_handler = _get_httpx_client()
                sync_response = sync_handler.post(
                    url=prepared_request.url,
                    headers=prepared_request.headers,  # type: ignore
                    data=prepared_request.body,
                    stream=stream,
                )

                if sync_response.status_code != 200:
                    raise SagemakerError(
                        status_code=sync_response.status_code,
                        message=str(sync_response.read()),
                    )

                decoder = AWSEventStreamDecoder(model="")

                completion_stream = decoder.iter_bytes(
                    sync_response.iter_bytes(chunk_size=1024)
                )
                streaming_response = CustomStreamWrapper(
                    completion_stream=completion_stream,
                    model=model,
                    custom_llm_provider="sagemaker",
                    logging_obj=logging_obj,
                )

            ## LOGGING
            logging_obj.post_call(
                input=messages,
                api_key="",
                original_response=streaming_response,
                additional_args={"complete_input_dict": data},
            )
            return streaming_response

        # Non-Streaming Requests

        # Async completion
        if acompletion is True:
            return self.async_completion(
                messages=messages,
                model=model,
                custom_prompt_dict=custom_prompt_dict,
                hf_model_name=hf_model_name,
                model_response=model_response,
                encoding=encoding,
                logging_obj=logging_obj,
                model_id=model_id,
                optional_params=optional_params,
                credentials=credentials,
                aws_region_name=aws_region_name,
                headers=headers,
                litellm_params=litellm_params,
            )

        ## Non-Streaming completion CALL
        _data = sagemaker_config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        prepared_request_args = {
            "model": model,
            "data": _data,
            "optional_params": optional_params,
            "litellm_params": litellm_params,
            "credentials": credentials,
            "aws_region_name": aws_region_name,
            "messages": messages,
        }
        prepared_request = self._prepare_request(**prepared_request_args)
        try:
            if model_id is not None:
                # Add model_id as InferenceComponentName header
                # boto3 doc: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html
                prepared_request.headers.update(
                    {"X-Amzn-SageMaker-Inference-Component": model_id}
                )

            ## LOGGING
            timeout = 300.0
            sync_handler = _get_httpx_client()
            ## LOGGING
            logging_obj.pre_call(
                input=[],
                api_key="",
                additional_args={
                    "complete_input_dict": _data,
                    "api_base": prepared_request.url,
                    "headers": prepared_request.headers,
                },
            )

            # make sync httpx post request here
            try:
                sync_response = sync_handler.post(
                    url=prepared_request.url,
                    headers=prepared_request.headers,  # type: ignore
                    data=prepared_request.body,
                    timeout=timeout,
                )

                if sync_response.status_code != 200:
                    raise SagemakerError(
                        status_code=sync_response.status_code,
                        message=sync_response.text,
                    )
            except Exception as e:
                ## LOGGING
                logging_obj.post_call(
                    input=[],
                    api_key="",
                    original_response=str(e),
                    additional_args={"complete_input_dict": _data},
                )
                raise e
        except Exception as e:
            verbose_logger.error("Sagemaker error %s", str(e))
            status_code = (
                getattr(e, "response", {})
                .get("ResponseMetadata", {})
                .get("HTTPStatusCode", 500)
            )
            error_message = (
                getattr(e, "response", {}).get("Error", {}).get("Message", str(e))
            )
            if "Inference Component Name header is required" in error_message:
                error_message += "\n pass in via `litellm.completion(..., model_id={InferenceComponentName})`"
            raise SagemakerError(status_code=status_code, message=error_message)

        return sagemaker_config.transform_response(
            model=model,
            raw_response=sync_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=_data,
            messages=messages,
            optional_params=optional_params,
            encoding=encoding,
            litellm_params=litellm_params,
        )

    async def make_async_call(
        self,
        api_base: str,
        headers: dict,
        data: str,
        logging_obj,
        client=None,
    ):
        try:
            if client is None:
                client = get_async_httpx_client(
                    llm_provider=litellm.LlmProviders.SAGEMAKER
                )  # Create a new client if none provided
            response = await client.post(
                api_base,
                headers=headers,
                data=data,
                stream=True,
            )

            if response.status_code != 200:
                raise SagemakerError(
                    status_code=response.status_code, message=response.text
                )

            decoder = AWSEventStreamDecoder(model="")
            completion_stream = decoder.aiter_bytes(
                response.aiter_bytes(chunk_size=1024)
            )

            return completion_stream

            # LOGGING
            logging_obj.post_call(
                input=[],
                api_key="",
                original_response="first stream response received",
                additional_args={"complete_input_dict": data},
            )

        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise SagemakerError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
            raise SagemakerError(status_code=408, message="Timeout error occurred.")
        except Exception as e:
            raise SagemakerError(status_code=500, message=str(e))

    async def async_streaming(
        self,
        messages: List[AllMessageValues],
        model: str,
        custom_prompt_dict: dict,
        hf_model_name: Optional[str],
        credentials,
        aws_region_name: str,
        optional_params,
        encoding,
        model_response: ModelResponse,
        model_id: Optional[str],
        logging_obj: Any,
        litellm_params: dict,
        headers: dict,
    ):
        data = await sagemaker_config.async_transform_request(
            model=model,
            messages=messages,
            optional_params={**optional_params, "stream": True},
            litellm_params=litellm_params,
            headers=headers,
        )
        asyncified_prepare_request = asyncify(self._prepare_request)
        prepared_request_args = {
            "model": model,
            "data": data,
            "optional_params": optional_params,
            "litellm_params": litellm_params,
            "credentials": credentials,
            "aws_region_name": aws_region_name,
            "messages": messages,
        }
        prepared_request = await asyncified_prepare_request(**prepared_request_args)
        if model_id is not None:  # Fixes https://github.com/BerriAI/litellm/issues/8889
            prepared_request.headers.update(
                {"X-Amzn-SageMaker-Inference-Component": model_id}
            )

        if not prepared_request.body:
            raise ValueError("Prepared request body is empty")

        completion_stream = await self.make_async_call(
            api_base=prepared_request.url,
            headers=prepared_request.headers,  # type: ignore
            data=cast(str, prepared_request.body),
            logging_obj=logging_obj,
        )
        streaming_response = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="sagemaker",
            logging_obj=logging_obj,
        )

        # LOGGING
        logging_obj.post_call(
            input=[],
            api_key="",
            original_response="first stream response received",
            additional_args={"complete_input_dict": data},
        )

        return streaming_response

    async def async_completion(
        self,
        messages: List[AllMessageValues],
        model: str,
        custom_prompt_dict: dict,
        hf_model_name: Optional[str],
        credentials,
        aws_region_name: str,
        encoding,
        model_response: ModelResponse,
        optional_params: dict,
        logging_obj: Any,
        model_id: Optional[str],
        headers: dict,
        litellm_params: dict,
    ):
        timeout = 300.0
        async_handler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.SAGEMAKER
        )

        data = await sagemaker_config.async_transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        asyncified_prepare_request = asyncify(self._prepare_request)
        prepared_request_args = {
            "model": model,
            "data": data,
            "optional_params": optional_params,
            "litellm_params": litellm_params,
            "credentials": credentials,
            "aws_region_name": aws_region_name,
            "messages": messages,
        }

        prepared_request = await asyncified_prepare_request(**prepared_request_args)
        ## LOGGING
        logging_obj.pre_call(
            input=[],
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": prepared_request.url,
                "headers": prepared_request.headers,
            },
        )
        try:
            if model_id is not None:
                # Add model_id as InferenceComponentName header
                # boto3 doc: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html
                prepared_request.headers.update(
                    {"X-Amzn-SageMaker-Inference-Component": model_id}
                )
            # make async httpx post request here
            try:
                response = await async_handler.post(
                    url=prepared_request.url,
                    headers=prepared_request.headers,  # type: ignore
                    data=prepared_request.body,
                    timeout=timeout,
                )

                if response.status_code != 200:
                    raise SagemakerError(
                        status_code=response.status_code, message=response.text
                    )
            except Exception as e:
                ## LOGGING
                logging_obj.post_call(
                    input=data["inputs"],
                    api_key="",
                    original_response=str(e),
                    additional_args={"complete_input_dict": data},
                )
                raise e
        except Exception as e:
            error_message = f"{str(e)}"
            if "Inference Component Name header is required" in error_message:
                error_message += "\n pass in via `litellm.completion(..., model_id={InferenceComponentName})`"
            raise SagemakerError(status_code=500, message=error_message)
        return sagemaker_config.transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=data,
            messages=messages,
            optional_params=optional_params,
            encoding=encoding,
            litellm_params=litellm_params,
        )

    def embedding(
        self,
        model: str,
        input: list,
        model_response: EmbeddingResponse,
        print_verbose: Callable,
        encoding,
        logging_obj,
        optional_params: dict,
        custom_prompt_dict={},
        litellm_params=None,
        logger_fn=None,
    ):
        """
        Supports Huggingface Jumpstart embeddings like GPT-6B
        """
        ### BOTO3 INIT
        import boto3

        # pop aws_secret_access_key, aws_access_key_id, aws_region_name from kwargs, since completion calls fail with them
        aws_secret_access_key = optional_params.pop("aws_secret_access_key", None)
        aws_access_key_id = optional_params.pop("aws_access_key_id", None)
        aws_region_name = optional_params.pop("aws_region_name", None)

        if aws_access_key_id is not None:
            # uses auth params passed to completion
            # aws_access_key_id is not None, assume user is trying to auth using litellm.completion
            client = boto3.client(
                service_name="sagemaker-runtime",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_region_name,
            )
        else:
            # aws_access_key_id is None, assume user is trying to auth using env variables
            # boto3 automaticaly reads env variables

            # we need to read region name from env
            # I assume majority of users use .env for auth
            region_name = (
                get_secret("AWS_REGION_NAME")
                or aws_region_name  # get region from config file if specified
                or "us-west-2"  # default to us-west-2 if region not specified
            )
            client = boto3.client(
                service_name="sagemaker-runtime",
                region_name=region_name,
            )

        # pop streaming if it's in the optional params as 'stream' raises an error with sagemaker
        inference_params = deepcopy(optional_params)
        inference_params.pop("stream", None)

        ## Load Config
        config = litellm.SagemakerConfig.get_config()
        for k, v in config.items():
            if (
                k not in inference_params
            ):  # completion(top_k=3) > sagemaker_config(top_k=3) <- allows for dynamic variables to be passed in
                inference_params[k] = v

        #### HF EMBEDDING LOGIC
        data = json.dumps({"inputs": input}).encode("utf-8")

        ## LOGGING
        request_str = f"""
        response = client.invoke_endpoint(
            EndpointName={model},
            ContentType="application/json",
            Body=f"{data!r}",  # Use !r for safe representation
            CustomAttributes="accept_eula=true",
        )"""  # type: ignore
        logging_obj.pre_call(
            input=input,
            api_key="",
            additional_args={"complete_input_dict": data, "request_str": request_str},
        )
        ## EMBEDDING CALL
        try:
            response = client.invoke_endpoint(
                EndpointName=model,
                ContentType="application/json",
                Body=data,
                CustomAttributes="accept_eula=true",
            )
        except Exception as e:
            status_code = (
                getattr(e, "response", {})
                .get("ResponseMetadata", {})
                .get("HTTPStatusCode", 500)
            )
            error_message = (
                getattr(e, "response", {}).get("Error", {}).get("Message", str(e))
            )
            raise SagemakerError(status_code=status_code, message=error_message)

        response = json.loads(response["Body"].read().decode("utf8"))
        ## LOGGING
        logging_obj.post_call(
            input=input,
            api_key="",
            original_response=response,
            additional_args={"complete_input_dict": data},
        )

        print_verbose(f"raw model_response: {response}")
        if "embedding" not in response:
            raise SagemakerError(
                status_code=500, message="embedding not found in response"
            )
        embeddings = response["embedding"]

        if not isinstance(embeddings, list):
            raise SagemakerError(
                status_code=422,
                message=f"Response not in expected format - {embeddings}",
            )

        output_data = []
        for idx, embedding in enumerate(embeddings):
            output_data.append(
                {"object": "embedding", "index": idx, "embedding": embedding}
            )

        model_response.object = "list"
        model_response.data = output_data
        model_response.model = model

        input_tokens = 0
        for text in input:
            input_tokens += len(encoding.encode(text))

        setattr(
            model_response,
            "usage",
            Usage(
                prompt_tokens=input_tokens,
                completion_tokens=0,
                total_tokens=input_tokens,
            ),
        )

        return model_response

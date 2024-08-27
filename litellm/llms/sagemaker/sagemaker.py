import io
import json
import os
import sys
import time
import traceback
import types
from copy import deepcopy
from enum import Enum
from functools import partial
from typing import Any, AsyncIterator, Callable, Iterator, List, Optional, Union

import httpx  # type: ignore
import requests  # type: ignore

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_async_httpx_client,
    _get_httpx_client,
)
from litellm.types.llms.openai import (
    ChatCompletionToolCallChunk,
    ChatCompletionUsageBlock,
)
from litellm.types.utils import GenericStreamingChunk as GChunk
from litellm.types.utils import StreamingChatCompletionChunk
from litellm.utils import (
    CustomStreamWrapper,
    EmbeddingResponse,
    ModelResponse,
    Usage,
    get_secret,
)

from ..base_aws_llm import BaseAWSLLM
from ..prompt_templates.factory import custom_prompt, prompt_factory

_response_stream_shape_cache = None


class SagemakerError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url="https://us-west-2.console.aws.amazon.com/sagemaker"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class SagemakerConfig:
    """
    Reference: https://d-uuwbxj1u4cnu.studio.us-west-2.sagemaker.aws/jupyter/default/lab/workspaces/auto-q/tree/DemoNotebooks/meta-textgeneration-llama-2-7b-SDK_1.ipynb
    """

    max_new_tokens: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    return_full_text: Optional[bool] = None

    def __init__(
        self,
        max_new_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        return_full_text: Optional[bool] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }


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
        aws_bedrock_runtime_endpoint = optional_params.pop(
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
        optional_params: dict,
        aws_region_name: str,
        extra_headers: Optional[dict] = None,
    ):
        try:
            import boto3
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            from botocore.credentials import Credentials
        except ImportError as e:
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
        headers = {"Content-Type": "application/json"}
        if extra_headers is not None:
            headers = {"Content-Type": "application/json", **extra_headers}
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

    def _transform_prompt(
        self,
        model: str,
        messages: List,
        custom_prompt_dict: dict,
        hf_model_name: Optional[str],
    ) -> str:
        if model in custom_prompt_dict:
            # check if the model has a registered custom prompt
            model_prompt_details = custom_prompt_dict[model]
            prompt = custom_prompt(
                role_dict=model_prompt_details.get("roles", None),
                initial_prompt_value=model_prompt_details.get(
                    "initial_prompt_value", ""
                ),
                final_prompt_value=model_prompt_details.get("final_prompt_value", ""),
                messages=messages,
            )
        elif hf_model_name in custom_prompt_dict:
            # check if the base huggingface model has a registered custom prompt
            model_prompt_details = custom_prompt_dict[hf_model_name]
            prompt = custom_prompt(
                role_dict=model_prompt_details.get("roles", None),
                initial_prompt_value=model_prompt_details.get(
                    "initial_prompt_value", ""
                ),
                final_prompt_value=model_prompt_details.get("final_prompt_value", ""),
                messages=messages,
            )
        else:
            if hf_model_name is None:
                if "llama-2" in model.lower():  # llama-2 model
                    if "chat" in model.lower():  # apply llama2 chat template
                        hf_model_name = "meta-llama/Llama-2-7b-chat-hf"
                    else:  # apply regular llama2 template
                        hf_model_name = "meta-llama/Llama-2-7b"
            hf_model_name = (
                hf_model_name or model
            )  # pass in hf model name for pulling it's prompt template - (e.g. `hf_model_name="meta-llama/Llama-2-7b-chat-hf` applies the llama2 chat template to the prompt)
            prompt = prompt_factory(model=hf_model_name, messages=messages)

        return prompt

    def completion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        logging_obj,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        custom_prompt_dict={},
        hf_model_name=None,
        optional_params=None,
        litellm_params=None,
        logger_fn=None,
        acompletion: bool = False,
        use_messages_api: Optional[bool] = None,
    ):

        # pop streaming if it's in the optional params as 'stream' raises an error with sagemaker
        credentials, aws_region_name = self._load_credentials(optional_params)
        inference_params = deepcopy(optional_params)
        stream = inference_params.pop("stream", None)
        model_id = optional_params.get("model_id", None)

        if use_messages_api is True:
            from litellm.llms.databricks import DatabricksChatCompletion

            openai_like_chat_completions = DatabricksChatCompletion()
            inference_params["stream"] = True if stream is True else False
            _data = {
                "model": model,
                "messages": messages,
                **inference_params,
            }

            prepared_request = self._prepare_request(
                model=model,
                data=_data,
                optional_params=optional_params,
                credentials=credentials,
                aws_region_name=aws_region_name,
            )

            custom_stream_decoder = AWSEventStreamDecoder(
                model="", is_messages_api=True
            )

            return openai_like_chat_completions.completion(
                model=model,
                messages=messages,
                api_base=prepared_request.url,
                api_key=None,
                custom_prompt_dict=custom_prompt_dict,
                model_response=model_response,
                print_verbose=print_verbose,
                logging_obj=logging_obj,
                optional_params=inference_params,
                acompletion=acompletion,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                timeout=timeout,
                encoding=encoding,
                headers=prepared_request.headers,
                custom_endpoint=True,
                custom_llm_provider="sagemaker_chat",
                streaming_decoder=custom_stream_decoder,  # type: ignore
            )

        ## Load Config
        config = litellm.SagemakerConfig.get_config()
        for k, v in config.items():
            if (
                k not in inference_params
            ):  # completion(top_k=3) > sagemaker_config(top_k=3) <- allows for dynamic variables to be passed in
                inference_params[k] = v

        if stream is True:
            data = {"parameters": inference_params, "stream": True}
            prepared_request = self._prepare_request(
                model=model,
                data=data,
                optional_params=optional_params,
                credentials=credentials,
                aws_region_name=aws_region_name,
            )
            if model_id is not None:
                # Add model_id as InferenceComponentName header
                # boto3 doc: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html
                prepared_request.headers.update(
                    {"X-Amzn-SageMaker-Inference-Component": model_id}
                )

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
                    data=data,
                    model_id=model_id,
                    aws_region_name=aws_region_name,
                    credentials=credentials,
                )
                return response
            else:
                prompt = self._transform_prompt(
                    model=model,
                    messages=messages,
                    custom_prompt_dict=custom_prompt_dict,
                    hf_model_name=hf_model_name,
                )
                data["inputs"] = prompt
                prepared_request = self._prepare_request(
                    model=model,
                    data=data,
                    optional_params=optional_params,
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
                    json=data,
                    stream=stream,
                )

                if sync_response.status_code != 200:
                    raise SagemakerError(
                        status_code=sync_response.status_code,
                        message=sync_response.read(),
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
        _data = {"parameters": inference_params}
        prepared_request_args = {
            "model": model,
            "data": _data,
            "optional_params": optional_params,
            "credentials": credentials,
            "aws_region_name": aws_region_name,
        }

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
                data=_data,
                model_id=model_id,
                optional_params=optional_params,
                credentials=credentials,
                aws_region_name=aws_region_name,
            )

        prompt = self._transform_prompt(
            model=model,
            messages=messages,
            custom_prompt_dict=custom_prompt_dict,
            hf_model_name=hf_model_name,
        )
        _data["inputs"] = prompt
        ## Non-Streaming completion CALL
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
                    headers=prepared_request.headers,
                    json=_data,
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

        completion_response = sync_response.json()
        ## LOGGING
        logging_obj.post_call(
            input=prompt,
            api_key="",
            original_response=completion_response,
            additional_args={"complete_input_dict": _data},
        )
        print_verbose(f"raw model_response: {completion_response}")
        ## RESPONSE OBJECT
        try:
            if isinstance(completion_response, list):
                completion_response_choices = completion_response[0]
            else:
                completion_response_choices = completion_response
            completion_output = ""
            if "generation" in completion_response_choices:
                completion_output += completion_response_choices["generation"]
            elif "generated_text" in completion_response_choices:
                completion_output += completion_response_choices["generated_text"]

            # check if the prompt template is part of output, if so - filter it out
            if completion_output.startswith(prompt) and "<s>" in prompt:
                completion_output = completion_output.replace(prompt, "", 1)

            model_response.choices[0].message.content = completion_output  # type: ignore
        except Exception:
            raise SagemakerError(
                message=f"LiteLLM Error: Unable to parse sagemaker RAW RESPONSE {json.dumps(completion_response)}",
                status_code=500,
            )

        ## CALCULATING USAGE - baseten charges on time, not tokens - have some mapping of cost here.
        prompt_tokens = len(encoding.encode(prompt))
        completion_tokens = len(
            encoding.encode(model_response["choices"][0]["message"].get("content", ""))
        )

        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)
        return model_response

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
                client = (
                    _get_async_httpx_client()
                )  # Create a new client if none provided
            response = await client.post(
                api_base,
                headers=headers,
                json=data,
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
        except httpx.TimeoutException as e:
            raise SagemakerError(status_code=408, message="Timeout error occurred.")
        except Exception as e:
            raise SagemakerError(status_code=500, message=str(e))

    async def async_streaming(
        self,
        messages: list,
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
        data,
    ):
        data["inputs"] = self._transform_prompt(
            model=model,
            messages=messages,
            custom_prompt_dict=custom_prompt_dict,
            hf_model_name=hf_model_name,
        )
        asyncified_prepare_request = asyncify(self._prepare_request)
        prepared_request_args = {
            "model": model,
            "data": data,
            "optional_params": optional_params,
            "credentials": credentials,
            "aws_region_name": aws_region_name,
        }
        prepared_request = await asyncified_prepare_request(**prepared_request_args)
        streaming_response = CustomStreamWrapper(
            completion_stream=None,
            make_call=partial(
                self.make_async_call,
                api_base=prepared_request.url,
                headers=prepared_request.headers,
                data=data,
                logging_obj=logging_obj,
            ),
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
        messages: list,
        model: str,
        custom_prompt_dict: dict,
        hf_model_name: Optional[str],
        credentials,
        aws_region_name: str,
        encoding,
        model_response: ModelResponse,
        optional_params: dict,
        logging_obj: Any,
        data: dict,
        model_id: Optional[str],
    ):
        timeout = 300.0
        async_handler = _get_async_httpx_client()

        async_transform_prompt = asyncify(self._transform_prompt)

        data["inputs"] = await async_transform_prompt(
            model=model,
            messages=messages,
            custom_prompt_dict=custom_prompt_dict,
            hf_model_name=hf_model_name,
        )
        asyncified_prepare_request = asyncify(self._prepare_request)
        prepared_request_args = {
            "model": model,
            "data": data,
            "optional_params": optional_params,
            "credentials": credentials,
            "aws_region_name": aws_region_name,
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
                    {"X-Amzn-SageMaker-Inference-Componen": model_id}
                )
            # make async httpx post request here
            try:
                response = await async_handler.post(
                    url=prepared_request.url,
                    headers=prepared_request.headers,
                    json=data,
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
        completion_response = response.json()
        ## LOGGING
        logging_obj.post_call(
            input=data["inputs"],
            api_key="",
            original_response=response,
            additional_args={"complete_input_dict": data},
        )
        ## RESPONSE OBJECT
        try:
            if isinstance(completion_response, list):
                completion_response_choices = completion_response[0]
            else:
                completion_response_choices = completion_response
            completion_output = ""
            if "generation" in completion_response_choices:
                completion_output += completion_response_choices["generation"]
            elif "generated_text" in completion_response_choices:
                completion_output += completion_response_choices["generated_text"]

            # check if the prompt template is part of output, if so - filter it out
            if completion_output.startswith(data["inputs"]) and "<s>" in data["inputs"]:
                completion_output = completion_output.replace(data["inputs"], "", 1)

            model_response.choices[0].message.content = completion_output  # type: ignore
        except Exception:
            raise SagemakerError(
                message=f"LiteLLM Error: Unable to parse sagemaker RAW RESPONSE {json.dumps(completion_response)}",
                status_code=500,
            )

        ## CALCULATING USAGE - baseten charges on time, not tokens - have some mapping of cost here.
        prompt_tokens = len(encoding.encode(data["inputs"]))
        completion_tokens = len(
            encoding.encode(model_response["choices"][0]["message"].get("content", ""))
        )

        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)
        return model_response

    def embedding(
        self,
        model: str,
        input: list,
        model_response: EmbeddingResponse,
        print_verbose: Callable,
        encoding,
        logging_obj,
        custom_prompt_dict={},
        optional_params=None,
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
        data = json.dumps({"text_inputs": input}).encode("utf-8")

        ## LOGGING
        request_str = f"""
        response = client.invoke_endpoint(
            EndpointName={model},
            ContentType="application/json",
            Body={data}, # type: ignore
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


def get_response_stream_shape():
    global _response_stream_shape_cache
    if _response_stream_shape_cache is None:

        from botocore.loaders import Loader
        from botocore.model import ServiceModel

        loader = Loader()
        sagemaker_service_dict = loader.load_service_model(
            "sagemaker-runtime", "service-2"
        )
        sagemaker_service_model = ServiceModel(sagemaker_service_dict)
        _response_stream_shape_cache = sagemaker_service_model.shape_for(
            "InvokeEndpointWithResponseStreamOutput"
        )
    return _response_stream_shape_cache


class AWSEventStreamDecoder:
    def __init__(self, model: str, is_messages_api: Optional[bool] = None) -> None:
        from botocore.parsers import EventStreamJSONParser

        self.model = model
        self.parser = EventStreamJSONParser()
        self.content_blocks: List = []
        self.is_messages_api = is_messages_api

    def _chunk_parser_messages_api(
        self, chunk_data: dict
    ) -> StreamingChatCompletionChunk:

        openai_chunk = StreamingChatCompletionChunk(**chunk_data)

        return openai_chunk

    def _chunk_parser(self, chunk_data: dict) -> GChunk:
        verbose_logger.debug("in sagemaker chunk parser, chunk_data %s", chunk_data)
        _token = chunk_data.get("token", {}) or {}
        _index = chunk_data.get("index", None) or 0
        is_finished = False
        finish_reason = ""

        _text = _token.get("text", "")
        if _text == "<|endoftext|>":
            return GChunk(
                text="",
                index=_index,
                is_finished=True,
                finish_reason="stop",
                usage=None,
            )

        return GChunk(
            text=_text,
            index=_index,
            is_finished=is_finished,
            finish_reason=finish_reason,
            usage=None,
        )

    def iter_bytes(
        self, iterator: Iterator[bytes]
    ) -> Iterator[Optional[Union[GChunk, StreamingChatCompletionChunk]]]:
        """Given an iterator that yields lines, iterate over it & yield every event encountered"""
        from botocore.eventstream import EventStreamBuffer

        event_stream_buffer = EventStreamBuffer()
        accumulated_json = ""

        for chunk in iterator:
            event_stream_buffer.add_data(chunk)
            for event in event_stream_buffer:
                message = self._parse_message_from_event(event)
                if message:
                    # remove data: prefix and "\n\n" at the end
                    message = message.replace("data:", "").replace("\n\n", "")

                    # Accumulate JSON data
                    accumulated_json += message

                    # Try to parse the accumulated JSON
                    try:
                        _data = json.loads(accumulated_json)
                        if self.is_messages_api:
                            yield self._chunk_parser_messages_api(chunk_data=_data)
                        else:
                            yield self._chunk_parser(chunk_data=_data)
                        # Reset accumulated_json after successful parsing
                        accumulated_json = ""
                    except json.JSONDecodeError:
                        # If it's not valid JSON yet, continue to the next event
                        continue

        # Handle any remaining data after the iterator is exhausted
        if accumulated_json:
            try:
                _data = json.loads(accumulated_json)
                if self.is_messages_api:
                    yield self._chunk_parser_messages_api(chunk_data=_data)
                else:
                    yield self._chunk_parser(chunk_data=_data)
            except json.JSONDecodeError as e:
                # Handle or log any unparseable data at the end
                verbose_logger.error(
                    f"Warning: Unparseable JSON data remained: {accumulated_json}"
                )
                yield None

    async def aiter_bytes(
        self, iterator: AsyncIterator[bytes]
    ) -> AsyncIterator[Optional[Union[GChunk, StreamingChatCompletionChunk]]]:
        """Given an async iterator that yields lines, iterate over it & yield every event encountered"""
        from botocore.eventstream import EventStreamBuffer

        event_stream_buffer = EventStreamBuffer()
        accumulated_json = ""

        async for chunk in iterator:
            event_stream_buffer.add_data(chunk)
            for event in event_stream_buffer:
                message = self._parse_message_from_event(event)
                if message:
                    verbose_logger.debug("sagemaker  parsed chunk bytes %s", message)
                    # remove data: prefix and "\n\n" at the end
                    message = message.replace("data:", "").replace("\n\n", "")

                    # Accumulate JSON data
                    accumulated_json += message

                    # Try to parse the accumulated JSON
                    try:
                        _data = json.loads(accumulated_json)
                        if self.is_messages_api:
                            yield self._chunk_parser_messages_api(chunk_data=_data)
                        else:
                            yield self._chunk_parser(chunk_data=_data)
                        # Reset accumulated_json after successful parsing
                        accumulated_json = ""
                    except json.JSONDecodeError:
                        # If it's not valid JSON yet, continue to the next event
                        continue

        # Handle any remaining data after the iterator is exhausted
        if accumulated_json:
            try:
                _data = json.loads(accumulated_json)
                if self.is_messages_api:
                    yield self._chunk_parser_messages_api(chunk_data=_data)
                else:
                    yield self._chunk_parser(chunk_data=_data)
            except json.JSONDecodeError:
                # Handle or log any unparseable data at the end
                verbose_logger.error(
                    f"Warning: Unparseable JSON data remained: {accumulated_json}"
                )
                yield None

    def _parse_message_from_event(self, event) -> Optional[str]:
        response_dict = event.to_response_dict()
        parsed_response = self.parser.parse(response_dict, get_response_stream_shape())

        if response_dict["status_code"] != 200:
            raise ValueError(f"Bad response code, expected 200: {response_dict}")

        if "chunk" in parsed_response:
            chunk = parsed_response.get("chunk")
            if not chunk:
                return None
            return chunk.get("bytes").decode()  # type: ignore[no-any-return]
        else:
            chunk = response_dict.get("body")
            if not chunk:
                return None

            return chunk.decode()  # type: ignore[no-any-return]

# What is this?
## Initial implementation of calling bedrock via httpx client (allows for async calls).
## V0 - just covers cohere command-r support

import os, types
import json
from enum import Enum
import requests, copy  # type: ignore
import time
from typing import Callable, Optional, List, Literal, Union, Any, TypedDict, Tuple
from litellm.utils import (
    ModelResponse,
    Usage,
    map_finish_reason,
    CustomStreamWrapper,
    Message,
    Choices,
    get_secret,
)
import litellm
from .prompt_templates.factory import prompt_factory, custom_prompt, cohere_message_pt
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from .base import BaseLLM
import httpx  # type: ignore
from .bedrock import BedrockError, convert_messages_to_prompt
from litellm.types.llms.bedrock import *


class AmazonCohereChatConfig:
    """
    Reference - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command-r-plus.html
    """

    documents: Optional[List[Document]] = None
    search_queries_only: Optional[bool] = None
    preamble: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    p: Optional[float] = None
    k: Optional[float] = None
    prompt_truncation: Optional[str] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None
    return_prompt: Optional[bool] = None
    stop_sequences: Optional[List[str]] = None
    raw_prompting: Optional[bool] = None

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        search_queries_only: Optional[bool] = None,
        preamble: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        p: Optional[float] = None,
        k: Optional[float] = None,
        prompt_truncation: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        return_prompt: Optional[bool] = None,
        stop_sequences: Optional[str] = None,
        raw_prompting: Optional[bool] = None,
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

    def get_supported_openai_params(self) -> List[str]:
        return [
            "max_tokens",
            "stream",
            "stop",
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "stop",
        ]

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["max_tokens"] = value
            if param == "stream":
                optional_params["stream"] = value
            if param == "stop":
                if isinstance(value, str):
                    value = [value]
                optional_params["stop_sequences"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["p"] = value
            if param == "frequency_penalty":
                optional_params["frequency_penalty"] = value
            if param == "presence_penalty":
                optional_params["presence_penalty"] = value
            if "seed":
                optional_params["seed"] = value
        return optional_params


class BedrockLLM(BaseLLM):
    """
    Example call

    ```
    curl --location --request POST 'https://bedrock-runtime.{aws_region_name}.amazonaws.com/model/{bedrock_model_name}/invoke' \
        --header 'Content-Type: application/json' \
        --header 'Accept: application/json' \
        --user "$AWS_ACCESS_KEY_ID":"$AWS_SECRET_ACCESS_KEY" \
        --aws-sigv4 "aws:amz:us-east-1:bedrock" \
        --data-raw '{
        "prompt": "Hi",
        "temperature": 0,
        "p": 0.9,
        "max_tokens": 4096
        }'
    ```
    """

    def __init__(self) -> None:
        super().__init__()

    def convert_messages_to_prompt(
        self, model, messages, provider, custom_prompt_dict
    ) -> Tuple[str, Optional[list]]:
        # handle anthropic prompts and amazon titan prompts
        prompt = ""
        chat_history: Optional[list] = None
        if provider == "anthropic" or provider == "amazon":
            if model in custom_prompt_dict:
                # check if the model has a registered custom prompt
                model_prompt_details = custom_prompt_dict[model]
                prompt = custom_prompt(
                    role_dict=model_prompt_details["roles"],
                    initial_prompt_value=model_prompt_details["initial_prompt_value"],
                    final_prompt_value=model_prompt_details["final_prompt_value"],
                    messages=messages,
                )
            else:
                prompt = prompt_factory(
                    model=model, messages=messages, custom_llm_provider="bedrock"
                )
        elif provider == "mistral":
            prompt = prompt_factory(
                model=model, messages=messages, custom_llm_provider="bedrock"
            )
        elif provider == "meta":
            prompt = prompt_factory(
                model=model, messages=messages, custom_llm_provider="bedrock"
            )
        elif provider == "cohere":
            prompt, chat_history = cohere_message_pt(messages=messages)
        else:
            prompt = ""
            for message in messages:
                if "role" in message:
                    if message["role"] == "user":
                        prompt += f"{message['content']}"
                    else:
                        prompt += f"{message['content']}"
                else:
                    prompt += f"{message['content']}"
        return prompt, chat_history  # type: ignore

    def get_credentials(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        aws_session_name: Optional[str] = None,
        aws_profile_name: Optional[str] = None,
        aws_role_name: Optional[str] = None,
    ):
        """
        Return a boto3.Credentials object
        """
        import boto3

        ## CHECK IS  'os.environ/' passed in
        params_to_check: List[Optional[str]] = [
            aws_access_key_id,
            aws_secret_access_key,
            aws_region_name,
            aws_session_name,
            aws_profile_name,
            aws_role_name,
        ]

        # Iterate over parameters and update if needed
        for i, param in enumerate(params_to_check):
            if param and param.startswith("os.environ/"):
                _v = get_secret(param)
                if _v is not None and isinstance(_v, str):
                    params_to_check[i] = _v
        # Assign updated values back to parameters
        (
            aws_access_key_id,
            aws_secret_access_key,
            aws_region_name,
            aws_session_name,
            aws_profile_name,
            aws_role_name,
        ) = params_to_check

        ### CHECK STS ###
        if aws_role_name is not None and aws_session_name is not None:
            sts_client = boto3.client(
                "sts",
                aws_access_key_id=aws_access_key_id,  # [OPTIONAL]
                aws_secret_access_key=aws_secret_access_key,  # [OPTIONAL]
            )

            sts_response = sts_client.assume_role(
                RoleArn=aws_role_name, RoleSessionName=aws_session_name
            )

            return sts_response["Credentials"]
        elif aws_profile_name is not None:  ### CHECK SESSION ###
            # uses auth values from AWS profile usually stored in ~/.aws/credentials
            client = boto3.Session(profile_name=aws_profile_name)

            return client.get_credentials()
        else:
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_region_name,
            )

            return session.get_credentials()

    def completion(
        self,
        model: str,
        messages: list,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        logging_obj,
        optional_params: dict,
        timeout: Optional[Union[float, httpx.Timeout]],
        litellm_params=None,
        logger_fn=None,
        extra_headers: Optional[dict] = None,
        client: Optional[HTTPHandler] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        try:
            import boto3

            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            from botocore.credentials import Credentials
        except ImportError as e:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        ## CREDENTIALS ##
        # pop aws_secret_access_key, aws_access_key_id, aws_region_name from kwargs, since completion calls fail with them
        aws_secret_access_key = optional_params.pop("aws_secret_access_key", None)
        aws_access_key_id = optional_params.pop("aws_access_key_id", None)
        aws_region_name = optional_params.pop("aws_region_name", None)
        aws_role_name = optional_params.pop("aws_role_name", None)
        aws_session_name = optional_params.pop("aws_session_name", None)
        aws_profile_name = optional_params.pop("aws_profile_name", None)
        aws_bedrock_runtime_endpoint = optional_params.pop(
            "aws_bedrock_runtime_endpoint", None
        )  # https://bedrock-runtime.{region_name}.amazonaws.com

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
            aws_region_name=aws_region_name,
            aws_session_name=aws_session_name,
            aws_profile_name=aws_profile_name,
            aws_role_name=aws_role_name,
        )

        ### SET RUNTIME ENDPOINT ###
        endpoint_url = ""
        env_aws_bedrock_runtime_endpoint = get_secret("AWS_BEDROCK_RUNTIME_ENDPOINT")
        if aws_bedrock_runtime_endpoint is not None and isinstance(
            aws_bedrock_runtime_endpoint, str
        ):
            endpoint_url = aws_bedrock_runtime_endpoint
        elif env_aws_bedrock_runtime_endpoint and isinstance(
            env_aws_bedrock_runtime_endpoint, str
        ):
            endpoint_url = env_aws_bedrock_runtime_endpoint
        else:
            endpoint_url = f"https://bedrock-runtime.{aws_region_name}.amazonaws.com"

        endpoint_url = f"{endpoint_url}/model/{model}/invoke"

        sigv4 = SigV4Auth(credentials, "bedrock", aws_region_name)

        provider = model.split(".")[0]
        prompt, chat_history = self.convert_messages_to_prompt(
            model, messages, provider, custom_prompt_dict
        )
        inference_params = copy.deepcopy(optional_params)
        stream = inference_params.pop("stream", False)

        if provider == "cohere":
            if model.startswith("cohere.command-r"):
                ## LOAD CONFIG
                config = litellm.AmazonCohereChatConfig().get_config()
                for k, v in config.items():
                    if (
                        k not in inference_params
                    ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                        inference_params[k] = v
                if optional_params.get("stream", False) == True:
                    inference_params["stream"] = (
                        True  # cohere requires stream = True in inference params
                    )

                _data = {"message": prompt, **inference_params}
                if chat_history is not None:
                    _data["chat_history"] = chat_history
                data = json.dumps(_data)
            else:
                ## LOAD CONFIG
                config = litellm.AmazonCohereConfig.get_config()
                for k, v in config.items():
                    if (
                        k not in inference_params
                    ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                        inference_params[k] = v
                if optional_params.get("stream", False) == True:
                    inference_params["stream"] = (
                        True  # cohere requires stream = True in inference params
                    )
                data = json.dumps({"prompt": prompt, **inference_params})
        else:
            raise Exception("UNSUPPORTED PROVIDER")

        ## COMPLETION CALL
        headers = {"Content-Type": "application/json"}
        request = AWSRequest(
            method="POST", url=endpoint_url, data=data, headers=headers
        )
        sigv4.add_auth(request)
        prepped = request.prepare()

        if client is None:
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            self.client = HTTPHandler(**_params)  # type: ignore
        else:
            self.client = client

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": prepped.url,
                "headers": prepped.headers,
            },
        )

        response = self.client.post(url=prepped.url, headers=prepped.headers, data=data)  # type: ignore

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise BedrockError(status_code=error_code, message=response.text)

        return response

    def embedding(self, *args, **kwargs):
        return super().embedding(*args, **kwargs)

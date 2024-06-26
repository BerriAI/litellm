# What is this?
## Initial implementation of calling bedrock via httpx client (allows for async calls).
## V1 - covers cohere + anthropic claude-3 support
import copy
import json
import os
import time
import types
import urllib.parse
import uuid
from enum import Enum
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import httpx  # type: ignore
import requests  # type: ignore

import litellm
from litellm.caching import DualCache
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_async_httpx_client,
    _get_httpx_client,
)
from litellm.types.llms.bedrock import *
from litellm.types.llms.openai import (
    ChatCompletionDeltaChunk,
    ChatCompletionResponseMessage,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
)
from litellm.types.utils import Choices, Message
from litellm.utils import CustomStreamWrapper, ModelResponse, Usage, get_secret

from .base import BaseLLM
from .bedrock import BedrockError, ModelResponseIterator, convert_messages_to_prompt
from .prompt_templates.factory import (
    _bedrock_converse_messages_pt,
    _bedrock_tools_pt,
    cohere_message_pt,
    construct_tool_use_system_prompt,
    contains_tag,
    custom_prompt,
    extract_between_tags,
    parse_xml_params,
    prompt_factory,
)

iam_cache = DualCache()


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
            "tools",
            "tool_choice",
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


async def make_call(
    client: Optional[AsyncHTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
):
    try:
        if client is None:
            client = _get_async_httpx_client()  # Create a new client if none provided

        response = await client.post(api_base, headers=headers, data=data, stream=True)

        if response.status_code != 200:
            raise BedrockError(status_code=response.status_code, message=response.text)

        decoder = AWSEventStreamDecoder(model=model)
        completion_stream = decoder.aiter_bytes(response.aiter_bytes(chunk_size=1024))

        # LOGGING
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response="first stream response received",
            additional_args={"complete_input_dict": data},
        )

        return completion_stream
    except httpx.HTTPStatusError as err:
        error_code = err.response.status_code
        raise BedrockError(status_code=error_code, message=str(err))
    except httpx.TimeoutException as e:
        raise BedrockError(status_code=408, message="Timeout error occurred.")
    except Exception as e:
        raise BedrockError(status_code=500, message=str(e))


def make_sync_call(
    client: Optional[HTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
):
    if client is None:
        client = _get_httpx_client()  # Create a new client if none provided

    response = client.post(api_base, headers=headers, data=data, stream=True)

    if response.status_code != 200:
        raise BedrockError(status_code=response.status_code, message=response.read())

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
        ## CUSTOM PROMPT
        if model in custom_prompt_dict:
            # check if the model has a registered custom prompt
            model_prompt_details = custom_prompt_dict[model]
            prompt = custom_prompt(
                role_dict=model_prompt_details["roles"],
                initial_prompt_value=model_prompt_details.get(
                    "initial_prompt_value", ""
                ),
                final_prompt_value=model_prompt_details.get("final_prompt_value", ""),
                messages=messages,
            )
            return prompt, None
        ## ELSE
        if provider == "anthropic" or provider == "amazon":
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
        aws_web_identity_token: Optional[str] = None,
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
            aws_web_identity_token,
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
            aws_web_identity_token,
        ) = params_to_check

        ### CHECK STS ###
        if (
            aws_web_identity_token is not None
            and aws_role_name is not None
            and aws_session_name is not None
        ):
            iam_creds_cache_key = json.dumps(
                {
                    "aws_web_identity_token": aws_web_identity_token,
                    "aws_role_name": aws_role_name,
                    "aws_session_name": aws_session_name,
                    "aws_region_name": aws_region_name,
                }
            )

            iam_creds_dict = iam_cache.get_cache(iam_creds_cache_key)
            if iam_creds_dict is None:
                oidc_token = get_secret(aws_web_identity_token)

                if oidc_token is None:
                    raise BedrockError(
                        message="OIDC token could not be retrieved from secret manager.",
                        status_code=401,
                    )

                sts_client = boto3.client(
                    "sts",
                    region_name=aws_region_name,
                    endpoint_url=f"https://sts.{aws_region_name}.amazonaws.com",
                )

                # https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRoleWithWebIdentity.html
                # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sts/client/assume_role_with_web_identity.html
                sts_response = sts_client.assume_role_with_web_identity(
                    RoleArn=aws_role_name,
                    RoleSessionName=aws_session_name,
                    WebIdentityToken=oidc_token,
                    DurationSeconds=3600,
                )

                iam_creds_dict = {
                    "aws_access_key_id": sts_response["Credentials"]["AccessKeyId"],
                    "aws_secret_access_key": sts_response["Credentials"][
                        "SecretAccessKey"
                    ],
                    "aws_session_token": sts_response["Credentials"]["SessionToken"],
                    "region_name": aws_region_name,
                }

                iam_cache.set_cache(
                    key=iam_creds_cache_key,
                    value=json.dumps(iam_creds_dict),
                    ttl=3600 - 60,
                )

            session = boto3.Session(**iam_creds_dict)

            iam_creds = session.get_credentials()

            return iam_creds
        elif aws_role_name is not None and aws_session_name is not None:
            sts_client = boto3.client(
                "sts",
                aws_access_key_id=aws_access_key_id,  # [OPTIONAL]
                aws_secret_access_key=aws_secret_access_key,  # [OPTIONAL]
            )

            sts_response = sts_client.assume_role(
                RoleArn=aws_role_name, RoleSessionName=aws_session_name
            )

            # Extract the credentials from the response and convert to Session Credentials
            sts_credentials = sts_response["Credentials"]
            from botocore.credentials import Credentials

            credentials = Credentials(
                access_key=sts_credentials["AccessKeyId"],
                secret_key=sts_credentials["SecretAccessKey"],
                token=sts_credentials["SessionToken"],
            )
            return credentials
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

    def process_response(
        self,
        model: str,
        response: Union[requests.Response, httpx.Response],
        model_response: ModelResponse,
        stream: bool,
        logging_obj: Logging,
        optional_params: dict,
        api_key: str,
        data: Union[dict, str],
        messages: List,
        print_verbose,
        encoding,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        provider = model.split(".")[0]
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )
        print_verbose(f"raw model_response: {response.text}")

        ## RESPONSE OBJECT
        try:
            completion_response = response.json()
        except:
            raise BedrockError(message=response.text, status_code=422)

        try:
            if provider == "cohere":
                if "text" in completion_response:
                    outputText = completion_response["text"]  # type: ignore
                elif "generations" in completion_response:
                    outputText = completion_response["generations"][0]["text"]
                    model_response["finish_reason"] = map_finish_reason(
                        completion_response["generations"][0]["finish_reason"]
                    )
            elif provider == "anthropic":
                if model.startswith("anthropic.claude-3"):
                    json_schemas: dict = {}
                    _is_function_call = False
                    ## Handle Tool Calling
                    if "tools" in optional_params:
                        _is_function_call = True
                        for tool in optional_params["tools"]:
                            json_schemas[tool["function"]["name"]] = tool[
                                "function"
                            ].get("parameters", None)
                    outputText = completion_response.get("content")[0].get("text", None)
                    if outputText is not None and contains_tag(
                        "invoke", outputText
                    ):  # OUTPUT PARSE FUNCTION CALL
                        function_name = extract_between_tags("tool_name", outputText)[0]
                        function_arguments_str = extract_between_tags(
                            "invoke", outputText
                        )[0].strip()
                        function_arguments_str = (
                            f"<invoke>{function_arguments_str}</invoke>"
                        )
                        function_arguments = parse_xml_params(
                            function_arguments_str,
                            json_schema=json_schemas.get(
                                function_name, None
                            ),  # check if we have a json schema for this function name)
                        )
                        _message = litellm.Message(
                            tool_calls=[
                                {
                                    "id": f"call_{uuid.uuid4()}",
                                    "type": "function",
                                    "function": {
                                        "name": function_name,
                                        "arguments": json.dumps(function_arguments),
                                    },
                                }
                            ],
                            content=None,
                        )
                        model_response.choices[0].message = _message  # type: ignore
                        model_response._hidden_params["original_response"] = (
                            outputText  # allow user to access raw anthropic tool calling response
                        )
                    if (
                        _is_function_call == True
                        and stream is not None
                        and stream == True
                    ):
                        print_verbose(
                            f"INSIDE BEDROCK STREAMING TOOL CALLING CONDITION BLOCK"
                        )
                        # return an iterator
                        streaming_model_response = ModelResponse(stream=True)
                        streaming_model_response.choices[0].finish_reason = getattr(
                            model_response.choices[0], "finish_reason", "stop"
                        )
                        # streaming_model_response.choices = [litellm.utils.StreamingChoices()]
                        streaming_choice = litellm.utils.StreamingChoices()
                        streaming_choice.index = model_response.choices[0].index
                        _tool_calls = []
                        print_verbose(
                            f"type of model_response.choices[0]: {type(model_response.choices[0])}"
                        )
                        print_verbose(
                            f"type of streaming_choice: {type(streaming_choice)}"
                        )
                        if isinstance(model_response.choices[0], litellm.Choices):
                            if getattr(
                                model_response.choices[0].message, "tool_calls", None
                            ) is not None and isinstance(
                                model_response.choices[0].message.tool_calls, list
                            ):
                                for tool_call in model_response.choices[
                                    0
                                ].message.tool_calls:
                                    _tool_call = {**tool_call.dict(), "index": 0}
                                    _tool_calls.append(_tool_call)
                            delta_obj = litellm.utils.Delta(
                                content=getattr(
                                    model_response.choices[0].message, "content", None
                                ),
                                role=model_response.choices[0].message.role,
                                tool_calls=_tool_calls,
                            )
                            streaming_choice.delta = delta_obj
                            streaming_model_response.choices = [streaming_choice]
                            completion_stream = ModelResponseIterator(
                                model_response=streaming_model_response
                            )
                            print_verbose(
                                f"Returns anthropic CustomStreamWrapper with 'cached_response' streaming object"
                            )
                            return litellm.CustomStreamWrapper(
                                completion_stream=completion_stream,
                                model=model,
                                custom_llm_provider="cached_response",
                                logging_obj=logging_obj,
                            )

                    model_response["finish_reason"] = map_finish_reason(
                        completion_response.get("stop_reason", "")
                    )
                    _usage = litellm.Usage(
                        prompt_tokens=completion_response["usage"]["input_tokens"],
                        completion_tokens=completion_response["usage"]["output_tokens"],
                        total_tokens=completion_response["usage"]["input_tokens"]
                        + completion_response["usage"]["output_tokens"],
                    )
                    setattr(model_response, "usage", _usage)
                else:
                    outputText = completion_response["completion"]

                    model_response["finish_reason"] = completion_response["stop_reason"]
            elif provider == "ai21":
                outputText = (
                    completion_response.get("completions")[0].get("data").get("text")
                )
            elif provider == "meta":
                outputText = completion_response["generation"]
            elif provider == "mistral":
                outputText = completion_response["outputs"][0]["text"]
                model_response["finish_reason"] = completion_response["outputs"][0][
                    "stop_reason"
                ]
            else:  # amazon titan
                outputText = completion_response.get("results")[0].get("outputText")
        except Exception as e:
            raise BedrockError(
                message="Error processing={}, Received error={}".format(
                    response.text, str(e)
                ),
                status_code=422,
            )

        try:
            if (
                len(outputText) > 0
                and hasattr(model_response.choices[0], "message")
                and getattr(model_response.choices[0].message, "tool_calls", None)
                is None
            ):
                model_response["choices"][0]["message"]["content"] = outputText
            elif (
                hasattr(model_response.choices[0], "message")
                and getattr(model_response.choices[0].message, "tool_calls", None)
                is not None
            ):
                pass
            else:
                raise Exception()
        except:
            raise BedrockError(
                message=json.dumps(outputText), status_code=response.status_code
            )

        if stream and provider == "ai21":
            streaming_model_response = ModelResponse(stream=True)
            streaming_model_response.choices[0].finish_reason = model_response.choices[  # type: ignore
                0
            ].finish_reason
            # streaming_model_response.choices = [litellm.utils.StreamingChoices()]
            streaming_choice = litellm.utils.StreamingChoices()
            streaming_choice.index = model_response.choices[0].index
            delta_obj = litellm.utils.Delta(
                content=getattr(model_response.choices[0].message, "content", None),
                role=model_response.choices[0].message.role,
            )
            streaming_choice.delta = delta_obj
            streaming_model_response.choices = [streaming_choice]
            mri = ModelResponseIterator(model_response=streaming_model_response)
            return CustomStreamWrapper(
                completion_stream=mri,
                model=model,
                custom_llm_provider="cached_response",
                logging_obj=logging_obj,
            )

        ## CALCULATING USAGE - bedrock returns usage in the headers
        bedrock_input_tokens = response.headers.get(
            "x-amzn-bedrock-input-token-count", None
        )
        bedrock_output_tokens = response.headers.get(
            "x-amzn-bedrock-output-token-count", None
        )

        prompt_tokens = int(
            bedrock_input_tokens or litellm.token_counter(messages=messages)
        )

        completion_tokens = int(
            bedrock_output_tokens
            or litellm.token_counter(
                text=model_response.choices[0].message.content,  # type: ignore
                count_response_tokens=True,
            )
        )

        model_response["created"] = int(time.time())
        model_response["model"] = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)

        return model_response

    def encode_model_id(self, model_id: str) -> str:
        """
        Double encode the model ID to ensure it matches the expected double-encoded format.
        Args:
            model_id (str): The model ID to encode.
        Returns:
            str: The double-encoded model ID.
        """
        return urllib.parse.quote(model_id, safe="")

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
        acompletion: bool,
        timeout: Optional[Union[float, httpx.Timeout]],
        litellm_params=None,
        logger_fn=None,
        extra_headers: Optional[dict] = None,
        client: Optional[Union[AsyncHTTPHandler, HTTPHandler]] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        try:
            import boto3
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            from botocore.credentials import Credentials
        except ImportError as e:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        ## SETUP ##
        stream = optional_params.pop("stream", None)
        modelId = optional_params.pop("model_id", None)
        if modelId is not None:
            modelId = self.encode_model_id(model_id=modelId)
        else:
            modelId = model

        provider = model.split(".")[0]

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
        aws_web_identity_token = optional_params.pop("aws_web_identity_token", None)

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
            aws_web_identity_token=aws_web_identity_token,
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

        if (stream is not None and stream == True) and provider != "ai21":
            endpoint_url = f"{endpoint_url}/model/{modelId}/invoke-with-response-stream"
        else:
            endpoint_url = f"{endpoint_url}/model/{modelId}/invoke"

        sigv4 = SigV4Auth(credentials, "bedrock", aws_region_name)

        prompt, chat_history = self.convert_messages_to_prompt(
            model, messages, provider, custom_prompt_dict
        )
        inference_params = copy.deepcopy(optional_params)
        json_schemas: dict = {}
        if provider == "cohere":
            if model.startswith("cohere.command-r"):
                ## LOAD CONFIG
                config = litellm.AmazonCohereChatConfig().get_config()
                for k, v in config.items():
                    if (
                        k not in inference_params
                    ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                        inference_params[k] = v
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
                if stream == True:
                    inference_params["stream"] = (
                        True  # cohere requires stream = True in inference params
                    )
                data = json.dumps({"prompt": prompt, **inference_params})
        elif provider == "anthropic":
            if model.startswith("anthropic.claude-3"):
                # Separate system prompt from rest of message
                system_prompt_idx: list[int] = []
                system_messages: list[str] = []
                for idx, message in enumerate(messages):
                    if message["role"] == "system":
                        system_messages.append(message["content"])
                        system_prompt_idx.append(idx)
                if len(system_prompt_idx) > 0:
                    inference_params["system"] = "\n".join(system_messages)
                    messages = [
                        i for j, i in enumerate(messages) if j not in system_prompt_idx
                    ]
                # Format rest of message according to anthropic guidelines
                messages = prompt_factory(
                    model=model, messages=messages, custom_llm_provider="anthropic_xml"
                )  # type: ignore
                ## LOAD CONFIG
                config = litellm.AmazonAnthropicClaude3Config.get_config()
                for k, v in config.items():
                    if (
                        k not in inference_params
                    ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                        inference_params[k] = v
                ## Handle Tool Calling
                if "tools" in inference_params:
                    _is_function_call = True
                    for tool in inference_params["tools"]:
                        json_schemas[tool["function"]["name"]] = tool["function"].get(
                            "parameters", None
                        )
                    tool_calling_system_prompt = construct_tool_use_system_prompt(
                        tools=inference_params["tools"]
                    )
                    inference_params["system"] = (
                        inference_params.get("system", "\n")
                        + tool_calling_system_prompt
                    )  # add the anthropic tool calling prompt to the system prompt
                    inference_params.pop("tools")
                data = json.dumps({"messages": messages, **inference_params})
            else:
                ## LOAD CONFIG
                config = litellm.AmazonAnthropicConfig.get_config()
                for k, v in config.items():
                    if (
                        k not in inference_params
                    ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                        inference_params[k] = v
                data = json.dumps({"prompt": prompt, **inference_params})
        elif provider == "ai21":
            ## LOAD CONFIG
            config = litellm.AmazonAI21Config.get_config()
            for k, v in config.items():
                if (
                    k not in inference_params
                ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                    inference_params[k] = v

            data = json.dumps({"prompt": prompt, **inference_params})
        elif provider == "mistral":
            ## LOAD CONFIG
            config = litellm.AmazonMistralConfig.get_config()
            for k, v in config.items():
                if (
                    k not in inference_params
                ):  # completion(top_k=3) > amazon_config(top_k=3) <- allows for dynamic variables to be passed in
                    inference_params[k] = v

            data = json.dumps({"prompt": prompt, **inference_params})
        elif provider == "amazon":  # amazon titan
            ## LOAD CONFIG
            config = litellm.AmazonTitanConfig.get_config()
            for k, v in config.items():
                if (
                    k not in inference_params
                ):  # completion(top_k=3) > amazon_config(top_k=3) <- allows for dynamic variables to be passed in
                    inference_params[k] = v

            data = json.dumps(
                {
                    "inputText": prompt,
                    "textGenerationConfig": inference_params,
                }
            )
        elif provider == "meta":
            ## LOAD CONFIG
            config = litellm.AmazonLlamaConfig.get_config()
            for k, v in config.items():
                if (
                    k not in inference_params
                ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                    inference_params[k] = v
            data = json.dumps({"prompt": prompt, **inference_params})
        else:
            ## LOGGING
            logging_obj.pre_call(
                input=messages,
                api_key="",
                additional_args={
                    "complete_input_dict": inference_params,
                },
            )
            raise Exception(
                "Bedrock HTTPX: Unsupported provider={}, model={}".format(
                    provider, model
                )
            )

        ## COMPLETION CALL

        headers = {"Content-Type": "application/json"}
        if extra_headers is not None:
            headers = {"Content-Type": "application/json", **extra_headers}
        request = AWSRequest(
            method="POST", url=endpoint_url, data=data, headers=headers
        )
        sigv4.add_auth(request)
        prepped = request.prepare()

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

        ### ROUTING (ASYNC, STREAMING, SYNC)
        if acompletion:
            if isinstance(client, HTTPHandler):
                client = None
            if stream == True and provider != "ai21":
                return self.async_streaming(
                    model=model,
                    messages=messages,
                    data=data,
                    api_base=prepped.url,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    encoding=encoding,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    stream=True,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    headers=prepped.headers,
                    timeout=timeout,
                    client=client,
                )  # type: ignore
            ### ASYNC COMPLETION
            return self.async_completion(
                model=model,
                messages=messages,
                data=data,
                api_base=prepped.url,
                model_response=model_response,
                print_verbose=print_verbose,
                encoding=encoding,
                logging_obj=logging_obj,
                optional_params=optional_params,
                stream=stream,  # type: ignore
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                headers=prepped.headers,
                timeout=timeout,
                client=client,
            )  # type: ignore

        if client is None or isinstance(client, AsyncHTTPHandler):
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            self.client = _get_httpx_client(_params)  # type: ignore
        else:
            self.client = client
        if (stream is not None and stream == True) and provider != "ai21":
            response = self.client.post(
                url=prepped.url,
                headers=prepped.headers,  # type: ignore
                data=data,
                stream=stream,
            )

            if response.status_code != 200:
                raise BedrockError(
                    status_code=response.status_code, message=response.text
                )

            decoder = AWSEventStreamDecoder(model=model)

            completion_stream = decoder.iter_bytes(response.iter_bytes(chunk_size=1024))
            streaming_response = CustomStreamWrapper(
                completion_stream=completion_stream,
                model=model,
                custom_llm_provider="bedrock",
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

        try:
            response = self.client.post(url=prepped.url, headers=prepped.headers, data=data)  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise BedrockError(status_code=error_code, message=response.text)
        except httpx.TimeoutException as e:
            raise BedrockError(status_code=408, message="Timeout error occurred.")

        return self.process_response(
            model=model,
            response=response,
            model_response=model_response,
            stream=stream,
            logging_obj=logging_obj,
            optional_params=optional_params,
            api_key="",
            data=data,
            messages=messages,
            print_verbose=print_verbose,
            encoding=encoding,
        )

    async def async_completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        model_response: ModelResponse,
        print_verbose: Callable,
        data: str,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        stream,
        optional_params: dict,
        litellm_params=None,
        logger_fn=None,
        headers={},
        client: Optional[AsyncHTTPHandler] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        if client is None:
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            client = _get_async_httpx_client(_params)  # type: ignore
        else:
            client = client  # type: ignore

        try:
            response = await client.post(api_base, headers=headers, data=data)  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise BedrockError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException as e:
            raise BedrockError(status_code=408, message="Timeout error occurred.")

        return self.process_response(
            model=model,
            response=response,
            model_response=model_response,
            stream=stream if isinstance(stream, bool) else False,
            logging_obj=logging_obj,
            api_key="",
            data=data,
            messages=messages,
            print_verbose=print_verbose,
            optional_params=optional_params,
            encoding=encoding,
        )

    async def async_streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        model_response: ModelResponse,
        print_verbose: Callable,
        data: str,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        stream,
        optional_params: dict,
        litellm_params=None,
        logger_fn=None,
        headers={},
        client: Optional[AsyncHTTPHandler] = None,
    ) -> CustomStreamWrapper:
        # The call is not made here; instead, we prepare the necessary objects for the stream.

        streaming_response = CustomStreamWrapper(
            completion_stream=None,
            make_call=partial(
                make_call,
                client=client,
                api_base=api_base,
                headers=headers,
                data=data,
                model=model,
                messages=messages,
                logging_obj=logging_obj,
            ),
            model=model,
            custom_llm_provider="bedrock",
            logging_obj=logging_obj,
        )
        return streaming_response

    def embedding(self, *args, **kwargs):
        return super().embedding(*args, **kwargs)


class AmazonConverseConfig:
    """
    Reference - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
    #2 - https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#conversation-inference-supported-models-features
    """

    maxTokens: Optional[int]
    stopSequences: Optional[List[str]]
    temperature: Optional[int]
    topP: Optional[int]

    def __init__(
        self,
        maxTokens: Optional[int] = None,
        stopSequences: Optional[List[str]] = None,
        temperature: Optional[int] = None,
        topP: Optional[int] = None,
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

    def get_supported_openai_params(self, model: str) -> List[str]:
        supported_params = [
            "max_tokens",
            "stream",
            "stream_options",
            "stop",
            "temperature",
            "top_p",
            "extra_headers",
        ]

        if (
            model.startswith("anthropic")
            or model.startswith("mistral")
            or model.startswith("cohere")
        ):
            supported_params.append("tools")

        if model.startswith("anthropic") or model.startswith("mistral"):
            # only anthropic and mistral support tool choice config. otherwise (E.g. cohere) will fail the call - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
            supported_params.append("tool_choice")

        return supported_params

    def map_tool_choice_values(
        self, model: str, tool_choice: Union[str, dict], drop_params: bool
    ) -> Optional[ToolChoiceValuesBlock]:
        if tool_choice == "none":
            if litellm.drop_params is True or drop_params is True:
                return None
            else:
                raise litellm.utils.UnsupportedParamsError(
                    message="Bedrock doesn't support tool_choice={}. To drop it from the call, set `litellm.drop_params = True.".format(
                        tool_choice
                    ),
                    status_code=400,
                )
        elif tool_choice == "required":
            return ToolChoiceValuesBlock(any={})
        elif tool_choice == "auto":
            return ToolChoiceValuesBlock(auto={})
        elif isinstance(tool_choice, dict):
            # only supported for anthropic + mistral models - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
            specific_tool = SpecificToolChoiceBlock(
                name=tool_choice.get("function", {}).get("name", "")
            )
            return ToolChoiceValuesBlock(tool=specific_tool)
        else:
            raise litellm.utils.UnsupportedParamsError(
                message="Bedrock doesn't support tool_choice={}. Supported tool_choice values=['auto', 'required', json object]. To drop it from the call, set `litellm.drop_params = True.".format(
                    tool_choice
                ),
                status_code=400,
            )

    def get_supported_image_types(self) -> List[str]:
        return ["png", "jpeg", "gif", "webp"]

    def map_openai_params(
        self,
        model: str,
        non_default_params: dict,
        optional_params: dict,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["maxTokens"] = value
            if param == "stream":
                optional_params["stream"] = value
            if param == "stop":
                if isinstance(value, str):
                    value = [value]
                optional_params["stop_sequences"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["topP"] = value
            if param == "tools":
                optional_params["tools"] = value
            if param == "tool_choice":
                _tool_choice_value = self.map_tool_choice_values(
                    model=model, tool_choice=value, drop_params=drop_params  # type: ignore
                )
                if _tool_choice_value is not None:
                    optional_params["tool_choice"] = _tool_choice_value
        return optional_params


class BedrockConverseLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    def process_response(
        self,
        model: str,
        response: Union[requests.Response, httpx.Response],
        model_response: ModelResponse,
        stream: bool,
        logging_obj: Logging,
        optional_params: dict,
        api_key: str,
        data: Union[dict, str],
        messages: List,
        print_verbose,
        encoding,
    ) -> Union[ModelResponse, CustomStreamWrapper]:

        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )
        print_verbose(f"raw model_response: {response.text}")

        ## RESPONSE OBJECT
        try:
            completion_response = ConverseResponseBlock(**response.json())  # type: ignore
        except Exception as e:
            raise BedrockError(
                message="Received={}, Error converting to valid response block={}. File an issue if litellm error - https://github.com/BerriAI/litellm/issues".format(
                    response.text, str(e)
                ),
                status_code=422,
            )

        """
        Bedrock Response Object has optional message block 

        completion_response["output"].get("message", None)

        A message block looks like this (Example 1): 
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "text": "Is there anything else you'd like to talk about? Perhaps I can help with some economic questions or provide some information about economic concepts?"
                    }
                ]
            }
        },
        (Example 2):
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_hbTgdi0CSLq_hM4P8csZJA",
                            "name": "top_song",
                            "input": {
                                "sign": "WZPZ"
                            }
                        }
                    }
                ]
            }
        }

        """
        message: Optional[MessageBlock] = completion_response["output"]["message"]
        chat_completion_message: ChatCompletionResponseMessage = {"role": "assistant"}
        content_str = ""
        tools: List[ChatCompletionToolCallChunk] = []
        if message is not None:
            for content in message["content"]:
                """
                - Content is either a tool response or text
                """
                if "text" in content:
                    content_str += content["text"]
                if "toolUse" in content:
                    _function_chunk = ChatCompletionToolCallFunctionChunk(
                        name=content["toolUse"]["name"],
                        arguments=json.dumps(content["toolUse"]["input"]),
                    )
                    _tool_response_chunk = ChatCompletionToolCallChunk(
                        id=content["toolUse"]["toolUseId"],
                        type="function",
                        function=_function_chunk,
                    )
                    tools.append(_tool_response_chunk)
        chat_completion_message["content"] = content_str
        chat_completion_message["tool_calls"] = tools

        ## CALCULATING USAGE - bedrock returns usage in the headers
        input_tokens = completion_response["usage"]["inputTokens"]
        output_tokens = completion_response["usage"]["outputTokens"]
        total_tokens = completion_response["usage"]["totalTokens"]

        model_response.choices = [
            litellm.Choices(
                finish_reason=map_finish_reason(completion_response["stopReason"]),
                index=0,
                message=litellm.Message(**chat_completion_message),
            )
        ]
        model_response["created"] = int(time.time())
        model_response["model"] = model
        usage = Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
        )
        setattr(model_response, "usage", usage)

        return model_response

    def encode_model_id(self, model_id: str) -> str:
        """
        Double encode the model ID to ensure it matches the expected double-encoded format.
        Args:
            model_id (str): The model ID to encode.
        Returns:
            str: The double-encoded model ID.
        """
        return urllib.parse.quote(model_id, safe="")

    def get_credentials(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        aws_session_name: Optional[str] = None,
        aws_profile_name: Optional[str] = None,
        aws_role_name: Optional[str] = None,
        aws_web_identity_token: Optional[str] = None,
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
            aws_web_identity_token,
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
            aws_web_identity_token,
        ) = params_to_check

        ### CHECK STS ###
        if (
            aws_web_identity_token is not None
            and aws_role_name is not None
            and aws_session_name is not None
        ):
            iam_creds_cache_key = json.dumps(
                {
                    "aws_web_identity_token": aws_web_identity_token,
                    "aws_role_name": aws_role_name,
                    "aws_session_name": aws_session_name,
                    "aws_region_name": aws_region_name,
                }
            )

            iam_creds_dict = iam_cache.get_cache(iam_creds_cache_key)
            if iam_creds_dict is None:
                oidc_token = get_secret(aws_web_identity_token)

                if oidc_token is None:
                    raise BedrockError(
                        message="OIDC token could not be retrieved from secret manager.",
                        status_code=401,
                    )

                sts_client = boto3.client(
                    "sts",
                    region_name=aws_region_name,
                    endpoint_url=f"https://sts.{aws_region_name}.amazonaws.com",
                )

                # https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRoleWithWebIdentity.html
                # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sts/client/assume_role_with_web_identity.html
                sts_response = sts_client.assume_role_with_web_identity(
                    RoleArn=aws_role_name,
                    RoleSessionName=aws_session_name,
                    WebIdentityToken=oidc_token,
                    DurationSeconds=3600,
                )

                iam_creds_dict = {
                    "aws_access_key_id": sts_response["Credentials"]["AccessKeyId"],
                    "aws_secret_access_key": sts_response["Credentials"][
                        "SecretAccessKey"
                    ],
                    "aws_session_token": sts_response["Credentials"]["SessionToken"],
                    "region_name": aws_region_name,
                }

                iam_cache.set_cache(
                    key=iam_creds_cache_key,
                    value=json.dumps(iam_creds_dict),
                    ttl=3600 - 60,
                )

            session = boto3.Session(**iam_creds_dict)

            iam_creds = session.get_credentials()

            return iam_creds
        elif aws_role_name is not None and aws_session_name is not None:
            sts_client = boto3.client(
                "sts",
                aws_access_key_id=aws_access_key_id,  # [OPTIONAL]
                aws_secret_access_key=aws_secret_access_key,  # [OPTIONAL]
            )

            sts_response = sts_client.assume_role(
                RoleArn=aws_role_name, RoleSessionName=aws_session_name
            )

            # Extract the credentials from the response and convert to Session Credentials
            sts_credentials = sts_response["Credentials"]
            from botocore.credentials import Credentials

            credentials = Credentials(
                access_key=sts_credentials["AccessKeyId"],
                secret_key=sts_credentials["SecretAccessKey"],
                token=sts_credentials["SessionToken"],
            )
            return credentials
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

    async def async_streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        model_response: ModelResponse,
        print_verbose: Callable,
        data: str,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        stream,
        optional_params: dict,
        litellm_params=None,
        logger_fn=None,
        headers={},
        client: Optional[AsyncHTTPHandler] = None,
    ) -> CustomStreamWrapper:
        streaming_response = CustomStreamWrapper(
            completion_stream=None,
            make_call=partial(
                make_call,
                client=client,
                api_base=api_base,
                headers=headers,
                data=data,
                model=model,
                messages=messages,
                logging_obj=logging_obj,
            ),
            model=model,
            custom_llm_provider="bedrock",
            logging_obj=logging_obj,
        )
        return streaming_response

    async def async_completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        model_response: ModelResponse,
        print_verbose: Callable,
        data: str,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        stream,
        optional_params: dict,
        litellm_params=None,
        logger_fn=None,
        headers={},
        client: Optional[AsyncHTTPHandler] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        if client is None:
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            client = _get_async_httpx_client(_params)  # type: ignore
        else:
            client = client  # type: ignore

        try:
            response = await client.post(api_base, headers=headers, data=data)  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise BedrockError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException as e:
            raise BedrockError(status_code=408, message="Timeout error occurred.")

        return self.process_response(
            model=model,
            response=response,
            model_response=model_response,
            stream=stream if isinstance(stream, bool) else False,
            logging_obj=logging_obj,
            api_key="",
            data=data,
            messages=messages,
            print_verbose=print_verbose,
            optional_params=optional_params,
            encoding=encoding,
        )

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
        acompletion: bool,
        timeout: Optional[Union[float, httpx.Timeout]],
        litellm_params=None,
        logger_fn=None,
        extra_headers: Optional[dict] = None,
        client: Optional[Union[AsyncHTTPHandler, HTTPHandler]] = None,
    ):
        try:
            import boto3
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            from botocore.credentials import Credentials
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        ## SETUP ##
        stream = optional_params.pop("stream", None)
        modelId = optional_params.pop("model_id", None)
        if modelId is not None:
            modelId = self.encode_model_id(model_id=modelId)
        else:
            modelId = model

        provider = model.split(".")[0]

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
        aws_web_identity_token = optional_params.pop("aws_web_identity_token", None)

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
            aws_web_identity_token=aws_web_identity_token,
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

        if (stream is not None and stream is True) and provider != "ai21":
            endpoint_url = f"{endpoint_url}/model/{modelId}/converse-stream"
        else:
            endpoint_url = f"{endpoint_url}/model/{modelId}/converse"

        sigv4 = SigV4Auth(credentials, "bedrock", aws_region_name)

        # Separate system prompt from rest of message
        system_prompt_indices = []
        system_content_blocks: List[SystemContentBlock] = []
        for idx, message in enumerate(messages):
            if message["role"] == "system":
                _system_content_block = SystemContentBlock(text=message["content"])
                system_content_blocks.append(_system_content_block)
                system_prompt_indices.append(idx)
        if len(system_prompt_indices) > 0:
            for idx in reversed(system_prompt_indices):
                messages.pop(idx)

        inference_params = copy.deepcopy(optional_params)
        additional_request_keys = []
        additional_request_params = {}
        supported_converse_params = AmazonConverseConfig.__annotations__.keys()
        supported_tool_call_params = ["tools", "tool_choice"]
        ## TRANSFORMATION ##
        # send all model-specific params in 'additional_request_params'
        for k, v in inference_params.items():
            if (
                k not in supported_converse_params
                and k not in supported_tool_call_params
            ):
                additional_request_params[k] = v
                additional_request_keys.append(k)
        for key in additional_request_keys:
            inference_params.pop(key, None)

        bedrock_messages: List[MessageBlock] = _bedrock_converse_messages_pt(
            messages=messages
        )
        bedrock_tools: List[ToolBlock] = _bedrock_tools_pt(
            inference_params.pop("tools", [])
        )
        bedrock_tool_config: Optional[ToolConfigBlock] = None
        if len(bedrock_tools) > 0:
            tool_choice_values: ToolChoiceValuesBlock = inference_params.pop(
                "tool_choice", None
            )
            bedrock_tool_config = ToolConfigBlock(
                tools=bedrock_tools,
            )
            if tool_choice_values is not None:
                bedrock_tool_config["toolChoice"] = tool_choice_values

        _data: RequestObject = {
            "messages": bedrock_messages,
            "additionalModelRequestFields": additional_request_params,
            "system": system_content_blocks,
            "inferenceConfig": InferenceConfig(**inference_params),
        }
        if bedrock_tool_config is not None:
            _data["toolConfig"] = bedrock_tool_config
        data = json.dumps(_data)
        ## COMPLETION CALL

        headers = {"Content-Type": "application/json"}
        if extra_headers is not None:
            headers = {"Content-Type": "application/json", **extra_headers}
        request = AWSRequest(
            method="POST", url=endpoint_url, data=data, headers=headers
        )
        sigv4.add_auth(request)
        prepped = request.prepare()

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

        ### ROUTING (ASYNC, STREAMING, SYNC)
        if acompletion:
            if isinstance(client, HTTPHandler):
                client = None
            if stream is True and provider != "ai21":
                return self.async_streaming(
                    model=model,
                    messages=messages,
                    data=data,
                    api_base=prepped.url,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    encoding=encoding,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    stream=True,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    headers=prepped.headers,
                    timeout=timeout,
                    client=client,
                )  # type: ignore
            ### ASYNC COMPLETION
            return self.async_completion(
                model=model,
                messages=messages,
                data=data,
                api_base=prepped.url,
                model_response=model_response,
                print_verbose=print_verbose,
                encoding=encoding,
                logging_obj=logging_obj,
                optional_params=optional_params,
                stream=stream,  # type: ignore
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                headers=prepped.headers,
                timeout=timeout,
                client=client,
            )  # type: ignore

        if (stream is not None and stream is True) and provider != "ai21":

            streaming_response = CustomStreamWrapper(
                completion_stream=None,
                make_call=partial(
                    make_sync_call,
                    client=None,
                    api_base=prepped.url,
                    headers=prepped.headers,  # type: ignore
                    data=data,
                    model=model,
                    messages=messages,
                    logging_obj=logging_obj,
                ),
                model=model,
                custom_llm_provider="bedrock",
                logging_obj=logging_obj,
            )

            return streaming_response
        ### COMPLETION

        if client is None or isinstance(client, AsyncHTTPHandler):
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            client = _get_httpx_client(_params)  # type: ignore
        else:
            client = client
        try:
            response = client.post(url=prepped.url, headers=prepped.headers, data=data)  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise BedrockError(status_code=error_code, message=response.text)
        except httpx.TimeoutException:
            raise BedrockError(status_code=408, message="Timeout error occurred.")

        return self.process_response(
            model=model,
            response=response,
            model_response=model_response,
            stream=stream,
            logging_obj=logging_obj,
            optional_params=optional_params,
            api_key="",
            data=data,
            messages=messages,
            print_verbose=print_verbose,
            encoding=encoding,
        )


def get_response_stream_shape():
    from botocore.loaders import Loader
    from botocore.model import ServiceModel

    loader = Loader()
    bedrock_service_dict = loader.load_service_model("bedrock-runtime", "service-2")
    bedrock_service_model = ServiceModel(bedrock_service_dict)
    return bedrock_service_model.shape_for("ResponseStream")


class AWSEventStreamDecoder:
    def __init__(self, model: str) -> None:
        from botocore.parsers import EventStreamJSONParser

        self.model = model
        self.parser = EventStreamJSONParser()

    def converse_chunk_parser(self, chunk_data: dict) -> GenericStreamingChunk:
        try:
            text = ""
            tool_use: Optional[ChatCompletionToolCallChunk] = None
            is_finished = False
            finish_reason = ""
            usage: Optional[ConverseTokenUsageBlock] = None

            index = int(chunk_data.get("contentBlockIndex", 0))
            if "start" in chunk_data:
                start_obj = ContentBlockStartEvent(**chunk_data["start"])
                if (
                    start_obj is not None
                    and "toolUse" in start_obj
                    and start_obj["toolUse"] is not None
                ):
                    tool_use = {
                        "id": start_obj["toolUse"]["toolUseId"],
                        "type": "function",
                        "function": {
                            "name": start_obj["toolUse"]["name"],
                            "arguments": "",
                        },
                    }
            elif "delta" in chunk_data:
                delta_obj = ContentBlockDeltaEvent(**chunk_data["delta"])
                if "text" in delta_obj:
                    text = delta_obj["text"]
                elif "toolUse" in delta_obj:
                    tool_use = {
                        "id": None,
                        "type": "function",
                        "function": {
                            "name": None,
                            "arguments": delta_obj["toolUse"]["input"],
                        },
                    }
            elif "stopReason" in chunk_data:
                finish_reason = map_finish_reason(chunk_data.get("stopReason", "stop"))
                is_finished = True
            elif "usage" in chunk_data:
                usage = ConverseTokenUsageBlock(**chunk_data["usage"])  # type: ignore

            response = GenericStreamingChunk(
                text=text,
                tool_use=tool_use,
                is_finished=is_finished,
                finish_reason=finish_reason,
                usage=usage,
                index=index,
            )
            return response
        except Exception as e:
            raise Exception("Received streaming error - {}".format(str(e)))

    def _chunk_parser(self, chunk_data: dict) -> GenericStreamingChunk:
        text = ""
        is_finished = False
        finish_reason = ""
        if "outputText" in chunk_data:
            text = chunk_data["outputText"]
        # ai21 mapping
        elif "ai21" in self.model:  # fake ai21 streaming
            text = chunk_data.get("completions")[0].get("data").get("text")  # type: ignore
            is_finished = True
            finish_reason = "stop"
        ######## bedrock.anthropic mappings ###############
        elif (
            "contentBlockIndex" in chunk_data
            or "stopReason" in chunk_data
            or "metrics" in chunk_data
        ):
            return self.converse_chunk_parser(chunk_data=chunk_data)
        ######## bedrock.mistral mappings ###############
        elif "outputs" in chunk_data:
            if (
                len(chunk_data["outputs"]) == 1
                and chunk_data["outputs"][0].get("text", None) is not None
            ):
                text = chunk_data["outputs"][0]["text"]
            stop_reason = chunk_data.get("stop_reason", None)
            if stop_reason is not None:
                is_finished = True
                finish_reason = stop_reason
        ######## bedrock.cohere mappings ###############
        # meta mapping
        elif "generation" in chunk_data:
            text = chunk_data["generation"]  # bedrock.meta
        # cohere mapping
        elif "text" in chunk_data:
            text = chunk_data["text"]  # bedrock.cohere
        # cohere mapping for finish reason
        elif "finish_reason" in chunk_data:
            finish_reason = chunk_data["finish_reason"]
            is_finished = True
        elif chunk_data.get("completionReason", None):
            is_finished = True
            finish_reason = chunk_data["completionReason"]
        return GenericStreamingChunk(
            text=text,
            is_finished=is_finished,
            finish_reason=finish_reason,
            usage=None,
            index=0,
            tool_use=None,
        )

    def iter_bytes(self, iterator: Iterator[bytes]) -> Iterator[GenericStreamingChunk]:
        """Given an iterator that yields lines, iterate over it & yield every event encountered"""
        from botocore.eventstream import EventStreamBuffer

        event_stream_buffer = EventStreamBuffer()
        for chunk in iterator:
            event_stream_buffer.add_data(chunk)
            for event in event_stream_buffer:
                message = self._parse_message_from_event(event)
                if message:
                    # sse_event = ServerSentEvent(data=message, event="completion")
                    _data = json.loads(message)
                    yield self._chunk_parser(chunk_data=_data)

    async def aiter_bytes(
        self, iterator: AsyncIterator[bytes]
    ) -> AsyncIterator[GenericStreamingChunk]:
        """Given an async iterator that yields lines, iterate over it & yield every event encountered"""
        from botocore.eventstream import EventStreamBuffer

        event_stream_buffer = EventStreamBuffer()
        async for chunk in iterator:
            event_stream_buffer.add_data(chunk)
            for event in event_stream_buffer:
                message = self._parse_message_from_event(event)
                if message:
                    _data = json.loads(message)
                    yield self._chunk_parser(chunk_data=_data)

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

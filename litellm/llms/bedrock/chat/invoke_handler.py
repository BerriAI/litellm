"""
TODO: DELETE FILE. Bedrock LLM is no longer used. Goto `litellm/llms/bedrock/chat/invoke_transformations/base_invoke_transformation.py`
"""

import copy
import time
import types
from functools import partial
from typing import (
    AsyncIterator,
    Callable,
    Iterator,
    Optional,
    Tuple,
    cast,
    get_args,
)

import httpx  # type: ignore

import litellm
from litellm import verbose_logger
from litellm._uuid import uuid
from litellm.caching.caching import InMemoryCache
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.logging_utils import track_llm_api_timing
from litellm.litellm_core_utils.prompt_templates.factory import (
    cohere_message_pt,
    construct_tool_use_system_prompt,
    contains_tag,
    custom_prompt,
    extract_between_tags,
    parse_xml_params,
    prompt_factory,
)
from litellm.llms.anthropic.chat.handler import (
    ModelResponseIterator as AnthropicModelResponseIterator,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.llms.bedrock import *
from litellm.types.llms.openai import (
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionUsageBlock,
)
from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Delta
from litellm.types.utils import GenericStreamingChunk as GChunk
from litellm.types.utils import (
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)
from litellm.utils import CustomStreamWrapper, get_secret

from ..base_aws_llm import BaseAWSLLM
from ..common_utils import BedrockError, ModelResponseIterator, get_bedrock_tool_name

_response_stream_shape_cache = None
bedrock_tool_name_mappings: InMemoryCache = InMemoryCache(
    max_size_in_memory=50, default_ttl=600
)
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

converse_config = AmazonConverseConfig()


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
        locals_ = locals().copy()
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
            "max_completion_tokens",
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
            if param == "max_tokens" or param == "max_completion_tokens":
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
    logging_obj: Logging,
    fake_stream: bool = False,
    json_mode: Optional[bool] = False,
    bedrock_invoke_provider: Optional[litellm.BEDROCK_INVOKE_PROVIDERS_LITERAL] = None,
):
    try:
        if client is None:
            client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.BEDROCK
            )  # Create a new client if none provided

        response = await client.post(
            api_base,
            headers=headers,
            data=data,
            stream=not fake_stream,
            logging_obj=logging_obj,
        )

        if response.status_code != 200:
            raise BedrockError(status_code=response.status_code, message=response.text)

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
        elif bedrock_invoke_provider == "anthropic":
            decoder: AWSEventStreamDecoder = AmazonAnthropicClaudeStreamDecoder(
                model=model,
                sync_stream=False,
                json_mode=json_mode,
            )
            completion_stream = decoder.aiter_bytes(
                response.aiter_bytes(chunk_size=1024)
            )
        elif bedrock_invoke_provider == "deepseek_r1":
            decoder = AmazonDeepSeekR1StreamDecoder(
                model=model,
                sync_stream=False,
            )
            completion_stream = decoder.aiter_bytes(
                response.aiter_bytes(chunk_size=1024)
            )
        else:
            decoder = AWSEventStreamDecoder(model=model)
            completion_stream = decoder.aiter_bytes(
                response.aiter_bytes(chunk_size=1024)
            )

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
        raise BedrockError(status_code=error_code, message=err.response.text)
    except httpx.TimeoutException:
        raise BedrockError(status_code=408, message="Timeout error occurred.")
    except Exception as e:
        raise BedrockError(status_code=500, message=str(e))


def make_sync_call(
    client: Optional[HTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    signed_json_body: Optional[bytes],
    model: str,
    messages: list,
    logging_obj: Logging,
    fake_stream: bool = False,
    json_mode: Optional[bool] = False,
    bedrock_invoke_provider: Optional[litellm.BEDROCK_INVOKE_PROVIDERS_LITERAL] = None,
):
    try:
        if client is None:
            client = _get_httpx_client(params={})

        response = client.post(
            api_base,
            headers=headers,
            data=signed_json_body if signed_json_body is not None else data,
            stream=not fake_stream,
            logging_obj=logging_obj,
        )

        if response.status_code != 200:
            raise BedrockError(status_code=response.status_code, message=response.text)

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
        elif bedrock_invoke_provider == "anthropic":
            decoder: AWSEventStreamDecoder = AmazonAnthropicClaudeStreamDecoder(
                model=model,
                sync_stream=True,
                json_mode=json_mode,
            )
            completion_stream = decoder.iter_bytes(response.iter_bytes(chunk_size=1024))
        elif bedrock_invoke_provider == "deepseek_r1":
            decoder = AmazonDeepSeekR1StreamDecoder(
                model=model,
                sync_stream=True,
            )
            completion_stream = decoder.iter_bytes(response.iter_bytes(chunk_size=1024))
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
    except httpx.HTTPStatusError as err:
        error_code = err.response.status_code
        raise BedrockError(status_code=error_code, message=err.response.text)
    except httpx.TimeoutException:
        raise BedrockError(status_code=408, message="Timeout error occurred.")
    except Exception as e:
        raise BedrockError(status_code=500, message=str(e))


class BedrockLLM(BaseAWSLLM):
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
        elif provider == "meta" or provider == "llama":
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

    def process_response(  # noqa: PLR0915
        self,
        model: str,
        response: httpx.Response,
        model_response: ModelResponse,
        stream: Optional[bool],
        logging_obj: Logging,
        optional_params: dict,
        api_key: str,
        data: Union[dict, str],
        messages: List,
        print_verbose,
        encoding,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        provider = self.get_bedrock_invoke_provider(model)
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
        except Exception:
            raise BedrockError(message=response.text, status_code=422)

        outputText: Optional[str] = None
        try:
            if provider == "cohere":
                if "text" in completion_response:
                    outputText = completion_response["text"]  # type: ignore
                elif "generations" in completion_response:
                    outputText = completion_response["generations"][0]["text"]
                    model_response.choices[0].finish_reason = map_finish_reason(
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
                        model_response._hidden_params[
                            "original_response"
                        ] = outputText  # allow user to access raw anthropic tool calling response
                    if (
                        _is_function_call is True
                        and stream is not None
                        and stream is True
                    ):
                        print_verbose(
                            "INSIDE BEDROCK STREAMING TOOL CALLING CONDITION BLOCK"
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
                            delta_obj = Delta(
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
                                "Returns anthropic CustomStreamWrapper with 'cached_response' streaming object"
                            )
                            return litellm.CustomStreamWrapper(
                                completion_stream=completion_stream,
                                model=model,
                                custom_llm_provider="cached_response",
                                logging_obj=logging_obj,
                            )

                    model_response.choices[0].finish_reason = map_finish_reason(
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

                    model_response.choices[0].finish_reason = completion_response[
                        "stop_reason"
                    ]
            elif provider == "ai21":
                outputText = (
                    completion_response.get("completions")[0].get("data").get("text")
                )
            elif provider == "meta" or provider == "llama":
                outputText = completion_response["generation"]
            elif provider == "mistral":
                outputText = completion_response["outputs"][0]["text"]
                model_response.choices[0].finish_reason = completion_response[
                    "outputs"
                ][0]["stop_reason"]
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
                outputText is not None
                and len(outputText) > 0
                and hasattr(model_response.choices[0], "message")
                and getattr(model_response.choices[0].message, "tool_calls", None)  # type: ignore
                is None
            ):
                model_response.choices[0].message.content = outputText  # type: ignore
            elif (
                hasattr(model_response.choices[0], "message")
                and getattr(model_response.choices[0].message, "tool_calls", None)  # type: ignore
                is not None
            ):
                pass
            else:
                raise Exception()
        except Exception as e:
            raise BedrockError(
                message="Error parsing received text={}.\nError-{}".format(
                    outputText, str(e)
                ),
                status_code=response.status_code,
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
                content=getattr(model_response.choices[0].message, "content", None),  # type: ignore
                role=model_response.choices[0].message.role,  # type: ignore
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

        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)

        return model_response

    def completion(  # noqa: PLR0915
        self,
        model: str,
        messages: list,
        api_base: Optional[str],
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        logging_obj: Logging,
        optional_params: dict,
        acompletion: bool,
        timeout: Optional[Union[float, httpx.Timeout]],
        litellm_params=None,
        logger_fn=None,
        extra_headers: Optional[dict] = None,
        client: Optional[Union[AsyncHTTPHandler, HTTPHandler]] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        try:
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            from botocore.credentials import Credentials
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        ## SETUP ##
        stream = optional_params.pop("stream", None)

        provider = self.get_bedrock_invoke_provider(model)
        modelId = self.get_bedrock_model_id(
            model=model,
            provider=provider,
            optional_params=optional_params,
        )

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

        ### SET RUNTIME ENDPOINT ###
        endpoint_url, proxy_endpoint_url = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=aws_bedrock_runtime_endpoint,
            aws_region_name=aws_region_name,
        )

        if (stream is not None and stream is True) and provider != "ai21":
            endpoint_url = f"{endpoint_url}/model/{modelId}/invoke-with-response-stream"
            proxy_endpoint_url = (
                f"{proxy_endpoint_url}/model/{modelId}/invoke-with-response-stream"
            )
        else:
            endpoint_url = f"{endpoint_url}/model/{modelId}/invoke"
            proxy_endpoint_url = f"{proxy_endpoint_url}/model/{modelId}/invoke"

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
                if stream is True:
                    inference_params[
                        "stream"
                    ] = True  # cohere requires stream = True in inference params
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
                config = litellm.AmazonAnthropicClaudeConfig.get_config()
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
        elif provider == "meta" or provider == "llama":
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
            raise BedrockError(
                status_code=404,
                message="Bedrock Invoke HTTPX: Unknown provider={}, model={}. Try calling via converse route - `bedrock/converse/<model>`.".format(
                    provider, model
                ),
            )

        ## COMPLETION CALL

        headers = {"Content-Type": "application/json"}
        if extra_headers is not None:
            headers = {"Content-Type": "application/json", **extra_headers}
        request = AWSRequest(
            method="POST", url=endpoint_url, data=data, headers=headers
        )
        sigv4.add_auth(request)
        if (
            extra_headers is not None and "Authorization" in extra_headers
        ):  # prevent sigv4 from overwriting the auth header
            request.headers["Authorization"] = extra_headers["Authorization"]
        prepped = request.prepare()

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": proxy_endpoint_url,
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
                    api_base=proxy_endpoint_url,
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
                api_base=proxy_endpoint_url,
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
        if (stream is not None and stream is True) and provider != "ai21":
            response = self.client.post(
                url=proxy_endpoint_url,
                headers=prepped.headers,  # type: ignore
                data=data,
                stream=stream,
                logging_obj=logging_obj,
            )

            if response.status_code != 200:
                raise BedrockError(
                    status_code=response.status_code, message=str(response.read())
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
            response = self.client.post(
                url=proxy_endpoint_url,
                headers=dict(prepped.headers),
                data=data,
                logging_obj=logging_obj,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise BedrockError(status_code=error_code, message=err.response.text)
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
        logging_obj: Logging,
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
            client = get_async_httpx_client(params=_params, llm_provider=litellm.LlmProviders.BEDROCK)  # type: ignore
        else:
            client = client  # type: ignore

        try:
            response = await client.post(
                api_base,
                headers=headers,
                data=data,
                timeout=timeout,
                logging_obj=logging_obj,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise BedrockError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
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

    @track_llm_api_timing()  # for streaming, we need to instrument the function calling the wrapper
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
        logging_obj: Logging,
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
                data=data,  # type: ignore
                model=model,
                messages=messages,
                logging_obj=logging_obj,
                fake_stream=True if "ai21" in api_base else False,
            ),
            model=model,
            custom_llm_provider="bedrock",
            logging_obj=logging_obj,
        )
        return streaming_response

    @staticmethod
    def _get_provider_from_model_path(
        model_path: str,
    ) -> Optional[litellm.BEDROCK_INVOKE_PROVIDERS_LITERAL]:
        """
        Helper function to get the provider from a model path with format: provider/model-name

        Args:
            model_path (str): The model path (e.g., 'llama/arn:aws:bedrock:us-east-1:086734376398:imported-model/r4c4kewx2s0n' or 'anthropic/model-name')

        Returns:
            Optional[str]: The provider name, or None if no valid provider found
        """
        parts = model_path.split("/")
        if len(parts) >= 1:
            provider = parts[0]
            if provider in get_args(litellm.BEDROCK_INVOKE_PROVIDERS_LITERAL):
                return cast(litellm.BEDROCK_INVOKE_PROVIDERS_LITERAL, provider)
        return None


def get_response_stream_shape():
    global _response_stream_shape_cache
    if _response_stream_shape_cache is None:
        from botocore.loaders import Loader
        from botocore.model import ServiceModel

        loader = Loader()
        bedrock_service_dict = loader.load_service_model("bedrock-runtime", "service-2")
        bedrock_service_model = ServiceModel(bedrock_service_dict)
        _response_stream_shape_cache = bedrock_service_model.shape_for("ResponseStream")

    return _response_stream_shape_cache


class AWSEventStreamDecoder:
    def __init__(self, model: str) -> None:
        from botocore.parsers import EventStreamJSONParser

        self.model = model
        self.parser = EventStreamJSONParser()
        self.content_blocks: List[ContentBlockDeltaEvent] = []
        self.tool_calls_index: Optional[int] = None
        self.response_id: Optional[str] = None

    def check_empty_tool_call_args(self) -> bool:
        """
        Check if the tool call block so far has been an empty string
        """
        args = ""
        # if text content block -> skip
        if len(self.content_blocks) == 0:
            return False

        if (
            "toolUse" not in self.content_blocks[0]
        ):  # be explicit - only do this if tool use block, as this is to prevent json decoding errors
            return False

        for block in self.content_blocks:
            if "toolUse" in block:
                args += block["toolUse"]["input"]

        if len(args) == 0:
            return True
        return False

    def extract_reasoning_content_str(
        self, reasoning_content_block: BedrockConverseReasoningContentBlockDelta
    ) -> Optional[str]:
        if "text" in reasoning_content_block:
            return reasoning_content_block["text"]
        return None

    def translate_thinking_blocks(
        self, thinking_block: BedrockConverseReasoningContentBlockDelta
    ) -> Optional[
        List[Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]]
    ]:
        """
        Translate the thinking blocks to a string
        """

        thinking_blocks_list: List[
            Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
        ] = []
        _thinking_block: Optional[
            Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
        ] = None

        if "text" in thinking_block:
            _thinking_block = ChatCompletionThinkingBlock(type="thinking")
            _thinking_block["thinking"] = thinking_block["text"]
        elif "signature" in thinking_block:
            _thinking_block = ChatCompletionThinkingBlock(type="thinking")
            _thinking_block["signature"] = thinking_block["signature"]
            _thinking_block["thinking"] = ""  # consistent with anthropic response
        elif "redactedContent" in thinking_block:
            _thinking_block = ChatCompletionRedactedThinkingBlock(
                type="redacted_thinking", data=thinking_block["redactedContent"]
            )
        if _thinking_block is not None:
            thinking_blocks_list.append(_thinking_block)
        return thinking_blocks_list

    def converse_chunk_parser(self, chunk_data: dict) -> ModelResponseStream:
        try:
            # Capture the conversationId from the first messageStart event
            # and use it as the consistent ID for all subsequent chunks.
            if self.response_id is None:
                if "messageStart" in chunk_data:
                    conversation_id = chunk_data["messageStart"].get("conversationId")
                    if conversation_id:
                        self.response_id = f"chatcmpl-{conversation_id}"
                else:
                    # Fallback to generating a UUID if the first chunk is not messageStart
                    self.response_id = f"chatcmpl-{uuid.uuid4()}"

            verbose_logger.debug("\n\nRaw Chunk: {}\n\n".format(chunk_data))
            text = ""
            tool_use: Optional[ChatCompletionToolCallChunk] = None
            finish_reason = ""
            usage: Optional[Usage] = None
            provider_specific_fields: dict = {}
            reasoning_content: Optional[str] = None
            thinking_blocks: Optional[
                List[
                    Union[
                        ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock
                    ]
                ]
            ] = None

            index = int(chunk_data.get("contentBlockIndex", 0))
            if "start" in chunk_data:
                start_obj = ContentBlockStartEvent(**chunk_data["start"])
                self.content_blocks = []  # reset
                if start_obj is not None:
                    if "toolUse" in start_obj and start_obj["toolUse"] is not None:
                        ## check tool name was formatted by litellm
                        _response_tool_name = start_obj["toolUse"]["name"]
                        response_tool_name = get_bedrock_tool_name(
                            response_tool_name=_response_tool_name
                        )
                        self.tool_calls_index = (
                            0
                            if self.tool_calls_index is None
                            else self.tool_calls_index + 1
                        )
                        tool_use = {
                            "id": start_obj["toolUse"]["toolUseId"],
                            "type": "function",
                            "function": {
                                "name": response_tool_name,
                                "arguments": "",
                            },
                            "index": self.tool_calls_index,
                        }
                    elif (
                        "reasoningContent" in start_obj
                        and start_obj["reasoningContent"] is not None
                    ):  # redacted thinking can be in start object
                        thinking_blocks = self.translate_thinking_blocks(
                            start_obj["reasoningContent"]
                        )
                        provider_specific_fields = {
                            "reasoningContent": start_obj["reasoningContent"],
                        }
            elif "delta" in chunk_data:
                delta_obj = ContentBlockDeltaEvent(**chunk_data["delta"])
                self.content_blocks.append(delta_obj)
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
                        "index": (
                            self.tool_calls_index
                            if self.tool_calls_index is not None
                            else index
                        ),
                    }
                elif "reasoningContent" in delta_obj:
                    provider_specific_fields = {
                        "reasoningContent": delta_obj["reasoningContent"],
                    }
                    reasoning_content = self.extract_reasoning_content_str(
                        delta_obj["reasoningContent"]
                    )
                    thinking_blocks = self.translate_thinking_blocks(
                        delta_obj["reasoningContent"]
                    )
                    if (
                        thinking_blocks
                        and len(thinking_blocks) > 0
                        and reasoning_content is None
                    ):
                        reasoning_content = ""  # set to non-empty string to ensure consistency with Anthropic
            elif (
                "contentBlockIndex" in chunk_data
            ):  # stop block, no 'start' or 'delta' object
                is_empty = self.check_empty_tool_call_args()
                if is_empty:
                    tool_use = {
                        "id": None,
                        "type": "function",
                        "function": {
                            "name": None,
                            "arguments": "{}",
                        },
                        "index": (
                            self.tool_calls_index
                            if self.tool_calls_index is not None
                            else index
                        ),
                    }
            elif "stopReason" in chunk_data:
                finish_reason = map_finish_reason(chunk_data.get("stopReason", "stop"))
            elif "usage" in chunk_data:
                usage = converse_config._transform_usage(chunk_data.get("usage", {}))

            model_response_provider_specific_fields = {}
            if "trace" in chunk_data:
                trace = chunk_data.get("trace")
                model_response_provider_specific_fields["trace"] = trace
            response = ModelResponseStream(
                choices=[
                    StreamingChoices(
                        finish_reason=finish_reason,
                        index=index,
                        delta=Delta(
                            content=text,
                            role="assistant",
                            tool_calls=[tool_use] if tool_use else None,
                            provider_specific_fields=(
                                provider_specific_fields
                                if provider_specific_fields
                                else None
                            ),
                            thinking_blocks=thinking_blocks,
                            reasoning_content=reasoning_content,
                        ),
                    )
                ],
                id=self.response_id,
                usage=usage,
                provider_specific_fields=model_response_provider_specific_fields,
            )

            return response
        except Exception as e:
            raise Exception("Received streaming error - {}".format(str(e)))

    def _chunk_parser(
        self, chunk_data: dict
    ) -> Union[GChunk, ModelResponseStream, dict]:
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
        ######## /bedrock/converse mappings ###############
        elif (
            "contentBlockIndex" in chunk_data
            or "stopReason" in chunk_data
            or "metrics" in chunk_data
            or "trace" in chunk_data
        ):
            return self.converse_chunk_parser(chunk_data=chunk_data)
        ######### /bedrock/invoke nova mappings ###############
        elif "contentBlockDelta" in chunk_data:
            # when using /bedrock/invoke/nova, the chunk_data is nested under "contentBlockDelta"
            _chunk_data = chunk_data.get("contentBlockDelta", {})
            return self.converse_chunk_parser(chunk_data=_chunk_data)
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
        return GChunk(
            text=text,
            is_finished=is_finished,
            finish_reason=finish_reason,
            usage=None,
            index=0,
            tool_use=None,
        )

    def iter_bytes(
        self, iterator: Iterator[bytes]
    ) -> Iterator[Union[GChunk, ModelResponseStream, dict]]:
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
    ) -> AsyncIterator[Union[GChunk, ModelResponseStream, dict]]:
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
            decoded_body = response_dict["body"].decode()
            if isinstance(decoded_body, dict):
                error_message = decoded_body.get("message")
            elif isinstance(decoded_body, str):
                error_message = decoded_body
            else:
                error_message = ""
            exception_status = response_dict["headers"].get(":exception-type")
            error_message = exception_status + " " + error_message
            raise BedrockError(
                status_code=response_dict["status_code"],
                message=(
                    json.dumps(error_message)
                    if isinstance(error_message, dict)
                    else error_message
                ),
            )
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


class AmazonAnthropicClaudeStreamDecoder(AWSEventStreamDecoder):
    def __init__(
        self,
        model: str,
        sync_stream: bool,
        json_mode: Optional[bool] = None,
    ) -> None:
        """
        Child class of AWSEventStreamDecoder that handles the streaming response from the Anthropic family of models

        The only difference between AWSEventStreamDecoder and AmazonAnthropicClaudeStreamDecoder is the `chunk_parser` method
        """
        super().__init__(model=model)
        self.anthropic_model_response_iterator = AnthropicModelResponseIterator(
            streaming_response=None,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )

    def _chunk_parser(self, chunk_data: dict) -> ModelResponseStream:
        return self.anthropic_model_response_iterator.chunk_parser(chunk=chunk_data)


class AmazonDeepSeekR1StreamDecoder(AWSEventStreamDecoder):
    def __init__(
        self,
        model: str,
        sync_stream: bool,
    ) -> None:
        super().__init__(model=model)
        from litellm.llms.bedrock.chat.invoke_transformations.amazon_deepseek_transformation import (
            AmazonDeepseekR1ResponseIterator,
        )

        self.deepseek_model_response_iterator = AmazonDeepseekR1ResponseIterator(
            streaming_response=None,
            sync_stream=sync_stream,
        )

    def _chunk_parser(
        self, chunk_data: dict
    ) -> Union[GChunk, ModelResponseStream, dict]:
        return self.deepseek_model_response_iterator.chunk_parser(chunk=chunk_data)


class MockResponseIterator:  # for returning ai21 streaming responses
    def __init__(self, model_response, json_mode: Optional[bool] = False):
        self.model_response = model_response
        self.json_mode = json_mode
        self.is_done = False

    # Sync iterator
    def __iter__(self):
        return self

    def _handle_json_mode_chunk(
        self, text: str, tool_calls: Optional[List[ChatCompletionToolCallChunk]]
    ) -> Tuple[str, Optional[ChatCompletionToolCallChunk]]:
        """
        If JSON mode is enabled, convert the tool call to a message.

        Bedrock returns the JSON schema as part of the tool call
        OpenAI returns the JSON schema as part of the content, this handles placing it in the content

        Args:
            text: str
            tool_use: Optional[ChatCompletionToolCallChunk]
        Returns:
            Tuple[str, Optional[ChatCompletionToolCallChunk]]

            text: The text to use in the content
            tool_use: The ChatCompletionToolCallChunk to use in the chunk response
        """
        tool_use: Optional[ChatCompletionToolCallChunk] = None
        if self.json_mode is True and tool_calls is not None:
            message = litellm.AnthropicConfig()._convert_tool_response_to_message(
                tool_calls=tool_calls
            )
            if message is not None:
                text = message.content or ""
                tool_use = None
        elif tool_calls is not None and len(tool_calls) > 0:
            tool_use = tool_calls[0]
        return text, tool_use

    def _chunk_parser(self, chunk_data: ModelResponse) -> GChunk:
        try:
            chunk_usage: Usage = getattr(chunk_data, "usage")
            text = chunk_data.choices[0].message.content or ""  # type: ignore
            tool_use = None
            _model_response_tool_call = cast(
                Optional[List[ChatCompletionMessageToolCall]],
                cast(Choices, chunk_data.choices[0]).message.tool_calls,
            )
            if self.json_mode is True:
                text, tool_use = self._handle_json_mode_chunk(
                    text=text,
                    tool_calls=chunk_data.choices[0].message.tool_calls,  # type: ignore
                )
            elif _model_response_tool_call is not None:
                tool_use = ChatCompletionToolCallChunk(
                    id=_model_response_tool_call[0].id,
                    type="function",
                    function=ChatCompletionToolCallFunctionChunk(
                        name=_model_response_tool_call[0].function.name,
                        arguments=_model_response_tool_call[0].function.arguments,
                    ),
                    index=0,
                )
            processed_chunk = GChunk(
                text=text,
                tool_use=tool_use,
                is_finished=True,
                finish_reason=map_finish_reason(
                    finish_reason=chunk_data.choices[0].finish_reason or ""
                ),
                usage=ChatCompletionUsageBlock(
                    prompt_tokens=chunk_usage.prompt_tokens,
                    completion_tokens=chunk_usage.completion_tokens,
                    total_tokens=chunk_usage.total_tokens,
                ),
                index=0,
            )
            return processed_chunk
        except Exception as e:
            raise ValueError(f"Failed to decode chunk: {chunk_data}. Error: {e}")

    def __next__(self):
        if self.is_done:
            raise StopIteration
        self.is_done = True
        return self._chunk_parser(self.model_response)

    # Async iterator
    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.is_done:
            raise StopAsyncIteration
        self.is_done = True
        return self._chunk_parser(self.model_response)

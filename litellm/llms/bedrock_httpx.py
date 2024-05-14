# What is this?
## Initial implementation of calling bedrock via httpx client (allows for async calls).
## V0 - just covers cohere command-r support

import os, types
import json
from enum import Enum
import requests, copy  # type: ignore
import time
from typing import (
    Callable,
    Optional,
    List,
    Literal,
    Union,
    Any,
    TypedDict,
    Tuple,
    Iterator,
    AsyncIterator,
)
from litellm.utils import (
    ModelResponse,
    Usage,
    map_finish_reason,
    CustomStreamWrapper,
    Message,
    Choices,
    get_secret,
    Logging,
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
    ) -> ModelResponse:
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
            model_response.choices[0].message.content = completion_response["text"]  # type: ignore
        except Exception as e:
            raise BedrockError(message=response.text, status_code=422)

        ## CALCULATING USAGE - bedrock returns usage in the headers
        prompt_tokens = int(
            response.headers.get(
                "x-amzn-bedrock-input-token-count",
                len(encoding.encode("".join(m.get("content", "") for m in messages))),
            )
        )
        completion_tokens = int(
            response.headers.get(
                "x-amzn-bedrock-output-token-count",
                len(
                    encoding.encode(
                        model_response.choices[0].message.content,  # type: ignore
                        disallowed_special=(),
                    )
                ),
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

        if stream is not None and stream == True:
            endpoint_url = f"{endpoint_url}/model/{model}/invoke-with-response-stream"
        else:
            endpoint_url = f"{endpoint_url}/model/{model}/invoke"

        sigv4 = SigV4Auth(credentials, "bedrock", aws_region_name)

        provider = model.split(".")[0]
        prompt, chat_history = self.convert_messages_to_prompt(
            model, messages, provider, custom_prompt_dict
        )
        inference_params = copy.deepcopy(optional_params)

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
        else:
            raise Exception("UNSUPPORTED PROVIDER")

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
            if stream:
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
                stream=False,
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
            self.client = HTTPHandler(**_params)  # type: ignore
        else:
            self.client = client
        if stream is not None and stream == True:
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

            decoder = AWSEventStreamDecoder()

            completion_stream = decoder.iter_bytes(response.iter_bytes(chunk_size=1024))
            streaming_response = CustomStreamWrapper(
                completion_stream=completion_stream,
                model=model,
                custom_llm_provider="bedrock",
                logging_obj=logging_obj,
            )
            return streaming_response

        response = self.client.post(url=prepped.url, headers=prepped.headers, data=data)  # type: ignore

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise BedrockError(status_code=error_code, message=response.text)

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
    ) -> ModelResponse:
        if client is None:
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            self.client = AsyncHTTPHandler(**_params)  # type: ignore
        else:
            self.client = client  # type: ignore

        response = await self.client.post(api_base, headers=headers, data=data)  # type: ignore
        return self.process_response(
            model=model,
            response=response,
            model_response=model_response,
            stream=stream,
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
        if client is None:
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            self.client = AsyncHTTPHandler(**_params)  # type: ignore
        else:
            self.client = client  # type: ignore

        response = await self.client.post(api_base, headers=headers, data=data, stream=True)  # type: ignore

        if response.status_code != 200:
            raise BedrockError(status_code=response.status_code, message=response.text)

        decoder = AWSEventStreamDecoder()

        completion_stream = decoder.aiter_bytes(response.aiter_bytes(chunk_size=1024))
        streaming_response = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="bedrock",
            logging_obj=logging_obj,
        )
        return streaming_response

    def embedding(self, *args, **kwargs):
        return super().embedding(*args, **kwargs)


def get_response_stream_shape():
    from botocore.model import ServiceModel
    from botocore.loaders import Loader

    loader = Loader()
    bedrock_service_dict = loader.load_service_model("bedrock-runtime", "service-2")
    bedrock_service_model = ServiceModel(bedrock_service_dict)
    return bedrock_service_model.shape_for("ResponseStream")


class AWSEventStreamDecoder:
    def __init__(self) -> None:
        from botocore.parsers import EventStreamJSONParser

        self.parser = EventStreamJSONParser()

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
                    streaming_chunk: GenericStreamingChunk = GenericStreamingChunk(
                        text=_data.get("text", ""),
                        is_finished=_data.get("is_finished", False),
                        finish_reason=_data.get("finish_reason", ""),
                    )
                    yield streaming_chunk

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
                    streaming_chunk: GenericStreamingChunk = GenericStreamingChunk(
                        text=_data.get("text", ""),
                        is_finished=_data.get("is_finished", False),
                        finish_reason=_data.get("finish_reason", ""),
                    )
                    yield streaming_chunk

    def _parse_message_from_event(self, event) -> Optional[str]:
        response_dict = event.to_response_dict()
        parsed_response = self.parser.parse(response_dict, get_response_stream_shape())
        if response_dict["status_code"] != 200:
            raise ValueError(f"Bad response code, expected 200: {response_dict}")

        chunk = parsed_response.get("chunk")
        if not chunk:
            return None

        return chunk.get("bytes").decode()  # type: ignore[no-any-return]

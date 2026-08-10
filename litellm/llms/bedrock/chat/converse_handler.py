import json
from typing import Any, Final

import httpx

import litellm
from litellm.anthropic_beta_headers_manager import (
    update_headers_with_filtered_beta,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObject
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper

from ..base_aws_llm import BaseAWSLLM, Credentials
from ..common_utils import BedrockError, _get_all_bedrock_regions
from .invoke_handler import AWSEventStreamDecoder, MockResponseIterator, make_call


def make_sync_call(
    client: HTTPHandler | None,
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj: LiteLLMLoggingObject,
    json_mode: bool | None = False,
    fake_stream: bool = False,
    stream_chunk_size: int | None = None,
):
    if client is None:
        client = _get_httpx_client()  # Create a new client if none provided

    response: Final = client.post(
        api_base,
        headers=headers,
        data=data,
        stream=not fake_stream,
        logging_obj=logging_obj,
    )

    if response.status_code != 200:
        raise BedrockError(status_code=response.status_code, message=str(response.read()))

    if fake_stream:
        model_response: Final[ModelResponse] = litellm.AmazonConverseConfig()._transform_response(
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
        )
        completion_stream: Any = MockResponseIterator(model_response=model_response, json_mode=json_mode)
    else:
        decoder: Final = AWSEventStreamDecoder(model=model, json_mode=json_mode)
        completion_stream = decoder.iter_bytes(response.iter_bytes(chunk_size=stream_chunk_size))

    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response="first stream response received",
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


class BedrockConverseLLM(BaseAWSLLM):
    def __init__(self) -> None:
        super().__init__()

    async def async_streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        model_response: ModelResponse,
        timeout: float | httpx.Timeout | None,
        encoding,
        logging_obj,
        stream,
        optional_params: dict,
        litellm_params: dict,
        credentials: Credentials,
        logger_fn=None,
        headers={},
        client: AsyncHTTPHandler | None = None,
        fake_stream: bool = False,
        json_mode: bool | None = False,
        api_key: str | None = None,
        stream_chunk_size: int | None = None,
    ) -> CustomStreamWrapper:
        request_data: Final = await litellm.AmazonConverseConfig()._async_transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        data: Final = json.dumps(request_data)

        prepped: Final = self.get_request_headers(
            credentials=credentials,
            aws_region_name=litellm_params.get("aws_region_name") or "us-west-2",
            extra_headers=headers,
            endpoint_url=api_base,
            data=data,
            headers=headers,
            api_key=api_key,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": dict(prepped.headers),
            },
        )

        completion_stream: Final = await make_call(
            client=client,
            api_base=api_base,
            headers=dict(prepped.headers),
            data=data,
            model=model,
            messages=messages,
            logging_obj=logging_obj,
            fake_stream=fake_stream,
            json_mode=json_mode,
            stream_chunk_size=stream_chunk_size,
        )
        streaming_response: Final = CustomStreamWrapper(
            completion_stream=completion_stream,
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
        timeout: float | httpx.Timeout | None,
        encoding,
        logging_obj: LiteLLMLoggingObject,
        stream,
        optional_params: dict,
        litellm_params: dict,
        credentials: Credentials,
        logger_fn=None,
        headers: dict = {},
        client: AsyncHTTPHandler | None = None,
        api_key: str | None = None,
    ) -> ModelResponse | CustomStreamWrapper:
        request_data: Final = await litellm.AmazonConverseConfig()._async_transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        data: Final = json.dumps(request_data)

        prepped: Final = self.get_request_headers(
            credentials=credentials,
            aws_region_name=litellm_params.get("aws_region_name") or "us-west-2",
            extra_headers=headers,
            endpoint_url=api_base,
            data=data,
            headers=headers,
            api_key=api_key,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": prepped.headers,
            },
        )

        headers = dict(prepped.headers)
        if client is None or not isinstance(client, AsyncHTTPHandler):
            _params: Final = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            client = get_async_httpx_client(params=_params, llm_provider=litellm.LlmProviders.BEDROCK)
        else:
            client = client

        try:
            response: Final = await client.post(
                url=api_base,
                headers=headers,
                data=data,
                logging_obj=logging_obj,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code: Final = err.response.status_code
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

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str | None,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        encoding,
        logging_obj: LiteLLMLoggingObject,
        optional_params: dict,
        acompletion: bool,
        timeout: float | httpx.Timeout | None,
        litellm_params: dict,
        logger_fn=None,
        extra_headers: dict | None = None,
        client: AsyncHTTPHandler | HTTPHandler | None = None,
        api_key: str | None = None,
    ):
        ## SETUP ##
        stream: Final = optional_params.pop("stream", None)
        stream_chunk_size: Final = optional_params.pop("stream_chunk_size", None)
        unencoded_model_id: Final = optional_params.pop("model_id", None)
        fake_stream = optional_params.pop("fake_stream", False)
        json_mode: Final = optional_params.get("json_mode", False)
        if unencoded_model_id is not None:
            modelId = self.encode_model_id(model_id=unencoded_model_id)
        else:
            # Strip nova spec prefixes before encoding model ID for API URL
            _model_for_id = model
            _stripped = _model_for_id
            for rp in ["bedrock/converse/", "bedrock/", "converse/"]:
                if _stripped.startswith(rp):
                    _stripped = _stripped[len(rp) :]
                    break
            # Strip embedded region prefix (e.g. "bedrock/us-east-1/model" -> "model")
            # and capture it so it can be used as aws_region_name below.
            _region_from_model: str | None = None
            _potential_region: Final = _stripped.split("/", 1)[0]
            if _potential_region in _get_all_bedrock_regions() and "/" in _stripped:
                _region_from_model = _potential_region
                _stripped = _stripped.split("/", 1)[1]
                _model_for_id = _stripped
            for _nova_prefix in ["nova-2/", "nova/"]:
                if _stripped.startswith(_nova_prefix):
                    _model_for_id = _model_for_id.replace(_nova_prefix, "", 1)
                    break
            modelId = self.encode_model_id(model_id=_model_for_id)
            # Inject region extracted from model path so _get_aws_region_name picks it up
            if _region_from_model is not None and "aws_region_name" not in optional_params:
                optional_params["aws_region_name"] = _region_from_model

        fake_stream = litellm.AmazonConverseConfig().should_fake_stream(
            fake_stream=fake_stream,
            model=model,
            stream=stream,
            custom_llm_provider="bedrock",
        )

        ### SET REGION NAME ###
        aws_region_name: Final = self._get_aws_region_name(
            optional_params=optional_params,
            model=model,
            model_id=unencoded_model_id,
        )

        ## CREDENTIALS ##
        # pop aws_secret_access_key, aws_access_key_id, aws_region_name from kwargs, since completion calls fail with them
        aws_secret_access_key: Final = optional_params.pop("aws_secret_access_key", None)
        aws_access_key_id: Final = optional_params.pop("aws_access_key_id", None)
        aws_session_token: Final = optional_params.pop("aws_session_token", None)
        aws_role_name: Final = optional_params.pop("aws_role_name", None)
        aws_session_name: Final = optional_params.pop("aws_session_name", None)
        aws_profile_name: Final = optional_params.pop("aws_profile_name", None)
        aws_bedrock_runtime_endpoint: Final = optional_params.pop(
            "aws_bedrock_runtime_endpoint", None
        )  # https://bedrock-runtime.{region_name}.amazonaws.com
        aws_web_identity_token: Final = optional_params.pop("aws_web_identity_token", None)
        aws_sts_endpoint: Final = optional_params.pop("aws_sts_endpoint", None)
        aws_external_id: Final = optional_params.pop("aws_external_id", None)
        optional_params.pop("aws_region_name", None)

        litellm_params["aws_region_name"] = aws_region_name  # [DO NOT DELETE] important for async calls

        credentials: Final[Credentials] = self.get_credentials(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            aws_region_name=aws_region_name,
            aws_session_name=aws_session_name,
            aws_profile_name=aws_profile_name,
            aws_role_name=aws_role_name,
            aws_web_identity_token=aws_web_identity_token,
            aws_sts_endpoint=aws_sts_endpoint,
            aws_external_id=aws_external_id,
        )

        ### SET RUNTIME ENDPOINT ###
        endpoint_url, proxy_endpoint_url = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=aws_bedrock_runtime_endpoint,
            aws_region_name=aws_region_name,
        )
        if (stream is not None and stream is True) and not fake_stream:
            endpoint_url = f"{endpoint_url}/model/{modelId}/converse-stream"
            proxy_endpoint_url = f"{proxy_endpoint_url}/model/{modelId}/converse-stream"
        else:
            endpoint_url = f"{endpoint_url}/model/{modelId}/converse"
            proxy_endpoint_url = f"{proxy_endpoint_url}/model/{modelId}/converse"

        ## COMPLETION CALL
        headers = {"Content-Type": "application/json"}
        if extra_headers is not None:
            headers = {"Content-Type": "application/json", **extra_headers}

        # Filter beta headers in HTTP headers before making the request
        headers = update_headers_with_filtered_beta(headers=headers, provider="bedrock_converse")
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
                    api_key=api_key,
                    stream_chunk_size=stream_chunk_size,
                )
            ### ASYNC COMPLETION
            return self.async_completion(
                model=model,
                messages=messages,
                api_base=proxy_endpoint_url,
                model_response=model_response,
                encoding=encoding,
                logging_obj=logging_obj,
                optional_params=optional_params,
                stream=stream,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                headers=headers,
                timeout=timeout,
                client=client,
                credentials=credentials,
                api_key=api_key,
            )

        ## TRANSFORMATION ##

        _data: Final = litellm.AmazonConverseConfig()._transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=extra_headers,
        )
        data: Final = json.dumps(_data)

        prepped: Final = self.get_request_headers(
            credentials=credentials,
            aws_region_name=aws_region_name,
            extra_headers=extra_headers,
            endpoint_url=proxy_endpoint_url,
            data=data,
            headers=headers,
            api_key=api_key,
        )

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
        if client is None or isinstance(client, AsyncHTTPHandler):
            _params: Final = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            client = _get_httpx_client(_params)
        else:
            client = client

        if stream is not None and stream is True:
            completion_stream: Final = make_sync_call(
                client=(client if client is not None and isinstance(client, HTTPHandler) else None),
                api_base=proxy_endpoint_url,
                headers=prepped.headers,
                data=data,
                model=model,
                messages=messages,
                logging_obj=logging_obj,
                json_mode=json_mode,
                fake_stream=fake_stream,
                stream_chunk_size=stream_chunk_size,
            )
            streaming_response: Final = CustomStreamWrapper(
                completion_stream=completion_stream,
                model=model,
                custom_llm_provider="bedrock",
                logging_obj=logging_obj,
            )

            return streaming_response

        ### COMPLETION

        try:
            response: Final = client.post(
                url=proxy_endpoint_url,
                headers=prepped.headers,
                data=data,
                logging_obj=logging_obj,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code: Final = err.response.status_code
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

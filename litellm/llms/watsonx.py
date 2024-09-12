import json
import types
import time
import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Dict,
    Generator,
    Iterator,
    Optional,
    Tuple,
    Union,
    List,
)
from contextlib import contextmanager, asynccontextmanager

import httpx
import requests
import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.utils import (
    EmbeddingResponse,
    ModelResponse,
    Usage,
    get_secret,
    map_finish_reason,
)

from .base import BaseLLM
from .prompt_templates import factory as ptf


class WatsonXAIError(Exception):
    def __init__(self, status_code, message, url: Optional[str] = None):
        self.status_code = status_code
        self.message = message
        url = url or "https://https://us-south.ml.cloud.ibm.com"
        self.request = httpx.Request(method="POST", url=url)
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class IBMWatsonXAIConfig:
    """
    Reference: https://cloud.ibm.com/apidocs/watsonx-ai#text-generation
    (See ibm_watsonx_ai.metanames.GenTextParamsMetaNames for a list of all available params)

    Supported params for all available watsonx.ai foundational models.

    - `decoding_method` (str): One of "greedy" or "sample"

    - `temperature` (float): Sets the model temperature for sampling - not available when decoding_method='greedy'.

    - `max_new_tokens` (integer): Maximum length of the generated tokens.

    - `min_new_tokens` (integer): Maximum length of input tokens. Any more than this will be truncated.

    - `length_penalty` (dict): A dictionary with keys "decay_factor" and "start_index".

    - `stop_sequences` (string[]): list of strings to use as stop sequences.

    - `top_k` (integer): top k for sampling - not available when decoding_method='greedy'.

    - `top_p` (integer): top p for sampling - not available when decoding_method='greedy'.

    - `repetition_penalty` (float): token repetition penalty during text generation.

    - `truncate_input_tokens` (integer): Truncate input tokens to this length.

    - `include_stop_sequences` (bool): If True, the stop sequence will be included at the end of the generated text in the case of a match.

    - `return_options` (dict): A dictionary of options to return. Options include "input_text", "generated_tokens", "input_tokens", "token_ranks". Values are boolean.

    - `random_seed` (integer): Random seed for text generation.

    - `moderations` (dict): Dictionary of properties that control the moderations, for usages such as Hate and profanity (HAP) and PII filtering.

    - `stream` (bool): If True, the model will return a stream of responses.
    """

    decoding_method: Optional[str] = "sample"
    temperature: Optional[float] = None
    max_new_tokens: Optional[int] = None  # litellm.max_tokens
    min_new_tokens: Optional[int] = None
    length_penalty: Optional[dict] = None  # e.g {"decay_factor": 2.5, "start_index": 5}
    stop_sequences: Optional[List[str]] = None  # e.g ["}", ")", "."]
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    truncate_input_tokens: Optional[int] = None
    include_stop_sequences: Optional[bool] = False
    return_options: Optional[Dict[str, bool]] = None
    random_seed: Optional[int] = None  # e.g 42
    moderations: Optional[dict] = None
    stream: Optional[bool] = False

    def __init__(
        self,
        decoding_method: Optional[str] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        min_new_tokens: Optional[int] = None,
        length_penalty: Optional[dict] = None,
        stop_sequences: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        truncate_input_tokens: Optional[int] = None,
        include_stop_sequences: Optional[bool] = None,
        return_options: Optional[dict] = None,
        random_seed: Optional[int] = None,
        moderations: Optional[dict] = None,
        stream: Optional[bool] = None,
        **kwargs,
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

    def get_supported_openai_params(self):
        return [
            "temperature",  # equivalent to temperature
            "max_tokens",  # equivalent to max_new_tokens
            "top_p",  # equivalent to top_p
            "frequency_penalty",  # equivalent to repetition_penalty
            "stop",  # equivalent to stop_sequences
            "seed",  # equivalent to random_seed
            "stream",  # equivalent to stream
        ]

    def get_mapped_special_auth_params(self) -> dict:
        """
        Common auth params across bedrock/vertex_ai/azure/watsonx
        """
        return {
            "project": "watsonx_project",
            "region_name": "watsonx_region_name",
            "token": "watsonx_token",
        }

    def map_special_auth_params(self, non_default_params: dict, optional_params: dict):
        mapped_params = self.get_mapped_special_auth_params()

        for param, value in non_default_params.items():
            if param in mapped_params:
                optional_params[mapped_params[param]] = value
        return optional_params

    def get_eu_regions(self) -> List[str]:
        """
        Source: https://www.ibm.com/docs/en/watsonx/saas?topic=integrations-regional-availability
        """
        return [
            "eu-de",
            "eu-gb",
        ]


def convert_messages_to_prompt(model, messages, provider, custom_prompt_dict):
    # handle anthropic prompts and amazon titan prompts
    if model in custom_prompt_dict:
        # check if the model has a registered custom prompt
        model_prompt_dict = custom_prompt_dict[model]
        prompt = ptf.custom_prompt(
            messages=messages,
            role_dict=model_prompt_dict.get(
                "role_dict", model_prompt_dict.get("roles")
            ),
            initial_prompt_value=model_prompt_dict.get("initial_prompt_value", ""),
            final_prompt_value=model_prompt_dict.get("final_prompt_value", ""),
            bos_token=model_prompt_dict.get("bos_token", ""),
            eos_token=model_prompt_dict.get("eos_token", ""),
        )
        return prompt
    elif provider == "ibm":
        prompt = ptf.prompt_factory(
            model=model, messages=messages, custom_llm_provider="watsonx"
        )
    elif provider == "ibm-mistralai":
        prompt = ptf.mistral_instruct_pt(messages=messages)
    else:
        prompt = ptf.prompt_factory(
            model=model, messages=messages, custom_llm_provider="watsonx"
        )
    return prompt


class WatsonXAIEndpoint(str, Enum):
    TEXT_GENERATION = "/ml/v1/text/generation"
    TEXT_GENERATION_STREAM = "/ml/v1/text/generation_stream"
    DEPLOYMENT_TEXT_GENERATION = "/ml/v1/deployments/{deployment_id}/text/generation"
    DEPLOYMENT_TEXT_GENERATION_STREAM = (
        "/ml/v1/deployments/{deployment_id}/text/generation_stream"
    )
    EMBEDDINGS = "/ml/v1/text/embeddings"
    PROMPTS = "/ml/v1/prompts"
    AVAILABLE_MODELS = "/ml/v1/foundation_model_specs"

@dataclass
class IBMAuthToken:
    """
    IBM IAM token object.
    """
    access_token: str
    expiration: int

    @property
    def is_expired(self):
        return time.time() > self.expiration

class IBMWatsonXAI(BaseLLM):
    """
    Class to interface with IBM watsonx.ai API for text generation and embeddings.

    Reference: https://cloud.ibm.com/apidocs/watsonx-ai
    """

    api_version = "2024-03-13"

    def __init__(self) -> None:
        super().__init__()
        self.request_manager = RequestManager()

    def _prepare_text_generation_req(
        self,
        model_id: str,
        prompt: str,
        stream: bool,
        optional_params: dict,
    ) -> Tuple[dict, dict]:
        """
        Get the request parameters for text generation.
        """
        api_args = self._get_endpoint_args(optional_params)
        extra_body_params = optional_params.pop("extra_body", {})
        optional_params.update(extra_body_params)
        # init the payload to the text generation call
        payload = {
            "input": prompt,
            "moderations": optional_params.pop("moderations", {}),
            "parameters": optional_params,
        }
        # text generation endpoint deployment or model / stream or not
        if model_id.startswith("deployment/"):
            # deployment models are passed in as 'deployment/<deployment_id>'
            if api_args.get("space_id") is None:
                raise WatsonXAIError(
                    status_code=401,
                    url=api_args["url"],
                    message="Error: space_id is required for models called using the 'deployment/' endpoint. Pass in the space_id as a parameter or set it in the WX_SPACE_ID environment variable.",
                )
            deployment_id = "/".join(model_id.split("/")[1:])
            endpoint = (
                WatsonXAIEndpoint.DEPLOYMENT_TEXT_GENERATION_STREAM.value
                if stream
                else WatsonXAIEndpoint.DEPLOYMENT_TEXT_GENERATION.value
            )
            endpoint = endpoint.format(deployment_id=deployment_id)
        else:
            payload["model_id"] = model_id
            payload["project_id"] = api_args["project_id"]
            endpoint = (
                WatsonXAIEndpoint.TEXT_GENERATION_STREAM
                if stream
                else WatsonXAIEndpoint.TEXT_GENERATION
            )
        url = api_args["url"].rstrip("/") + endpoint
        req_params = dict(
            method="POST", url=url, json=payload, params=dict(version=api_args["api_version"])
        )
        return req_params, api_args

    def _get_endpoint_args(
        self, 
        params: dict,
    ) -> dict:
        """
        Find watsonx.ai credentials in the params or environment variables and return the headers for authentication.
        """
        # Load auth variables from params
        url = params.pop("url", params.pop("api_base", params.pop("base_url", None)))
        api_key = params.pop("apikey", None)
        token = params.pop("token", params.pop("watsonx_token", None))
        project_id = params.pop(
            "project_id", params.pop("watsonx_project", None)
        )  # watsonx.ai project_id - allow 'watsonx_project' to be consistent with how vertex project implementation works -> reduce provider-specific params
        space_id = params.pop("space_id", None)  # watsonx.ai deployment space_id
        region_name = params.pop("region_name", params.pop("region", None))
        if region_name is None:
            region_name = params.pop(
                "watsonx_region_name", params.pop("watsonx_region", None)
            )  # consistent with how vertex ai + aws regions are accepted
        wx_credentials = params.pop(
            "wx_credentials",
            params.pop(
                "watsonx_credentials", None
            ),  # follow {provider}_credentials, same as vertex ai
        )
        api_version = params.pop("api_version", IBMWatsonXAI.api_version)
        # Load auth variables from environment variables
        if url is None:
            url = (
                get_secret("WATSONX_API_BASE")  # consistent with 'AZURE_API_BASE'
                or get_secret("WATSONX_URL")
                or get_secret("WX_URL")
                or get_secret("WML_URL")
            )
        if api_key is None:
            api_key = (
                get_secret("WATSONX_APIKEY")
                or get_secret("WATSONX_API_KEY")
                or get_secret("WX_API_KEY")
            )
        if token is None:
            token = get_secret("WATSONX_TOKEN") or get_secret("WX_TOKEN")
        if project_id is None:
            project_id = (
                get_secret("WATSONX_PROJECT_ID")
                or get_secret("WX_PROJECT_ID")
                or get_secret("PROJECT_ID")
            )
        if region_name is None:
            region_name = (
                get_secret("WATSONX_REGION")
                or get_secret("WX_REGION")
                or get_secret("REGION")
            )
        if space_id is None:
            space_id = (
                get_secret("WATSONX_DEPLOYMENT_SPACE_ID")
                or get_secret("WATSONX_SPACE_ID")
                or get_secret("WX_SPACE_ID")
                or get_secret("SPACE_ID")
            )

        # credentials parsing
        if wx_credentials is not None:
            url = wx_credentials.get("url", url)
            api_key = wx_credentials.get(
                "apikey", wx_credentials.get("api_key", api_key)
            )
            token = wx_credentials.get(
                "token",
                wx_credentials.get(
                    "watsonx_token", token
                ),  # follow format of {provider}_token, same as azure - e.g. 'azure_ad_token=..'
            )
        # verify that all required credentials are present
        if url is None:
            raise WatsonXAIError(
                status_code=401,
                message="Error: Watsonx URL not set. Set WATSONX_URL in environment variables or pass in as a parameter.",
            )
        if project_id is None:
            raise WatsonXAIError(
                status_code=401,
                url=url,
                message="Error: Watsonx project_id not set. Set WATSONX_PROJECT_ID in environment variables or pass in as a parameter.",
            )
        return {
            "url": url,
            "api_key": api_key,
            "token": token,
            "project_id": project_id,
            "space_id": space_id,
            "region_name": region_name,
            "api_version": api_version,
        }

    def _process_text_gen_response(
        self, json_resp: dict, model_response: Union[ModelResponse, None] = None
    ) -> ModelResponse:
        if "results" not in json_resp:
            raise WatsonXAIError(
                status_code=500,
                message=f"Error: Invalid response from Watsonx.ai API: {json_resp}",
            )
        if model_response is None:
            model_response = ModelResponse(model=json_resp.get("model_id", None))
        generated_text = json_resp["results"][0]["generated_text"]
        prompt_tokens = json_resp["results"][0]["input_token_count"]
        completion_tokens = json_resp["results"][0]["generated_token_count"]
        model_response.choices[0].message.content = generated_text  # type: ignore
        model_response.choices[0].finish_reason = map_finish_reason(
            json_resp["results"][0]["stop_reason"]
        )
        if json_resp.get("created_at"):
            model_response.created = int(
                datetime.fromisoformat(json_resp["created_at"]).timestamp()
            )
        else:
            model_response.created = int(time.time())
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
        optional_params=None,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        timeout=None,
    ):
        """
        Send a text generation request to the IBM Watsonx.ai API.
        Reference: https://cloud.ibm.com/apidocs/watsonx-ai#text-generation
        """
        stream = optional_params.pop("stream", False)
        if timeout is not None and timeout != self.request_manager.timeout:
            self.request_manager.set_timeout(timeout)
        # Load default configs
        config = IBMWatsonXAIConfig.get_config()
        for k, v in config.items():
            if k not in optional_params and k!="stream":
                optional_params[k] = v
        # Make prompt to send to model
        provider = model.split("/")[0]
        # model_name = "/".join(model.split("/")[1:])
        prompt = convert_messages_to_prompt(
            model, messages, provider, custom_prompt_dict
        )
        model_response.model = 'watsonx/'+model if not model.startswith("watsonx/") else model
        request_params, api_args = self._prepare_text_generation_req(
            model_id=model,
            prompt=prompt,
            stream=stream,
            optional_params=optional_params,
        )

        def process_stream_response(
            stream_resp: Union[Iterator[str], AsyncIterator],
        ) -> litellm.CustomStreamWrapper:
            streamwrapper = litellm.CustomStreamWrapper(
                stream_resp,
                model=model,
                custom_llm_provider="watsonx",
                logging_obj=logging_obj,
            )
            return streamwrapper

        # create the function to handle the request to watsonx.ai
        def handle_text_request() -> ModelResponse:
            with self.request_manager.request(
                request_params, api_args,
                input=prompt,
                logging_obj=logging_obj
            ) as resp:
                json_resp = resp.json()

            return self._process_text_gen_response(json_resp, model_response)

        async def handle_text_request_async() -> ModelResponse:
            async with self.request_manager.async_request(
                request_params, api_args, 
                input=prompt,
                logging_obj=logging_obj
            ) as resp:
                json_resp = resp.json()
            return self._process_text_gen_response(json_resp, model_response)

        def handle_stream_request() -> litellm.CustomStreamWrapper:
            # stream the response - generated chunks will be handled
            # by litellm.utils.CustomStreamWrapper.handle_watsonx_stream
            with self.request_manager.request(
                request_params, api_args,
                stream=True,
                input=prompt,
                logging_obj=logging_obj
            ) as resp:
                streamwrapper = process_stream_response(resp.iter_lines())
            return streamwrapper

        async def handle_stream_request_async() -> litellm.CustomStreamWrapper:
            # stream the response - generated chunks will be handled
            # by litellm.utils.CustomStreamWrapper.handle_watsonx_stream
            async with self.request_manager.async_request(
                request_params,
                api_args,
                stream=True,
                input=prompt,
                logging_obj=logging_obj
            ) as resp:
                streamwrapper = process_stream_response(resp.aiter_lines())
            return streamwrapper

        try:
            ## Get the response from the model
            if stream and (acompletion is True):
                # stream and async text generation
                return handle_stream_request_async()
            elif stream:
                # streaming text generation
                return handle_stream_request()
            elif acompletion is True:
                # async text generation
                return handle_text_request_async()
            else:
                # regular text generation
                return handle_text_request()
        except WatsonXAIError as e:
            raise e
        except Exception as e:
            raise WatsonXAIError(status_code=500, message=str(e))

    def _process_embedding_response(
        self, json_resp: dict, model_response: Optional[EmbeddingResponse] = None
    ) -> EmbeddingResponse:
        if model_response is None:
            model_response = EmbeddingResponse(model='watsonx/'+json_resp.get("model_id", ''))
        results = json_resp.get("results", [])
        embedding_response = []
        for idx, result in enumerate(results):
            embedding_response.append(
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": result["embedding"],
                }
            )
        model_response.object = "list"
        model_response.data = embedding_response
        input_tokens = json_resp.get("input_token_count", 0)
        usage = Usage(
            prompt_tokens=input_tokens,
            completion_tokens=0,
            total_tokens=input_tokens,
        )
        setattr(model_response, "usage", usage)
        return model_response

    def embedding(
        self,
        model: str,
        input: Union[list, str],
        model_response: litellm.EmbeddingResponse,
        api_key: Optional[str] = None,
        logging_obj=None,
        optional_params=None,
        encoding=None,
        print_verbose=None,
        aembedding=None,
    ):
        """
        Send a text embedding request to the IBM Watsonx.ai API.
        """
        if optional_params is None:
            optional_params = {}
        # Load default configs
        config = IBMWatsonXAIConfig.get_config()
        for k, v in config.items():
            if k not in optional_params:
                optional_params[k] = v

        model_response.model = model
        if not model_response.model.startswith("watsonx/"):
            model_response.model = "watsonx/" + model_response.model
        
        # Load auth variables from env
        if isinstance(input, str):
            input = [input]
        if api_key is not None:
            optional_params["api_key"] = api_key
        api_args = self._get_endpoint_args(optional_params)
        # init the payload to the text generation call
        payload = {
            "inputs": input,
            "model_id": model,
            "project_id": api_args["project_id"],
            "parameters": optional_params,
        }
        request_params = dict(version=api_args["api_version"])
        url = api_args["url"].rstrip("/") + WatsonXAIEndpoint.EMBEDDINGS
        req_params = {
            "method": "POST",
            "url": url,
            "json": payload,
            "params": request_params,
        }
        def handle_embedding(request_params: dict) -> litellm.EmbeddingResponse:
            with self.request_manager.request(
                request_params,
                api_args,
                input=input,
                logging_obj=logging_obj
            ) as resp:
                json_resp = resp.json()
            return self._process_embedding_response(json_resp, model_response)

        async def handle_aembedding(request_params: dict) -> litellm.EmbeddingResponse:
            async with self.request_manager.async_request(
                request_params, 
                api_args,
                input=input,
                logging_obj=logging_obj
            ) as resp:
                json_resp = resp.json()
            return self._process_embedding_response(json_resp, model_response)

        try:
            if aembedding is True:
                return handle_aembedding(req_params)
            else:
                return handle_embedding(req_params)
        except WatsonXAIError as e:
            raise e
        except Exception as e:
            raise WatsonXAIError(status_code=500, message=str(e))

    def generate_iam_token(self, api_key=None, async_:bool=False, **params):
        iam_token = self.request_manager.get_auth_token(api_key, async_=async_, **params)
        if async_:
            async def get_token_async():
                return (await iam_token).access_token
            return get_token_async()
        return iam_token.access_token

    def get_available_models(self, *, ids_only: bool = True, async_:bool=False, **params):
        api_args = self._get_endpoint_args(params)
        url = api_args["url"].rstrip("/") + WatsonXAIEndpoint.AVAILABLE_MODELS
        req_params = dict(
            method="GET", 
            url=url, 
            params=dict(version=api_args["api_version"]),
        )

        def get_models():
            with self.request_manager.request(req_params, api_args) as resp:
                json_resp = resp.json()
            if not ids_only:
                return json_resp
            return [res["model_id"] for res in json_resp["resources"]]
        
        async def get_models_async():
            async with self.request_manager.async_request(req_params, api_args) as resp:
                json_resp = resp.json()
            if not ids_only:
                return json_resp
            return [res["model_id"] for res in json_resp["resources"]]
        
        if async_:
            return get_models_async()
        return get_models()


class RequestManager:
    """
    A class to handle sync/async HTTP requests to the IBM Watsonx.ai API.

    Usage:
    ```python
    request_params = dict(method="POST", url="https://api.example.com", headers={"Authorization" : "Bearer token"}, json={"key": "value"})
    request_manager = RequestManager(logging_obj=logging_obj)
    with request_manager.request(request_params) as resp:
        ...
    # or
    async with request_manager.async_request(request_params) as resp:
        ...
    ```
    """

    def __init__(self, timeout: Optional[float] = None) -> None:
        self._auth_token = None
        if timeout is None:
            timeout = float(get_secret("WATSONX_TIMEOUT") or 600)
        self.timeout = timeout
        self.async_handler = AsyncHTTPHandler(
            timeout=httpx.Timeout(
                timeout=timeout,
                connect=5.0
            ),
        )
    
    def set_timeout(self, timeout: float):
        self.timeout = timeout
        self.async_handler.timeout = httpx.Timeout(
            timeout=timeout,
            connect=5.0
        )

    @contextmanager
    def request(
        self,
        request_params: dict,
        api_args: dict,
        logging_obj=None,
        stream: bool = False,
        input: Optional[Any] = None
    ) -> Generator[requests.Response, None, None]:
        """
        Returns a context manager that yields the response from the request.
        """
        if stream:
            request_params["stream"] = stream
        request_params["timeout"] = self.timeout
        request_params = self._pre_call(
            request_params, api_args,
            input=input,
            logging_obj=logging_obj,
            is_async=False,
        )
        try:
            retries = 0
            while retries < 3:
                resp = requests.request(**request_params)
                if resp.status_code in [429, 503, 504, 520]:
                    # to handle rate limiting and service unavailable errors
                    # see: ibm_watsonx_ai.foundation_models.inference.base_model_inference.BaseModelInference._send_inference_payload
                    time.sleep(2**retries)
                    retries += 1
                else:
                    break
            if not resp.ok:
                raise WatsonXAIError(
                    status_code=resp.status_code,
                    message=f"Error {resp.status_code} ({resp.reason}): {resp.text}",
                )
            yield resp
        except Exception as e:
            raise WatsonXAIError(status_code=500, message=str(e))
        if not stream:
            self._post_call(resp, request_params, input=input, logging_obj=logging_obj)

    @asynccontextmanager
    async def async_request(
        self,
        request_params: dict,
        api_args,
        logging_obj=None,
        stream: bool = False,
        input: Optional[Any] = None
    ) -> AsyncGenerator[httpx.Response, None]:
        if stream:
            request_params["stream"] = stream
        request_params = self._pre_call(
            request_params, api_args,
            input=input, 
            logging_obj=logging_obj,
            is_async=True,
        )
        try:
            method = request_params.pop("method")
            retries = 0
            while retries < 3:
                if method.upper() == "POST":
                    resp = await self.async_handler.post(**request_params)
                else:
                    resp = await self.async_handler.get(**request_params)
                if resp.status_code in [429, 503, 504, 520]:
                    # to handle rate limiting and service unavailable errors
                    # see: ibm_watsonx_ai.foundation_models.inference.base_model_inference.BaseModelInference._send_inference_payload
                    await asyncio.sleep(2**retries)
                    retries += 1
                else:
                    break
            if resp.is_error:
                raise WatsonXAIError(
                    status_code=resp.status_code,
                    message=f"Error {resp.status_code} ({resp.reason}): {resp.text}",
                )
            yield resp
            # await async_handler.close()
        except Exception as e:
            raise e
            # raise WatsonXAIError(status_code=500, message=str(e))
        if not stream:
            self._post_call(resp, request_params, input=input, logging_obj=logging_obj)

    def get_auth_token(self, api_key=None, async_:bool=False, **params):
        
        try:
            req_params = self._get_auth_token_request_params(api_key=api_key, **params)
        except Exception as e:
            raise WatsonXAIError(
                status_code=500,
                message=f"Error generating auth token: {str(e)}",
            )

        async def get_token_async():
            if isinstance(self._auth_token, IBMAuthToken) and not self._auth_token.is_expired:
                return self._auth_token
            # renew the token
            try:
                resp = await self.async_handler.post(**req_params)
                resp.raise_for_status()
                json_data = resp.json()
                self._auth_token = IBMAuthToken(
                    access_token=json_data["access_token"],
                    expiration=json_data["expiration"],
                )
            except Exception as e:
                raise WatsonXAIError(
                    status_code=500,
                    message=f"Error generating auth token: {str(e)}",
                )
            return self._auth_token
        
        def get_token():
            if isinstance(self._auth_token, IBMAuthToken) and not self._auth_token.is_expired:
                return self._auth_token
            # renew the token
            try:
                resp = requests.post(**req_params)
                resp.raise_for_status()
                json_data = resp.json()
                self._auth_token = IBMAuthToken(
                    access_token=json_data["access_token"],
                    expiration=json_data["expiration"],
                )
            except Exception as e:
                raise WatsonXAIError(
                    status_code=500,
                    message=f"Error generating auth token: {str(e)}",
                )
            return self._auth_token

        if async_:
            return get_token_async()
        return get_token()
    
    def _get_auth_token_request_params(self, **kwargs):
        auth_type = (get_secret("WATSONX_AUTH_TYPE") or "iam").lower()
        if auth_type == "iam":
            api_key = kwargs.pop("api_key", None)
            if api_key is None:
                api_key = get_secret("WATSONX_APIKEY") or get_secret("WATSONX_API_KEY") or get_secret("WX_API_KEY")
            if api_key is None:
                raise ValueError("API key is required for IAM Auth token generation")
            headers = {}
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            data = {
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": api_key,
            }
            iam_token_url = get_secret("WATSONX_IAM_TOKEN_URL") or "https://iam.cloud.ibm.com/identity/token"
            req_params = dict(url=iam_token_url, data=data, headers=headers, params=kwargs)
            return req_params
        elif auth_type == "cp4d":
            raise NotImplementedError("CP4D authentication is not yet supported. Pass token=<your-bearer-token> instead.")
            # cp4d_username = kwargs.pop("username", get_secret("WATSONX_USERNAME"))
            # cp4d_password = kwargs.pop("password", get_secret("WATSONX_PASSWORD"))
            # cp4d_url = kwargs.pop("url", get_secret("WATSONX_CP4D_URL")) or get_secret("WATSONX_URL")
            # if cp4d_username is None or cp4d_password is None or cp4d_url is None:
            #     raise ValueError(
            #         "CP4D authentication requires passing username, password and url "
            #         "(env: WATSONX_USERNAME, WATSONX_PASSWORD, WATSONX_CP4D_URL)"
            #     )

    def _pre_call(
        self,
        request_params: dict,
        api_args: dict,
        input: Optional[Any] = None,
        logging_obj = None,
        is_async: Optional[bool] = False
    ):
        """
        Set the Authorization header for the request and log the pre-call details.
        """

        if request_params.get("headers", {}).get("Authorization") is None:
            if request_params.get('headers') is None:
                request_params['headers'] = {}
            if api_args.get("token") is not None:
                access_token = str(api_args.get("token"))
            else:
                # iam token is generated sync to avoid issues with multiple async calls generating multiple tokens
                access_token = self.get_auth_token(api_args.get('api_key')).access_token
            request_params["headers"]["Authorization"] = f"Bearer {access_token}"
        
        if logging_obj is not None:
            request_str = (
                f"response = {'await ' if is_async else ''}{request_params['method']}(\n"
                f"\turl={request_params['url']},\n"
                f"\tjson={request_params.get('json')},\n"
                f")"
            )
            logging_obj.pre_call(
                input=input,
                api_key=request_params["headers"].get("Authorization"),
                additional_args={
                    "complete_input_dict": request_params.get("json"),
                    "request_str": request_str,
                },
            )  
        return request_params

    def _post_call(self, resp, request_params, input, logging_obj=None):
        """
        Log the post-call details.
        """
        if logging_obj is None:
            return
        logging_obj.post_call(
            input=input,
            api_key=request_params["headers"].get("Authorization"),
            original_response=json.dumps(resp.json()),
            additional_args={
                "status_code": resp.status_code,
                "complete_input_dict": request_params.get(
                    "data", request_params.get("json")
                ),
            },
        )
import asyncio
import json  # noqa: E401
import time
import types
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    AsyncContextManager,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    ContextManager,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Union,
)

import httpx  # type: ignore
import requests  # type: ignore

import litellm
from litellm.litellm_core_utils.prompt_templates import factory as ptf
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.llms.watsonx import WatsonXAIEndpoint
from litellm.utils import EmbeddingResponse, ModelResponse, Usage, map_finish_reason

from ...base import BaseLLM
from ..common_utils import WatsonXAIError, _get_api_params, generate_iam_token
from .transformation import IBMWatsonXAIConfig


def convert_messages_to_prompt(model, messages, provider, custom_prompt_dict) -> str:
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
    elif provider == "ibm-mistralai":
        prompt = ptf.mistral_instruct_pt(messages=messages)
    else:
        prompt: str = ptf.prompt_factory(  # type: ignore
            model=model, messages=messages, custom_llm_provider="watsonx"
        )
    return prompt


class IBMWatsonXAI(BaseLLM):
    """
    Class to interface with IBM watsonx.ai API for text generation and embeddings.

    Reference: https://cloud.ibm.com/apidocs/watsonx-ai
    """

    api_version = "2024-03-13"

    def __init__(self) -> None:
        super().__init__()

    def _prepare_text_generation_req(
        self,
        model_id: str,
        messages: List[AllMessageValues],
        prompt: str,
        stream: bool,
        optional_params: dict,
        print_verbose: Optional[Callable] = None,
    ) -> dict:
        """
        Get the request parameters for text generation.
        """
        api_params = _get_api_params(optional_params, print_verbose=print_verbose)
        # build auth headers
        api_token = api_params.get("token")
        self.token = api_token
        headers = IBMWatsonXAIConfig().validate_environment(
            headers={},
            model=model_id,
            messages=messages,
            optional_params=optional_params,
            api_key=api_token,
        )
        extra_body_params = optional_params.pop("extra_body", {})
        optional_params.update(extra_body_params)
        # init the payload to the text generation call
        payload = {
            "input": prompt,
            "moderations": optional_params.pop("moderations", {}),
            "parameters": optional_params,
        }
        request_params = dict(version=api_params["api_version"])
        # text generation endpoint deployment or model / stream or not
        if model_id.startswith("deployment/"):
            # deployment models are passed in as 'deployment/<deployment_id>'
            if api_params.get("space_id") is None:
                raise WatsonXAIError(
                    status_code=401,
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
            payload["project_id"] = api_params["project_id"]
            endpoint = (
                WatsonXAIEndpoint.TEXT_GENERATION_STREAM
                if stream
                else WatsonXAIEndpoint.TEXT_GENERATION
            )
        url = api_params["url"].rstrip("/") + endpoint
        return dict(
            method="POST", url=url, headers=headers, json=payload, params=request_params
        )

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
        logging_obj: Any,
        optional_params: dict,
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

        # Load default configs
        config = IBMWatsonXAIConfig.get_config()
        for k, v in config.items():
            if k not in optional_params:
                optional_params[k] = v

        # Make prompt to send to model
        provider = model.split("/")[0]
        # model_name = "/".join(model.split("/")[1:])
        prompt = convert_messages_to_prompt(
            model, messages, provider, custom_prompt_dict
        )
        model_response.model = model

        def process_stream_response(
            stream_resp: Union[Iterator[str], AsyncIterator],
        ) -> CustomStreamWrapper:
            streamwrapper = litellm.CustomStreamWrapper(
                stream_resp,
                model=model,
                custom_llm_provider="watsonx",
                logging_obj=logging_obj,
            )
            return streamwrapper

        # create the function to manage the request to watsonx.ai
        self.request_manager = RequestManager(logging_obj)

        def handle_text_request(request_params: dict) -> ModelResponse:
            with self.request_manager.request(
                request_params,
                input=prompt,
                timeout=timeout,
            ) as resp:
                json_resp = resp.json()

            return self._process_text_gen_response(json_resp, model_response)

        async def handle_text_request_async(request_params: dict) -> ModelResponse:
            async with self.request_manager.async_request(
                request_params,
                input=prompt,
                timeout=timeout,
            ) as resp:
                json_resp = resp.json()
            return self._process_text_gen_response(json_resp, model_response)

        def handle_stream_request(request_params: dict) -> CustomStreamWrapper:
            # stream the response - generated chunks will be handled
            # by litellm.utils.CustomStreamWrapper.handle_watsonx_stream
            with self.request_manager.request(
                request_params,
                stream=True,
                input=prompt,
                timeout=timeout,
            ) as resp:
                streamwrapper = process_stream_response(resp.iter_lines())
            return streamwrapper

        async def handle_stream_request_async(
            request_params: dict,
        ) -> CustomStreamWrapper:
            # stream the response - generated chunks will be handled
            # by litellm.utils.CustomStreamWrapper.handle_watsonx_stream
            async with self.request_manager.async_request(
                request_params,
                stream=True,
                input=prompt,
                timeout=timeout,
            ) as resp:
                streamwrapper = process_stream_response(resp.aiter_lines())
            return streamwrapper

        try:
            ## Get the response from the model
            req_params = self._prepare_text_generation_req(
                model_id=model,
                prompt=prompt,
                messages=messages,
                stream=stream,
                optional_params=optional_params,
                print_verbose=print_verbose,
            )
            if stream and (acompletion is True):
                # stream and async text generation
                return handle_stream_request_async(req_params)
            elif stream:
                # streaming text generation
                return handle_stream_request(req_params)
            elif acompletion is True:
                # async text generation
                return handle_text_request_async(req_params)
            else:
                # regular text generation
                return handle_text_request(req_params)
        except WatsonXAIError as e:
            raise e
        except Exception as e:
            raise WatsonXAIError(status_code=500, message=str(e))

    def _process_embedding_response(
        self, json_resp: dict, model_response: Optional[EmbeddingResponse] = None
    ) -> EmbeddingResponse:
        if model_response is None:
            model_response = EmbeddingResponse(model=json_resp.get("model_id", None))
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

    def embedding(
        self,
        model: str,
        input: Union[list, str],
        model_response: EmbeddingResponse,
        api_key: Optional[str],
        logging_obj: Any,
        optional_params: dict,
        encoding=None,
        print_verbose=None,
        aembedding=None,
    ) -> EmbeddingResponse:
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

        # Load auth variables from environment variables
        if isinstance(input, str):
            input = [input]
        if api_key is not None:
            optional_params["api_key"] = api_key
        api_params = _get_api_params(optional_params)
        # build auth headers
        api_token = api_params.get("token")
        self.token = api_token
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # init the payload to the text generation call
        payload = {
            "inputs": input,
            "model_id": model,
            "project_id": api_params["project_id"],
            "parameters": optional_params,
        }
        request_params = dict(version=api_params["api_version"])
        url = api_params["url"].rstrip("/") + WatsonXAIEndpoint.EMBEDDINGS
        req_params = {
            "method": "POST",
            "url": url,
            "headers": headers,
            "json": payload,
            "params": request_params,
        }
        request_manager = RequestManager(logging_obj)

        def handle_embedding(request_params: dict) -> EmbeddingResponse:
            with request_manager.request(request_params, input=input) as resp:
                json_resp = resp.json()
            return self._process_embedding_response(json_resp, model_response)

        async def handle_aembedding(request_params: dict) -> EmbeddingResponse:
            async with request_manager.async_request(
                request_params, input=input
            ) as resp:
                json_resp = resp.json()
            return self._process_embedding_response(json_resp, model_response)

        try:
            if aembedding is True:
                return handle_aembedding(req_params)  # type: ignore
            else:
                return handle_embedding(req_params)
        except WatsonXAIError as e:
            raise e
        except Exception as e:
            raise WatsonXAIError(status_code=500, message=str(e))

    def get_available_models(self, *, ids_only: bool = True, **params):
        api_params = _get_api_params(params)
        self.token = api_params["token"]
        headers = {
            "Authorization": f"Bearer {api_params['token']}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        request_params = dict(version=api_params["api_version"])
        url = api_params["url"].rstrip("/") + WatsonXAIEndpoint.AVAILABLE_MODELS
        req_params = dict(method="GET", url=url, headers=headers, params=request_params)
        with RequestManager(logging_obj=None).request(req_params) as resp:
            json_resp = resp.json()
        if not ids_only:
            return json_resp
        return [res["model_id"] for res in json_resp["resources"]]


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

    def __init__(self, logging_obj=None):
        self.logging_obj = logging_obj

    def pre_call(
        self,
        request_params: dict,
        input: Optional[Any] = None,
        is_async: Optional[bool] = False,
    ):
        if self.logging_obj is None:
            return
        request_str = (
            f"response = {'await ' if is_async else ''}{request_params['method']}(\n"
            f"\turl={request_params['url']},\n"
            f"\tjson={request_params.get('json')},\n"
            f")"
        )
        self.logging_obj.pre_call(
            input=input,
            api_key=request_params["headers"].get("Authorization"),
            additional_args={
                "complete_input_dict": request_params.get("json"),
                "request_str": request_str,
            },
        )

    def post_call(self, resp, request_params):
        if self.logging_obj is None:
            return
        self.logging_obj.post_call(
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

    @contextmanager
    def request(
        self,
        request_params: dict,
        stream: bool = False,
        input: Optional[Any] = None,
        timeout=None,
    ) -> Generator[requests.Response, None, None]:
        """
        Returns a context manager that yields the response from the request.
        """
        self.pre_call(request_params, input)
        if timeout:
            request_params["timeout"] = timeout
        if stream:
            request_params["stream"] = stream
        try:
            resp = requests.request(**request_params)
            if not resp.ok:
                raise WatsonXAIError(
                    status_code=resp.status_code,
                    message=f"Error {resp.status_code} ({resp.reason}): {resp.text}",
                )
            yield resp
        except Exception as e:
            raise WatsonXAIError(status_code=500, message=str(e))
        if not stream:
            self.post_call(resp, request_params)

    @asynccontextmanager
    async def async_request(
        self,
        request_params: dict,
        stream: bool = False,
        input: Optional[Any] = None,
        timeout=None,
    ) -> AsyncGenerator[httpx.Response, None]:
        self.pre_call(request_params, input, is_async=True)
        if timeout:
            request_params["timeout"] = timeout
        if stream:
            request_params["stream"] = stream
        try:
            self.async_handler = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.WATSONX,
                params={
                    "timeout": httpx.Timeout(
                        timeout=request_params.pop("timeout", 600.0), connect=5.0
                    ),
                },
            )
            if "json" in request_params:
                request_params["data"] = json.dumps(request_params.pop("json", {}))
            method = request_params.pop("method")
            retries = 0
            resp: Optional[httpx.Response] = None
            while retries < 3:
                if method.upper() == "POST":
                    resp = await self.async_handler.post(**request_params)
                else:
                    resp = await self.async_handler.get(**request_params)
                if resp is not None and resp.status_code in [429, 503, 504, 520]:
                    # to handle rate limiting and service unavailable errors
                    # see: ibm_watsonx_ai.foundation_models.inference.base_model_inference.BaseModelInference._send_inference_payload
                    await asyncio.sleep(2**retries)
                    retries += 1
                else:
                    break
            if resp is None:
                raise WatsonXAIError(
                    status_code=500,
                    message="No response from the server",
                )
            if resp.is_error:
                error_reason = getattr(resp, "reason", "")
                raise WatsonXAIError(
                    status_code=resp.status_code,
                    message=f"Error {resp.status_code} ({error_reason}): {resp.text}",
                )
            yield resp
            # await async_handler.close()
        except Exception as e:
            raise e
            raise WatsonXAIError(status_code=500, message=str(e))
        if not stream:
            self.post_call(resp, request_params)

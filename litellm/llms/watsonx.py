import json, types, time  # noqa: E401
import asyncio
from datetime import datetime
from enum import Enum
from contextlib import asynccontextmanager, contextmanager
from typing import (
    Callable,
    Dict,
    Generator,
    AsyncGenerator,
    Iterator,
    AsyncIterator,
    Optional,
    Any,
    Union,
    List,
)

import httpx  # type: ignore
import requests  # type: ignore
import litellm
from litellm.utils import ModelResponse, Usage, get_secret
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

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
        prompt: str,
        stream: bool,
        optional_params: dict,
        print_verbose: Optional[Callable] = None,
    ) -> dict:
        """
        Get the request parameters for text generation.
        """
        api_params = self._get_api_params(optional_params, print_verbose=print_verbose)
        # build auth headers
        api_token = api_params.get("token")

        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
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
                    url=api_params["url"],
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

    def _get_api_params(
        self,
        params: dict,
        print_verbose: Optional[Callable] = None,
        generate_token: Optional[bool] = True,
    ) -> dict:
        """
        Find watsonx.ai credentials in the params or environment variables and return the headers for authentication.
        """
        # Load auth variables from params
        url = params.pop("url", params.pop("api_base", params.pop("base_url", None)))
        api_key = params.pop("apikey", None)
        token = params.pop("token", None)
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
                message="Error: Watsonx URL not set. Set WX_URL in environment variables or pass in as a parameter.",
            )
        if token is None and api_key is not None and generate_token:
            # generate the auth token
            if print_verbose is not None:
                print_verbose("Generating IAM token for Watsonx.ai")
            token = self.generate_iam_token(api_key)
        elif token is None and api_key is None:
            raise WatsonXAIError(
                status_code=401,
                url=url,
                message="Error: API key or token not found. Set WX_API_KEY or WX_TOKEN in environment variables or pass in as a parameter.",
            )
        if project_id is None:
            raise WatsonXAIError(
                status_code=401,
                url=url,
                message="Error: Watsonx project_id not set. Set WX_PROJECT_ID in environment variables or pass in as a parameter.",
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
        model_response["choices"][0]["message"]["content"] = generated_text
        model_response["finish_reason"] = json_resp["results"][0]["stop_reason"]
        if json_resp.get("created_at"):
            model_response["created"] = datetime.fromisoformat(
                json_resp["created_at"]
            ).timestamp()
        else:
            model_response["created"] = int(time.time())
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
        client=None,
    ):
        """
        Send a text generation request to the IBM Watsonx.ai API.
        Reference: https://cloud.ibm.com/apidocs/watsonx-ai#text-generation
        """
        if client is not None:
            print_verbose(
                "'client' parameter passed: Using IBM Watsonx.ai client for text generation."
            )
            return self.client_completion(
                client=client,
                model=model,
                messages=messages,
                custom_prompt_dict=custom_prompt_dict,
                model_response=model_response,
                print_verbose=print_verbose,
                encoding=encoding,
                logging_obj=logging_obj,
                optional_params=optional_params,
                acompletion=acompletion,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                timeout=timeout,
            )

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
        model_response["model"] = model

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

        def handle_stream_request(request_params: dict) -> litellm.CustomStreamWrapper:
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
        ) -> litellm.CustomStreamWrapper:
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

    def client_completion(
        self,
        client,
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
        Send a text generation request to the IBM Watsonx.ai API via an ibm_watsonx_ai.client.Client object.
        """
        from ibm_watsonx_ai.foundation_models import ModelInference
        # from ibm_watsonx_ai.client import APIClient

        # client: APIClient
        stream = optional_params.pop("stream", False)
        # Load default configs
        config = IBMWatsonXAIConfig.get_config()
        for k, v in config.items():
            if k not in optional_params:
                optional_params[k] = v

        # Make prompt to send to model
        provider = model.split("/")[0]
        prompt = convert_messages_to_prompt(
            model, messages, provider, custom_prompt_dict
        )
        api_params = self._get_api_params(
            optional_params, print_verbose=print_verbose, generate_token=False
        )
        project_id = (
            getattr(client, "default_project_id", None) or
            getattr(client, "project_id", None) or 
            api_params.get("project_id")
        )
        model_inference: ModelInference = self.init_watsonx_model_with_client(
            model_id=model,
            watsonx_client=client,
            project_id=project_id,
            space_id=api_params.get("space_id"),
            verify=api_params.get("verify", False),
            validate=api_params.get("validate", False),
            model_params=optional_params,
        )
        model_response["model"] = model

        def handle_text_request(prompt: str) -> ModelResponse:
            raw_resp = model_inference.generate(prompt, async_mode=False)
            return self._process_text_gen_response(raw_resp, model_response)

        async def handle_text_request_async(prompt: str) -> ModelResponse:
            raw_resp = model_inference.generate(prompt, async_mode=True)
            return self._process_text_gen_response(raw_resp, model_response)

        def handle_stream_request(prompt: str) -> litellm.CustomStreamWrapper:
            # stream the response - generated chunks will be handled
            # by litellm.utils.CustomStreamWrapper.handle_watsonx_stream
            stream = model_inference.generate_text_stream(prompt, raw_response=True)
            streamwrapper = litellm.CustomStreamWrapper(
                stream,
                model=model,
                custom_llm_provider="watsonx",
                logging_obj=logging_obj,
            )
            return streamwrapper

        async def handle_stream_request_async(
            prompt: str,
        ) -> litellm.CustomStreamWrapper:
            # stream the response - generated chunks will be handled
            # by litellm.utils.CustomStreamWrapper.handle_watsonx_stream
            stream = model_inference.generate_text_stream(prompt, raw_response=True)
            streamwrapper = litellm.CustomStreamWrapper(
                stream,
                model=model,
                custom_llm_provider="watsonx",
                logging_obj=logging_obj,
            )
            return streamwrapper

        if stream and (acompletion is True):
            # stream and async text generation
            return handle_stream_request_async(prompt)
        elif stream:
            # streaming text generation
            return handle_stream_request(prompt)
        elif acompletion is True:
            # async text generation
            return handle_text_request_async(prompt)
        else:
            # regular text generation
            return handle_text_request(prompt)

    def _process_embedding_response(self, json_resp: dict, model_response:Union[ModelResponse,None]=None) -> ModelResponse:
        if model_response is None:
            model_response = ModelResponse(model=json_resp.get("model_id", None))
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
        model_response["object"] = "list"
        model_response["data"] = embedding_response
        input_tokens = json_resp.get("input_token_count", 0)
        model_response.usage = Usage(
            prompt_tokens=input_tokens,
            completion_tokens=0,
            total_tokens=input_tokens,
        )
        return model_response

    def embedding(
        self,
        model: str,
        input: Union[list, str],
        api_key: Optional[str] = None,
        logging_obj=None,
        model_response=None,
        optional_params=None,
        encoding=None,
        print_verbose=None,
        aembedding=None,
        client=None,
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

        model_response['model'] = model
        if client is not None:
            print_verbose(
                "'client' parameter passed: Using IBM Watsonx.ai client for text embeddings."
            )
            return self.client_embedding(
                client=client,
                model=model,
                input=input,
                api_key=api_key,
                logging_obj=logging_obj,
                model_response=model_response,
                optional_params=optional_params,
                encoding=encoding,
                aembedding=aembedding,
            )
        
        # Load auth variables from environment variables
        if isinstance(input, str):
            input = [input]
        if api_key is not None:
            optional_params["api_key"] = api_key
        api_params = self._get_api_params(optional_params)
        # build auth headers
        api_token = api_params.get("token")
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

        def handle_embedding(request_params: dict) -> ModelResponse:
            with request_manager.request(request_params, input=input) as resp:
                json_resp = resp.json()
            logging_obj.post_call(
                original_response=json.dumps(json_resp),
                input=input,
                api_key=api_params.get("api_key"),
            )
            return self._process_embedding_response(json_resp, model_response)

        async def handle_aembedding(request_params: dict) -> ModelResponse:
            async with request_manager.async_request(
                request_params, input=input
            ) as resp:
                json_resp = resp.json()
            logging_obj.post_call(
                original_response=json.dumps(json_resp),
                input=input,
                api_key=api_params.get("api_key"),
            )
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
        
    def client_embedding(
        self,
        client,
        model: str,
        input: Union[list, str],
        api_key: Optional[str] = None,
        logging_obj=None,
        model_response=None,
        optional_params=None,
        encoding=None,
        aembedding=None,
    ):
        from ibm_watsonx_ai.foundation_models import Embeddings

        if optional_params is None:
            optional_params = {}
        
        concurrency_limit = optional_params.pop("concurrency_limit", 10)
        project_id = (
            getattr(client, "default_project_id", None) or
            getattr(client, "project_id", None) or
            optional_params.pop("project_id", None)
        )

        embeddings = Embeddings(
            model_id=model,
            api_client=client,
            project_id=project_id,
            space_id=optional_params.get("space_id"),
            params=optional_params,
        )
        if isinstance(input, str):
            input = [input]
        
        async def handle_aembedding(input: list) -> ModelResponse:
            results:List[List[float]] = embeddings.embed_documents(input, concurrency_limit=concurrency_limit)
            json_resp = {"results": [{"embedding": result} for result in results], "model_id": model}
            logging_obj.post_call(
                original_response=json.dumps(json_resp),
                input=input
            )
            return self._process_embedding_response(json_resp, model_response)
        
        def handle_embedding(input: list) -> ModelResponse:
            results:List[List[float]] = embeddings.embed_documents(input, concurrency_limit=concurrency_limit)
            json_resp = {"results": [{"embedding": result} for result in results], "model_id": model}
            logging_obj.post_call(
                original_response=json.dumps(json_resp),
                input=input
            )
            return self._process_embedding_response(json_resp, model_response)
        
        if aembedding is True:
            return handle_aembedding(input)
        else:
            return handle_embedding(input)
        

    def generate_iam_token(self, api_key=None, **params):
        headers = {}
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        if api_key is None:
            api_key = get_secret("WX_API_KEY") or get_secret("WATSONX_API_KEY")
        if api_key is None:
            raise ValueError("API key is required")
        headers["Accept"] = "application/json"
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": api_key,
        }
        response = httpx.post(
            "https://iam.cloud.ibm.com/identity/token", data=data, headers=headers
        )
        response.raise_for_status()
        json_data = response.json()
        iam_access_token = json_data["access_token"]
        self.token = iam_access_token
        return iam_access_token

    def get_available_models(self, *, ids_only: bool = True, **params):
        api_params = self._get_api_params(params)
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

    def init_watsonx_model_with_client(
        self,
        model_id,
        watsonx_client,
        project_id: Optional[str] = None,
        space_id: Optional[str] = None,
        verify: Optional[Union[bool, str]] = False,
        validate: Optional[bool] = False,
        model_params: Optional[dict] = None,
    ):
        """
        Initialize a watsonx.ai model for inference.
        Args:
        model_id (str): The model ID to use for inference. If this is a model deployed in a deployment space, the model_id should be in the format 'deployment/<deployment_id>' and the space_id to the deploymend space should be provided.
        wx_credentials (dict): A dictionary containing 'apikey' and 'url' keys for the watsonx.ai instance.
        watsonx_client (object): An instance of the ibm_watsonx_ai.APIClient class.
        project_id (str): The project ID for the watsonx.ai instance.
        space_id (str): The space ID for the deployment space.
        verify (bool): Whether to verify the SSL certificate of calls to the watsonx url.
        validate (bool): Whether to validate the model_id at initialization.
        watsonx_client (object): An instance of the ibm_watsonx_ai.APIClient class. If this is provided, the model will be initialized using the provided client.
        model_params (dict): A dictionary containing additional parameters to pass to the model (see IBMWatsonXConfig for a list of supported parameters).
        """

        from ibm_watsonx_ai.foundation_models import ModelInference
        # from ibm_watsonx_ai import APIClient

        # watsonx_client: APIClient = watsonx_client
        if project_id is None:
            project_id = getattr(watsonx_client, "default_project_id", None) or getattr(
                watsonx_client, "project_id", None
            )

        if model_id.startswith("deployment/"):
            # deployment models are passed in as 'deployment/<deployment_id>'
            assert space_id is not None, "space_id is required for deployment models"
            deployment_id = "/".join(model_id.split("/")[1:])
            model_id = None
        else:
            space_id = None
            deployment_id = None

        model = ModelInference(
            model_id=model_id,
            params=model_params,
            api_client=watsonx_client,
            project_id=project_id,
            deployment_id=deployment_id,
            verify=verify,
            validate=validate,
            space_id=space_id,
        )
        return model


class RequestManager:
    """
    Returns a context manager that manages the response from the request.
    if async_ is True, returns an async context manager, otherwise returns a regular context manager.

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
    ):
        if self.logging_obj is None:
            return
        request_str = (
            f"response = {request_params['method']}(\n"
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
            self.post_call(resp, request_params)

    @asynccontextmanager
    async def async_request(
        self,
        request_params: dict,
        stream: bool = False,
        input: Optional[Any] = None,
        timeout=None,
    ) -> AsyncGenerator[httpx.Response, None]:
        self.pre_call(request_params, input)
        if timeout:
            request_params["timeout"] = timeout
        if stream:
            request_params["stream"] = stream
        try:
            # async with AsyncHTTPHandler(timeout=timeout) as client:
            self.async_handler = AsyncHTTPHandler(
                timeout=httpx.Timeout(
                    timeout=request_params.pop("timeout", 600.0), connect=5.0
                ),
            )
            # async_handler.client.verify = False
            if "json" in request_params:
                request_params["data"] = json.dumps(request_params.pop("json", {}))
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
            raise WatsonXAIError(status_code=500, message=str(e))
        if not stream:
            self.post_call(resp, request_params)

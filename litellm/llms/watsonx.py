from enum import Enum
import json, types, time  # noqa: E401
from contextlib import contextmanager
from typing import Callable, Dict, Optional, Any, Union, List

import httpx  # type: ignore
import requests  # type: ignore
import litellm
from litellm.utils import ModelResponse, get_secret, Usage

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


class IBMWatsonXAI(BaseLLM):
    """
    Class to interface with IBM Watsonx.ai API for text generation and embeddings.

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
        self, params: dict, print_verbose: Optional[Callable] = None
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
        if token is None and api_key is not None:
            # generate the auth token
            if print_verbose:
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
        litellm_params: Optional[dict] = None,
        logger_fn=None,
        timeout: Optional[float] = None,
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

        def process_text_request(request_params: dict) -> ModelResponse:
            with self._manage_response(
                request_params, logging_obj=logging_obj, input=prompt, timeout=timeout
            ) as resp:
                json_resp = resp.json()

            generated_text = json_resp["results"][0]["generated_text"]
            prompt_tokens = json_resp["results"][0]["input_token_count"]
            completion_tokens = json_resp["results"][0]["generated_token_count"]
            model_response["choices"][0]["message"]["content"] = generated_text
            model_response["finish_reason"] = json_resp["results"][0]["stop_reason"]
            model_response["created"] = int(time.time())
            model_response["model"] = model
            setattr(
                model_response,
                "usage",
                Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
            return model_response

        def process_stream_request(
            request_params: dict,
        ) -> litellm.CustomStreamWrapper:
            # stream the response - generated chunks will be handled
            # by litellm.utils.CustomStreamWrapper.handle_watsonx_stream
            with self._manage_response(
                request_params,
                logging_obj=logging_obj,
                stream=True,
                input=prompt,
                timeout=timeout,
            ) as resp:
                response = litellm.CustomStreamWrapper(
                    resp.iter_lines(),
                    model=model,
                    custom_llm_provider="watsonx",
                    logging_obj=logging_obj,
                )
            return response

        try:
            ## Get the response from the model
            req_params = self._prepare_text_generation_req(
                model_id=model,
                prompt=prompt,
                stream=stream,
                optional_params=optional_params,
                print_verbose=print_verbose,
            )
            if stream:
                return process_stream_request(req_params)
            else:
                return process_text_request(req_params)
        except WatsonXAIError as e:
            raise e
        except Exception as e:
            raise WatsonXAIError(status_code=500, message=str(e))

    def embedding(
        self,
        model: str,
        input: Union[list, str],
        api_key: Optional[str] = None,
        logging_obj=None,
        model_response=None,
        optional_params=None,
        encoding=None,
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
        # request = httpx.Request(
        #     "POST", url, headers=headers, json=payload, params=request_params
        # )
        req_params = {
            "method": "POST",
            "url": url,
            "headers": headers,
            "json": payload,
            "params": request_params,
        }
        with self._manage_response(
            req_params, logging_obj=logging_obj, input=input
        ) as resp:
            json_resp = resp.json()

        results = json_resp.get("results", [])
        embedding_response = []
        for idx, result in enumerate(results):
            embedding_response.append(
                {"object": "embedding", "index": idx, "embedding": result["embedding"]}
            )
        model_response["object"] = "list"
        model_response["data"] = embedding_response
        model_response["model"] = model
        input_tokens = json_resp.get("input_token_count", 0)
        model_response.usage = Usage(
            prompt_tokens=input_tokens, completion_tokens=0, total_tokens=input_tokens
        )
        return model_response

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

    @contextmanager
    def _manage_response(
        self,
        request_params: dict,
        logging_obj: Any,
        stream: bool = False,
        input: Optional[Any] = None,
        timeout: Optional[float] = None,
    ):
        request_str = (
            f"response = {request_params['method']}(\n"
            f"\turl={request_params['url']},\n"
            f"\tjson={request_params['json']},\n"
            f")"
        )
        logging_obj.pre_call(
            input=input,
            api_key=request_params["headers"].get("Authorization"),
            additional_args={
                "complete_input_dict": request_params["json"],
                "request_str": request_str,
            },
        )
        if timeout:
            request_params["timeout"] = timeout
        try:
            if stream:
                resp = requests.request(
                    **request_params,
                    stream=True,
                )
                resp.raise_for_status()
                yield resp
            else:
                resp = requests.request(**request_params)
                resp.raise_for_status()
                yield resp
        except Exception as e:
            raise WatsonXAIError(status_code=500, message=str(e))
        if not stream:
            logging_obj.post_call(
                input=input,
                api_key=request_params["headers"].get("Authorization"),
                original_response=json.dumps(resp.json()),
                additional_args={
                    "status_code": resp.status_code,
                    "complete_input_dict": request_params["json"],
                },
            )

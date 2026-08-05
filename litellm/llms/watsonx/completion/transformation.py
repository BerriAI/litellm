import time
from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final

import httpx

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.types.llms.openai import AllMessageValues, ChatCompletionUsageBlock
from litellm.types.llms.watsonx import WatsonXAIEndpoint
from litellm.types.utils import GenericStreamingChunk, ModelResponse, Usage
from litellm.utils import map_finish_reason

from ...base_llm.chat.transformation import BaseConfig
from ..common_utils import (
    IBMWatsonXMixin,
    WatsonXAIError,
    _get_api_params,
    convert_watsonx_messages_to_prompt,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class IBMWatsonXAIConfig(IBMWatsonXMixin, BaseConfig):
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

    decoding_method: str | None = "sample"
    temperature: float | None = None
    max_new_tokens: int | None = None  # litellm.max_tokens
    min_new_tokens: int | None = None
    length_penalty: dict | None = None  # e.g {"decay_factor": 2.5, "start_index": 5}
    stop_sequences: list[str] | None = None  # e.g ["}", ")", "."]
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    truncate_input_tokens: int | None = None
    include_stop_sequences: bool | None = False
    return_options: dict[str, bool] | None = None
    random_seed: int | None = None  # e.g 42
    moderations: dict | None = None
    stream: bool | None = False

    def __init__(
        self,
        decoding_method: str | None = None,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        min_new_tokens: int | None = None,
        length_penalty: dict | None = None,
        stop_sequences: list[str] | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        truncate_input_tokens: int | None = None,
        include_stop_sequences: bool | None = None,
        return_options: dict | None = None,
        random_seed: int | None = None,
        moderations: dict | None = None,
        stream: bool | None = None,
        **kwargs,
    ) -> None:
        locals_: Final = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def is_watsonx_text_param(self, param: str) -> bool:
        """
        Determine if user passed in a watsonx.ai text generation param
        """
        text_generation_params: Final = [
            "decoding_method",
            "max_new_tokens",
            "min_new_tokens",
            "length_penalty",
            "stop_sequences",
            "top_k",
            "repetition_penalty",
            "truncate_input_tokens",
            "include_stop_sequences",
            "return_options",
            "random_seed",
            "moderations",
            "decoding_method",
            "min_tokens",
        ]

        return param in text_generation_params

    def get_supported_openai_params(self, model: str):
        return [
            "temperature",  # equivalent to temperature
            "max_tokens",  # equivalent to max_new_tokens
            "top_p",  # equivalent to top_p
            "frequency_penalty",  # equivalent to repetition_penalty
            "stop",  # equivalent to stop_sequences
            "seed",  # equivalent to random_seed
            "stream",  # equivalent to stream
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        extra_body: Final = {}
        for k, v in non_default_params.items():
            if k == "max_tokens":
                optional_params["max_new_tokens"] = v
            elif k == "stream":
                optional_params["stream"] = v
            elif k == "temperature":
                optional_params["temperature"] = v
            elif k == "top_p":
                optional_params["top_p"] = v
            elif k == "frequency_penalty":
                optional_params["repetition_penalty"] = v
            elif k == "seed":
                optional_params["random_seed"] = v
            elif k == "stop":
                optional_params["stop_sequences"] = v
            elif k == "decoding_method":
                extra_body["decoding_method"] = v
            elif k == "min_tokens":
                extra_body["min_new_tokens"] = v
            elif k == "top_k":
                extra_body["top_k"] = v
            elif k == "truncate_input_tokens":
                extra_body["truncate_input_tokens"] = v
            elif k == "length_penalty":
                extra_body["length_penalty"] = v
            elif k == "time_limit":
                extra_body["time_limit"] = v
            elif k == "return_options":
                extra_body["return_options"] = v

        if extra_body:
            optional_params["extra_body"] = extra_body
        return optional_params

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
        mapped_params: Final = self.get_mapped_special_auth_params()

        for param, value in non_default_params.items():
            if param in mapped_params:
                optional_params[mapped_params[param]] = value
        return optional_params

    def get_eu_regions(self) -> list[str]:
        """
        Source: https://www.ibm.com/docs/en/watsonx/saas?topic=integrations-regional-availability
        """
        return [
            "eu-de",
            "eu-gb",
        ]

    def get_us_regions(self) -> list[str]:
        """
        Source: https://www.ibm.com/docs/en/watsonx/saas?topic=integrations-regional-availability
        """
        return [
            "us-south",
        ]

    def _build_request_payload(self, model: str, prompt: str, optional_params: dict) -> dict:
        """Shared logic to build request payload"""
        extra_body_params: Final = optional_params.pop("extra_body", {})
        optional_params.update(extra_body_params)
        watsonx_api_params: Final = _get_api_params(params=optional_params, model=model)
        watsonx_auth_payload: Final = self._prepare_payload(model=model, api_params=watsonx_api_params)

        return {
            "input": prompt,
            "moderations": optional_params.pop("moderations", {}),
            "parameters": optional_params,
            **watsonx_auth_payload,
        }

    async def atransform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """Async version of transform_request"""
        from litellm.llms.watsonx.common_utils import (
            aconvert_watsonx_messages_to_prompt,
        )

        provider: Final = model.split("/")[0]
        prompt: Final = await aconvert_watsonx_messages_to_prompt(
            model=model, messages=messages, provider=provider, custom_prompt_dict={}
        )
        return self._build_request_payload(model=model, prompt=prompt, optional_params=optional_params)

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """Sync version of transform_request"""
        provider: Final = model.split("/")[0]
        prompt: Final = convert_watsonx_messages_to_prompt(
            model=model, messages=messages, provider=provider, custom_prompt_dict={}
        )
        return self._build_request_payload(model=model, prompt=prompt, optional_params=optional_params)

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: str,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=raw_response.text,
        )

        json_resp: Final = raw_response.json()

        if "results" not in json_resp:
            raise WatsonXAIError(
                status_code=500,
                message=f"Error: Invalid response from Watsonx.ai API: {json_resp}",
            )
        if model_response is None:
            model_response = ModelResponse(model=json_resp.get("model_id", None))
        generated_text: Final = json_resp["results"][0]["generated_text"]
        prompt_tokens: Final = json_resp["results"][0]["input_token_count"]
        completion_tokens: Final = json_resp["results"][0]["generated_token_count"]
        model_response.choices[0].message.content = generated_text
        model_response.choices[0].finish_reason = map_finish_reason(json_resp["results"][0]["stop_reason"])
        if json_resp.get("created_at"):
            try:
                created_datetime = datetime.fromisoformat(json_resp["created_at"])
            except ValueError:
                # datetime.fromisoformat cannot handle 'Z' in Python 3.10
                created_datetime = datetime.fromisoformat(f"{json_resp['created_at'].rstrip('Z')}+00:00")
            model_response.created = int(created_datetime.timestamp())
        else:
            model_response.created = int(time.time())
        usage: Final = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)
        return model_response

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        url = self._get_base_url(api_base=api_base)
        if model.startswith("deployment/"):
            # deployment models are passed in as 'deployment/<deployment_id>'
            deployment_id: Final = "/".join(model.split("/")[1:])
            endpoint = (
                WatsonXAIEndpoint.DEPLOYMENT_TEXT_GENERATION_STREAM.value
                if stream
                else WatsonXAIEndpoint.DEPLOYMENT_TEXT_GENERATION.value
            )
            endpoint = endpoint.format(deployment_id=deployment_id)
        else:
            endpoint = WatsonXAIEndpoint.TEXT_GENERATION_STREAM if stream else WatsonXAIEndpoint.TEXT_GENERATION
        url = url.rstrip("/") + endpoint

        ## add api version
        url = self._add_api_version_to_url(url=url, api_version=optional_params.pop("api_version", None))
        return url

    def get_model_response_iterator(
        self,
        streaming_response: Iterator[str] | AsyncIterator[str] | ModelResponse,
        sync_stream: bool,
        json_mode: bool | None = False,
    ):
        return WatsonxTextCompletionResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class WatsonxTextCompletionResponseIterator(BaseModelResponseIterator):
    # def _handle_string_chunk(self, str_line: str) -> GenericStreamingChunk:
    #     return self.chunk_parser(json.loads(str_line))

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        try:
            results: Final = chunk.get("results", [])
            if len(results) > 0:
                text: Final = results[0].get("generated_text", "")
                finish_reason: Final = results[0].get("stop_reason")
                is_finished: Final = finish_reason != "not_finished"

                return GenericStreamingChunk(
                    text=text,
                    is_finished=is_finished,
                    finish_reason=finish_reason,
                    usage=ChatCompletionUsageBlock(
                        prompt_tokens=results[0].get("input_token_count", 0),
                        completion_tokens=results[0].get("generated_token_count", 0),
                        total_tokens=results[0].get("input_token_count", 0)
                        + results[0].get("generated_token_count", 0),
                    ),
                )
            return GenericStreamingChunk(
                text="",
                is_finished=False,
                finish_reason="stop",
                usage=None,
            )
        except Exception as e:
            raise e

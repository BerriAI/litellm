import time
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

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
        locals_ = locals().copy()
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
        text_generation_params = [
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
        non_default_params: Dict,
        optional_params: Dict,
        model: str,
        drop_params: bool,
    ) -> Dict:
        extra_body = {}
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

    def get_us_regions(self) -> List[str]:
        """
        Source: https://www.ibm.com/docs/en/watsonx/saas?topic=integrations-regional-availability
        """
        return [
            "us-south",
        ]

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: Dict,
        headers: Dict,
    ) -> Dict:
        provider = model.split("/")[0]
        prompt = convert_watsonx_messages_to_prompt(
            model=model,
            messages=messages,
            provider=provider,
            custom_prompt_dict={},
        )
        extra_body_params = optional_params.pop("extra_body", {})
        optional_params.update(extra_body_params)
        watsonx_api_params = _get_api_params(params=optional_params)

        watsonx_auth_payload = self._prepare_payload(
            model=model,
            api_params=watsonx_api_params,
        )

        # init the payload to the text generation call
        payload = {
            "input": prompt,
            "moderations": optional_params.pop("moderations", {}),
            "parameters": optional_params,
            **watsonx_auth_payload,
        }

        return payload

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: Dict,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: Dict,
        encoding: str,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=raw_response.text,
        )

        json_resp = raw_response.json()

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

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        url = self._get_base_url(api_base=api_base)
        if model.startswith("deployment/"):
            # deployment models are passed in as 'deployment/<deployment_id>'
            deployment_id = "/".join(model.split("/")[1:])
            endpoint = (
                WatsonXAIEndpoint.DEPLOYMENT_TEXT_GENERATION_STREAM.value
                if stream
                else WatsonXAIEndpoint.DEPLOYMENT_TEXT_GENERATION.value
            )
            endpoint = endpoint.format(deployment_id=deployment_id)
        else:
            endpoint = (
                WatsonXAIEndpoint.TEXT_GENERATION_STREAM
                if stream
                else WatsonXAIEndpoint.TEXT_GENERATION
            )
        url = url.rstrip("/") + endpoint

        ## add api version
        url = self._add_api_version_to_url(
            url=url, api_version=optional_params.pop("api_version", None)
        )
        return url

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
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
            results = chunk.get("results", [])
            if len(results) > 0:
                text = results[0].get("generated_text", "")
                finish_reason = results[0].get("stop_reason")
                is_finished = finish_reason != "not_finished"

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

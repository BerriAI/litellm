"""
Translates from OpenAI's `/v1/chat/completions` endpoint to Triton's `/generate` endpoint.
"""

import json
from typing import Any, Dict, List, Literal, Optional, Union

from httpx import Headers, Response

from litellm.litellm_core_utils.prompt_templates.factory import prompt_factory
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import (
    BaseConfig,
    BaseLLMException,
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    ChatCompletionToolCallChunk,
    ChatCompletionUsageBlock,
    Choices,
    GenericStreamingChunk,
    Message,
    ModelResponse,
)

from ..common_utils import TritonError


class TritonConfig(BaseConfig):
    """
    Base class for Triton configurations.

    Handles routing between /infer and /generate triton completion llms
    """

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[Dict, Headers]
    ) -> BaseLLMException:
        return TritonError(
            status_code=status_code, message=error_message, headers=headers
        )

    def validate_environment(
        self,
        headers: Dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: Dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict:
        return {"Content-Type": "application/json"}

    def get_supported_openai_params(self, model: str) -> List:
        return ["max_tokens", "max_completion_tokens"]

    def map_openai_params(
        self,
        non_default_params: Dict,
        optional_params: Dict,
        model: str,
        drop_params: bool,
    ) -> Dict:
        for param, value in non_default_params.items():
            if param == "max_tokens" or param == "max_completion_tokens":
                optional_params[param] = value
        return optional_params

    def transform_response(
        self,
        model: str,
        raw_response: Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: Dict,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: Dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        api_base = litellm_params.get("api_base", "")
        llm_type = self._get_triton_llm_type(api_base)
        if llm_type == "generate":
            return TritonGenerateConfig().transform_response(
                model=model,
                raw_response=raw_response,
                model_response=model_response,
                logging_obj=logging_obj,
                request_data=request_data,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                encoding=encoding,
                api_key=api_key,
                json_mode=json_mode,
            )
        elif llm_type == "infer":
            return TritonInferConfig().transform_response(
                model=model,
                raw_response=raw_response,
                model_response=model_response,
                logging_obj=logging_obj,
                request_data=request_data,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                encoding=encoding,
                api_key=api_key,
                json_mode=json_mode,
            )
        return model_response

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        api_base = litellm_params.get("api_base", "")
        llm_type = self._get_triton_llm_type(api_base)
        if llm_type == "generate":
            return TritonGenerateConfig().transform_request(
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
            )
        elif llm_type == "infer":
            return TritonInferConfig().transform_request(
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
            )
        return {}

    def _get_triton_llm_type(self, api_base: str) -> Literal["generate", "infer"]:
        if api_base.endswith("/generate"):
            return "generate"
        elif api_base.endswith("/infer"):
            return "infer"
        else:
            raise ValueError(f"Invalid Triton API base: {api_base}")


class TritonGenerateConfig(TritonConfig):
    """
    Transformations for triton /generate endpoint (This is a trtllm model)
    """

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        inference_params = optional_params.copy()
        stream = inference_params.pop("stream", False)
        data_for_triton: Dict[str, Any] = {
            "text_input": prompt_factory(model=model, messages=messages),
            "parameters": {
                "max_tokens": int(optional_params.get("max_tokens", 2000)),
                "bad_words": [""],
                "stop_words": [""],
            },
            "stream": bool(stream),
        }
        data_for_triton["parameters"].update(inference_params)
        return data_for_triton

    def transform_response(
        self,
        model: str,
        raw_response: Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: Dict,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: Dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise TritonError(
                message=raw_response.text, status_code=raw_response.status_code
            )
        model_response.choices = [
            Choices(index=0, message=Message(content=raw_response_json["text_output"]))
        ]

        return model_response


class TritonInferConfig(TritonGenerateConfig):
    """
    Transformations for triton /infer endpoint (his is an infer model with a custom model on triton)
    """

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:

        text_input = messages[0].get("content", "")
        data_for_triton = {
            "inputs": [
                {
                    "name": "text_input",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [text_input],
                }
            ]
        }

        for k, v in optional_params.items():
            if not (k == "stream" or k == "max_retries"):
                datatype = "INT32" if isinstance(v, int) else "BYTES"
                datatype = "FP32" if isinstance(v, float) else datatype
                data_for_triton["inputs"].append(
                    {"name": k, "shape": [1], "datatype": datatype, "data": [v]}
                )

        if "max_tokens" not in optional_params:
            data_for_triton["inputs"].append(
                {
                    "name": "max_tokens",
                    "shape": [1],
                    "datatype": "INT32",
                    "data": [20],
                }
            )
        return data_for_triton

    def transform_response(
        self,
        model: str,
        raw_response: Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: Dict,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: Dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise TritonError(
                message=raw_response.text, status_code=raw_response.status_code
            )

        _triton_response_data = raw_response_json["outputs"][0]["data"]
        triton_response_data: Optional[str] = None
        if isinstance(_triton_response_data, list):
            triton_response_data = "".join(_triton_response_data)
        else:
            triton_response_data = _triton_response_data

        model_response.choices = [
            Choices(
                index=0,
                message=Message(content=triton_response_data),
            )
        ]

        return model_response


class TritonResponseIterator(BaseModelResponseIterator):
    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        try:
            text = ""
            tool_use: Optional[ChatCompletionToolCallChunk] = None
            is_finished = False
            finish_reason = ""
            usage: Optional[ChatCompletionUsageBlock] = None
            provider_specific_fields = None
            index = int(chunk.get("index", 0))

            # set values
            text = chunk.get("text_output", "")
            finish_reason = chunk.get("stop_reason", "")
            is_finished = chunk.get("is_finished", False)

            return GenericStreamingChunk(
                text=text,
                tool_use=tool_use,
                is_finished=is_finished,
                finish_reason=finish_reason,
                usage=usage,
                index=index,
                provider_specific_fields=provider_specific_fields,
            )
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from chunk: {chunk}")

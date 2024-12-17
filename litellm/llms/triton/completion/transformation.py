"""
Translates from OpenAI's `/v1/chat/completions` endpoint to Triton's `/generate` endpoint.
"""

from typing import Any, Dict, List, Optional, Union

from httpx import Headers, Response

from litellm.litellm_core_utils.prompt_templates.factory import prompt_factory
from litellm.llms.base_llm.transformation import (
    BaseConfig,
    BaseLLMException,
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ..common_utils import TritonError


class TritonConfig(BaseConfig):
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

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[Dict, Headers]
    ) -> BaseLLMException:
        return TritonError(
            status_code=status_code, message=error_message, headers=headers
        )

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
        raise NotImplementedError(
            "response transformation done in handler.py. [TODO] Migrate here."
        )

    def validate_environment(
        self,
        headers: Dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: Dict,
        api_key: Optional[str] = None,
    ) -> Dict:
        return {"Content-Type": "application/json"}

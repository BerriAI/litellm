"""
Transformation logic from OpenAI /v1/embeddings format to BurnCloud's  `/v1/embeddings` format.
"""

import types
from typing import List, Optional, Union, cast
from litellm.secret_managers.main import get_secret_str

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm import BaseEmbeddingConfig
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse
from litellm.utils import is_base64_encoded

from ..common_utils import BurnCloudError


class BurnCloudEmbeddingConfig(BaseEmbeddingConfig):

    def __init__(
            self,
    ) -> None:
        locals_ = locals().copy()
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

    def get_complete_url(
            self,
            api_base: Optional[str],
            api_key: Optional[str],
            model: str,
            optional_params: dict,
            litellm_params: dict,
            stream: Optional[bool] = None,
    ) -> str:
        if api_base is None:
            api_base = get_secret_str("BURNCLOUD_API_BASE")

        # Remove trailing slashes and ensure clean base URL
        api_base = api_base.rstrip("/") if api_base else api_base

        # if endswith "/v1"
        if api_base and api_base.endswith("/v1"):
            api_base = f"{api_base}/embeddings"
        else:
            api_base = f"{api_base}/v1/embeddings"
        return api_base

    def validate_environment(
            self,
            headers: dict,
            model: str,
            messages: List[AllMessageValues],
            optional_params: dict,
            litellm_params: dict,
            api_key: Optional[str] = None,
            api_base: Optional[str] = None,
    ) -> dict:
        if api_key is None:
            api_key = get_secret_str("BURNCLOUD_API_KEY")

        default_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # If 'Authorization' is provided in headers, it overrides the default.
        if "Authorization" in headers:
            default_headers["Authorization"] = headers["Authorization"]

        # Merge other headers, overriding any default ones except Authorization
        return {**default_headers, **headers}

    def get_supported_openai_params(self, model: str):
        return ["dimensions","encoding_format"]

    def map_openai_params(
            self,
            non_default_params: dict,
            optional_params: dict,
            model: str,
            drop_params: bool,
    ):
        if "dimensions" in non_default_params:
            optional_params["dimensions"] = non_default_params["dimensions"]

        if "encoding_format" in non_default_params:
            optional_params["encoding_format"] = non_default_params["encoding_format"]
        return optional_params

    def transform_embedding_request(
            self,
            model: str,
            input: AllEmbeddingInputValues,
            optional_params: dict,
            headers: dict,
    ) -> dict:
        data = {"model": model, **optional_params}
        input = cast(List[str], input) if isinstance(input, List) else [input]
        if any((is_base64_encoded(x) for x in input)):
            transformed_input = []
            for value in input:
                if isinstance(value, str):
                    if is_base64_encoded(value):
                        img_data = value.split(",")[1]
                        transformed_input.append({"image": img_data})
                    else:
                        transformed_input.append({"text": value})
            data["input"] = transformed_input
        else:
            data["input"] = input
        return data

    def transform_embedding_response(
            self,
            model: str,
            raw_response: httpx.Response,
            model_response: EmbeddingResponse,
            logging_obj: LiteLLMLoggingObj,
            api_key: Optional[str],
            request_data: dict,
            optional_params: dict,
            litellm_params: dict,
    ) -> EmbeddingResponse:
        response_json = raw_response.json()
        ## LOGGING
        logging_obj.post_call(
            input=input,
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},
            original_response=response_json,
        )
        return EmbeddingResponse(**response_json)

    def get_error_class(
            self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return BurnCloudError(
            message=error_message, status_code=status_code, headers=headers
        )

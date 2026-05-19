"""
Transformation logic from OpenAI /v1/embeddings format to DashScope's /v1/embeddings format.

Supports
- text-embedding-v4
- text-embedding-v3

Endpoint
- https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings

Docs - https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api
"""

from typing import List, Optional, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage

from ..common_utils import DashScopeError

DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class DashScopeEmbeddingConfig(BaseEmbeddingConfig):
    """
    Reference: https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api

    DashScope exposes an OpenAI-compatible /v1/embeddings endpoint, so the
    request and response shapes are nearly identical to OpenAI's.
    """

    def __init__(self) -> None:
        pass

    def get_supported_openai_params(self, model: str) -> List[str]:
        # DashScope's compatible-mode embeddings API accepts the same params as OpenAI.
        # `dimensions` / `encoding_format` are only honored by text-embedding-v3 / v4;
        # earlier versions silently ignore them server-side.
        return ["dimensions", "encoding_format", "user"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool = False,
    ) -> dict:
        supported = self.get_supported_openai_params(model)
        for k, v in non_default_params.items():
            if v is None:
                continue
            if k in supported:
                optional_params[k] = v
            # unsupported params are dropped when drop_params=True;
            # the upstream _check_valid_arg already raised UnsupportedParamsError
            # for drop_params=False before this method is called.
        return optional_params

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
            api_key = get_secret_str("DASHSCOPE_API_KEY")
        if api_key is None:
            raise ValueError(
                "DashScope API key is required. Set 'DASHSCOPE_API_KEY' env var or pass api_key explicitly."
            )
        default_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        return {**default_headers, **headers}

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base = api_base or get_secret_str("DASHSCOPE_API_BASE") or DEFAULT_API_BASE
        base = base.rstrip("/")
        if base.endswith("/embeddings"):
            return base
        return f"{base}/embeddings"

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        data: dict = {
            "model": model,
            "input": input,
        }
        for key in ("dimensions", "encoding_format", "user"):
            value = optional_params.get(key)
            if value is not None:
                data[key] = value
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
        try:
            response_json = raw_response.json()
        except Exception as e:
            raise DashScopeError(
                status_code=raw_response.status_code,
                message=f"Failed to parse DashScope response as JSON: {str(e)}",
            )

        logging_obj.post_call(
            input=request_data.get("input"),
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},
            original_response=response_json,
        )

        if "error" in response_json:
            error = response_json["error"]
            message = (
                error.get("message", str(error))
                if isinstance(error, dict)
                else str(error)
            )
            raise DashScopeError(
                status_code=raw_response.status_code,
                message=message,
            )

        model_response.object = "list"
        model_response.data = response_json.get("data", [])
        model_response.model = response_json.get("model", model)

        usage = response_json.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens)
        setattr(
            model_response,
            "usage",
            Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=total_tokens,
            ),
        )

        if "id" in response_json:
            setattr(model_response, "id", response_json["id"])

        return model_response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        if isinstance(headers, dict):
            headers = httpx.Headers(headers)
        return DashScopeError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

from typing import Optional, cast

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage

from ..common_utils import SiliconFlowException, get_dict, get_int, get_list, get_str


class SiliconFlowEmbeddingConfig(BaseEmbeddingConfig):
    DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        stream: Optional[bool] = None,
    ) -> str:
        return "{}{}".format(
            (api_base or get_secret_str("SILICONFLOW_API_BASE") or self.DEFAULT_BASE_URL).rstrip("/"),
            "/embeddings",
        )

    def validate_environment(
        self,
        headers: dict[str, str],
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict[str, str]:
        final_api_key = api_key or get_secret_str("SILICONFLOW_API_KEY")
        if final_api_key is None:
            raise ValueError("SILICONFLOW_API_KEY is not set")
        return {
            **headers,
            "Authorization": "Bearer {}".format(final_api_key),
            "accept": "application/json",
            "Content-Type": "application/json",
        }

    def get_supported_openai_params(self, model: str) -> list[str]:
        return ["dimensions", "encoding_format", "user"]

    def map_openai_params(
        self,
        non_default_params: dict[str, object],
        optional_params: dict[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        supported_openai_params = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param in supported_openai_params:
                optional_params[param] = value
        return optional_params

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, object]:
        return {"input": input, "model": model, **optional_params}

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        request_data: dict[str, object],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
    ) -> EmbeddingResponse:
        try:
            raw_response_json = get_dict(cast(object, raw_response.json()))
        except Exception:
            raise SiliconFlowException(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        usage = get_dict(raw_response_json.get("usage"))
        prompt_tokens = get_int(usage.get("prompt_tokens"))
        if prompt_tokens is None:
            prompt_tokens = get_int(usage.get("input_tokens")) or 0
        total_tokens = get_int(usage.get("total_tokens")) or prompt_tokens
        model_response.model = get_str(raw_response_json.get("model"))
        model_response.data = get_list(raw_response_json.get("data"))
        model_response.object = "list"
        model_response.usage = Usage(
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
        )
        return model_response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,
    ) -> SiliconFlowException:
        return SiliconFlowException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

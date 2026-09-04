from typing import Final

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.types.llms.openai import AllEmbeddingInputValues
from litellm.types.utils import EmbeddingResponse

from ..utils import SnowflakeBaseConfig, SnowflakeException


class SnowflakeEmbeddingConfig(SnowflakeBaseConfig, BaseEmbeddingConfig):
    """
    source: https://docs.snowflake.com/developer-guide/snowflake-rest-api/reference/cortex-embed
    """

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        api_base = self._get_api_base(api_base, optional_params)

        return f"{api_base}/cortex/inference:embed"

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        return {"text": input, "model": model, **optional_params}

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
    ) -> EmbeddingResponse:
        response_json: Final = raw_response.json()
        # convert embeddings to 1d array
        for item in response_json["data"]:
            item["embedding"] = item["embedding"][0]
        returned_response: Final = EmbeddingResponse(**response_json)

        returned_response.model = "snowflake/" + (returned_response.model or "")

        if model is not None:
            returned_response._hidden_params["model"] = model
        return returned_response

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return SnowflakeException(message=error_message, status_code=status_code, headers=headers)

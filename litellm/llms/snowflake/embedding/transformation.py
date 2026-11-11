from collections.abc import Mapping
from typing import Final, Protocol, runtime_checkable

import httpx
from pydantic import ConfigDict, TypeAdapter

from litellm.litellm_core_utils.core_helpers import iter_items
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.types.llms.openai import AllEmbeddingInputValues
from litellm.types.utils import EmbeddingResponse

from ..utils import SnowflakeBaseConfig, SnowflakeException

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_DECODED_JSON: Final = TypeAdapter(object)


@runtime_checkable
class _SupportsGetItem(Protocol):
    def __getitem__(self, key: str | int, /) -> object: ...


@runtime_checkable
class _SupportsSetItem(Protocol):
    def __setitem__(self, key: str, value: object, /) -> None: ...


def _subscript(value: object, key: str | int) -> object:
    if isinstance(value, _SupportsGetItem):
        return value[key]
    raise TypeError(f"'{type(value).__name__}' object is not subscriptable")


def _item_assignment_target(value: object) -> _SupportsSetItem:
    if isinstance(value, _SupportsSetItem):
        return value
    raise TypeError(f"'{type(value).__name__}' object does not support item assignment")


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

        return join_api_base(api_base, "cortex/inference:embed")

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
        response_json: Final = _DECODED_JSON.validate_python(raw_response.json())
        # convert embeddings to 1d array
        for item in iter_items(_subscript(response_json, "data")):
            _item_assignment_target(item)["embedding"] = _subscript(_subscript(item, "embedding"), 0)
        returned_response: Final = EmbeddingResponse.model_validate(_JSON_OBJECT.validate_python(response_json))

        returned_response.model = "snowflake/" + (returned_response.model or "")

        if model is not None:
            returned_response.hidden_params["model"] = model
        return returned_response

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return SnowflakeException(message=error_message, status_code=status_code, headers=headers)

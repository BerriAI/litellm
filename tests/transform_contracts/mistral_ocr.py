from __future__ import annotations

from typing import Final, Protocol, cast

import httpx
from pydantic import BaseModel, TypeAdapter

from tests.transform_contracts.schema import (
    GetSupportedOCRParamsCase,
    JsonObject,
    JsonValue,
    MapOCRParamsCase,
    TransformOCRRequestCase,
    TransformOCRResponseCase,
    TransformationCase,
)

_JSON_OBJECT_ADAPTER: Final[TypeAdapter[JsonObject]] = TypeAdapter(JsonObject)
_JSON_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_DOCUMENT_ADAPTER: Final[TypeAdapter[dict[str, str]]] = TypeAdapter(dict[str, str])


class _MistralOCRConfig(Protocol):
    def get_supported_ocr_params(self, model: str) -> object: ...

    def map_ocr_params(
        self,
        non_default_params: JsonObject,
        optional_params: JsonObject,
        model: str,
    ) -> object: ...

    def transform_ocr_request(
        self,
        model: str,
        document: dict[str, str],
        optional_params: JsonObject,
        headers: dict[str, str],
    ) -> BaseModel: ...

    def transform_ocr_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: object | None,
    ) -> BaseModel: ...


def _config() -> _MistralOCRConfig:
    from litellm.llms.mistral.ocr.transformation import MistralOCRConfig

    return cast(_MistralOCRConfig, MistralOCRConfig())


def _model_dump(model: BaseModel) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(cast(object, model.model_dump(mode="json")))


def run_get_supported_ocr_params(case: TransformationCase) -> JsonValue:
    if not isinstance(case, GetSupportedOCRParamsCase):
        raise TypeError(f"invalid case type for {case.operation}: {type(case).__name__}")
    return _JSON_ADAPTER.validate_python(_config().get_supported_ocr_params(model=case.input.model))


def run_map_ocr_params(case: TransformationCase) -> JsonValue:
    if not isinstance(case, MapOCRParamsCase):
        raise TypeError(f"invalid case type for {case.operation}: {type(case).__name__}")
    result: Final = _config().map_ocr_params(
        non_default_params=case.input.non_default_params,
        optional_params=case.input.optional_params,
        model=case.input.model,
    )
    return _JSON_OBJECT_ADAPTER.validate_python(result)


def run_transform_ocr_request(case: TransformationCase) -> JsonValue:
    if not isinstance(case, TransformOCRRequestCase):
        raise TypeError(f"invalid case type for {case.operation}: {type(case).__name__}")
    result: Final = _config().transform_ocr_request(
        model=case.input.model,
        document=_DOCUMENT_ADAPTER.validate_python(cast(object, case.input.document.model_dump(mode="json"))),
        optional_params=case.input.optional_params,
        headers=case.input.headers,
    )
    return _model_dump(result)


def run_transform_ocr_response(case: TransformationCase) -> JsonValue:
    if not isinstance(case, TransformOCRResponseCase):
        raise TypeError(f"invalid case type for {case.operation}: {type(case).__name__}")
    response: Final = httpx.Response(
        status_code=case.input.response.status_code,
        json=case.input.response.body,
        headers=case.input.response.headers,
    )
    result: Final = _config().transform_ocr_response(
        model=case.input.model,
        raw_response=response,
        logging_obj=None,
    )
    return _model_dump(result)

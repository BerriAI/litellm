from typing import Any, Optional, cast

import httpx

from litellm.llms.base_llm.ocr.transformation import (
    BaseOCRConfig,
    DocumentType,
    OCRRequestData,
    OCRResponse,
)
from litellm.rust_bridge.ocr.providers import RustOcrProvider, call_ocr


def get_rust_ocr_provider_config(
    custom_llm_provider: Optional[str],
    fallback_config: BaseOCRConfig,
) -> BaseOCRConfig:
    if custom_llm_provider is None:
        return fallback_config

    provider_value = getattr(custom_llm_provider, "value", custom_llm_provider)
    try:
        rust_ocr_provider = RustOcrProvider(str(provider_value))
    except ValueError:
        return fallback_config

    return cast(
        BaseOCRConfig,
        _RustOCRProviderConfig(
            rust_ocr_provider=rust_ocr_provider,
            fallback_config=fallback_config,
        ),
    )


class _RustOCRProviderConfig:
    def __init__(
        self,
        rust_ocr_provider: RustOcrProvider,
        fallback_config: BaseOCRConfig,
    ) -> None:
        self.rust_ocr_provider = rust_ocr_provider
        self.fallback_config = fallback_config

    def __getattr__(self, name: str) -> Any:
        return getattr(self.fallback_config, name)

    def map_ocr_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
    ) -> dict:
        mapped_params = call_ocr(
            {
                "provider": self.rust_ocr_provider.value,
                "operation": "map_params",
                "non_default_params": non_default_params,
            }
        )
        if mapped_params is not None:
            return mapped_params
        return self.fallback_config.map_ocr_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
        )

    def transform_ocr_request(
        self,
        model: str,
        document: DocumentType,
        optional_params: dict,
        headers: dict,
        **kwargs,
    ) -> OCRRequestData:
        if isinstance(document, dict):
            transformed_request = call_ocr(
                {
                    "provider": self.rust_ocr_provider.value,
                    "operation": "transform_request",
                    "model": model,
                    "document": document,
                    "optional_params": optional_params,
                },
            )
            if transformed_request is not None:
                request_data = transformed_request.get("data")
                if not isinstance(request_data, dict):
                    raise ValueError(
                        f"Rust OCR provider {self.rust_ocr_provider.value} "
                        "returned invalid request data"
                    )
                return OCRRequestData(
                    data=request_data,
                    files=transformed_request.get("files"),
                )

        return self.fallback_config.transform_ocr_request(
            model=model,
            document=document,
            optional_params=optional_params,
            headers=headers,
            **kwargs,
        )

    async def async_transform_ocr_request(
        self,
        model: str,
        document: DocumentType,
        optional_params: dict,
        headers: dict,
        **kwargs,
    ) -> OCRRequestData:
        return self.transform_ocr_request(
            model=model,
            document=document,
            optional_params=optional_params,
            headers=headers,
            **kwargs,
        )

    def transform_ocr_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
        **kwargs,
    ) -> OCRResponse:
        transformed_response = call_ocr(
            {
                "provider": self.rust_ocr_provider.value,
                "operation": "transform_response",
                "model": model,
                "response_json": raw_response.json(),
            },
        )
        if transformed_response is not None:
            return OCRResponse(**transformed_response)

        return self.fallback_config.transform_ocr_response(
            model=model,
            raw_response=raw_response,
            logging_obj=logging_obj,
            **kwargs,
        )

    async def async_transform_ocr_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
        **kwargs,
    ) -> OCRResponse:
        return self.transform_ocr_response(
            model=model,
            raw_response=raw_response,
            logging_obj=logging_obj,
            **kwargs,
        )

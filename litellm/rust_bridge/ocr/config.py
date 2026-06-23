from typing import Any, Optional

import httpx

from litellm.llms.base_llm.ocr.transformation import (
    BaseOCRConfig,
    DocumentType,
    OCRRequestData,
    OCRResponse,
)
from litellm.rust_bridge.ocr.providers import call_ocr


def get_rust_ocr_provider_config(
    model: str,
    fallback_config: BaseOCRConfig,
) -> BaseOCRConfig:
    rust_ocr_provider = fallback_config.get_rust_ocr_provider(model=model)
    if not rust_ocr_provider:
        return fallback_config

    return RustOCRProviderConfig(
        rust_ocr_provider=rust_ocr_provider,
        fallback_config=fallback_config,
    )


class RustOCRProviderConfig(BaseOCRConfig):
    def __init__(
        self,
        rust_ocr_provider: str,
        fallback_config: BaseOCRConfig,
    ) -> None:
        super().__init__()
        self.rust_ocr_provider = rust_ocr_provider
        self.fallback_config = fallback_config

    def get_supported_ocr_params(self, model: str) -> list:
        return self.fallback_config.get_supported_ocr_params(model=model)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        litellm_params: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        return self.fallback_config.validate_environment(
            headers=headers,
            model=model,
            api_key=api_key,
            api_base=api_base,
            litellm_params=litellm_params,
            **kwargs,
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: Optional[dict] = None,
        **kwargs,
    ) -> str:
        return self.fallback_config.get_complete_url(
            api_base=api_base,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
            **kwargs,
        )

    def map_ocr_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
    ) -> dict:
        mapped_params = call_ocr(
            {
                "provider": self.rust_ocr_provider,
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
                    "provider": self.rust_ocr_provider,
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
                        f"Rust OCR provider {self.rust_ocr_provider} "
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
                "provider": self.rust_ocr_provider,
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

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict,
    ) -> Exception:
        return self.fallback_config.get_error_class(
            error_message=error_message,
            status_code=status_code,
            headers=headers,
        )

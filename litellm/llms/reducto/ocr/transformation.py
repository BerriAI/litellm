from typing import Any, Final

import httpx

from litellm.llms.base_llm.ocr.transformation import (
    BaseOCRConfig,
    DocumentType,
    OCRRequestData,
    OCRResponse,
    OCRUsageInfo,
)
from litellm.llms.reducto.common import (
    REDUCTO_API_BASE,
    build_pages_from_reducto,
    extract_file_id_or_bytes,
    upload_bytes_async,
    upload_bytes_sync,
)


class _BaseReductoOCRConfig(BaseOCRConfig):
    def map_ocr_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
    ) -> dict:
        mapped_params: Final = dict(optional_params)
        supported_params: Final = self.get_supported_ocr_params(model=model)
        for param, value in non_default_params.items():
            if param in supported_params:
                mapped_params[param] = value
        return mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        litellm_params: dict | None = None,
        **kwargs,
    ) -> dict:
        from litellm.secret_managers.main import get_secret_str

        resolved_key: Final = api_key or get_secret_str("REDUCTO_API_KEY")
        if resolved_key is None:
            raise ValueError(
                "Missing REDUCTO_API_KEY - set it in the environment or pass api_key to litellm.ocr()/litellm.aocr()"
            )

        return {
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
            **headers,
        }

    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict | None = None,
        **kwargs,
    ) -> str:
        return "{}/parse".format((api_base or REDUCTO_API_BASE).rstrip("/"))

    def _get_source_url(self, document: DocumentType, model: str) -> str:
        source_url: Final = document.get("document_url") or document.get("image_url")
        if source_url is None:
            raise ValueError(
                f"Reducto expected OCR preprocessing to produce document_url or image_url for model={model}"
            )
        return source_url

    @staticmethod
    def _resolve_credentials(api_key: str | None, api_base: str | None) -> tuple[str, str]:
        from litellm.secret_managers.main import get_secret_str

        resolved_key: Final = api_key or get_secret_str("REDUCTO_API_KEY")
        if resolved_key is None:
            raise ValueError(
                "Missing REDUCTO_API_KEY - set it in the environment or pass api_key to litellm.ocr()/litellm.aocr()"
            )
        resolved_base: Final = (api_base or REDUCTO_API_BASE).rstrip("/")
        return resolved_key, resolved_base

    def _ensure_file_id_sync(
        self,
        model: str,
        document: DocumentType,
        api_key: str | None,
        api_base: str | None,
    ) -> str:
        source_url: Final = self._get_source_url(document=document, model=model)
        file_id, raw_bytes, mime = extract_file_id_or_bytes(source_url, model=model)
        if file_id is not None:
            return file_id
        resolved_key, resolved_base = self._resolve_credentials(api_key, api_base)
        return upload_bytes_sync(
            raw_bytes=raw_bytes or b"",
            mime=mime,
            api_key=resolved_key,
            api_base=resolved_base,
        )

    async def _ensure_file_id_async(
        self,
        model: str,
        document: DocumentType,
        api_key: str | None,
        api_base: str | None,
    ) -> str:
        source_url: Final = self._get_source_url(document=document, model=model)
        file_id, raw_bytes, mime = extract_file_id_or_bytes(source_url, model=model)
        if file_id is not None:
            return file_id
        resolved_key, resolved_base = self._resolve_credentials(api_key, api_base)
        return await upload_bytes_async(
            raw_bytes=raw_bytes or b"",
            mime=mime,
            api_key=resolved_key,
            api_base=resolved_base,
        )

    def transform_ocr_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
        **kwargs,
    ) -> OCRResponse:
        response_json: Final = raw_response.json()
        result: Final = response_json.get("result", response_json) or {}
        usage: Final = response_json.get("usage", {}) or {}
        response: Final = OCRResponse(
            pages=build_pages_from_reducto(result),
            model=model,
            usage_info=OCRUsageInfo(
                pages_processed=usage.get("num_pages"),
                credits=usage.get("credits"),
            ),
            object="ocr",
        )
        response._hidden_params["reducto_raw"] = response_json
        return response


class ReductoParseV3Config(_BaseReductoOCRConfig):
    def get_supported_ocr_params(self, model: str) -> list:
        return ["formatting", "retrieval", "settings"]

    def transform_ocr_request(
        self,
        model: str,
        document: DocumentType,
        optional_params: dict,
        headers: dict,
        **kwargs,
    ) -> OCRRequestData:
        file_id: Final = self._ensure_file_id_sync(
            model=model,
            document=document,
            api_key=kwargs.get("api_key"),
            api_base=kwargs.get("api_base"),
        )
        return OCRRequestData(data={"input": file_id, **optional_params}, files=None)

    async def async_transform_ocr_request(
        self,
        model: str,
        document: DocumentType,
        optional_params: dict,
        headers: dict,
        **kwargs,
    ) -> OCRRequestData:
        file_id: Final = await self._ensure_file_id_async(
            model=model,
            document=document,
            api_key=kwargs.get("api_key"),
            api_base=kwargs.get("api_base"),
        )
        return OCRRequestData(data={"input": file_id, **optional_params}, files=None)


class ReductoParseLegacyConfig(_BaseReductoOCRConfig):
    def get_supported_ocr_params(self, model: str) -> list:
        return ["enhance"]

    def _build_legacy_body(self, file_id: str, optional_params: dict) -> dict[str, Any]:
        body: Final[dict[str, Any]] = {"document_url": file_id}
        enhance: Final = optional_params.get("enhance")
        if enhance is not None:
            body["options"] = {"enhance": enhance}
        return body

    def transform_ocr_request(
        self,
        model: str,
        document: DocumentType,
        optional_params: dict,
        headers: dict,
        **kwargs,
    ) -> OCRRequestData:
        file_id: Final = self._ensure_file_id_sync(
            model=model,
            document=document,
            api_key=kwargs.get("api_key"),
            api_base=kwargs.get("api_base"),
        )
        return OCRRequestData(
            data=self._build_legacy_body(file_id=file_id, optional_params=optional_params),
            files=None,
        )

    async def async_transform_ocr_request(
        self,
        model: str,
        document: DocumentType,
        optional_params: dict,
        headers: dict,
        **kwargs,
    ) -> OCRRequestData:
        file_id: Final = await self._ensure_file_id_async(
            model=model,
            document=document,
            api_key=kwargs.get("api_key"),
            api_base=kwargs.get("api_base"),
        )
        return OCRRequestData(
            data=self._build_legacy_body(file_id=file_id, optional_params=optional_params),
            files=None,
        )

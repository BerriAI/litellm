from typing import Any, Dict, Optional

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
from litellm.secret_managers.main import get_secret_str


class _BaseReductoOCRConfig(BaseOCRConfig):
    def __init__(self) -> None:
        super().__init__()
        self._api_key: Optional[str] = None
        self._api_base: Optional[str] = None

    def map_ocr_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
    ) -> dict:
        mapped_params = dict(optional_params)
        supported_params = self.get_supported_ocr_params(model=model)
        for param, value in non_default_params.items():
            if param in supported_params:
                mapped_params[param] = value
        return mapped_params

    def validate_environment(
        self,
        headers: Dict,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        litellm_params: Optional[dict] = None,
        **kwargs,
    ) -> Dict:
        resolved_key = api_key or get_secret_str("REDUCTO_API_KEY")
        if resolved_key is None:
            raise ValueError(
                "Missing REDUCTO_API_KEY - set it in the environment or pass api_key to litellm.ocr()/litellm.aocr()"
            )

        self._api_key = resolved_key
        self._api_base = (api_base or REDUCTO_API_BASE).rstrip("/")

        return {
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
            **headers,
        }

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: Optional[dict] = None,
        **kwargs,
    ) -> str:
        return "{}/parse".format((api_base or REDUCTO_API_BASE).rstrip("/"))

    def _get_source_url(self, document: DocumentType, model: str) -> str:
        source_url = document.get("document_url") or document.get("image_url")
        if source_url is None:
            raise ValueError(
                "Reducto expected OCR preprocessing to produce document_url or image_url for model={}".format(
                    model
                )
            )
        return source_url

    def _ensure_file_id_sync(self, model: str, document: DocumentType) -> str:
        source_url = self._get_source_url(document=document, model=model)
        file_id, raw_bytes, mime = extract_file_id_or_bytes(source_url, model=model)
        if file_id is not None:
            return file_id
        if self._api_key is None:
            raise ValueError("Reducto API key was not initialized before OCR upload.")
        return upload_bytes_sync(
            raw_bytes=raw_bytes or b"",
            mime=mime,
            api_key=self._api_key,
            api_base=self._api_base,
        )

    async def _ensure_file_id_async(self, model: str, document: DocumentType) -> str:
        source_url = self._get_source_url(document=document, model=model)
        file_id, raw_bytes, mime = extract_file_id_or_bytes(source_url, model=model)
        if file_id is not None:
            return file_id
        if self._api_key is None:
            raise ValueError("Reducto API key was not initialized before OCR upload.")
        return await upload_bytes_async(
            raw_bytes=raw_bytes or b"",
            mime=mime,
            api_key=self._api_key,
            api_base=self._api_base,
        )

    def transform_ocr_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
        **kwargs,
    ) -> OCRResponse:
        response_json = raw_response.json()
        result = response_json.get("result", response_json) or {}
        usage = response_json.get("usage", {}) or {}
        response = OCRResponse(
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
        file_id = self._ensure_file_id_sync(model=model, document=document)
        return OCRRequestData(data={"input": file_id, **optional_params}, files=None)

    async def async_transform_ocr_request(
        self,
        model: str,
        document: DocumentType,
        optional_params: dict,
        headers: dict,
        **kwargs,
    ) -> OCRRequestData:
        file_id = await self._ensure_file_id_async(model=model, document=document)
        return OCRRequestData(data={"input": file_id, **optional_params}, files=None)


class ReductoParseLegacyConfig(_BaseReductoOCRConfig):
    def get_supported_ocr_params(self, model: str) -> list:
        return ["enhance"]

    def _build_legacy_body(self, file_id: str, optional_params: dict) -> Dict[str, Any]:
        body: Dict[str, Any] = {"document_url": file_id}
        enhance = optional_params.get("enhance")
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
        file_id = self._ensure_file_id_sync(model=model, document=document)
        return OCRRequestData(
            data=self._build_legacy_body(
                file_id=file_id, optional_params=optional_params
            ),
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
        file_id = await self._ensure_file_id_async(model=model, document=document)
        return OCRRequestData(
            data=self._build_legacy_body(
                file_id=file_id, optional_params=optional_params
            ),
            files=None,
        )

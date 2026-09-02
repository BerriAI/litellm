"""Azure AI Cohere Parse OCR transformation."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, Literal

import httpx
from pydantic import ConfigDict, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm.llms.azure_ai.ocr.transformation import AzureAIOCRConfig
from litellm.llms.base_llm.ocr.transformation import (
    DocumentType,
    OCRPage,
    OCRPageImage,
    OCRRequestData,
    OCRResponse,
    OCRUsageInfo,
)
from litellm.types.llms.base import LiteLLMPydanticObjectBase

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class _CohereParseMarkdown(LiteLLMPydanticObjectBase):
    content: str = ""
    images: Sequence[OCRPageImage] | None = None

    model_config = ConfigDict(extra="allow", frozen=True)


class _CohereParsePage(LiteLLMPydanticObjectBase):
    index: int | None = None
    markdown: _CohereParseMarkdown | str | None = None
    blocks: Sequence[Mapping[str, object]] | None = None

    model_config = ConfigDict(extra="allow", frozen=True)


class _CohereParseBilledUnits(LiteLLMPydanticObjectBase):
    pages: int | None = None

    model_config = ConfigDict(extra="allow", frozen=True)


class _CohereParseMeta(LiteLLMPydanticObjectBase):
    billed_units: _CohereParseBilledUnits | None = None

    model_config = ConfigDict(extra="allow", frozen=True)


class _CohereParseResponse(LiteLLMPydanticObjectBase):
    pages: Sequence[_CohereParsePage]
    model: str | None = None
    meta: _CohereParseMeta | None = None

    model_config = ConfigDict(extra="allow", frozen=True)


class _CohereImageDocument(TypedDict):
    type: ReadOnly[Literal["image_url"]]
    image_url: ReadOnly[str]


class _CohereRequestPayload(TypedDict):
    model: ReadOnly[str]
    document: ReadOnly[_CohereImageDocument]
    output_format: ReadOnly[object]


class _OCRRequestDataPayload(TypedDict):
    data: ReadOnly[_CohereRequestPayload]
    files: ReadOnly[None]


class _NormalizedPageData(TypedDict):
    index: ReadOnly[int]
    markdown: ReadOnly[str]
    images: ReadOnly[Sequence[OCRPageImage] | None]
    blocks: ReadOnly[Sequence[Mapping[str, object]] | None]


class _NormalizedResponseData(TypedDict):
    pages: ReadOnly[Sequence[OCRPage]]
    model: ReadOnly[str]
    usage_info: ReadOnly[OCRUsageInfo]
    object: ReadOnly[Literal["ocr"]]


_NATIVE_RESPONSE_ADAPTER: Final = TypeAdapter(Mapping[str, object])


class AzureAICohereParseConfig(AzureAIOCRConfig):
    """Translate LiteLLM OCR requests and Cohere Parse responses on Azure AI."""

    def get_supported_ocr_params(self, model: str) -> list[str]:  # mutable-ok: BaseOCRConfig requires a list
        return ["output_format"]  # mutable-ok: BaseOCRConfig requires a list

    def map_ocr_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
    ) -> dict[str, object]:  # mutable-ok: BaseOCRConfig requires a dict
        output_format: Final = non_default_params.get("output_format")
        if output_format is None:
            return dict(optional_params)  # mutable-ok: BaseOCRConfig requires a dict
        if output_format not in ("markdown", "blocks"):
            raise ValueError("Cohere Parse output_format must be either 'markdown' or 'blocks'.")
        return {**optional_params, "output_format": output_format}  # mutable-ok: BaseOCRConfig requires a dict

    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object] | None = None,
        **kwargs: object,  # kwargs-ok: BaseOCRConfig forwards provider-specific extras
    ) -> str:
        if api_base is None:
            raise ValueError(
                "Missing Azure AI API Base - Set AZURE_AI_API_BASE environment variable or pass api_base parameter"
            )

        original_url: Final = httpx.URL(api_base)
        if not original_url.is_absolute_url:
            raise ValueError(f"Azure AI API Base must be an absolute URL including scheme. Got api_base={api_base!r}.")

        normalized_path: Final = original_url.path.rstrip("/")
        if normalized_path.endswith("/v2/parse"):
            return str(original_url.copy_with(path=normalized_path))

        if original_url.host and original_url.host.endswith(".models.ai.azure.com"):
            dedicated_path: Final = f"{normalized_path}/v2/parse" if normalized_path else "/v2/parse"
            return str(original_url.copy_with(path=dedicated_path))

        foundry_path: Final = normalized_path.removesuffix("/models")
        provider_path: Final = f"{foundry_path}/providers/cohere/v2/parse"
        return str(original_url.copy_with(path=provider_path))

    @staticmethod
    def _validate_image_document(document: DocumentType) -> str:
        if document.get("type") != "image_url":
            raise ValueError(
                "Cohere Parse currently supports only document.type='image_url'; "
                "document_url and PDF inputs are not supported"
            )

        image_url: Final = document.get("image_url", "")
        if not image_url:
            raise ValueError("Cohere Parse image_url must not be empty")
        if image_url.lower().startswith("data:application/pdf"):
            raise ValueError("Cohere Parse does not support PDF inputs; provide an image_url")
        return image_url

    def transform_ocr_request(
        self,
        model: str,
        document: DocumentType,
        optional_params: Mapping[str, object],
        headers: Mapping[str, object],
        **kwargs: object,  # kwargs-ok: BaseOCRConfig forwards provider-specific extras
    ) -> OCRRequestData:
        image_url: Final = self._validate_image_document(document)
        converted_url: Final = (
            image_url if image_url.startswith("data:") else self._convert_url_to_data_uri_sync(url=image_url)
        )
        return self._build_request(model=model, image_url=converted_url, optional_params=optional_params)

    async def async_transform_ocr_request(
        self,
        model: str,
        document: DocumentType,
        optional_params: Mapping[str, object],
        headers: Mapping[str, object],
        **kwargs: object,  # kwargs-ok: BaseOCRConfig forwards provider-specific extras
    ) -> OCRRequestData:
        image_url: Final = self._validate_image_document(document)
        converted_url: Final = (
            image_url if image_url.startswith("data:") else await self._convert_url_to_data_uri_async(url=image_url)
        )
        return self._build_request(model=model, image_url=converted_url, optional_params=optional_params)

    @staticmethod
    def _build_request(model: str, image_url: str, optional_params: Mapping[str, object]) -> OCRRequestData:
        document: Final[_CohereImageDocument] = {"type": "image_url", "image_url": image_url}
        payload: Final[_CohereRequestPayload] = {
            "model": model,
            "document": document,
            "output_format": optional_params.get("output_format", "markdown"),
        }
        request_data: Final[_OCRRequestDataPayload] = {"data": payload, "files": None}
        return OCRRequestData.model_validate(request_data)

    @staticmethod
    def _normalize_page(page: _CohereParsePage, page_number: int) -> OCRPage:
        markdown_value: Final = page.markdown
        markdown: Final = markdown_value.content if isinstance(markdown_value, _CohereParseMarkdown) else markdown_value
        images: Final = markdown_value.images if isinstance(markdown_value, _CohereParseMarkdown) else None
        page_data: Final[_NormalizedPageData] = {
            "index": page.index if page.index is not None else page_number,
            "markdown": markdown or "",
            "images": images,
            "blocks": page.blocks,
        }
        return OCRPage.model_validate(page_data)

    def transform_ocr_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
        **kwargs: object,  # kwargs-ok: BaseOCRConfig forwards provider-specific extras
    ) -> OCRResponse:
        native_response: Final = _NATIVE_RESPONSE_ADAPTER.validate_json(raw_response.content)
        response_data: Final = _CohereParseResponse.model_validate(native_response)
        normalized_pages: Final = tuple(
            self._normalize_page(page=page, page_number=page_number)
            for page_number, page in enumerate(response_data.pages)
        )
        billed_units: Final = response_data.meta.billed_units if response_data.meta is not None else None
        normalized_response: Final[_NormalizedResponseData] = {
            "pages": normalized_pages,
            "model": response_data.model or model,
            "usage_info": OCRUsageInfo(pages_processed=billed_units.pages if billed_units is not None else None),
            "object": "ocr",
        }
        response: Final = OCRResponse.model_validate(normalized_response)
        response.set_provider_native_response(  # pyright: ignore[reportUnknownMemberType]  # base mapping values are untyped
            native_response
        )
        return response

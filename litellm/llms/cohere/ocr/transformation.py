"""Cohere Parse (`POST /v2/parse`) exposed through LiteLLM's OCR interface."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm.exceptions import BadRequestError, UnsupportedParamsError
from litellm.llms.base_llm.ocr.transformation import (
    OCR_REQUEST_FORMAT_PARAM,
    BaseOCRConfig,
    DocumentType,
    OCRPage,
    OCRPageImage,
    OCRRequestData,
    OCRRequestFormat,
    OCRResponse,
    OCRUsageInfo,
    parse_ocr_request_format,
)
from litellm.llms.cohere.common_utils import CohereError
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

COHERE_API_KEY_ENV_VAR: Final = "COHERE_API_KEY"
COHERE_PARSE_API_BASE: Final = "https://api.cohere.com"
COHERE_PARSE_PATH: Final = "/v2/parse"
COHERE_PARSE_OUTPUT_FORMAT_PARAM: Final = "output_format"
COHERE_PARSE_OUTPUT_FORMATS: Final = ("markdown", "blocks")
COHERE_PARSE_DEFAULT_OUTPUT_FORMAT: Final = "markdown"
COHERE_PARSE_SUPPORTED_PARAMS: Final = (COHERE_PARSE_OUTPUT_FORMAT_PARAM, OCR_REQUEST_FORMAT_PARAM)
COHERE_PARSE_HEALTH_CHECK_IMAGE_DATA_URI: Final = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
)
COHERE_PARSE_IMAGE_ONLY_MESSAGE: Final = (
    "Cohere Parse only accepts `image_url` documents (an image URL or a base64 image data URI); "
    "`document_url` and PDF inputs are not supported."
)

_NATIVE_RESPONSE_ADAPTER: Final = TypeAdapter(dict[str, object])
_BOUNDING_BOX_ADAPTER: Final = TypeAdapter(Mapping[str, object])


class _CohereParseDocument(TypedDict):
    type: ReadOnly[Literal["image_url"]]
    image_url: ReadOnly[str]


class _CohereParseRequestBody(TypedDict):
    model: ReadOnly[str]
    document: ReadOnly[_CohereParseDocument]
    output_format: ReadOnly[str]


class _MarkdownPage(TypedDict):
    index: ReadOnly[int]
    markdown: ReadOnly[str]
    images: ReadOnly[Sequence[OCRPageImage] | None]


class _BlocksPage(_MarkdownPage):
    blocks: ReadOnly[Sequence[Mapping[str, object]]]


class _CohereParseMarkdown(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    content: str = ""
    images: Sequence[Mapping[str, object]] | None = None


class _CohereParsePage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    index: int | None = None
    markdown: _CohereParseMarkdown | None = None
    blocks: Sequence[Mapping[str, object]] | None = None


class _CohereParseBilledUnits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    pages: int | None = None


class _CohereParseMeta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    billed_units: _CohereParseBilledUnits | None = None


class _CohereParseResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    pages: Sequence[_CohereParsePage] = ()
    meta: _CohereParseMeta | None = None


def _requested_format(optional_params: Mapping[str, object] | None) -> OCRRequestFormat:
    if optional_params is None:
        return "litellm"
    return "native" if optional_params.get(OCR_REQUEST_FORMAT_PARAM) == "native" else "litellm"


def _page_image(image: Mapping[str, object]) -> OCRPageImage:
    bounding_box: Final = image.get("bounding_box")
    if not isinstance(bounding_box, Mapping):
        return OCRPageImage.model_validate(image)
    bbox: Final = _BOUNDING_BOX_ADAPTER.validate_python(bounding_box)
    return OCRPageImage.model_validate(MappingProxyType({**image, "bbox": bbox}))


def _normalize_page(page: _CohereParsePage, position: int) -> OCRPage:
    markdown: Final = page.markdown
    images: Final = tuple(_page_image(image) for image in markdown.images) if markdown and markdown.images else None
    normalized: Final[_MarkdownPage] = {
        "index": page.index if page.index is not None else position,
        "markdown": markdown.content if markdown else "",
        "images": images,
    }
    if page.blocks is None:
        return OCRPage.model_validate(normalized)
    with_blocks: Final[_BlocksPage] = {**normalized, "blocks": page.blocks}
    return OCRPage.model_validate(with_blocks)


def _billed_pages(parsed: _CohereParseResponse) -> int | None:
    if parsed.meta is None or parsed.meta.billed_units is None:
        return None
    return parsed.meta.billed_units.pages


class CohereParseConfig(BaseOCRConfig):
    """Cohere Parse, an image-only document understanding endpoint returning markdown or blocks."""

    def get_supported_ocr_params(self, model: str) -> list[str]:  # mutable-ok: BaseOCRConfig signature
        return list(COHERE_PARSE_SUPPORTED_PARAMS)  # mutable-ok: BaseOCRConfig signature

    def get_api_key_env_var(self) -> str | None:
        return COHERE_API_KEY_ENV_VAR

    def supports_rust_bridge(self) -> bool:
        return False

    def get_health_check_document(self) -> DocumentType:
        return {  # mutable-ok: litellm.aocr rejects any document that is not a dict
            "type": "image_url",
            "image_url": COHERE_PARSE_HEALTH_CHECK_IMAGE_DATA_URI,
        }

    def _llm_provider(self) -> str:
        return "cohere"

    def map_ocr_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
    ) -> dict[str, object]:  # mutable-ok: BaseOCRConfig signature
        output_format: Final = non_default_params.get(COHERE_PARSE_OUTPUT_FORMAT_PARAM)
        if output_format is not None and output_format not in COHERE_PARSE_OUTPUT_FORMATS:
            raise UnsupportedParamsError(
                message=(
                    f"Invalid `{COHERE_PARSE_OUTPUT_FORMAT_PARAM}`: {output_format!r}. "
                    f"Expected one of {', '.join(COHERE_PARSE_OUTPUT_FORMATS)}."
                ),
                model=model,
                llm_provider=self._llm_provider(),
            )
        requested_format: Final = non_default_params.get(OCR_REQUEST_FORMAT_PARAM)
        request_format: Final = parse_ocr_request_format(requested_format) if requested_format is not None else None
        overrides: Final = tuple(
            (key, value)
            for key, value in (
                (COHERE_PARSE_OUTPUT_FORMAT_PARAM, output_format),
                (OCR_REQUEST_FORMAT_PARAM, request_format),
            )
            if value is not None
        )
        return {**optional_params, **dict(overrides)}  # mutable-ok: BaseOCRConfig signature

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        litellm_params: Mapping[str, object] | None = None,
        **kwargs: object,  # kwargs-ok: BaseOCRConfig.validate_environment signature
    ) -> dict[str, str]:  # mutable-ok: BaseOCRConfig signature
        resolved_key: Final = api_key or get_secret_str(COHERE_API_KEY_ENV_VAR)
        if resolved_key is None:
            raise ValueError(
                f"Missing {COHERE_API_KEY_ENV_VAR} - set it in the environment or pass api_key to "
                "litellm.ocr()/litellm.aocr()"
            )
        return {  # mutable-ok: BaseOCRConfig signature
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
            **headers,
        }

    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object] | None = None,
        **kwargs: object,  # kwargs-ok: BaseOCRConfig.get_complete_url signature
    ) -> str:
        url: Final = httpx.URL(api_base or COHERE_PARSE_API_BASE)
        path: Final = url.path.rstrip("/")
        if path.endswith(COHERE_PARSE_PATH):
            return str(url.copy_with(path=path))
        if path.endswith("/v2"):
            return str(url.copy_with(path=f"{path}/parse"))
        return str(url.copy_with(path=f"{path}{COHERE_PARSE_PATH}"))

    def _image_url(self, document: DocumentType, model: str) -> str:
        image_url: Final = document.get("image_url", "")
        if document.get("type") != "image_url" or not image_url or image_url.startswith("data:application/pdf"):
            raise BadRequestError(
                message=COHERE_PARSE_IMAGE_ONLY_MESSAGE,
                model=model,
                llm_provider=self._llm_provider(),
            )
        return image_url

    def _resolve_image_url_sync(self, image_url: str) -> str:
        return image_url

    async def _resolve_image_url_async(self, image_url: str) -> str:
        return image_url

    def _build_request(self, model: str, image_url: str, optional_params: Mapping[str, object]) -> OCRRequestData:
        body: Final[_CohereParseRequestBody] = {
            "model": model,
            "document": {"type": "image_url", "image_url": image_url},
            "output_format": str(
                optional_params.get(COHERE_PARSE_OUTPUT_FORMAT_PARAM, COHERE_PARSE_DEFAULT_OUTPUT_FORMAT)
            ),
        }
        return OCRRequestData(data=dict(body), files=None)  # mutable-ok: OCRRequestData.data is a dict

    def transform_ocr_request(
        self,
        model: str,
        document: DocumentType,
        optional_params: Mapping[str, object],
        headers: Mapping[str, str],
        **kwargs: object,  # kwargs-ok: BaseOCRConfig.transform_ocr_request signature
    ) -> OCRRequestData:
        image_url: Final = self._resolve_image_url_sync(self._image_url(document, model))
        return self._build_request(model=model, image_url=image_url, optional_params=optional_params)

    async def async_transform_ocr_request(
        self,
        model: str,
        document: DocumentType,
        optional_params: Mapping[str, object],
        headers: Mapping[str, str],
        **kwargs: object,  # kwargs-ok: BaseOCRConfig.async_transform_ocr_request signature
    ) -> OCRRequestData:
        image_url: Final = await self._resolve_image_url_async(self._image_url(document, model))
        return self._build_request(model=model, image_url=image_url, optional_params=optional_params)

    def transform_ocr_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
        optional_params: Mapping[str, object] | None = None,
        **kwargs: object,  # kwargs-ok: BaseOCRConfig.transform_ocr_response signature
    ) -> OCRResponse:
        native: Final = _NATIVE_RESPONSE_ADAPTER.validate_python(raw_response.json())
        parsed: Final = _CohereParseResponse.model_validate(native)
        pages: Final = [  # mutable-ok: OCRResponse.pages is a list
            _normalize_page(page, position) for position, page in enumerate(parsed.pages)
        ]
        billed_pages: Final = _billed_pages(parsed)
        response: Final = OCRResponse(
            pages=pages,
            model=model,
            usage_info=OCRUsageInfo(pages_processed=billed_pages if billed_pages is not None else len(pages)),
        )
        if _requested_format(optional_params) == "native":
            response.set_provider_native_response(native)
        return response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Mapping[str, str],
    ) -> Exception:
        return CohereError(status_code=status_code, message=error_message)

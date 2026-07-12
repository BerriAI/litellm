"""
Azure Document Intelligence OCR transformation implementation.

Azure Document Intelligence (formerly Form Recognizer) provides advanced document analysis capabilities.
This implementation transforms between Mistral OCR format and Azure Document Intelligence API v4.0.

Note: Azure Document Intelligence API is async - POST returns 202 Accepted with Operation-Location header.
The operation location must be polled until the analysis completes.
"""

import asyncio
import re
import time
from typing import Any, Dict
from urllib.parse import quote

import httpx
from pydantic import BaseModel

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.url_utils import SSRFError, assert_same_origin
from litellm.constants import (
    AZURE_DOCUMENT_INTELLIGENCE_API_VERSION,
    AZURE_DOCUMENT_INTELLIGENCE_DEFAULT_DPI,
    AZURE_OPERATION_POLLING_TIMEOUT,
)
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.ocr.transformation import (
    BaseOCRConfig,
    DocumentType,
    OCRPage,
    OCRPageDimensions,
    OCRRequestData,
    OCRResponse,
    OCRUsageInfo,
)
from litellm.secret_managers.main import get_secret_str

AZURE_DOCUMENT_INTELLIGENCE_API_KEY_ENV_VAR = "AZURE_DOCUMENT_INTELLIGENCE_API_KEY"


class AzureDocumentIntelligenceLine(BaseModel):
    content: str | None = None


class AzureDocumentIntelligencePage(BaseModel):
    pageNumber: int | None = None
    width: float | None = None
    height: float | None = None
    unit: str | None = None
    lines: tuple[AzureDocumentIntelligenceLine, ...] = ()


class AzureDocumentIntelligenceAnalyzeResult(BaseModel):
    content: str | None = None
    pages: tuple[AzureDocumentIntelligencePage, ...] = ()
    tables: list[dict[str, object]] | None = None
    keyValuePairs: list[dict[str, object]] | None = None


class AzureDocumentIntelligenceOperation(BaseModel):
    status: str | None = None
    analyzeResult: AzureDocumentIntelligenceAnalyzeResult | None = None


class AzureDocumentIntelligenceOCRConfig(BaseOCRConfig):
    """
    Azure Document Intelligence OCR transformation configuration.

    Supports Azure Document Intelligence v4.0 (2024-11-30) API.
    Model route: azure_ai/doc-intelligence/<model>

    Supported models:
    - prebuilt-layout: Extracts text with markdown, tables, and structure (closest to Mistral OCR)
    - prebuilt-read: Basic text extraction optimized for reading
    - prebuilt-document: General document analysis

    Reference: https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/
    """

    def __init__(self) -> None:
        super().__init__()

    def get_api_key_env_var(self) -> str | None:
        return AZURE_DOCUMENT_INTELLIGENCE_API_KEY_ENV_VAR

    def get_supported_ocr_params(self, model: str) -> list:
        """
        Get supported OCR parameters for Azure Document Intelligence.

        Azure DI exposes a `pages` query parameter on the analyze endpoint
        (1-based, e.g. "1-3,5,7-9"). To keep the public request shape
        aligned with Mistral OCR, callers pass `pages` using Mistral
        semantics — a list of 0-based integers — or a pre-formatted
        Azure-style string. Azure DI also exposes a `features` query
        parameter enabling add-on capabilities (e.g. "keyValuePairs",
        "languages"), passed as a list of feature names or a
        comma-separated string. Other Mistral-specific params (e.g.
        `include_image_base64`) are not supported by Azure DI and are
        ignored during transformation.
        """
        return ["pages", "features"]

    def map_ocr_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
    ) -> dict:
        """
        Map OCR params to Azure DI format.

        Translates Mistral-style `pages` (list[int], 0-based) into Azure's
        `pages` query string (1-based, e.g. "1,2,3" or "1-3,5"). A raw
        string that already matches Azure's format is passed through
        unchanged. `features` (list[str] or comma-separated string) is
        normalized into Azure's comma-joined `features` query string.
        """
        pages = non_default_params.get("pages")
        features = non_default_params.get("features")
        normalized_pages = self._normalize_pages_param(pages) if pages is not None else ""
        normalized_features = self._normalize_features_param(features) if features is not None else ""
        return {
            **optional_params,
            **({"pages": normalized_pages} if normalized_pages else {}),
            **({"features": normalized_features} if normalized_features else {}),
        }

    @staticmethod
    def _normalize_pages_param(pages: Any) -> str:
        """
        Convert a caller-provided `pages` value to Azure DI's query-string
        form. Azure expects 1-based page numbers, grammar: `^(\\d+(-\\d+)?)(,\\s*(\\d+(-\\d+)?))*$`.

        Accepted inputs:
          - list[int]: Mistral-style 0-based indices. Converted to 1-based
            and joined (e.g. [0,1,2] -> "1,2,3").
          - list[str]: tokens like "1" or "3-5". Validated, joined as-is
            (treated as Azure-native, i.e. 1-based).
          - str: already in Azure format. Validated and whitespace-stripped.
        """
        pages_pattern = re.compile(r"^\s*\d+(-\d+)?(\s*,\s*\d+(-\d+)?)*\s*$")

        if isinstance(pages, str):
            if not pages_pattern.match(pages):
                raise ValueError(
                    f"Invalid `pages` string for Azure Document Intelligence: "
                    f"{pages!r}. Expected format like '1-3,5,7-9'."
                )
            return pages.replace(" ", "")

        if isinstance(pages, list):
            if len(pages) == 0:
                return ""
            if any(isinstance(p, bool) for p in pages):
                raise ValueError("`pages` must be integers, not booleans")
            if all(isinstance(p, int) for p in pages):
                if any(p < 0 for p in pages):
                    raise ValueError("`pages` integers must be >= 0 (Mistral 0-based indices)")
                # Mistral 0-based -> Azure 1-based.
                return ",".join(str(p + 1) for p in sorted(set(pages)))
            if all(isinstance(p, str) for p in pages):
                joined = ",".join(p.strip() for p in pages)
                if not pages_pattern.match(joined):
                    raise ValueError(
                        f"Invalid `pages` list for Azure Document Intelligence: "
                        f"{pages!r}. Expected tokens like '1' or '3-5'."
                    )
                return joined

        raise ValueError("`pages` must be a list[int] (0-based, Mistral-style) or a string like '1-3,5,7-9'.")

    @staticmethod
    def _normalize_features_param(features: object) -> str:
        """
        Convert a caller-provided `features` value to Azure DI's query-string
        form (comma-joined feature names, e.g. "keyValuePairs,languages").

        Accepted inputs:
          - list[str]: feature names like ["keyValuePairs", "languages"].
          - str: a single feature name or comma-separated names.
        """
        invalid_features_error = ValueError(
            f"Invalid `features` for Azure Document Intelligence: {features!r}. "
            f"Expected a list of feature names or a comma-separated string like "
            f"'keyValuePairs' or 'keyValuePairs,languages'."
        )

        if isinstance(features, str):
            raw_tokens = features.split(",")
        elif isinstance(features, list):
            if len(features) == 0:
                return ""
            raw_tokens = [feature for feature in features if isinstance(feature, str)]
            if len(raw_tokens) != len(features):
                raise invalid_features_error
        else:
            raise invalid_features_error

        tokens = tuple(token.strip() for token in raw_tokens)
        feature_pattern = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")
        if not all(feature_pattern.match(token) for token in tokens):
            raise invalid_features_error
        return ",".join(tokens)

    def validate_environment(
        self,
        headers: Dict,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        litellm_params: dict | None = None,
        **kwargs,
    ) -> Dict:
        """
        Validate environment and return headers for Azure Document Intelligence.

        Authentication uses Ocp-Apim-Subscription-Key header.
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = get_secret_str(AZURE_DOCUMENT_INTELLIGENCE_API_KEY_ENV_VAR)

        if api_key is None:
            raise ValueError(
                "Missing Azure Document Intelligence API Key - Set AZURE_DOCUMENT_INTELLIGENCE_API_KEY environment variable or pass api_key parameter"
            )

        # Validate API base/endpoint is provided
        if api_base is None:
            api_base = get_secret_str("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")

        if api_base is None:
            raise ValueError(
                "Missing Azure Document Intelligence Endpoint - Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT environment variable or pass api_base parameter"
            )

        headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "Content-Type": "application/json",
            **headers,
        }

        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict | None = None,
        **kwargs,
    ) -> str:
        """
        Get complete URL for Azure Document Intelligence endpoint.

        Format: {endpoint}/documentintelligence/documentModels/{modelId}:analyze?api-version=2024-11-30

        Note: API version 2024-11-30 uses /documentintelligence/ path (not /formrecognizer/)

        Args:
            api_base: Azure Document Intelligence endpoint (e.g., https://your-resource.cognitiveservices.azure.com)
            model: Model ID (e.g., "prebuilt-layout", "prebuilt-read")
            optional_params: Optional parameters

        Returns: Complete URL for Azure DI analyze endpoint
        """
        if api_base is None:
            api_base = get_secret_str("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")

        if api_base is None:
            raise ValueError(
                "Missing Azure Document Intelligence Endpoint - Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT environment variable or pass api_base parameter"
            )

        # Ensure no trailing slash
        api_base = api_base.rstrip("/")

        # Extract model ID from full model path if needed
        # Model can be "prebuilt-layout" or "azure_ai/doc-intelligence/prebuilt-layout"
        model_id = model
        if "/" in model:
            # Extract the last part after the last slash
            model_id = model.split("/")[-1]
        encoded_model_id = encode_url_path_segment(model_id, field_name="model_id")

        # Azure Document Intelligence analyze endpoint
        # Note: API version 2024-11-30+ uses /documentintelligence/ (not /formrecognizer/)
        url = (
            f"{api_base}/documentintelligence/documentModels/{encoded_model_id}:analyze"
            f"?api-version={AZURE_DOCUMENT_INTELLIGENCE_API_VERSION}"
        )

        # Azure DI accepts `pages` (1-based, e.g. "1-3,5") and `features`
        # (comma-joined names, e.g. "keyValuePairs") as query params.
        # `optional_params` has already been normalized in `map_ocr_params`.
        pages = optional_params.get("pages") if optional_params else None
        features = optional_params.get("features") if optional_params else None
        pages_query = f"&pages={quote(str(pages), safe=',-')}" if pages else ""
        features_query = f"&features={quote(str(features), safe=',')}" if features else ""

        return f"{url}{pages_query}{features_query}"

    def _extract_base64_from_data_uri(self, data_uri: str) -> str:
        """
        Extract base64 content from a data URI.

        Args:
            data_uri: Data URI like "data:application/pdf;base64,..."

        Returns:
            Base64 string without the data URI prefix
        """
        # Match pattern: data:[<mediatype>][;base64],<data>
        match = re.match(r"data:([^;]+)(?:;base64)?,(.+)", data_uri)
        if match:
            return match.group(2)
        return data_uri

    def transform_ocr_request(
        self,
        model: str,
        document: DocumentType,
        optional_params: dict,
        headers: dict,
        **kwargs,
    ) -> OCRRequestData:
        """
        Transform OCR request to Azure Document Intelligence format.

        Mistral OCR format:
        {
            "document": {
                "type": "document_url",
                "document_url": "https://example.com/doc.pdf"
            }
        }

        Azure DI format:
        {
            "urlSource": "https://example.com/doc.pdf"
        }
        OR
        {
            "base64Source": "base64_encoded_content"
        }

        Args:
            model: Model name
            document: Document dict from user (Mistral format)
            optional_params: Already mapped optional parameters
            headers: Request headers

        Returns:
            OCRRequestData with JSON data
        """
        verbose_logger.debug(f"Azure Document Intelligence transform_ocr_request - model: {model}")

        if not isinstance(document, dict):
            raise ValueError(f"Expected document dict, got {type(document)}")

        # Extract document URL from Mistral format
        doc_type = document.get("type")
        document_url = None

        if doc_type == "document_url":
            document_url = document.get("document_url", "")
        elif doc_type == "image_url":
            document_url = document.get("image_url", "")
        else:
            raise ValueError(f"Invalid document type: {doc_type}. Must be 'document_url' or 'image_url'")

        if not document_url:
            raise ValueError("Document URL is required")

        # Build Azure DI request
        data: Dict[str, Any] = {}

        # Check if it's a data URI (base64)
        if document_url.startswith("data:"):
            # Extract base64 content
            base64_content = self._extract_base64_from_data_uri(document_url)
            data["base64Source"] = base64_content
            verbose_logger.debug("Using base64Source for Azure Document Intelligence")
        else:
            # Regular URL
            data["urlSource"] = document_url
            verbose_logger.debug("Using urlSource for Azure Document Intelligence")

        # Azure DI: `pages` is a query param (wired in get_complete_url),
        # not a body field. Other Mistral-specific params (e.g.
        # include_image_base64, image_limit) are unsupported and ignored.

        return OCRRequestData(data=data, files=None)

    def _transform_azure_page(self, azure_page: AzureDocumentIntelligencePage) -> OCRPage:
        page_number = azure_page.pageNumber if azure_page.pageNumber is not None else 1
        markdown = "\n".join(line.content or "" for line in azure_page.lines)
        dimensions = self._convert_dimensions(
            width=azure_page.width if azure_page.width is not None else 8.5,
            height=azure_page.height if azure_page.height is not None else 11,
            unit=azure_page.unit if azure_page.unit is not None else "inch",
        )
        return OCRPage(index=page_number - 1, markdown=markdown, dimensions=dimensions)

    def _convert_dimensions(self, width: float, height: float, unit: str) -> OCRPageDimensions:
        """
        Convert Azure DI dimensions to pixels.

        Azure DI provides dimensions in inches. We convert to pixels using configured DPI.

        Args:
            width: Width in specified unit
            height: Height in specified unit
            unit: Unit of measurement (e.g., "inch")

        Returns:
            OCRPageDimensions with pixel values
        """
        # Convert to pixels using configured DPI
        dpi = AZURE_DOCUMENT_INTELLIGENCE_DEFAULT_DPI
        if unit == "inch":
            width_px = int(width * dpi)
            height_px = int(height * dpi)
        else:
            # If unit is not inches, assume it's already in pixels
            width_px = int(width)
            height_px = int(height)

        return OCRPageDimensions(width=width_px, height=height_px, dpi=dpi)

    @staticmethod
    def _check_timeout(start_time: float, timeout_secs: int) -> None:
        """
        Check if operation has timed out.

        Args:
            start_time: Start time of the operation
            timeout_secs: Timeout duration in seconds

        Raises:
            TimeoutError: If operation has exceeded timeout
        """
        if time.time() - start_time > timeout_secs:
            raise TimeoutError(f"Azure Document Intelligence operation polling timed out after {timeout_secs} seconds")

    @staticmethod
    def _get_retry_after(response: httpx.Response) -> int:
        """
        Get retry-after duration from response headers.

        Args:
            response: HTTP response

        Returns:
            Retry-after duration in seconds (default: 2)
        """
        retry_after = int(response.headers.get("retry-after", "2"))
        verbose_logger.debug(f"Retry polling after: {retry_after} seconds")
        return retry_after

    @staticmethod
    def _check_operation_status(response: httpx.Response) -> str:
        """
        Check Azure DI operation status from response.

        Args:
            response: HTTP response from operation endpoint

        Returns:
            Operation status string

        Raises:
            ValueError: If operation failed or status is unknown
        """
        try:
            result = response.json()
            status = result.get("status")

            verbose_logger.debug(f"Azure DI operation status: {status}")

            if status == "succeeded":
                return "succeeded"
            elif status == "failed":
                error_msg = result.get("error", {}).get("message", "Unknown error")
                raise ValueError(f"Azure Document Intelligence analysis failed: {error_msg}")
            elif status in ["running", "notStarted"]:
                return "running"
            else:
                raise ValueError(f"Unknown operation status: {status}")

        except Exception as e:
            if "succeeded" in str(e) or "failed" in str(e):
                raise
            # If we can't parse JSON, something went wrong
            raise ValueError(f"Failed to parse Azure DI operation response: {e}")

    def _poll_operation_sync(
        self,
        operation_url: str,
        headers: Dict[str, str],
        timeout_secs: int,
    ) -> httpx.Response:
        """
        Poll Azure Document Intelligence operation until completion (sync).

        Azure DI POST returns 202 with Operation-Location header.
        We need to poll that URL until status is "succeeded" or "failed".

        Args:
            operation_url: The Operation-Location URL to poll
            headers: Request headers (including auth)
            timeout_secs: Total timeout in seconds

        Returns:
            Final response with completed analysis
        """
        from litellm.llms.custom_httpx.http_handler import _get_httpx_client

        client = _get_httpx_client()
        start_time = time.time()

        verbose_logger.debug(f"Polling Azure DI operation: {operation_url}")

        while True:
            self._check_timeout(start_time=start_time, timeout_secs=timeout_secs)

            # Poll the operation status
            response = client.get(url=operation_url, headers=headers)

            # Check operation status
            status = self._check_operation_status(response=response)

            if status == "succeeded":
                return response
            elif status == "running":
                # Wait before polling again
                retry_after = self._get_retry_after(response=response)
                time.sleep(retry_after)

    async def _poll_operation_async(
        self,
        operation_url: str,
        headers: Dict[str, str],
        timeout_secs: int,
    ) -> httpx.Response:
        """
        Poll Azure Document Intelligence operation until completion (async).

        Args:
            operation_url: The Operation-Location URL to poll
            headers: Request headers (including auth)
            timeout_secs: Total timeout in seconds

        Returns:
            Final response with completed analysis
        """
        import litellm
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

        client = get_async_httpx_client(llm_provider=litellm.LlmProviders.AZURE_AI)
        start_time = time.time()

        verbose_logger.debug(f"Polling Azure DI operation (async): {operation_url}")

        while True:
            self._check_timeout(start_time=start_time, timeout_secs=timeout_secs)

            # Poll the operation status
            response = await client.get(url=operation_url, headers=headers)

            # Check operation status
            status = self._check_operation_status(response=response)

            if status == "succeeded":
                return response
            elif status == "running":
                # Wait before polling again
                retry_after = self._get_retry_after(response=response)
                await asyncio.sleep(retry_after)

    def _get_polling_target(self, raw_response: httpx.Response) -> tuple[str, Dict[str, str]]:
        operation_url = raw_response.headers.get("Operation-Location")
        if not operation_url:
            raise ValueError("Azure Document Intelligence returned 202 but no Operation-Location header found")

        # Reject cross-origin polling URLs — the auth headers
        # below would otherwise leak to whatever URL the upstream
        # (or an attacker-controlled upstream) returns. VERIA-51.
        try:
            assert_same_origin(operation_url, str(raw_response.request.url))
        except SSRFError as ssrf_err:
            raise ValueError(f"Azure Document Intelligence: rejected polling URL ({ssrf_err})")

        poll_headers = {"Ocp-Apim-Subscription-Key": raw_response.request.headers.get("Ocp-Apim-Subscription-Key", "")}
        return operation_url, poll_headers

    def _transform_completed_response(self, model: str, raw_response: httpx.Response) -> OCRResponse:
        """
        Transform a completed Azure Document Intelligence analyze operation
        into the Mistral OCR response shape, preserving Azure-native
        `analyzeResult` fields (`content`, `tables`, `keyValuePairs`) as
        top-level response fields.
        """
        operation = AzureDocumentIntelligenceOperation.model_validate(raw_response.json())

        verbose_logger.debug(f"Azure Document Intelligence response status: {operation.status}")

        if operation.status != "succeeded":
            raise ValueError(f"Azure Document Intelligence analysis failed with status: {operation.status}")

        analyze_result = (
            operation.analyzeResult if operation.analyzeResult is not None else AzureDocumentIntelligenceAnalyzeResult()
        )
        mistral_pages = [self._transform_azure_page(azure_page) for azure_page in analyze_result.pages]
        usage_info = OCRUsageInfo(pages_processed=len(mistral_pages), doc_size_bytes=None)

        return OCRResponse(
            pages=mistral_pages,
            model=model,
            usage_info=usage_info,
            object="ocr",
            content=analyze_result.content,
            tables=analyze_result.tables,
            keyValuePairs=analyze_result.keyValuePairs,
        )

    def transform_ocr_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
        **kwargs,
    ) -> OCRResponse:
        """
        Transform Azure Document Intelligence response to Mistral OCR format.

        Handles async operation polling: If response is 202 Accepted, polls Operation-Location
        until analysis completes.

        Azure DI response (after polling):
        {
            "status": "succeeded",
            "analyzeResult": {
                "content": "Full document text...",
                "pages": [
                    {
                        "pageNumber": 1,
                        "width": 8.5,
                        "height": 11,
                        "unit": "inch",
                        "lines": [{"content": "text", "boundingBox": [...]}]
                    }
                ],
                "tables": [...],
                "keyValuePairs": [...]
            }
        }

        Mistral OCR format (with Azure-native fields preserved):
        {
            "pages": [
                {
                    "index": 0,
                    "markdown": "extracted text",
                    "dimensions": {"width": 816, "height": 1056, "dpi": 96}
                }
            ],
            "model": "azure_ai/doc-intelligence/prebuilt-layout",
            "usage_info": {"pages_processed": 1},
            "object": "ocr",
            "content": "Full document text...",
            "tables": [...],
            "keyValuePairs": [...]
        }

        Args:
            model: Model name
            raw_response: Raw HTTP response from Azure DI (may be 202 Accepted)
            logging_obj: Logging object

        Returns:
            OCRResponse in Mistral format
        """
        if raw_response.status_code != 202:
            return self._transform_completed_response(model=model, raw_response=raw_response)

        verbose_logger.debug("Azure DI returned 202 Accepted, polling operation...")
        operation_url, poll_headers = self._get_polling_target(raw_response)
        completed_response = self._poll_operation_sync(
            operation_url=operation_url,
            headers=poll_headers,
            timeout_secs=AZURE_OPERATION_POLLING_TIMEOUT,
        )
        return self._transform_completed_response(model=model, raw_response=completed_response)

    async def async_transform_ocr_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
        **kwargs,
    ) -> OCRResponse:
        """
        Async transform Azure Document Intelligence response to Mistral OCR format.

        Handles async operation polling: If response is 202 Accepted, polls Operation-Location
        until analysis completes using async polling.

        Args:
            model: Model name
            raw_response: Raw HTTP response from Azure DI (may be 202 Accepted)
            logging_obj: Logging object

        Returns:
            OCRResponse in Mistral format
        """
        if raw_response.status_code != 202:
            return self._transform_completed_response(model=model, raw_response=raw_response)

        verbose_logger.debug("Azure DI returned 202 Accepted, polling operation (async)...")
        operation_url, poll_headers = self._get_polling_target(raw_response)
        completed_response = await self._poll_operation_async(
            operation_url=operation_url,
            headers=poll_headers,
            timeout_secs=AZURE_OPERATION_POLLING_TIMEOUT,
        )
        return self._transform_completed_response(model=model, raw_response=completed_response)

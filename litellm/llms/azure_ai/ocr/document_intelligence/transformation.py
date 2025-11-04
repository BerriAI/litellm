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
from typing import Any, Dict, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.constants import (
    AZURE_DOCUMENT_INTELLIGENCE_API_VERSION,
    AZURE_DOCUMENT_INTELLIGENCE_DEFAULT_DPI,
    AZURE_OPERATION_POLLING_TIMEOUT,
)
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

    def get_supported_ocr_params(self, model: str) -> list:
        """
        Get supported OCR parameters for Azure Document Intelligence.
        
        Azure DI has minimal optional parameters compared to Mistral OCR.
        Most Mistral-specific params are ignored during transformation.
        """
        return []

    def validate_environment(
        self,
        headers: Dict,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        litellm_params: Optional[dict] = None,
        **kwargs,
    ) -> Dict:
        """
        Validate environment and return headers for Azure Document Intelligence.
        
        Authentication uses Ocp-Apim-Subscription-Key header.
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = get_secret_str("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")

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
        api_base: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: Optional[dict] = None,
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

        # Azure Document Intelligence analyze endpoint
        # Note: API version 2024-11-30+ uses /documentintelligence/ (not /formrecognizer/)
        return f"{api_base}/documentintelligence/documentModels/{model_id}:analyze?api-version={AZURE_DOCUMENT_INTELLIGENCE_API_VERSION}"

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
        verbose_logger.debug(
            f"Azure Document Intelligence transform_ocr_request - model: {model}"
        )

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
            raise ValueError(
                f"Invalid document type: {doc_type}. Must be 'document_url' or 'image_url'"
            )

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

        # Azure DI doesn't support most Mistral-specific params
        # Ignore pages, include_image_base64, etc.

        return OCRRequestData(data=data, files=None)

    def _extract_page_markdown(self, page_data: Dict[str, Any]) -> str:
        """
        Extract text from Azure DI page and format as markdown.
        
        Azure DI provides text in 'lines' array. We concatenate them with newlines.
        
        Args:
            page_data: Azure DI page object
            
        Returns:
            Markdown-formatted text
        """
        lines = page_data.get("lines", [])
        if not lines:
            return ""

        # Extract text content from each line
        text_lines = [line.get("content", "") for line in lines]

        # Join with newlines to preserve structure
        return "\n".join(text_lines)

    def _convert_dimensions(
        self, width: float, height: float, unit: str
    ) -> OCRPageDimensions:
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
            raise TimeoutError(
                f"Azure Document Intelligence operation polling timed out after {timeout_secs} seconds"
            )

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
                raise ValueError(
                    f"Azure Document Intelligence analysis failed: {error_msg}"
                )
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
                ]
            }
        }
        
        Mistral OCR format:
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
            "object": "ocr"
        }
        
        Args:
            model: Model name
            raw_response: Raw HTTP response from Azure DI (may be 202 Accepted)
            logging_obj: Logging object
            
        Returns:
            OCRResponse in Mistral format
        """
        try:
            # Check if we got 202 Accepted (async operation started)
            if raw_response.status_code == 202:
                verbose_logger.debug(
                    "Azure DI returned 202 Accepted, polling operation..."
                )

                # Get Operation-Location header
                operation_url = raw_response.headers.get("Operation-Location")
                if not operation_url:
                    raise ValueError(
                        "Azure Document Intelligence returned 202 but no Operation-Location header found"
                    )

                # Get headers for polling (need auth)
                poll_headers = {
                    "Ocp-Apim-Subscription-Key": raw_response.request.headers.get(
                        "Ocp-Apim-Subscription-Key", ""
                    )
                }

                # Get timeout from kwargs or use default
                timeout_secs = AZURE_OPERATION_POLLING_TIMEOUT

                # Poll until operation completes
                raw_response = self._poll_operation_sync(
                    operation_url=operation_url,
                    headers=poll_headers,
                    timeout_secs=timeout_secs,
                )

            # Now parse the completed response
            response_json = raw_response.json()

            verbose_logger.debug(
                f"Azure Document Intelligence response status: {response_json.get('status')}"
            )

            # Check if request succeeded
            status = response_json.get("status")
            if status != "succeeded":
                raise ValueError(
                    f"Azure Document Intelligence analysis failed with status: {status}"
                )

            # Extract analyze result
            analyze_result = response_json.get("analyzeResult", {})
            azure_pages = analyze_result.get("pages", [])

            # Transform pages to Mistral format
            mistral_pages = []
            for azure_page in azure_pages:
                page_number = azure_page.get("pageNumber", 1)
                index = page_number - 1  # Convert to 0-based index

                # Extract markdown text
                markdown = self._extract_page_markdown(azure_page)

                # Convert dimensions
                width = azure_page.get("width", 8.5)
                height = azure_page.get("height", 11)
                unit = azure_page.get("unit", "inch")
                dimensions = self._convert_dimensions(
                    width=width, height=height, unit=unit
                )

                # Build OCR page
                ocr_page = OCRPage(
                    index=index, markdown=markdown, dimensions=dimensions
                )
                mistral_pages.append(ocr_page)

            # Build usage info
            usage_info = OCRUsageInfo(
                pages_processed=len(mistral_pages), doc_size_bytes=None
            )

            # Return Mistral OCR response
            return OCRResponse(
                pages=mistral_pages,
                model=model,
                usage_info=usage_info,
                object="ocr",
            )

        except Exception as e:
            verbose_logger.error(
                f"Error parsing Azure Document Intelligence response: {e}"
            )
            raise e

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
        try:
            # Check if we got 202 Accepted (async operation started)
            if raw_response.status_code == 202:
                verbose_logger.debug(
                    "Azure DI returned 202 Accepted, polling operation (async)..."
                )

                # Get Operation-Location header
                operation_url = raw_response.headers.get("Operation-Location")
                if not operation_url:
                    raise ValueError(
                        "Azure Document Intelligence returned 202 but no Operation-Location header found"
                    )

                # Get headers for polling (need auth)
                poll_headers = {
                    "Ocp-Apim-Subscription-Key": raw_response.request.headers.get(
                        "Ocp-Apim-Subscription-Key", ""
                    )
                }

                # Get timeout from kwargs or use default
                timeout_secs = AZURE_OPERATION_POLLING_TIMEOUT

                # Poll until operation completes (async)
                raw_response = await self._poll_operation_async(
                    operation_url=operation_url,
                    headers=poll_headers,
                    timeout_secs=timeout_secs,
                )

            # Now parse the completed response
            response_json = raw_response.json()

            verbose_logger.debug(
                f"Azure Document Intelligence response status: {response_json.get('status')}"
            )

            # Check if request succeeded
            status = response_json.get("status")
            if status != "succeeded":
                raise ValueError(
                    f"Azure Document Intelligence analysis failed with status: {status}"
                )

            # Extract analyze result
            analyze_result = response_json.get("analyzeResult", {})
            azure_pages = analyze_result.get("pages", [])

            # Transform pages to Mistral format
            mistral_pages = []
            for azure_page in azure_pages:
                page_number = azure_page.get("pageNumber", 1)
                index = page_number - 1  # Convert to 0-based index

                # Extract markdown text
                markdown = self._extract_page_markdown(azure_page)

                # Convert dimensions
                width = azure_page.get("width", 8.5)
                height = azure_page.get("height", 11)
                unit = azure_page.get("unit", "inch")
                dimensions = self._convert_dimensions(
                    width=width, height=height, unit=unit
                )

                # Build OCR page
                ocr_page = OCRPage(
                    index=index, markdown=markdown, dimensions=dimensions
                )
                mistral_pages.append(ocr_page)

            # Build usage info
            usage_info = OCRUsageInfo(
                pages_processed=len(mistral_pages), doc_size_bytes=None
            )

            # Return Mistral OCR response
            return OCRResponse(
                pages=mistral_pages,
                model=model,
                usage_info=usage_info,
                object="ocr",
            )

        except Exception as e:
            verbose_logger.error(
                f"Error parsing Azure Document Intelligence response (async): {e}"
            )
            raise e


"""
Vertex AI DeepSeek OCR transformation implementation.
"""
import json
from typing import TYPE_CHECKING, Any, Dict, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.llms.base_llm.ocr.transformation import (
    BaseOCRConfig,
    DocumentType,
    OCRPage,
    OCRRequestData,
    OCRResponse,
    OCRUsageInfo,
)
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class VertexAIDeepSeekOCRConfig(BaseOCRConfig):
    """
    Vertex AI DeepSeek OCR transformation configuration.
    
    Vertex AI DeepSeek OCR uses the chat completion API format through the openapi endpoint.
    This transformation converts OCR requests to chat completion format and vice versa.
    """

    def __init__(self) -> None:
        super().__init__()
        self.vertex_base = VertexBase()

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
        Validate environment and return headers for Vertex AI OCR.
        
        Vertex AI uses Bearer token authentication with access token from credentials.
        """
        # Extract Vertex AI parameters using safe helpers from VertexBase
        # Use safe_get_* methods that don't mutate litellm_params dict
        litellm_params = litellm_params or {}
        
        vertex_project = VertexBase.safe_get_vertex_ai_project(litellm_params=litellm_params)
        vertex_credentials = VertexBase.safe_get_vertex_ai_credentials(litellm_params=litellm_params)
        
        # Get access token from Vertex credentials
        access_token, project_id = self.vertex_base.get_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
        )

        headers = {
            "Authorization": f"Bearer {access_token}",
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
        Get complete URL for Vertex AI DeepSeek OCR endpoint.
        
        Vertex AI endpoint format: 
        https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi/chat/completions
        
        Args:
            api_base: Vertex AI API base URL (optional)
            model: Model name (e.g., "deepseek-ai/deepseek-ocr-maas")
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters containing vertex_project, vertex_location
            
        Returns: Complete URL for Vertex AI OCR endpoint
        """
        # Extract Vertex AI parameters using safe helpers from VertexBase
        # Use safe_get_* methods that don't mutate litellm_params dict
        litellm_params = litellm_params or {}
        
        vertex_project = VertexBase.safe_get_vertex_ai_project(litellm_params=litellm_params)
        vertex_location = VertexBase.safe_get_vertex_ai_location(litellm_params=litellm_params)
        
        if vertex_project is None:
            raise ValueError(
                "Missing vertex_project - Set VERTEXAI_PROJECT environment variable or pass vertex_project parameter"
            )

        if vertex_location is None:
            vertex_location = "us-central1"

        # Get API base URL
        if api_base is None:
            api_base = "https://aiplatform.googleapis.com"

        # Ensure no trailing slash
        api_base = api_base.rstrip("/")
        
        # Vertex AI DeepSeek OCR endpoint format
        # Format: https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/endpoints/openapi/chat/completions
        return f"{api_base}/v1/projects/{vertex_project}/locations/{vertex_location}/endpoints/openapi/chat/completions"

    def transform_ocr_request(
        self,
        model: str,
        document: DocumentType,
        optional_params: dict,
        headers: dict,
        **kwargs,
    ) -> OCRRequestData:
        """
        Transform OCR request to chat completion format for Vertex AI DeepSeek OCR.
        
        Converts OCR document format to chat completion messages format:
        - Input: {"type": "image_url", "image_url": "gs://..."}
        - Output: {"model": "deepseek-ai/deepseek-ocr-maas", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "gs://..."}]}]}
        
        Args:
            model: Model name (e.g., "deepseek-ai/deepseek-ocr-maas")
            document: Document dict from user (Mistral OCR format)
            optional_params: Already mapped optional parameters
            headers: Request headers
            **kwargs: Additional arguments
            
        Returns:
            OCRRequestData with JSON data in chat completion format
        """
        verbose_logger.debug("Vertex AI DeepSeek OCR transform_ocr_request (sync) called")
        
        if not isinstance(document, dict):
            raise ValueError(f"Expected document dict, got {type(document)}")
        
        # Extract document type and URL
        doc_type = document.get("type")
        image_url = None
        document_url = None
        
        if doc_type == "image_url":
            image_url = document.get("image_url", "")
        elif doc_type == "document_url":
            document_url = document.get("document_url", "")
        else:
            raise ValueError(f"Unsupported document type: {doc_type}. Expected 'image_url' or 'document_url'")
        
        # Build chat completion message content
        content_item = {}
        if image_url:
            content_item = {
                "type": "image_url",
                "image_url": image_url
            }
        elif document_url:
            # For document URLs, we use image_url type as well (Vertex AI supports both)
            content_item = {
                "type": "image_url",
                "image_url": document_url
            }
        
        # Build chat completion request
        data = {
            "model": "deepseek-ai/" + model,
            "messages": [
                {
                    "role": "user",
                    "content": [content_item]
                }
            ]
        }
        
        # Add optional parameters (stream, temperature, etc.)
        # Filter out OCR-specific params that don't apply to chat completion
        chat_completion_params = {}
        for key, value in optional_params.items():
            # Include common chat completion params
            if key in ["stream", "temperature", "max_tokens", "top_p", "n", "stop"]:
                chat_completion_params[key] = value
        
        data.update(chat_completion_params)
        
        verbose_logger.debug("Vertex AI DeepSeek OCR: Transformed request to chat completion format")
        
        return OCRRequestData(data=data, files=None)

    async def async_transform_ocr_request(
        self,
        model: str,
        document: DocumentType,
        optional_params: dict,
        headers: dict,
        **kwargs,
    ) -> OCRRequestData:
        """
        Transform OCR request to chat completion format for Vertex AI DeepSeek OCR (async).
        
        Same as sync version - no async-specific logic needed.
        
        Args:
            model: Model name
            document: Document dict from user
            optional_params: Already mapped optional parameters
            headers: Request headers
            **kwargs: Additional arguments
            
        Returns:
            OCRRequestData with JSON data in chat completion format
        """
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
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> OCRResponse:
        """
        Transform chat completion response to OCR format.
        
        Vertex AI DeepSeek OCR returns chat completion format:
        {
            "id": "...",
            "object": "chat.completion",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "<OCR result as JSON string or markdown>"
                }
            }],
            "usage": {...}
        }
        
        We need to extract the content and convert it to OCRResponse format.
        
        Args:
            model: Model name
            raw_response: Raw HTTP response from Vertex AI
            logging_obj: Logging object
            **kwargs: Additional arguments
            
        Returns:
            OCRResponse in standard format
        """
        verbose_logger.debug("Vertex AI DeepSeek OCR transform_ocr_response called")
        verbose_logger.debug(f"Raw response: {raw_response.text}")
        
        try:
            response_json = raw_response.json()
            
            # Extract content from chat completion response
            choices = response_json.get("choices", [])
            if not choices:
                raise ValueError("No choices in chat completion response")
            
            message = choices[0].get("message", {})
            content = message.get("content", "")
            
            if not content:
                raise ValueError("No content in chat completion response")
            
            # Try to parse content as JSON (OCR result might be JSON string)
            ocr_data = None
            try:
                # If content is a JSON string, parse it
                if isinstance(content, str) and content.strip().startswith("{"):
                    ocr_data = json.loads(content)
                elif isinstance(content, dict):
                    ocr_data = content
                else:
                    # If content is markdown text, create a single page with the markdown
                    ocr_data = {
                        "pages": [
                            {
                                "index": 0,
                                "markdown": content
                            }
                        ],
                        "model": model,
                        "usage_info": response_json.get("usage", {})
                    }
            except json.JSONDecodeError:
                # If JSON parsing fails, treat content as markdown
                ocr_data = {
                    "pages": [
                        {
                            "index": 0,
                            "markdown": content
                        }
                    ],
                    "model": model,
                    "usage_info": response_json.get("usage", {})
                }
            
            # Ensure we have the expected structure
            if "pages" not in ocr_data:
                # If OCR data doesn't have pages, wrap the content in a page
                ocr_data = {
                    "pages": [
                        {
                            "index": 0,
                            "markdown": content if isinstance(content, str) else json.dumps(content)
                        }
                    ],
                    "model": ocr_data.get("model", model),
                    "usage_info": ocr_data.get("usage_info", response_json.get("usage", {}))
                }
            
            # Convert usage info if present
            usage_info = None
            if "usage_info" in ocr_data:
                usage_dict = ocr_data["usage_info"]
                if isinstance(usage_dict, dict):
                    usage_info = OCRUsageInfo(**usage_dict)
            
            # Build OCRResponse
            pages = []
            for page_data in ocr_data.get("pages", []):
                # Ensure page has required fields
                if isinstance(page_data, dict):
                    page = OCRPage(
                        index=page_data.get("index", 0),
                        markdown=page_data.get("markdown", ""),
                        images=page_data.get("images"),
                        dimensions=page_data.get("dimensions")
                    )
                    pages.append(page)
            
            if not pages:
                # Create a default page if none exist
                pages = [OCRPage(index=0, markdown=content if isinstance(content, str) else "")]
            
            return OCRResponse(
                pages=pages,
                model=ocr_data.get("model", model),
                document_annotation=ocr_data.get("document_annotation"),
                usage_info=usage_info,
                object="ocr",
            )
            
        except Exception as e:
            verbose_logger.error(f"Error parsing Vertex AI DeepSeek OCR response: {e}")
            raise e

    async def async_transform_ocr_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> OCRResponse:
        """
        Async transform chat completion response to OCR format.
        
        Same as sync version - no async-specific logic needed.
        
        Args:
            model: Model name
            raw_response: Raw HTTP response
            logging_obj: Logging object
            **kwargs: Additional arguments
            
        Returns:
            OCRResponse in standard format
        """
        return self.transform_ocr_response(
            model=model,
            raw_response=raw_response,
            logging_obj=logging_obj,
            **kwargs,
        )


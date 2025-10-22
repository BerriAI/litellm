from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, IO, cast
from io import BufferedReader
import json

import httpx
from httpx._types import RequestFiles

from litellm.types.videos.main import VideoCreateOptionalRequestParams
from litellm.types.llms.azure import AzureCreateVideoRequest
from litellm.types.videos.main import VideoResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.secret_managers.main import get_secret_str
from litellm.types.videos.main import VideoObject
from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm._logging import verbose_logger
from litellm.utils import _add_path_to_api_base
import litellm

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    from ...base_llm.videos.transformation import BaseVideoConfig as _BaseVideoConfig
    from ...base_llm.chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseVideoConfig = _BaseVideoConfig
    BaseLLMException = _BaseLLMException
else:
    LiteLLMLoggingObj = Any
    BaseVideoConfig = Any
    BaseLLMException = Any


class AzureVideoConfig(BaseVideoConfig):
    """
    Configuration class for Azure video generation.
    """

    def __init__(self):
        super().__init__()

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the list of supported Azure parameters for video generation.
        """
        return [
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "n_variants",
            "inpaint_items",
            "user",
            "extra_headers",
        ]

    def _convert_input_reference_to_inpaint_items(
        self, input_reference: Any
    ) -> List[Dict[str, Any]]:
        """
        Convert input_reference to inpaint_items format for Azure video generation.
        
        Args:
            input_reference: File path, file object, or BufferedReader
            
        Returns:
            List of inpaint item dictionaries
        """
        inpaint_items = []
        
        if input_reference is None:
            return inpaint_items
            
        # Determine file name and type
        if isinstance(input_reference, BufferedReader):
            file_name = input_reference.name
        elif isinstance(input_reference, str):
            file_name = input_reference
        else:
            file_name = "input_reference.png"
        
        # Determine file type based on extension
        file_type = "image"
        if file_name.lower().endswith(('.mp4', '.mov', '.avi')):
            file_type = "video"
        
        # Create inpaint item
        inpaint_item = {
            "type": file_type,
            "file_name": file_name,
            "frame_index": 0,  # Default to first frame
            "crop_bounds": {
                "left_fraction": 0.0,
                "top_fraction": 0.0,
                "right_fraction": 1.0,
                "bottom_fraction": 1.0
            }
        }
        
        inpaint_items.append(inpaint_item)
        return inpaint_items

    def _get_file_content_and_type(
        self, file_path: str
    ) -> Tuple[bytes, str]:
        """
        Helper method to read file content and determine MIME type.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (file_content, mime_type)
        """
        try:
            with open(file_path, "rb") as f:
                file_content = f.read()
            
            # Determine MIME type based on extension
            if file_path.lower().endswith('.mp4'):
                mime_type = "video/mp4"
            elif file_path.lower().endswith('.mov'):
                mime_type = "video/quicktime"
            elif file_path.lower().endswith('.png'):
                mime_type = "image/png"
            elif file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg'):
                mime_type = "image/jpeg"
            else:
                mime_type = "image/jpeg"  # Default fallback
                
            return file_content, mime_type
        except Exception as e:
            verbose_logger.debug(f"Could not read file {file_path}: {e}")
            return b"", "image/jpeg"

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        """
        Map OpenAI parameters to Azure parameters.
        
        Azure uses different parameter names:
        - seconds (OpenAI) -> n_seconds (Azure)
        - size (OpenAI "WxH") -> height + width (Azure separate fields)
        - input_reference (OpenAI) -> inpaint_items (Azure)
        """
        mapped_params = dict(video_create_optional_params)
        
        # Handle size parameter conversion
        if "size" in mapped_params and "size" not in ("height", "width"):
            size = mapped_params.pop("size")
            if size:
                try:
                    # Format: "720x1280" -> width=720, height=1280
                    parts = str(size).split("x")
                    if len(parts) == 2:
                        mapped_params["width"] = int(parts[0])
                        mapped_params["height"] = int(parts[1])
                except (ValueError, IndexError):
                    pass
        
        # Handle seconds parameter conversion
        if "seconds" in mapped_params and "n_seconds" not in mapped_params:
            mapped_params["n_seconds"] = mapped_params.pop("seconds")
        
        # Handle input_reference to inpaint_items conversion
        if "input_reference" in mapped_params and "inpaint_items" not in mapped_params:
            input_reference = mapped_params.pop("input_reference")
            if input_reference is not None:
                inpaint_items = self._convert_input_reference_to_inpaint_items(input_reference)
                if inpaint_items:
                    mapped_params["inpaint_items"] = inpaint_items
        
        return mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        api_key = (
            api_key
            or litellm.api_key
            or litellm.azure_key
            or get_secret_str("AZURE_OPENAI_API_KEY")
            or get_secret_str("AZURE_API_KEY")
        )

        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
            }
        )
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for Azure video generation.
        
        Azure Video Generation URL format:
        {api_base}/openai/v1/video/generations/jobs?api-version=preview
        """
        return BaseAzureLLM._get_base_azure_url(
            api_base=api_base,
            litellm_params=litellm_params,
            route="/openai/v1/video/generations/jobs",
        )

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles]:
        """
        Transform the video creation request for Azure API.
        
        Handles both regular video generation and inpainting with multipart form data.
        Azure API expects form-data with JSON-encoded inpaint_items and file uploads.
        """
        # Create a copy to avoid modifying the original
        optional_params = dict(video_create_optional_request_params)
        
        # Remove parameters that shouldn't be in the request body
        optional_params.pop("extra_headers", None)
        optional_params.pop("input_reference", None)  # Handle separately
        
        # Map OpenAI params to Azure params
        optional_params = self.map_openai_params(
            optional_params, model, drop_params=False
        )
        
        # Create the request data
        data_dict: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
        }
        
        # Add optional parameters to data
        for key, value in optional_params.items():
            if value is not None and key not in ["inpaint_items"]:
                if key == "n_seconds":
                    data_dict["n_seconds"] = str(value)
                elif key == "height":
                    data_dict["height"] = str(value)
                elif key == "width":
                    data_dict["width"] = str(value)
                elif key == "n_variants":
                    data_dict["n_variants"] = str(value)
                else:
                    data_dict[key] = value
        
        # Handle file uploads for inpainting
        files_list: List[Tuple[str, Tuple[str, Union[IO[bytes], bytes], str]]] = []
        
        # Handle inpaint_items parameter if provided
        inpaint_items = optional_params.get("inpaint_items")
        if inpaint_items:
            # For multipart requests, inpaint_items must be JSON string
            data_dict["inpaint_items"] = json.dumps(inpaint_items)
            
            # Extract files from inpaint_items using helper method
            for item in inpaint_items:
                if isinstance(item, dict) and "file_name" in item:
                    file_name = item["file_name"]
                    file_content, mime_type = self._get_file_content_and_type(file_name)
                    if file_content:  # Only add if file was successfully read
                        files_list.append(
                            ("files", (file_name, file_content, mime_type))
                        )
        
        # Handle input_reference parameter if provided (this should now be converted to inpaint_items)
        # But we still need to handle it for backward compatibility
        _input_reference = video_create_optional_request_params.get("input_reference")
        if _input_reference is not None and not inpaint_items:
            # Convert input_reference to inpaint_items format
            inpaint_items = self._convert_input_reference_to_inpaint_items(_input_reference)
            if inpaint_items:
                data_dict["inpaint_items"] = json.dumps(inpaint_items)
                
                # Handle file upload for the input_reference
                if isinstance(_input_reference, BufferedReader):
                    file_name = _input_reference.name
                    file_content, mime_type = self._get_file_content_and_type(file_name)
                    if file_content:
                        files_list.append(
                            ("files", (file_name, file_content, mime_type))
                        )
                elif isinstance(_input_reference, str):
                    file_content, mime_type = self._get_file_content_and_type(_input_reference)
                    if file_content:
                        files_list.append(
                            ("files", (_input_reference, file_content, mime_type))
                        )
                else:
                    # Handle file-like object
                    files_list.append(
                        ("files", ("input_reference.png", _input_reference, "image/png"))
                    )
        
        verbose_logger.debug(f"Azure video request data: {data_dict}")
        verbose_logger.debug(f"Azure video request files: {[f[0] for f in files_list]}")
        
        return data_dict, files_list

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        """
        Transform the Azure video creation response.
        """
        response_data = raw_response.json()
        
        # Transform the response data
    
        video_obj = VideoObject(**response_data)
        
        # Create usage object with duration information for cost calculation
        usage_data = {}
        if video_obj:
            if hasattr(video_obj, "seconds") and video_obj.seconds:
                try:
                    usage_data["duration_seconds"] = float(video_obj.seconds)
                except (ValueError, TypeError):
                    pass
        
        video_obj.usage = usage_data
        
        return video_obj
    
    def transform_video_content_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        """
        Transform the Azure video content download response.
        Returns raw video content as bytes.
        """
        return raw_response.content

    def transform_video_remix_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        """
        Transform the Azure video remix response.
        """
        response_data = raw_response.json()
        
        video_obj = VideoObject(**response_data)
        
        # Create usage object with duration information for cost calculation
        usage_data = {}
        if video_obj:
            if hasattr(video_obj, "seconds") and video_obj.seconds:
                try:
                    usage_data["duration_seconds"] = float(video_obj.seconds)
                except (ValueError, TypeError):
                    pass
        
        video_obj.usage = usage_data

        return video_obj

    def transform_video_list_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> List[VideoObject]:
        """
        Transform the Azure video list response.
        """
        response_data = raw_response.json()
        video_response = VideoResponse(**response_data)
        # Convert VideoResponse object to dictionary to match base class return type
        return [VideoObject(**video) for video in video_response.data]

    def transform_video_delete_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        """
        Transform the Azure video delete response.
        """
        response_data = raw_response.json()
        
        video_obj = VideoObject(**response_data)

        return video_obj

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        from ...base_llm.chat.transformation import BaseLLMException

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

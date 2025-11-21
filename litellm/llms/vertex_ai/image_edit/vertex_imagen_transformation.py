import base64
import json
import os
from io import BufferedRandom, BufferedReader, BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import httpx
from httpx._types import RequestFiles

import litellm

from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageObject, ImageResponse, OpenAIImage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class VertexAIImagenImageEditConfig(BaseImageEditConfig, VertexLLM):
    """
    Vertex AI Imagen Image Edit Configuration
    
    Uses predict API for Imagen models on Vertex AI
    """
    SUPPORTED_PARAMS: List[str] = ["n", "size", "mask"]

    def __init__(self) -> None:
        BaseImageEditConfig.__init__(self)
        VertexLLM.__init__(self)

    def get_supported_openai_params(self, model: str) -> List[str]:
        return list(self.SUPPORTED_PARAMS)

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict[str, Any]:
        supported_params = self.get_supported_openai_params(model)
        filtered_params = {
            key: value
            for key, value in image_edit_optional_params.items()
            if key in supported_params
        }

        mapped_params: Dict[str, Any] = {}

        # Map OpenAI parameters to Imagen format
        if "n" in filtered_params:
            mapped_params["sampleCount"] = filtered_params["n"]
        
        if "size" in filtered_params:
            mapped_params["aspectRatio"] = self._map_size_to_aspect_ratio(
                filtered_params["size"]  # type: ignore[arg-type]
            )
            
        if "mask" in filtered_params:
            mapped_params["mask"] = filtered_params["mask"]

        return mapped_params

    def _resolve_vertex_project(self) -> Optional[str]:
        return (
            getattr(self, "_vertex_project", None)
            or os.environ.get("VERTEXAI_PROJECT")
            or getattr(litellm, "vertex_project", None)
            or get_secret_str("VERTEXAI_PROJECT")
        )

    def _resolve_vertex_location(self) -> Optional[str]:
        return (
            getattr(self, "_vertex_location", None)
            or os.environ.get("VERTEXAI_LOCATION")
            or os.environ.get("VERTEX_LOCATION")
            or getattr(litellm, "vertex_location", None)
            or get_secret_str("VERTEXAI_LOCATION")
            or get_secret_str("VERTEX_LOCATION")
        )

    def _resolve_vertex_credentials(self) -> Optional[str]:
        return (
            getattr(self, "_vertex_credentials", None)
            or os.environ.get("VERTEXAI_CREDENTIALS")
            or getattr(litellm, "vertex_credentials", None)
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            or get_secret_str("VERTEXAI_CREDENTIALS")
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        headers = headers or {}
        vertex_project = self._resolve_vertex_project()
        vertex_credentials = self._resolve_vertex_credentials()
        access_token, _ = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )
        return self.set_headers(access_token, headers)

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for Vertex AI Imagen predict API
        """
        vertex_project = self._resolve_vertex_project()
        vertex_location = self._resolve_vertex_location()

        if not vertex_project or not vertex_location:
            raise ValueError("vertex_project and vertex_location are required for Vertex AI")

        # Use the model name as provided, handling vertex_ai prefix
        model_name = model
        if model.startswith("vertex_ai/"):
            model_name = model.replace("vertex_ai/", "")

        if api_base:
            base_url = api_base.rstrip("/")
        else:
            base_url = f"https://{vertex_location}-aiplatform.googleapis.com"

        return f"{base_url}/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model_name}:predict"

    def transform_image_edit_request(  # type: ignore[override]
        self,
        model: str,
        prompt: str,
        image: FileTypes,
        image_edit_optional_request_params: Dict[str, Any],
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict[str, Any], Optional[RequestFiles]]:
        # Prepare reference images in the correct Imagen format
        reference_images = self._prepare_reference_images(image, image_edit_optional_request_params)
        if not reference_images:
            raise ValueError("Vertex AI Imagen image edit requires at least one reference image.")

        # Correct Imagen instances format
        instances = [
            {
                "prompt": prompt,
                "referenceImages": reference_images
            }
        ]

        # Extract OpenAI parameters and set sensible defaults for Vertex AI-specific parameters
        sample_count = image_edit_optional_request_params.get("sampleCount", 1)
        # Use sensible defaults for Vertex AI-specific parameters (not exposed to users)
        edit_mode = "EDIT_MODE_INPAINT_INSERTION"  # Default edit mode
        base_steps = 50  # Default number of steps
        
        # Imagen parameters with correct structure
        parameters = {
            "sampleCount": sample_count,
            "editMode": edit_mode,
            "editConfig": {
                "baseSteps": base_steps
            }
        }

        # Set default values for Vertex AI-specific parameters (not configurable by users via OpenAI API)
        parameters["guidanceScale"] = 7.5  # Default guidance scale
        parameters["seed"] = None  # Let Vertex AI choose random seed

        request_body: Dict[str, Any] = {
            "instances": instances,
            "parameters": parameters
        }

        payload: Any = json.dumps(request_body)
        empty_files = cast(RequestFiles, [])
        return cast(Tuple[Dict[str, Any], Optional[RequestFiles]], (payload, empty_files))

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
    ) -> ImageResponse:
        model_response = ImageResponse()
        try:
            response_json = raw_response.json()
        except Exception as exc:
            raise self.get_error_class(
                error_message=f"Error transforming image edit response: {exc}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        predictions = response_json.get("predictions", [])
        data_list: List[ImageObject] = []

        for prediction in predictions:
            # Imagen returns images as bytesBase64Encoded
            if "bytesBase64Encoded" in prediction:
                data_list.append(
                    ImageObject(
                        b64_json=prediction["bytesBase64Encoded"],
                        url=None,
                    )
                )

        model_response.data = cast(List[OpenAIImage], data_list)
        return model_response

    def _map_size_to_aspect_ratio(self, size: str) -> str:
        """Map OpenAI size format to Imagen aspect ratio format"""
        aspect_ratio_map = {
            "1024x1024": "1:1",
            "1792x1024": "16:9", 
            "1024x1792": "9:16",
            "1280x896": "4:3",
            "896x1280": "3:4",
        }
        return aspect_ratio_map.get(size, "1:1")

    def _prepare_reference_images(
        self, image: Union[FileTypes, List[FileTypes]], 
        image_edit_optional_request_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Prepare reference images in the correct Imagen API format
        """
        images: List[FileTypes]
        if isinstance(image, list):
            images = image
        else:
            images = [image]

        reference_images: List[Dict[str, Any]] = []
        
        for idx, img in enumerate(images):
            if img is None:
                continue

            image_bytes = self._read_all_bytes(img)
            base64_data = base64.b64encode(image_bytes).decode("utf-8")
            
            # Create reference image structure
            reference_image = {
                "referenceType": "REFERENCE_TYPE_RAW",
                "referenceId": idx + 1,
                "referenceImage": {
                    "bytesBase64Encoded": base64_data
                }
            }
            
            reference_images.append(reference_image)
        
        # Handle mask image if provided (for inpainting)
        mask_image = image_edit_optional_request_params.get("mask")
        if mask_image is not None:
            mask_bytes = self._read_all_bytes(mask_image)
            mask_base64 = base64.b64encode(mask_bytes).decode("utf-8")
            
            mask_reference = {
                "referenceType": "REFERENCE_TYPE_MASK",
                "referenceId": len(reference_images) + 1,
                "referenceImage": {
                    "bytesBase64Encoded": mask_base64
                },
                "maskImageConfig": {
                    "maskMode": "MASK_MODE_USER_PROVIDED",
                    "dilation": 0.03  # Default dilation value (not configurable via OpenAI API)
                }
            }
            reference_images.append(mask_reference)

        return reference_images

    def _read_all_bytes(self, image: Any) -> bytes:
        if isinstance(image, (list, tuple)):
            for item in image:
                if item is not None:
                    return self._read_all_bytes(item)
            raise ValueError("Unsupported image type for Vertex AI Imagen image edit.")

        if isinstance(image, dict):
            for key in ("data", "bytes", "content"):
                if key in image and image[key] is not None:
                    value = image[key]
                    if isinstance(value, str):
                        try:
                            return base64.b64decode(value)
                        except Exception:
                            continue
                    return self._read_all_bytes(value)
            if "path" in image:
                return self._read_all_bytes(image["path"])

        if isinstance(image, bytes):
            return image
        if isinstance(image, bytearray):
            return bytes(image)
        if isinstance(image, BytesIO):
            current_pos = image.tell()
            image.seek(0)
            data = image.read()
            image.seek(current_pos)
            return data
        if isinstance(image, (BufferedReader, BufferedRandom)):
            stream_pos: Optional[int] = None
            try:
                stream_pos = image.tell()
            except Exception:
                stream_pos = None
            if stream_pos is not None:
                image.seek(0)
            data = image.read()
            if stream_pos is not None:
                image.seek(stream_pos)
            return data
        if isinstance(image, (str, Path)):
            path_obj = Path(image)
            if not path_obj.exists():
                raise ValueError(
                    f"Mask/image path does not exist for Vertex AI Imagen image edit: {path_obj}"
                )
            return path_obj.read_bytes()
        if hasattr(image, "read"):
            data = image.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
            return data
        raise ValueError(
            f"Unsupported image type for Vertex AI Imagen image edit. Got type={type(image)}"
        )

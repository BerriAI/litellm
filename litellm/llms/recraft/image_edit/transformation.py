from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import httpx
from httpx._types import RequestFiles

from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.responses.main import *
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class RecraftImageEditConfig(BaseImageEditConfig):
    DEFAULT_BASE_URL: str = "https://external.api.recraft.ai"
    IMAGE_EDIT_ENDPOINT: str = "v1/images/imageToImage"
    
    def get_supported_openai_params(
        self, model: str
    ) -> List:
        """
        https://www.recraft.ai/docs#generate-image
        """
        return [
            "n",
            "response_format",
            "size",
            "style"
        ]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        return {}

    
    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        complete_url: str = (
            api_base 
            or get_secret_str("RECRAFT_API_BASE") 
            or self.DEFAULT_BASE_URL
        )

        complete_url = complete_url.rstrip("/")
        complete_url = f"{complete_url}/{self.IMAGE_EDIT_ENDPOINT}"
        return complete_url

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        final_api_key: Optional[str] = (
            api_key or 
            get_secret_str("RECRAFT_API_KEY")
        )
        if not final_api_key:
            raise ValueError("RECRAFT_API_KEY is not set")
        
        headers["Authorization"] = f"Bearer {final_api_key}"        
        return headers


    def transform_image_edit_request(
        self,
        model: str,
        prompt: str,
        image: FileTypes,
        image_edit_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles]:
        """
        Transform the image generation response to the litellm image response

        https://www.recraft.ai/docs#generate-image
        """
        return {}, {}
    
    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ImageResponse:
        model_response = ImageResponse()
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error transforming image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        if not model_response.data:
            model_response.data = []
        
        for image_data in response_data["data"]:
            model_response.data.append(ImageObject(
                url=image_data.get("url", None),
                b64_json=image_data.get("b64_json", None),
            ))
        
        return model_response
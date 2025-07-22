from io import BufferedReader
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import httpx
from httpx._types import RequestFiles

from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.llms.recraft import RecraftImageEditRequestParams
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
    DEFAULT_STRENGTH: float = 0.2
    
    def get_supported_openai_params(
        self, model: str
    ) -> List:
        """
        Supported OpenAI parameters that can be mapped to Recraft image edit API.
        
        Based on Recraft API docs: https://www.recraft.ai/docs#image-to-image
        """
        return [
            "n",              # Maps to n (number of images)
            "response_format", # Maps to response_format (url or b64_json)
            "style"            # Maps to style parameter
        ]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        """
        Map OpenAI image edit parameters to Recraft parameters.
        Reuses OpenAI logic but filters to supported params only.
        """
        # Start with all params like OpenAI does
        all_params = dict(image_edit_optional_params)
        
        # Filter to only supported Recraft parameters
        supported_params = self.get_supported_openai_params(model)
        filtered_params = {k: v for k, v in all_params.items() if k in supported_params}
        
        return filtered_params

    
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
        Transform the image edit request to Recraft's multipart form format.
        Direct implementation for Recraft API.

        https://www.recraft.ai/docs#image-to-image
        """
    
        # Build Recraft form data directly
        #########################################################
        data: RecraftImageEditRequestParams = RecraftImageEditRequestParams(
            model=model,
            prompt=prompt,
            strength=image_edit_optional_request_params.get("strength", self.DEFAULT_STRENGTH),
            **image_edit_optional_request_params,
        )
        data_dict: Dict = dict(data)
        #########################################################
        # Prepare image file for multipart upload
        #########################################################
        files = []
        if image:
            image_content_type: str = ImageEditRequestUtils.get_image_content_type(image)
            if isinstance(image, BufferedReader):
                files.append(("image", (image.name, image, image_content_type)))
            else:
                files.append(("image", ("image.png", image, image_content_type)))
        
        #########################################################
        # Return the data and files
        #########################################################
        return data_dict, files
    
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
                error_message=f"Error transforming image edit response: {e}",
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
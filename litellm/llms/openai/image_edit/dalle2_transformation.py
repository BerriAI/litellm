from io import BufferedReader
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, cast

from httpx._types import RequestFiles

import litellm
from litellm.images.utils import ImageEditRequestUtils
from litellm.types.images.main import ImageEditRequestParams
from litellm.types.llms.openai import FileTypes
from litellm.types.router import GenericLiteLLMParams

from .transformation import OpenAIImageEditConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class DallE2ImageEditConfig(OpenAIImageEditConfig):
    """
    DALL-E-2 specific configuration for image edit API.
    
    DALL-E-2 only supports editing a single image (not an array).
    Uses "image" field name instead of "image[]".
    """

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
        Transform image edit request for DALL-E-2.

        DALL-E-2 only accepts a single image with field name "image" (not "image[]").
        """
        request = ImageEditRequestParams(
            model=model,
            image=image,
            prompt=prompt,
            **image_edit_optional_request_params,
        )
        request_dict = cast(Dict, request)

        #########################################################
        # Separate images and masks as `files` and send other parameters as `data`
        #########################################################
        _image_list = request_dict.get("image")
        _mask = request_dict.get("mask")
        data_without_files = {
            k: v for k, v in request_dict.items() if k not in ["image", "mask"]
        }
        files_list: List[Tuple[str, Any]] = []

        # Handle image parameter - DALL-E-2 only supports single image
        if _image_list is not None:
            image_list = (
                [_image_list] if not isinstance(_image_list, list) else _image_list
            )

            # Validate only one image is provided
            if len(image_list) > 1:
                raise litellm.BadRequestError(
                    message="DALL-E-2 only supports editing a single image. Please provide one image.",
                    model=model,
                    llm_provider="openai",
                )

            # Use "image" field name (singular) for DALL-E-2
            for _image in image_list:
                if _image is not None:
                    self._add_image_to_files(
                        files_list=files_list,
                        image=_image,
                        field_name="image",
                    )

        # Handle mask parameter if provided
        if _mask is not None:
            # Handle case where mask can be a list (extract first mask)
            if isinstance(_mask, list):
                _mask = _mask[0] if _mask else None

            if _mask is not None:
                mask_content_type: str = ImageEditRequestUtils.get_image_content_type(
                    _mask
                )
                if isinstance(_mask, BufferedReader):
                    files_list.append(("mask", (_mask.name, _mask, mask_content_type)))
                else:
                    files_list.append(("mask", ("mask.png", _mask, mask_content_type)))

        return data_without_files, files_list


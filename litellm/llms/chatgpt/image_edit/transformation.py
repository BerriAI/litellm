import base64
from io import BufferedReader, BytesIO
from os import PathLike
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx
from httpx._types import RequestFiles

from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.chatgpt.image_generation.generation_transformation import (
    ChatGPTImageGenerationConfig,
)
from litellm.llms.openai.common_utils import OpenAIError
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.llms.openai import FileTypes
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class ChatGPTImageEditConfig(BaseImageEditConfig):
    """
    Bridge OpenAI-style Images Edits calls to ChatGPT/Codex Responses image generation.
    """

    def __init__(self) -> None:
        self.image_generation_config = ChatGPTImageGenerationConfig()

    def get_supported_openai_params(self, model: str) -> List[str]:
        return ["size"]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict[str, Any]:
        supported_params = self.get_supported_openai_params(model)
        return {
            key: value
            for key, value in image_edit_optional_params.items()
            if key in supported_params
        }

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[dict] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return self.image_generation_config.validate_environment(
            headers=headers,
            model=model,
            messages=[],
            optional_params={},
            litellm_params=litellm_params or {},
            api_key=api_key,
            api_base=api_base,
        )

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        return self.image_generation_config.get_complete_url(
            api_base=api_base,
            api_key=litellm_params.get("api_key"),
            model=model,
            optional_params={},
            litellm_params=litellm_params,
        )

    def use_multipart_form_data(self) -> bool:
        return False

    def transform_image_edit_request(
        self,
        model: str,
        prompt: Optional[str],
        image: Optional[FileTypes],
        image_edit_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict[str, Any], RequestFiles]:
        optional_params = dict(image_edit_optional_request_params)
        self.image_generation_config._validate_openai_image_generation_params(
            model, optional_params
        )

        input_images = self._prepare_input_images(image)
        if not input_images:
            raise ValueError("ChatGPT image edit requires at least one image.")

        request = self.image_generation_config._build_responses_image_request(
            model=model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params=dict(litellm_params),
            input_images=input_images,
        )
        return request, []

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> ImageResponse:
        return self.image_generation_config.transform_image_generation_response(
            model=model,
            raw_response=raw_response,
            model_response=ImageResponse(),
            logging_obj=logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

    def _prepare_input_images(
        self, image: Optional[Union[FileTypes, List[FileTypes]]]
    ) -> List[Dict[str, Any]]:
        if image is None:
            return []

        images = image if isinstance(image, list) else [image]
        input_images: List[Dict[str, Any]] = []
        for img in images:
            if img is None:
                continue
            mime_type = ImageEditRequestUtils.get_image_content_type(img)
            image_bytes = self._read_image_bytes(img)
            b64_data = base64.b64encode(image_bytes).decode("utf-8")
            input_images.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{b64_data}",
                }
            )
        return input_images

    @staticmethod
    def _read_image_bytes(image: FileTypes) -> bytes:
        if isinstance(image, bytes):
            return image
        if isinstance(image, BytesIO):
            current_pos = image.tell()
            image.seek(0)
            data = image.read()
            image.seek(current_pos)
            return data
        if isinstance(image, BufferedReader):
            current_pos = image.tell()
            image.seek(0)
            data = image.read()
            image.seek(current_pos)
            return data
        if isinstance(image, tuple):
            return ChatGPTImageEditConfig._read_image_bytes(image[1])
        if isinstance(image, PathLike):
            with open(image, "rb") as image_file:
                return image_file.read()
        raise ValueError("Unsupported image type for ChatGPT image edit.")

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> OpenAIError:
        return self.image_generation_config.get_error_class(
            error_message=error_message,
            status_code=status_code,
            headers=headers,
        )

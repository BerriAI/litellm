import base64
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final

from httpx._types import FileTypes as HTTPFileTypes
from httpx._types import RequestFiles
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from litellm.llms.openai.image_generation.gpt_transformation import GPTImageGenerationConfig
from litellm.types.llms.openai import AllMessageValues, FileTypes
from litellm.types.router import GenericLiteLLMParams

from .common_utils import CHATGPT_API_BASE
from .responses.transformation import ChatGPTResponsesAPIConfig


class ReferenceImage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_url: str = Field(pattern=r"^(data:image/(png|jpeg|webp);base64,|https://)")


def encode_reference(
    file: HTTPFileTypes | FileTypes,
) -> dict[str, str]:  # mutable-ok: image handler requires dictionaries
    content: Final = file[1] if isinstance(file, tuple) else file
    raw: Final = (
        Path(os.fsdecode(content)).read_bytes()
        if isinstance(content, os.PathLike)
        else content.encode()
        if isinstance(content, str)
        else content
        if isinstance(content, bytes)
        else content.read()
    )
    content_type: Final = (
        file[2]
        if isinstance(file, tuple) and len(file) >= 3 and file[2]
        else ImageEditRequestUtils.get_image_content_type(raw)
    )
    if content_type not in ("image/png", "image/jpeg", "image/webp"):
        raise ValueError("Reference images must be PNG, JPEG, or WEBP")
    return {  # mutable-ok: JSON request serialization
        "image_url": f"data:{content_type};base64," + base64.b64encode(raw).decode("ascii")
    }


def image_headers(
    headers: Mapping[str, object], model: str, params: Mapping[str, object]
) -> dict[str, object]:  # mutable-ok: image handler requires dictionaries
    auth_headers: Final = ChatGPTResponsesAPIConfig().validate_environment(
        headers={},  # mutable-ok: Responses adapter header contract
        model=model,
        litellm_params=GenericLiteLLMParams.model_validate(params),
    )
    return {**headers, **auth_headers, "accept": "application/json"}  # mutable-ok: JSON request serialization


class ChatGPTImageGenerationConfig(GPTImageGenerationConfig):
    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: image handler requires dictionaries
        return image_headers(headers, model, litellm_params)

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return f"{(api_base or CHATGPT_API_BASE).rstrip('/')}/images/generations"

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, object],
    ) -> dict[str, object]:  # mutable-ok: image handler requires dictionaries
        return {"model": model, "prompt": prompt, **optional_params}  # mutable-ok: JSON request serialization


class ChatGPTImageEditConfig(OpenAIImageEditConfig):
    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        api_key: str | None = None,
        litellm_params: Mapping[str, object] | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: image handler requires dictionaries
        return image_headers(headers, model, litellm_params or MappingProxyType({}))

    def get_complete_url(self, model: str, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        return f"{(api_base or CHATGPT_API_BASE).rstrip('/')}/images/edits"

    def use_multipart_form_data(self) -> bool:
        return False

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: FileTypes | Sequence[FileTypes] | None,
        image_edit_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, object],
    ) -> tuple[dict[str, object], RequestFiles]:  # mutable-ok: image handler requires dictionaries
        if image_edit_optional_request_params.get("mask") is not None:
            raise ValueError("ChatGPT image editing does not support masks")
        references: Final = getattr(litellm_params, "images", None)
        if references is not None:
            if image:
                raise ValueError("Specify only one of image or images")
            validated: Final = TypeAdapter(tuple[ReferenceImage, ...]).validate_python(references)
            if not 1 <= len(validated) <= 5:
                raise ValueError("images must contain between 1 and 5 reference images")
            return {  # mutable-ok: JSON request serialization
                "model": model,
                "prompt": prompt,
                **image_edit_optional_request_params,
                "images": tuple(item.model_dump() for item in validated),
            }, ()

        inputs: Final = tuple(image) if isinstance(image, list) else (image,) if image is not None else ()
        encoded: Final = tuple(encode_reference(file) for file in inputs)
        if not 1 <= len(encoded) <= 5:
            raise ValueError("images must contain between 1 and 5 reference images")
        return {  # mutable-ok: JSON request serialization
            "model": model,
            "prompt": prompt,
            **image_edit_optional_request_params,
            "images": encoded,
        }, ()

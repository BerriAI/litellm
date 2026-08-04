import mimetypes
from io import BufferedReader, BytesIO
from typing import TYPE_CHECKING, Any, Final, cast
from urllib.parse import quote

import httpx
from httpx._types import FileContent, FileTypes, RequestFiles

import litellm
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.openai.image_edit.transformation import ImageEditRequestUtils
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import CreateVideoRequest
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import (
    CharacterObject,
    VideoCreateOptionalRequestParams,
    VideoObject,
)
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_character_id,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    from ...base_llm.chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseLLMException = _BaseLLMException
else:
    LiteLLMLoggingObj = Any
    BaseLLMException = Any


class OpenAIVideoConfig(BaseVideoConfig):
    """
    Configuration class for OpenAI video generation.
    """

    def __init__(self):
        super().__init__()

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the list of supported OpenAI parameters for video generation.
        """
        return [
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "characters",
            "user",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        """No mapping applied since inputs are in OpenAI spec already"""
        return dict(video_create_optional_params)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:
        # Use api_key from litellm_params if available, otherwise fall back to other sources
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key

        api_key = api_key or litellm.api_key or litellm.openai_key or get_secret_str("OPENAI_API_KEY")
        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
            }
        )
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for OpenAI video generation.
        """
        if api_base is None:
            api_base = "https://api.openai.com/v1"

        return f"{api_base.rstrip('/')}/videos"

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles, str]:
        """
        Transform the video creation request for OpenAI API.
        """
        # Remove model and extra_headers from optional params as they're handled separately
        video_create_optional_request_params = {
            k: v
            for k, v in video_create_optional_request_params.items()
            if k not in ["model", "extra_headers", "prompt"]
        }

        # Create the request data
        video_create_request = CreateVideoRequest(model=model, prompt=prompt, **video_create_optional_request_params)
        request_dict = cast(dict, video_create_request)
        request_dict = self._decode_character_ids_in_create_video_request(request_dict)

        # Handle input_reference parameter if provided
        _input_reference: Final = video_create_optional_request_params.get("input_reference")
        data_without_files: Final = {k: v for k, v in request_dict.items() if k not in ["input_reference"]}
        files_list: Final[list[tuple[str, FileTypes]]] = []

        # Handle input_reference parameter
        if _input_reference is not None:
            self._add_image_to_files(
                files_list=files_list,
                image=_input_reference,
                field_name="input_reference",
            )
        return data_without_files, files_list, api_base

    def _decode_character_ids_in_create_video_request(self, request_dict: dict) -> dict:
        """
        Decode LiteLLM-managed encoded character ids for provider requests.

        OpenAI expects character ids like `char_...`. If a caller sends
        `character_<base64-encoded-provider-payload>`, convert it back to the
        original provider id before forwarding upstream.
        """
        raw_characters: Final = request_dict.get("characters")
        if not isinstance(raw_characters, list):
            return request_dict

        decoded_characters: Final[list[Any]] = []
        for character in raw_characters:
            if not isinstance(character, dict):
                decoded_characters.append(character)
                continue

            character_id = character.get("id")
            if isinstance(character_id, str):
                decoded_character = dict(character)
                decoded_character["id"] = extract_original_character_id(character_id)
                decoded_characters.append(decoded_character)
            else:
                decoded_characters.append(character)

        request_dict["characters"] = decoded_characters
        return request_dict

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict | None = None,
    ) -> VideoObject:
        """Transform the OpenAI video creation response."""
        video_obj: Final = VideoObject.model_validate(raw_response.json())

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, model)

        usage_data: Final = {}
        if video_obj:
            if hasattr(video_obj, "seconds") and video_obj.seconds:
                try:
                    usage_data["duration_seconds"] = float(video_obj.seconds)
                except (ValueError, TypeError):
                    pass
        video_obj.usage = usage_data

        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: str | None = None,
    ) -> tuple[str, dict]:
        """
        Transform the video content request for OpenAI API.

        OpenAI API expects the following request:
        - GET /v1/videos/{video_id}/content
        - GET /v1/videos/{video_id}/content?variant=thumbnail
        """
        original_video_id: Final = extract_original_video_id(video_id)
        encoded_video_id: Final = encode_url_path_segment(original_video_id, field_name="video_id")

        # Construct the URL for video content download
        url = f"{api_base.rstrip('/')}/{encoded_video_id}/content"
        if variant is not None:
            # Encode the user-controlled ``variant`` so a value like
            # ``thumbnail&extra=1`` cannot inject additional query params
            # into the upstream request — same hardening rationale as the
            # path-segment encoding above.
            url = f"{url}?variant={quote(variant, safe='')}"

        # No additional data needed for GET content request
        data: Final[dict[str, object]] = {}

        return url, data

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, dict]:
        """
        Transform the video remix request for OpenAI API.

        OpenAI API expects the following request:
        - POST /v1/videos/{video_id}/remix
        """
        original_video_id: Final = extract_original_video_id(video_id)
        encoded_video_id: Final = encode_url_path_segment(original_video_id, field_name="video_id")

        # Construct the URL for video remix
        url: Final = f"{api_base.rstrip('/')}/{encoded_video_id}/remix"

        # Prepare the request data
        data: Final = {"prompt": prompt}

        # Add any extra body parameters
        if extra_body:
            data.update(extra_body)

        return url, data

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        """Transform the OpenAI video content download response."""
        return raw_response.content

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        """
        Transform the OpenAI video remix response.
        """
        # Transform the response data
        video_obj: Final = VideoObject.model_validate(raw_response.json())

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, None)

        # Create usage object with duration information for cost calculation
        # Video remix API doesn't provide usage, so we create one with duration
        usage_data: Final = {}
        if video_obj:
            if hasattr(video_obj, "seconds") and video_obj.seconds:
                try:
                    usage_data["duration_seconds"] = float(video_obj.seconds)
                except (ValueError, TypeError):
                    pass
        # Create the response
        video_obj.usage = usage_data

        return video_obj

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: dict[str, Any] | None = None,
    ) -> tuple[str, dict]:
        """
        Transform the video list request for OpenAI API.

        OpenAI API expects the following request:
        - GET /v1/videos
        """
        # Use the api_base directly for video list
        url: Final = api_base

        # Prepare query parameters
        params: Final = {}
        if after is not None:
            # Decode the wrapped video ID back to the original provider ID
            params["after"] = extract_original_video_id(after)
        if limit is not None:
            params["limit"] = str(limit)
        if order is not None:
            params["order"] = order

        # Add any extra query parameters
        if extra_query:
            params.update(extra_query)

        return url, params

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:
        response_data: Final = raw_response.json()

        if custom_llm_provider and "data" in response_data:
            for video_obj in response_data.get("data", []):
                if isinstance(video_obj, dict) and "id" in video_obj:
                    video_obj["id"] = encode_video_id_with_provider(
                        video_obj["id"],
                        custom_llm_provider,
                        video_obj.get("model"),
                    )

            # Encode pagination cursor IDs so they remain consistent
            # with the wrapped data[].id format
            data_list: Final = response_data.get("data", [])
            if response_data.get("first_id"):
                first_model = None
                if data_list and isinstance(data_list[0], dict):
                    first_model = data_list[0].get("model")
                response_data["first_id"] = encode_video_id_with_provider(
                    response_data["first_id"],
                    custom_llm_provider,
                    first_model,
                )
            if response_data.get("last_id"):
                last_model = None
                if data_list and isinstance(data_list[-1], dict):
                    last_model = data_list[-1].get("model")
                response_data["last_id"] = encode_video_id_with_provider(
                    response_data["last_id"],
                    custom_llm_provider,
                    last_model,
                )

        return response_data

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        """
        Transform the video delete request for OpenAI API.

        OpenAI API expects the following request:
        - DELETE /v1/videos/{video_id}
        """
        original_video_id: Final = extract_original_video_id(video_id)
        encoded_video_id: Final = encode_url_path_segment(original_video_id, field_name="video_id")

        # Construct the URL for video delete
        url: Final = f"{api_base.rstrip('/')}/{encoded_video_id}"

        # No data needed for DELETE request
        data: Final[dict[str, object]] = {}

        return url, data

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        """
        Transform the OpenAI video delete response.
        """
        # Transform the response data
        video_obj: Final = VideoObject.model_validate(raw_response.json())

        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        """
        Transform the OpenAI video retrieve request.
        """
        # Extract the original video_id (remove provider encoding if present)
        original_video_id: Final = extract_original_video_id(video_id)
        encoded_video_id: Final = encode_url_path_segment(original_video_id, field_name="video_id")

        # For video retrieve, we just need to construct the URL
        url: Final = f"{api_base.rstrip('/')}/{encoded_video_id}"

        # No additional data needed for GET request
        data: Final[dict[str, object]] = {}

        return url, data

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        """
        Transform the OpenAI video retrieve response.
        """
        # Transform the response data
        video_obj: Final = VideoObject.model_validate(raw_response.json())

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, None)

        return video_obj

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        from ...base_llm.chat.transformation import BaseLLMException

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def transform_video_create_character_request(
        self,
        name: str,
        video: FileContent,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, list]:
        url: Final = f"{api_base.rstrip('/')}/characters"
        files_list: Final[list[tuple[str, FileTypes]]] = [("name", (None, name))]
        self._add_video_to_files(files_list, video, "video")
        return url, files_list

    def transform_video_create_character_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> CharacterObject:
        return CharacterObject.model_validate(raw_response.json())

    def transform_video_get_character_request(
        self,
        character_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        original_character_id: Final = extract_original_character_id(character_id)
        encoded_character_id: Final = encode_url_path_segment(original_character_id, field_name="character_id")
        url: Final = f"{api_base.rstrip('/')}/characters/{encoded_character_id}"
        return url, {}

    def transform_video_get_character_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> CharacterObject:
        return CharacterObject.model_validate(raw_response.json())

    def transform_video_edit_request(
        self,
        prompt: str,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: dict[str, object] | None = None,
        prefetched_source_data: dict[str, object] | None = None,
    ) -> tuple[str, dict]:
        original_video_id: Final = extract_original_video_id(video_id)
        url: Final = f"{api_base.rstrip('/')}/edits"
        data: Final[dict[str, object]] = {"prompt": prompt, "video": {"id": original_video_id}}
        if extra_body:
            data.update(extra_body)
        return url, data

    def transform_video_edit_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict | None = None,
    ) -> VideoObject:
        video_obj: Final = VideoObject.model_validate(raw_response.json())
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, None)
        return video_obj

    def transform_video_extension_request(
        self,
        prompt: str,
        video_id: str,
        seconds: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: dict[str, object] | None = None,
    ) -> tuple[str, dict]:
        original_video_id: Final = extract_original_video_id(video_id)
        url: Final = f"{api_base.rstrip('/')}/extensions"
        data: Final[dict[str, object]] = {
            "prompt": prompt,
            "seconds": seconds,
            "video": {"id": original_video_id},
        }
        if extra_body:
            data.update(extra_body)
        return url, data

    def transform_video_extension_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        video_obj: Final = VideoObject.model_validate(raw_response.json())
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, None)
        return video_obj

    def _add_image_to_files(
        self,
        files_list: list[tuple[str, Any]],
        image: Any,
        field_name: str,
    ) -> None:
        """Add an image to the files list with appropriate content type"""
        image_content_type: Final = ImageEditRequestUtils.get_image_content_type(image)

        if isinstance(image, BufferedReader):
            files_list.append((field_name, (image.name, image, image_content_type)))
        else:
            files_list.append((field_name, ("input_reference.png", image, image_content_type)))

    def _add_video_to_files(
        self,
        files_list: list[tuple[str, FileTypes]],
        video: FileContent,
        field_name: str,
    ) -> None:
        """
        Add a video to files with proper video MIME type detection.

        This path is used by POST /videos/characters and must send video/mp4,
        not image/* content types.
        """
        filename: Final = getattr(video, "name", None) or "input_video.mp4"
        content_type: Final = self._get_video_content_type(video=video, filename=filename)
        files_list.append((field_name, (filename, video, content_type)))

    def _get_video_content_type(self, video: FileContent, filename: str) -> str:
        guessed_content_type, _ = mimetypes.guess_type(filename)
        if guessed_content_type and guessed_content_type.startswith("video/"):
            return guessed_content_type

        # Fast-path detection for common MP4 signatures when filename is missing/incorrect.
        try:
            header_bytes = b""
            if isinstance(video, BytesIO) or isinstance(video, BufferedReader):
                current_pos: Final = video.tell()
                video.seek(0)
                header_bytes = video.read(64)
                video.seek(current_pos)
            elif isinstance(video, bytes):
                header_bytes = video[:64]

            # MP4 typically includes ftyp in first box.
            if b"ftyp" in header_bytes:
                return "video/mp4"
        except Exception:
            pass

        # OpenAI create-character currently supports mp4.
        return "video/mp4"

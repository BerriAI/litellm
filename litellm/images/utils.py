import os
import struct
from collections.abc import Collection, Mapping, Sequence
from io import BufferedReader, BytesIO
from typing import IO, Any, Final, cast, get_type_hints

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.token_counter import get_image_type
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.types.files import FILE_MIME_TYPES, FileType
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.llms.openai import FileTypes

_JPEG_SOF_MARKERS: Final = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
_MAX_JPEG_SEGMENTS: Final = 64
_JPEG_FIRST_SEGMENT_OFFSET: Final = 2
_JPEG_SOF_PAYLOAD_SIZE: Final = 5
_HEADER_READ_SIZE: Final = 32
_MIN_PNG_HEADER_SIZE: Final = 24
_MIN_WEBP_HEADER_SIZE: Final = 30


def _content_stream(image: FileTypes) -> IO[bytes] | None:
    content: Final = image[1] if isinstance(image, tuple) else image
    if isinstance(content, bytes):
        return BytesIO(content)
    if isinstance(content, (str, os.PathLike)):
        return None
    return content if content.seekable() else None


def _png_dimensions(head: bytes) -> tuple[int, int] | None:
    if len(head) < _MIN_PNG_HEADER_SIZE:
        return None
    width, height = struct.unpack(">II", head[16:24])
    return width, height


def _webp_dimensions(head: bytes) -> tuple[int, int] | None:
    if len(head) < _MIN_WEBP_HEADER_SIZE:
        return None
    match head[12:16]:
        case b"VP8X":
            return int.from_bytes(head[24:27], "little") + 1, int.from_bytes(head[27:30], "little") + 1
        case b"VP8 ":
            width, height = struct.unpack("<HH", head[26:30])
            return width & 0x3FFF, height & 0x3FFF
        case b"VP8L":
            bits: Final = int.from_bytes(head[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        case _:
            return None


def _jpeg_sof_dimensions(stream: IO[bytes], start: int, offset: int, segments_left: int) -> tuple[int, int] | None:
    segment_offset = offset  # rebind-ok: advanced one JPEG segment per hop
    for _ in range(segments_left):
        stream.seek(start + segment_offset)
        marker = stream.read(4)
        if len(marker) < 4 or marker[0] != 0xFF:
            return None
        if marker[1] == 0xFF:
            segment_offset += 1
            continue
        if marker[1] in _JPEG_SOF_MARKERS:
            sof = stream.read(_JPEG_SOF_PAYLOAD_SIZE)
            if len(sof) < _JPEG_SOF_PAYLOAD_SIZE:
                return None
            _precision, height, width = struct.unpack(">BHH", sof)
            return width, height
        segment_length = int.from_bytes(marker[2:4], "big")
        if segment_length < 2:
            return None
        segment_offset += 2 + segment_length
    return None


def _header_dimensions(stream: IO[bytes], position: int) -> tuple[int, int] | None:
    stream.seek(position)
    head: Final = stream.read(_HEADER_READ_SIZE)
    match get_image_type(head):
        case "png":
            return _png_dimensions(head)
        case "webp":
            return _webp_dimensions(head)
        case "jpeg":
            return _jpeg_sof_dimensions(stream, position, _JPEG_FIRST_SEGMENT_OFFSET, _MAX_JPEG_SEGMENTS)
        case _:
            return None


def measure_reference_image(image: FileTypes) -> tuple[int, int] | None:
    stream: Final = _content_stream(image)
    if stream is None:
        return None
    position: Final = stream.tell()
    try:
        return _header_dimensions(stream, position)
    except (OSError, ValueError, struct.error):
        return None
    finally:
        stream.seek(position)


def measure_reference_pixels(images: Sequence[FileTypes]) -> int | None:
    measured: Final = tuple(measure_reference_image(image) for image in images)
    dimensions: Final = tuple(size for size in measured if size is not None)
    if len(dimensions) != len(measured):
        verbose_logger.debug(
            "Reference image %d has no readable PNG, JPEG or WebP header; billing generated pixels only",
            measured.index(None),
        )
        return None
    return sum(width * height for width, height in dimensions)


class ImageEditRequestUtils:
    @staticmethod
    def get_optional_params_image_edit(
        model: str,
        image_edit_provider_config: BaseImageEditConfig,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        drop_params: bool | None = None,
        additional_drop_params: list[str] | None = None,
    ) -> dict:
        """
        Get optional parameters for the image edit API.

        Args:
            model: The model name
            image_edit_provider_config: The provider configuration for image edit API
            image_edit_optional_params: The optional parameters for the image edit API
            drop_params: If True, silently drop unsupported parameters instead of raising
            additional_drop_params: List of additional parameter names to drop

        Returns:
            A dictionary of supported parameters for the image edit API
        """
        supported_params: Final = image_edit_provider_config.get_supported_openai_params(model)

        should_drop: Final = litellm.drop_params is True or drop_params is True

        filtered_optional_params: Final = dict(image_edit_optional_params)
        if additional_drop_params:
            for param in additional_drop_params:
                filtered_optional_params.pop(param, None)

        unsupported_params: Final = [param for param in filtered_optional_params if param not in supported_params]

        if unsupported_params:
            if should_drop:
                for param in unsupported_params:
                    filtered_optional_params.pop(param, None)
            else:
                raise litellm.UnsupportedParamsError(
                    model=model,
                    message=f"The following parameters are not supported for model {model}: {', '.join(unsupported_params)}",
                )

        mapped_params: Final = image_edit_provider_config.map_openai_params(
            image_edit_optional_params=cast(ImageEditOptionalRequestParams, filtered_optional_params),
            model=model,
            drop_params=should_drop,
        )

        return mapped_params

    @staticmethod
    def get_requested_image_edit_optional_param(
        params: Mapping[str, object],
        provider_supported_params: Collection[str] = (),
    ) -> ImageEditOptionalRequestParams:
        """
        Filter parameters to only include those defined in ImageEditOptionalRequestParams.

        Args:
            params: Dictionary of parameters to filter

        Returns:
            ImageEditOptionalRequestParams instance with only the valid parameters
        """
        valid_keys: Final = frozenset(get_type_hints(ImageEditOptionalRequestParams)) | frozenset(
            provider_supported_params
        )
        filtered_params: Final = {k: v for k, v in params.items() if k in valid_keys and v is not None}
        return cast(ImageEditOptionalRequestParams, filtered_params)

    @staticmethod
    def get_image_content_type(image_data: Any) -> str:
        """
        Detect the content type of image data using existing LiteLLM utils.

        Args:
            image_data: Can be BytesIO, bytes, BufferedReader, or other file-like objects

        Returns:
            The MIME type string (e.g., "image/png", "image/jpeg")
        """
        try:
            # Extract bytes for content type detection
            if isinstance(image_data, BytesIO):
                # Save current position
                current_pos = image_data.tell()
                image_data.seek(0)
                bytes_data = image_data.read(100)  # First 100 bytes are enough for detection
                # Restore position
                image_data.seek(current_pos)
            elif isinstance(image_data, BufferedReader):
                # Save current position
                current_pos = image_data.tell()
                image_data.seek(0)
                bytes_data = image_data.read(100)
                # Restore position
                image_data.seek(current_pos)
            elif isinstance(image_data, bytes):
                bytes_data = image_data[:100]
            else:
                # For other types, try to read if possible
                if hasattr(image_data, "read"):
                    current_pos = getattr(image_data, "tell", lambda: 0)()
                    if hasattr(image_data, "seek"):
                        image_data.seek(0)
                    bytes_data = image_data.read(100)
                    if hasattr(image_data, "seek"):
                        image_data.seek(current_pos)
                else:
                    return FILE_MIME_TYPES[FileType.PNG]  # Default fallback

            # Use the existing get_image_type function to detect image type
            image_type_str: Final = get_image_type(bytes_data)

            if image_type_str is None:
                return FILE_MIME_TYPES[FileType.PNG]  # Default if detection fails

            # Map detected type string to FileType enum and get MIME type
            type_mapping: Final = {
                "png": FileType.PNG,
                "jpeg": FileType.JPEG,
                "gif": FileType.GIF,
                "webp": FileType.WEBP,
                "heic": FileType.HEIC,
            }

            file_type: Final = type_mapping.get(image_type_str)
            if file_type is None:
                return FILE_MIME_TYPES[FileType.PNG]  # Default to PNG if unknown

            return FILE_MIME_TYPES[file_type]

        except Exception:
            # If anything goes wrong, default to PNG
            return FILE_MIME_TYPES[FileType.PNG]

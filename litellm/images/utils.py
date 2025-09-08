from io import BufferedReader, BytesIO
from typing import Any, Dict, cast, get_type_hints

import litellm
from litellm.litellm_core_utils.token_counter import get_image_type
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.types.files import FILE_MIME_TYPES, FileType
from litellm.types.images.main import ImageEditOptionalRequestParams


class ImageEditRequestUtils:
    @staticmethod
    def get_optional_params_image_edit(
        model: str,
        image_edit_provider_config: BaseImageEditConfig,
        image_edit_optional_params: ImageEditOptionalRequestParams,
    ) -> Dict:
        """
        Get optional parameters for the image edit API.

        Args:
            params: Dictionary of all parameters
            model: The model name
            image_edit_provider_config: The provider configuration for image edit API

        Returns:
            A dictionary of supported parameters for the image edit API
        """
        # Remove None values and internal parameters

        # Get supported parameters for the model
        supported_params = image_edit_provider_config.get_supported_openai_params(model)

        # Check for unsupported parameters
        unsupported_params = [
            param
            for param in image_edit_optional_params
            if param not in supported_params
        ]

        if unsupported_params:
            raise litellm.UnsupportedParamsError(
                model=model,
                message=f"The following parameters are not supported for model {model}: {', '.join(unsupported_params)}",
            )

        # Map parameters to provider-specific format
        mapped_params = image_edit_provider_config.map_openai_params(
            image_edit_optional_params=image_edit_optional_params,
            model=model,
            drop_params=litellm.drop_params,
        )

        return mapped_params

    @staticmethod
    def get_requested_image_edit_optional_param(
        params: Dict[str, Any],
    ) -> ImageEditOptionalRequestParams:
        """
        Filter parameters to only include those defined in ImageEditOptionalRequestParams.

        Args:
            params: Dictionary of parameters to filter

        Returns:
            ImageEditOptionalRequestParams instance with only the valid parameters
        """
        valid_keys = get_type_hints(ImageEditOptionalRequestParams).keys()
        filtered_params = {
            k: v for k, v in params.items() if k in valid_keys and v is not None
        }

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
                bytes_data = image_data.read(
                    100
                )  # First 100 bytes are enough for detection
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
            image_type_str = get_image_type(bytes_data)

            if image_type_str is None:
                return FILE_MIME_TYPES[FileType.PNG]  # Default if detection fails

            # Map detected type string to FileType enum and get MIME type
            type_mapping = {
                "png": FileType.PNG,
                "jpeg": FileType.JPEG,
                "gif": FileType.GIF,
                "webp": FileType.WEBP,
                "heic": FileType.HEIC,
            }

            file_type = type_mapping.get(image_type_str)
            if file_type is None:
                return FILE_MIME_TYPES[FileType.PNG]  # Default to PNG if unknown

            return FILE_MIME_TYPES[file_type]

        except Exception:
            # If anything goes wrong, default to PNG
            return FILE_MIME_TYPES[FileType.PNG]

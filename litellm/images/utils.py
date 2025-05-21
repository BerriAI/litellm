from typing import Any, Dict, cast, get_type_hints

import litellm
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
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

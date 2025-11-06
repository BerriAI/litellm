from typing import Any, Dict, cast, get_type_hints

import litellm
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.types.videos.main import VideoCreateOptionalRequestParams


class VideoGenerationRequestUtils:
    """Helper utils for constructing video generation requests"""

    @staticmethod
    def get_optional_params_video_generation(
        model: str,
        video_generation_provider_config: BaseVideoConfig,
        video_generation_optional_params: VideoCreateOptionalRequestParams,
    ) -> Dict:
        """
        Get optional parameters for the video generation API.

        Args:
            model: The model name
            video_generation_provider_config: The provider configuration for video generation API
            video_generation_optional_params: The optional parameters for video generation

        Returns:
            A dictionary of supported parameters for the video generation API
        """
        # Map parameters to provider-specific format
        mapped_params = video_generation_provider_config.map_openai_params(
            video_create_optional_params=video_generation_optional_params,
            model=model,
            drop_params=litellm.drop_params,
        )

        # Merge extra_body params if present (for provider-specific parameters)
        if "extra_body" in video_generation_optional_params:
            extra_body = video_generation_optional_params["extra_body"]
            if extra_body and isinstance(extra_body, dict):
                # extra_body params override mapped params
                mapped_params.update(extra_body)
            # Remove extra_body from mapped_params since it's not sent to the API
            mapped_params.pop("extra_body", None)

        return mapped_params

    @staticmethod
    def get_requested_video_generation_optional_param(
        params: Dict[str, Any],
    ) -> VideoCreateOptionalRequestParams:
        """
        Filter parameters to only include those defined in VideoCreateOptionalRequestParams.

        Args:
            params: Dictionary of parameters to filter

        Returns:
            VideoCreateOptionalRequestParams instance with only the valid parameters
        """
        filtered_params = {
            k: v for k, v in params.items() if v is not None
        }

        return cast(VideoCreateOptionalRequestParams, filtered_params)

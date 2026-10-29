from typing import TypedDict

from typing_extensions import ReadOnly


class ImageDescriptionInterceptionConfig(TypedDict, total=False):
    """
    Configuration for ImageDescriptionInterceptionLogger, read from proxy_config.yaml:

        litellm_settings:
          callbacks: ["image_description_interception"]
        callback_settings:
          image_description_interception:
            image_model: vision-describer
            timeout: 120
    """

    image_model: ReadOnly[str]
    prompt: ReadOnly[str]
    timeout: ReadOnly[float]

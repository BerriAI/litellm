"""
Docker Model Runner Image Generation API Configuration.

Docker Model Runner provides image generation via the Diffusers backend:
/engines/diffusers/v1/images/generations

Docs: https://docs.docker.com/ai/model-runner/api-reference/
"""

from typing import Final

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse


class DockerModelRunnerImageGenerationConfig(BaseImageGenerationConfig):
    DEFAULT_BASE_URL: str = "http://localhost:12434"

    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: signature dictated by BaseImageGenerationConfig
        return [  # mutable-ok: supported-params list, mirrors hosted_vllm
            "n",
            "size",
            "response_format",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,  # mutable-ok: signature dictated by BaseImageGenerationConfig
        optional_params: dict,  # mutable-ok: signature dictated by BaseImageGenerationConfig
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: signature dictated by BaseImageGenerationConfig
        supported_params: Final = self.get_supported_openai_params(model)
        mapped: Final = {  # mutable-ok: API request payload
            k: v for k, v in non_default_params.items() if k in supported_params
        }
        return {**optional_params, **mapped}  # mutable-ok: API request payload

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,  # mutable-ok: signature dictated by BaseImageGenerationConfig
        litellm_params: dict,  # mutable-ok: signature dictated by BaseImageGenerationConfig
        stream: bool | None = None,
    ) -> str:
        base_url: str = (  # rebind-ok: progressively normalized into the endpoint URL
            api_base or get_secret_str("DOCKER_MODEL_RUNNER_API_BASE") or self.DEFAULT_BASE_URL
        )
        base_url = base_url.rstrip("/")  # rebind-ok: progressively normalized into the endpoint URL

        return f"{base_url}/engines/diffusers/v1/images/generations"

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: signature dictated by BaseImageGenerationConfig
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: signature dictated by BaseImageGenerationConfig
        optional_params: dict,  # mutable-ok: signature dictated by BaseImageGenerationConfig
        litellm_params: dict,  # mutable-ok: signature dictated by BaseImageGenerationConfig
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: signature dictated by BaseImageGenerationConfig
        default_headers: Final = {  # mutable-ok: API request payload
            "Content-Type": "application/json",
        }

        if api_key:
            default_headers["Authorization"] = f"Bearer {api_key}"
        else:
            default_headers["Authorization"] = "Bearer dummy-key"

        return {**default_headers, **headers}  # mutable-ok: API request payload

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,  # mutable-ok: signature dictated by BaseImageGenerationConfig
        litellm_params: dict,  # mutable-ok: signature dictated by BaseImageGenerationConfig
        headers: dict,  # mutable-ok: signature dictated by BaseImageGenerationConfig
    ) -> dict:  # mutable-ok: signature dictated by BaseImageGenerationConfig
        return {  # mutable-ok: API request payload
            "model": model,
            "prompt": prompt,
            **optional_params,
        }

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,  # mutable-ok: signature dictated by BaseImageGenerationConfig
        optional_params: dict,  # mutable-ok: signature dictated by BaseImageGenerationConfig
        litellm_params: dict,  # mutable-ok: signature dictated by BaseImageGenerationConfig
        encoding: object,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        try:
            response_data: Final = raw_response.json()
        except Exception as e:
            raise ValueError(
                f"Error parsing response: {raw_response.text}, status_code={raw_response.status_code}"
            ) from e

        if not model_response.data:
            model_response.data = []  # rebind-ok: fills the caller's response object in place, the transform contract  # mutable-ok: API response payload

        data_list: Final = response_data.get("data", [])  # mutable-ok: API response payload
        for item in data_list:
            b64_json = item.get("b64_json")
            if b64_json:
                model_response.data.append(
                    ImageObject(
                        b64_json=b64_json,
                        url=None,
                        revised_prompt=None,
                    )
                )

        return model_response

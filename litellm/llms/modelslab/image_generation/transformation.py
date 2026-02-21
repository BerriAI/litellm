"""
ModelsLab Image Generation Provider

API docs: https://docs.modelslab.com/image-generation/community-models/text2img

Auth note: ModelsLab uses key-in-body authentication — the API key is embedded
in the JSON request body as "key" rather than via a Bearer header. This means
the key will appear in LiteLLM's request logging (pre_call) and any observability
backends (LangFuse, etc.). Users should treat MODELSLAB_API_KEY with appropriate
care and rotate it if it is inadvertently logged.
"""

import time
from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

MODELSLAB_POLLING_INTERVAL = 3  # seconds between polls
MODELSLAB_POLLING_TIMEOUT = 300  # 5 minutes max


class ModelsLabImageGenerationConfig(BaseImageGenerationConfig):
    DEFAULT_BASE_URL: str = "https://modelslab.com/api/v6"
    IMAGE_GENERATION_ENDPOINT: str = "images/text2img"
    FETCH_ENDPOINT: str = "images/fetch"

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return [
            "n",
            "size",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)

        for k in non_default_params.keys():
            if k not in optional_params.keys():
                if k in supported_params:
                    if k == "n":
                        optional_params["samples"] = non_default_params[k]
                    elif k == "size":
                        size_str = non_default_params[k]
                        if "x" in str(size_str):
                            w, h = size_str.split("x", 1)
                            optional_params["width"] = int(w)
                            optional_params["height"] = int(h)
                    else:
                        optional_params[k] = non_default_params[k]
                elif drop_params:
                    pass
                else:
                    raise ValueError(
                        f"Parameter {k} is not supported for model {model}. "
                        f"Supported parameters are {supported_params}. "
                        f"Set drop_params=True to drop unsupported parameters."
                    )

        return optional_params

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        complete_url: str = (
            api_base
            or get_secret_str("MODELSLAB_API_BASE")
            or self.DEFAULT_BASE_URL
        )
        complete_url = complete_url.rstrip("/")
        complete_url = f"{complete_url}/{self.IMAGE_GENERATION_ENDPOINT}"
        return complete_url

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        final_api_key: Optional[str] = (
            api_key or get_secret_str("MODELSLAB_API_KEY")
        )
        if not final_api_key:
            raise ValueError(
                "MODELSLAB_API_KEY is not set. Please set the MODELSLAB_API_KEY "
                "environment variable or pass api_key."
            )
        headers["Content-Type"] = "application/json"
        return headers

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        # ModelsLab uses key-in-body auth. See module docstring for security note.
        api_key = litellm_params.get("api_key") or get_secret_str("MODELSLAB_API_KEY")

        # Strip provider prefix from model name (e.g. "modelslab/flux" → "flux")
        model_id = model
        if "/" in model_id:
            model_id = model_id.split("/", 1)[1]

        request_body: dict = {
            "key": api_key,
            "prompt": prompt,
            "model_id": model_id,
            **optional_params,
        }
        return request_body

    def _resolve_api_key(self, request_data: dict, litellm_params: dict) -> str:
        """Extract the API key from request data or environment."""
        return (
            request_data.get("key")
            or litellm_params.get("api_key")
            or get_secret_str("MODELSLAB_API_KEY")
            or ""
        )

    def _poll_sync(
        self,
        generation_id: int,
        api_key: str,
        base_url: str,
        timeout_secs: float = MODELSLAB_POLLING_TIMEOUT,
    ) -> dict:
        """
        Poll ModelsLab fetch endpoint until image generation completes.

        ModelsLab fetch endpoint: POST /api/v6/images/fetch/{id}
        Body: {"key": "<api_key>"}
        Returns same response schema as text2img.
        """
        from litellm.llms.custom_httpx.http_handler import _get_httpx_client

        client = _get_httpx_client()
        start_time = time.time()
        fetch_url = f"{base_url.rstrip('/')}/{self.FETCH_ENDPOINT}/{generation_id}"

        verbose_logger.debug(f"ModelsLab: polling fetch URL {fetch_url}")

        while True:
            if time.time() - start_time > timeout_secs:
                raise TimeoutError(
                    f"ModelsLab image generation timed out after {timeout_secs}s. "
                    f"Generation ID: {generation_id}"
                )

            response = client.post(
                url=fetch_url,
                json={"key": api_key},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
            status = data.get("status", "")

            verbose_logger.debug(f"ModelsLab: poll status={status}, id={generation_id}")

            if status == "success":
                return data
            elif status == "error":
                raise ValueError(
                    f"ModelsLab generation failed: {data.get('message', 'Unknown error')}"
                )
            elif status == "processing":
                time.sleep(MODELSLAB_POLLING_INTERVAL)
            else:
                raise ValueError(
                    f"ModelsLab unexpected status '{status}' for generation {generation_id}"
                )

    def _build_image_response(
        self,
        response_data: dict,
        model_response: ImageResponse,
    ) -> ImageResponse:
        """Map ModelsLab success response to LiteLLM ImageResponse."""
        if not model_response.data:
            model_response.data = []

        for url in response_data.get("output", []):
            model_response.data.append(ImageObject(url=url))

        return model_response

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error parsing ModelsLab response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        status = response_data.get("status", "")

        if status == "error":
            raise self.get_error_class(
                error_message=f"ModelsLab error: {response_data.get('message', 'Unknown error')}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if status == "processing":
            generation_id = response_data.get("id")
            if not generation_id:
                raise self.get_error_class(
                    error_message="ModelsLab returned 'processing' without a generation ID",
                    status_code=raw_response.status_code,
                    headers=raw_response.headers,
                )

            verbose_logger.debug(
                f"ModelsLab: generation {generation_id} is processing, starting poll..."
            )

            resolved_key = self._resolve_api_key(request_data, litellm_params)
            base_url = (
                get_secret_str("MODELSLAB_API_BASE") or self.DEFAULT_BASE_URL
            )

            response_data = self._poll_sync(
                generation_id=generation_id,
                api_key=resolved_key,
                base_url=base_url,
            )

        if status in ("success",) or response_data.get("status") == "success":
            return self._build_image_response(response_data, model_response)

        raise self.get_error_class(
            error_message=f"Unexpected ModelsLab response: {response_data}",
            status_code=raw_response.status_code,
            headers=raw_response.headers,
        )

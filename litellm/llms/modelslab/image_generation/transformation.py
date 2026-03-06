"""
ModelsLab Image Generation Provider

API docs: https://docs.modelslab.com/image-generation/community-models/text2img

Auth note: ModelsLab uses key-in-body authentication — the API key is embedded
in the JSON request body as "key" rather than via a Bearer header. This means
the key will appear in LiteLLM's request logging (pre_call) and any observability
backends (LangFuse, etc.). Users should treat MODELSLAB_API_KEY with appropriate
care and rotate it if it is inadvertently logged.
"""

import asyncio
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

# HTTP status codes that should be retried (transient errors)
TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries

# Fields that contain sensitive data and should be redacted from logs
MODELSLAB_SENSITIVE_FIELDS: List[str] = ["key"]


def _redact_sensitive_fields(data: dict) -> dict:
    """Return a copy of the data with sensitive fields redacted for logging."""
    redacted = data.copy()
    for field in MODELSLAB_SENSITIVE_FIELDS:
        if field in redacted:
            redacted[field] = "***REDACTED***"
    return redacted


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
            "negative_prompt",
        ]

    def get_sensitive_request_fields(self) -> List[str]:
        """
        Return list of field names that contain sensitive data and should be
        redacted from logs. This enables integration with LiteLLM's logging
        pipeline when/if it supports per-provider sensitive field redaction.

        For ModelsLab, the 'key' field contains the API key in the request body.
        """
        return MODELSLAB_SENSITIVE_FIELDS

    def get_redacted_request(self, request_data: dict) -> dict:
        """
        Return a copy of request data with sensitive fields redacted.
        Useful for safe logging when the full request would otherwise be logged.
        """
        return _redact_sensitive_fields(request_data)

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
                    elif k == "negative_prompt":
                        optional_params["negative_prompt"] = non_default_params[k]
                    elif k == "size":
                        size_str = str(non_default_params[k])
                        if "x" not in size_str:
                            raise ValueError(
                                f"Invalid size format '{size_str}'. "
                                f"Expected format is WIDTHxHEIGHT (e.g., '1024x1024')."
                            )
                        try:
                            w, h = size_str.split("x", 1)
                            optional_params["width"] = int(w)
                            optional_params["height"] = int(h)
                        except (ValueError, IndexError) as e:
                            raise ValueError(
                                f"Invalid size format '{size_str}'. "
                                f"Expected format is WIDTHxHEIGHT (e.g., '1024x1024')."
                            )
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
        # Fall back to "flux" if user passes just "modelslab" with no model suffix
        model_id = model
        if "/" in model_id:
            model_id = model_id.split("/", 1)[1]
        if not model_id or model_id.lower() == "modelslab":
            model_id = "flux"

        # Put optional_params first, then critical fields to prevent override
        request_body: dict = {
            **optional_params,
            "key": api_key,
            "prompt": prompt,
            "model_id": model_id,
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

            # Retry logic for transient HTTP errors.
            # Note: _get_httpx_client() returns an HTTPHandler whose .post() raises
            # httpx.HTTPStatusError internally (via raise_for_status) before returning
            # the response object.  We therefore inspect e.response rather than a
            # local `response` variable which would never be assigned on error.
            last_error = None
            response = None
            for attempt in range(MAX_RETRIES):
                try:
                    # HTTPHandler.post() calls raise_for_status() internally, so any
                    # non-2xx status arrives here as httpx.HTTPStatusError (not a return).
                    response = client.post(
                        url=fetch_url,
                        json={"key": api_key},
                        headers={"Content-Type": "application/json"},
                    )
                    break  # 2xx — success, exit retry loop
                except httpx.HTTPStatusError as e:
                    last_error = e
                    status_code = e.response.status_code if e.response is not None else 0
                    if status_code in TRANSIENT_HTTP_STATUSES and attempt < MAX_RETRIES - 1:
                        verbose_logger.warning(
                            f"ModelsLab: poll got {status_code}, retrying ({attempt + 1}/{MAX_RETRIES})..."
                        )
                        time.sleep(RETRY_DELAY)
                        continue
                    # Not a transient error or max retries reached, propagate
                    raise

            if response is None:
                raise last_error or ValueError("Failed to get response from ModelsLab fetch endpoint")

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

    async def _poll_async(
        self,
        generation_id: int,
        api_key: str,
        base_url: str,
        timeout_secs: float = MODELSLAB_POLLING_TIMEOUT,
    ) -> dict:
        """
        Poll ModelsLab fetch endpoint until image generation completes (async).

        ModelsLab fetch endpoint: POST /api/v6/images/fetch/{id}
        Body: {"key": "<api_key>"}
        Returns same response schema as text2img.
        """
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
        import litellm

        client = get_async_httpx_client(llm_provider=litellm.LlmProviders.MODELSLAB)
        start_time = time.time()
        fetch_url = f"{base_url.rstrip('/')}/{self.FETCH_ENDPOINT}/{generation_id}"

        verbose_logger.debug(f"ModelsLab: async polling fetch URL {fetch_url}")

        while True:
            if time.time() - start_time > timeout_secs:
                raise TimeoutError(
                    f"ModelsLab image generation timed out after {timeout_secs}s. "
                    f"Generation ID: {generation_id}"
                )

            # Mirror _poll_sync retry logic for transient HTTP errors.
            last_error = None
            response = None
            for attempt in range(MAX_RETRIES):
                try:
                    # AsyncHTTPHandler.post() raises internally on non-2xx — no separate raise_for_status needed
                    response = await client.post(
                        url=fetch_url,
                        json={"key": api_key},
                        headers={"Content-Type": "application/json"},
                    )
                    break  # 2xx — success, exit retry loop
                except httpx.HTTPStatusError as e:
                    last_error = e
                    status_code = e.response.status_code if e.response is not None else 0
                    if status_code in TRANSIENT_HTTP_STATUSES and attempt < MAX_RETRIES - 1:
                        verbose_logger.warning(
                            f"ModelsLab: async poll got {status_code}, retrying ({attempt + 1}/{MAX_RETRIES})..."
                        )
                        await asyncio.sleep(RETRY_DELAY)
                        continue
                    raise

            if response is None:
                raise last_error or ValueError("Failed to get async response from ModelsLab fetch endpoint")

            data = response.json()
            status = data.get("status", "")

            verbose_logger.debug(f"ModelsLab: async poll status={status}, id={generation_id}")

            if status == "success":
                return data
            elif status == "error":
                raise ValueError(
                    f"ModelsLab generation failed: {data.get('message', 'Unknown error')}"
                )
            elif status == "processing":
                await asyncio.sleep(MODELSLAB_POLLING_INTERVAL)
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

        # Use `or []` to guard against explicit null in the API response
        # (`get("output", [])` only substitutes the default when the key is absent,
        # not when its value is None/null)
        for url in (response_data.get("output") or []):
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
            # Try to get custom api_base from litellm_params, otherwise fall back to env/default
            # Note: transform_image_generation_response doesn't receive api_base directly,
            # so we check if it was passed in litellm_params
            base_url = (
                litellm_params.get("api_base")
                or get_secret_str("MODELSLAB_API_BASE")
                or self.DEFAULT_BASE_URL
            )

            response_data = self._poll_sync(
                generation_id=generation_id,
                api_key=resolved_key,
                base_url=base_url,
            )
            # Refresh status from polled response to avoid stale variable
            status = response_data.get("status", "")

        if status == "success":
            return self._build_image_response(response_data, model_response)

        # Avoid leaking API key — only surface safe fields in the error message
        raise self.get_error_class(
            error_message=(
                f"Unexpected ModelsLab response: status={response_data.get('status')!r}, "
                f"message={response_data.get('message')!r}"
            ),
            status_code=raw_response.status_code,
            headers=raw_response.headers,
        )

    async def async_transform_image_generation_response(
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
        """
        Async transform the image generation response to the litellm image response.

        ModelsLab returns a task immediately with status PENDING/RUNNING.
        We need to poll the task until it completes (status SUCCEEDED) using async polling.
        """
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
                f"ModelsLab: generation {generation_id} is processing, starting async poll..."
            )

            resolved_key = self._resolve_api_key(request_data, litellm_params)
            base_url = (
                litellm_params.get("api_base")
                or get_secret_str("MODELSLAB_API_BASE")
                or self.DEFAULT_BASE_URL
            )

            response_data = await self._poll_async(
                generation_id=generation_id,
                api_key=resolved_key,
                base_url=base_url,
            )
            # Refresh status from polled response to avoid stale variable
            status = response_data.get("status", "")

        if status == "success":
            return self._build_image_response(response_data, model_response)

        # Avoid leaking API key — only surface safe fields in the error message
        raise self.get_error_class(
            error_message=(
                f"Unexpected ModelsLab response: status={response_data.get('status')!r}, "
                f"message={response_data.get('message')!r}"
            ),
            status_code=raw_response.status_code,
            headers=raw_response.headers,
        )

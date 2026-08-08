"""
MuAPI image generation transformation for LiteLLM.

MuAPI (https://muapi.ai) is a generative media aggregator providing access to
50+ image and video models (Flux, Midjourney, HiDream, Qwen, GPT-4o, etc.)
through a unified REST API.

Pattern: POST /{endpoint} → {"request_id": ...}
         GET  /predictions/{id}/result → poll until status == "completed"

Auth: x-api-key header (set MUAPI_API_KEY env var or pass api_key).
"""

import time
from typing import TYPE_CHECKING, Any, List, Optional
from urllib.parse import urlparse

import httpx

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

BASE_URL = "https://api.muapi.ai/api/v1"

# Map litellm model IDs (muapi/<model-id>) to MuAPI endpoint paths.
# Image generation endpoints (text-to-image and image-to-image).
MUAPI_IMAGE_ENDPOINTS: dict = {
    # --- Flux ---
    "flux-dev": "flux-dev-image",
    "flux-schnell": "flux-schnell-image",
    "flux-krea": "flux-krea-dev",
    "flux-kontext-dev": "flux-kontext-dev-t2i",
    "flux-kontext-pro": "flux-kontext-pro-t2i",
    "flux-kontext-max": "flux-kontext-max-t2i",
    "flux-2-dev": "flux-2-dev",
    "flux-2-pro": "flux-2-pro",
    # --- HiDream ---
    "hidream-fast": "hidream_i1_fast_image",
    "hidream-dev": "hidream_i1_dev_image",
    "hidream-full": "hidream_i1_full_image",
    # --- Wan ---
    "wan2.1": "wan2.1-text-to-image",
    "wan2.5": "wan2.5-text-to-image",
    "wan2.6": "wan2.6-text-to-image",
    # --- GPT ---
    "gpt4o": "gpt4o-text-to-image",
    "gpt-image": "gpt-image-1.5",
    "gpt-image-2": "gpt-image-2-text-to-image",
    # --- Google ---
    "imagen4": "google-imagen4",
    "imagen4-fast": "google-imagen4-fast",
    "imagen4-ultra": "google-imagen4-ultra",
    # --- Midjourney ---
    "midjourney": "midjourney-v7-text-to-image",
    "midjourney-v7": "midjourney-v7",
    "midjourney-v8": "midjourney-v8",
    # --- Seedream ---
    "seedream": "bytedance-seedream-v4.5",
    "seedream-v3": "bytedance-seedream-image",
    "seedream-v4": "bytedance-seedream-v4",
    "seedream-5": "seedream-5.0",
    # --- Qwen ---
    "qwen": "qwen-image",
    "qwen-2": "qwen-image-2.0",
    "qwen-2-pro": "qwen-image-2.0-pro",
    # --- Others ---
    "hunyuan": "hunyuan-image-2.1",
    "ideogram": "ideogram-v3-t2i",
    "reve": "reve-text-to-image",
    "sdxl": "sdxl-image",
    "grok": "grok-imagine-text-to-image",
    "kling-o1": "kling-o1-text-to-image",
    "kling-o3": "kling-o3-image",
    # --- Image-to-image (editing) ---
    "flux-kontext-dev-i2i": "flux-kontext-dev-i2i",
    "flux-kontext-pro-i2i": "flux-kontext-pro-i2i",
    "flux-kontext-max-i2i": "flux-kontext-max-i2i",
    "gpt4o-i2i": "gpt4o-image-to-image",
    "gpt4o-edit": "gpt4o-edit",
    "seededit": "bytedance-seededit-image",
    "seedream-edit": "bytedance-seedream-edit-v4",
    "midjourney-i2i": "midjourney-v7-image-to-image",
    "midjourney-style": "midjourney-v7-style-reference",
    "qwen-edit": "qwen-image-edit",
    "reve-edit": "reve-image-edit",
}

# Poll interval in seconds between status checks
_POLL_INTERVAL = 2.0
# Maximum time to wait for a generation to complete (5 minutes)
_POLL_TIMEOUT = 300.0


class MuAPIImageConfig(BaseImageGenerationConfig):
    """
    LiteLLM provider config for MuAPI image generation.

    Model strings are expected in the form ``muapi/<model-id>`` where
    ``<model-id>`` is one of the keys in MUAPI_IMAGE_ENDPOINTS above,
    e.g. ``muapi/flux-schnell``, ``muapi/midjourney``.
    """

    DEFAULT_BASE_URL: str = BASE_URL

    # ------------------------------------------------------------------ #
    #  Environment / headers                                               #
    # ------------------------------------------------------------------ #

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
        # Prevent the server-configured key from being forwarded to an arbitrary
        # caller-controlled host.  If api_base is set to a non-MuAPI host, the
        # caller must supply their own api_key; otherwise we'd leak the server key.
        # Use exact hostname comparison — startswith is bypassable via subdomain
        # (api.muapi.ai.evil.com) or userinfo (api.muapi.ai@evil.com).
        is_official = not api_base or (
            urlparse(api_base).hostname == "api.muapi.ai"
            and urlparse(api_base).scheme == "https"
        )
        if not is_official and not api_key:
            raise ValueError(
                "A custom api_base was provided that does not match the MuAPI host "
                "(https://api.muapi.ai). Supply an explicit api_key when using a "
                "custom api_base to avoid leaking the server-configured MUAPI_API_KEY."
            )
        final_key = api_key or get_secret_str("MUAPI_API_KEY")
        if not final_key:
            raise ValueError(
                "MuAPI API key is required. Pass api_key or set MUAPI_API_KEY env var."
            )
        headers["x-api-key"] = final_key
        headers["Content-Type"] = "application/json"
        return headers

    # ------------------------------------------------------------------ #
    #  URL construction                                                    #
    # ------------------------------------------------------------------ #

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base = (api_base or self.DEFAULT_BASE_URL).rstrip("/")
        endpoint = self._resolve_endpoint(model)
        return f"{base}/{endpoint}"

    # ------------------------------------------------------------------ #
    #  OpenAI param mapping                                                #
    # ------------------------------------------------------------------ #

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return ["n", "size"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported = set(self.get_supported_openai_params(model))
        for k, v in non_default_params.items():
            if k in supported:
                if k == "size" and isinstance(v, str) and "x" in v:
                    w, h = v.split("x", 1)
                    optional_params["width"] = int(w)
                    optional_params["height"] = int(h)
                else:
                    optional_params[k] = v
            elif not drop_params:
                raise ValueError(
                    f"Parameter '{k}' is not supported for MuAPI image generation. "
                    f"Supported: {sorted(supported)}. Use drop_params=True to ignore."
                )
        return optional_params

    # ------------------------------------------------------------------ #
    #  Request / response transformation                                   #
    # ------------------------------------------------------------------ #

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        payload: dict = {"prompt": prompt}
        payload.update(optional_params)
        return payload

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
        """
        MuAPI uses a two-step async pattern:
          1. POST endpoint → {"request_id": "..."}
          2. GET /predictions/{id}/result → poll until completed

        The raw_response here is from step 1 (the submit call).  We poll
        synchronously inside this method to keep the litellm interface clean.
        """
        try:
            submit_data = raw_response.json()
        except Exception as exc:
            raise self.get_error_class(
                error_message=f"Failed to parse MuAPI submit response: {exc}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        request_id = submit_data.get("request_id")
        if not request_id:
            raise self.get_error_class(
                error_message=f"MuAPI did not return a request_id: {submit_data}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        api_key = api_key or get_secret_str("MUAPI_API_KEY") or ""
        # Extract api_base from the submit response URL if available
        api_base = str(raw_response.url).rsplit("/", 2)[0] if raw_response.url else BASE_URL
        outputs = self._poll_result(request_id, api_key, api_base=api_base)

        if not model_response.data:
            model_response.data = []

        for url in outputs:
            model_response.data.append(ImageObject(url=url, b64_json=None))

        return model_response

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_endpoint(model: str) -> str:
        """Strip the 'muapi/' prefix and look up the MuAPI endpoint path."""
        model_id = model.replace("muapi/", "", 1)
        endpoint = MUAPI_IMAGE_ENDPOINTS.get(model_id)
        if not endpoint:
            raise ValueError(
                f"Unknown MuAPI image model '{model_id}'. "
                f"Supported models: {sorted(MUAPI_IMAGE_ENDPOINTS.keys())}"
            )
        return endpoint

    @staticmethod
    def _poll_result(
        request_id: str,
        api_key: str,
        api_base: Optional[str] = None,
        timeout: float = _POLL_TIMEOUT,
    ) -> List[str]:
        """Poll /predictions/{id}/result until the job is completed or failed."""
        base = (api_base or BASE_URL).rstrip("/")
        # Normalise: if api_base already ends at /api/v1 use it, otherwise append
        if not base.endswith("/api/v1"):
            base = BASE_URL
        poll_url = f"{base}/predictions/{request_id}/result"
        headers = {"x-api-key": api_key}
        deadline = time.monotonic() + timeout

        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"MuAPI image generation timed out after {timeout:.0f}s "
                    f"(request_id={request_id})"
                )
            resp = httpx.get(poll_url, headers=headers, timeout=30)
            if not resp.is_success:
                raise Exception(
                    f"MuAPI poll request failed [{resp.status_code}]: {resp.text}"
                )
            data = resp.json()
            status = data.get("status")
            if status == "completed":
                return data.get("outputs", [])
            if status == "failed":
                raise Exception(
                    f"MuAPI image generation failed: {data.get('error', 'unknown error')}"
                )
            # queued / pending / processing — keep waiting
            time.sleep(_POLL_INTERVAL)


def get_muapi_image_generation_config(model: str) -> MuAPIImageConfig:
    """Factory used by ProviderConfigManager.get_provider_image_generation_config."""
    return MuAPIImageConfig()

"""
MuAPI video generation transformation for LiteLLM.

MuAPI (https://muapi.ai) is a generative media aggregator providing access to
50+ video models (Veo3, Kling, Wan, Seedance, Runway, Pixverse, Sora, etc.)
through a unified REST API.

Pattern: POST /{endpoint} → {"request_id": ...}
         GET  /predictions/{id}/result → poll until status == "completed"

Auth: x-api-key header (set MUAPI_API_KEY env var or pass api_key).

LiteLLM model IDs:  muapi/<model-id>
  e.g. muapi/veo3-fast, muapi/kling-master, muapi/wan2.7
"""

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from httpx._types import RequestFiles

from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

BASE_URL = "https://api.muapi.ai/api/v1"

# Text-to-video model → endpoint mapping
MUAPI_T2V_ENDPOINTS: Dict[str, str] = {
    # --- Google Veo ---
    "veo3": "veo3-text-to-video",
    "veo3-fast": "veo3-fast-text-to-video",
    "veo3.1": "veo3.1-text-to-video",
    "veo4": "veo-4-text-to-video",
    # --- Kling ---
    "kling-master": "kling-v2.1-master-t2v",
    "kling-v2.5-pro": "kling-v2.5-turbo-pro-t2v",
    "kling-v2.6-pro": "kling-v2.6-pro-t2v",
    "kling-v3-pro": "kling-v3.0-pro-text-to-video",
    # --- Wan ---
    "wan2.1": "wan2.1-text-to-video",
    "wan2.2": "wan2.2-text-to-video",
    "wan2.5": "wan2.5-text-to-video",
    "wan2.6": "wan2.6-text-to-video",
    "wan2.7": "wan2.7-text-to-video",
    # --- Seedance ---
    "seedance-pro": "seedance-pro-t2v",
    "seedance-pro-fast": "seedance-pro-t2v-fast",
    # --- Others ---
    "runway": "runway-text-to-video",
    "pixverse": "pixverse-v4.5-t2v",
    "pixverse-v5": "pixverse-v5-t2v",
    "pixverse-v6": "pixverse-v6-t2v",
    "vidu": "vidu-v2.0-t2v",
    "vidu-q2-pro": "vidu-q2-pro-text-to-video",
    "vidu-q3-pro": "vidu-q3-pro-text-to-video",
    "sora": "openai-sora",
    "sora-2": "openai-sora-2-text-to-video",
    "hunyuan": "hunyuan-text-to-video",
}

# Image-to-video model → endpoint mapping
MUAPI_I2V_ENDPOINTS: Dict[str, str] = {
    "kling-std": "kling-v2.1-standard-i2v",
    "kling-pro": "kling-v2.1-pro-i2v",
    "kling-master": "kling-v2.1-master-i2v",
    "kling-v2.5-pro": "kling-v2.5-turbo-pro-i2v",
    "veo3": "veo3-image-to-video",
    "veo3.1": "veo3.1-image-to-video",
    "veo4": "veo-4-image-to-video",
    "wan2.1": "wan2.1-image-to-video",
    "wan2.5": "wan2.5-image-to-video",
    "wan2.6": "wan2.6-image-to-video",
    "wan2.7": "wan2.7-image-to-video",
    "seedance-pro": "seedance-pro-i2v",
    "seedance-v2": "seedance-v2.0-i2v",
    "seedance-2": "seedance-2-image-to-video",
    "runway": "runway-image-to-video",
    "pixverse-v4.5": "pixverse-v4.5-i2v",
    "pixverse-v5": "pixverse-v5-i2v",
    "pixverse-v6": "pixverse-v6-i2v",
    "vidu": "vidu-v2.0-i2v",
    "vidu-q2-pro": "vidu-q2-pro-image-to-video",
    "midjourney": "midjourney-v7-image-to-video",
    "sora-2": "openai-sora-2-image-to-video",
    "hunyuan": "hunyuan-image-to-video",
}

# Models that expect images_list instead of image_url
LIST_INPUT_I2V = {
    "wan2.1", "wan2.5", "wan2.6", "wan2.7",
    "seedance-pro", "seedance-v2", "seedance-2",
    "vidu", "vidu-q2-pro",
    "pixverse-v4.5", "pixverse-v5", "pixverse-v6",
    "sora-2",
}

_POLL_INTERVAL = 3.0


def _muapi_status_to_litellm(status: str) -> str:
    mapping = {
        "queued": "queued",
        "pending": "queued",
        "processing": "in_progress",
        "completed": "completed",
        "failed": "failed",
        "cancelled": "failed",
    }
    return mapping.get(status, "in_progress")


class MuAPIVideoConfig(BaseVideoConfig):
    """
    LiteLLM provider config for MuAPI video generation.

    Supports both text-to-video and image-to-video.  The mode is encoded
    in the model string suffix:
      - ``muapi/<model>``          → text-to-video
      - ``muapi/<model>-i2v``      → image-to-video (pass image_url via
                                      parameters.image_url)
    """

    DEFAULT_BASE_URL: str = BASE_URL

    # ------------------------------------------------------------------ #
    #  Environment / headers                                               #
    # ------------------------------------------------------------------ #

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[GenericLiteLLMParams] = None,
    ) -> dict:
        # Use exact hostname comparison — startswith is bypassable via subdomain
        # (api.muapi.ai.evil.com) or userinfo (api.muapi.ai@evil.com).
        api_base = (litellm_params or {}).get("api_base") if litellm_params else None
        is_official = not api_base or (
            urlparse(str(api_base)).hostname == "api.muapi.ai"
            and urlparse(str(api_base)).scheme == "https"
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
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        base = (api_base or self.DEFAULT_BASE_URL).rstrip("/")
        model_id, is_i2v = self._parse_model(model)
        registry = MUAPI_I2V_ENDPOINTS if is_i2v else MUAPI_T2V_ENDPOINTS
        endpoint = registry.get(model_id)
        if not endpoint:
            available = sorted(registry.keys())
            raise ValueError(
                f"Unknown MuAPI {'image-to-video' if is_i2v else 'text-to-video'} "
                f"model '{model_id}'. Available: {available}"
            )
        return f"{base}/{endpoint}"

    # ------------------------------------------------------------------ #
    #  OpenAI param mapping                                                #
    # ------------------------------------------------------------------ #

    def get_supported_openai_params(self, model: str) -> list:
        return ["seconds", "size"]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        mapped: Dict[str, Any] = {}
        for k, v in video_create_optional_params.items():
            if k == "seconds":
                mapped["duration"] = int(v)  # type: ignore[arg-type]
            elif k == "size" and isinstance(v, str):
                mapped["aspect_ratio"] = v  # pass through e.g. "16:9"
            elif k in ("extra_headers", "extra_body", "user"):
                pass  # handled elsewhere
            elif not drop_params:
                raise ValueError(
                    f"Parameter '{k}' is not supported for MuAPI video generation. "
                    "Use drop_params=True to ignore unsupported parameters."
                )
        return mapped

    # ------------------------------------------------------------------ #
    #  Create request / response                                           #
    # ------------------------------------------------------------------ #

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles, str]:
        model_id, is_i2v = self._parse_model(model)

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "duration": video_create_optional_request_params.get("duration", 5),
            "aspect_ratio": video_create_optional_request_params.get(
                "aspect_ratio", "16:9"
            ),
        }

        # Image-to-video: attach the source image
        if is_i2v:
            # Callers pass image_url inside parameters dict or extra_body
            params_block = video_create_optional_request_params.get("parameters", {}) or {}
            extra_body = video_create_optional_request_params.get("extra_body", {}) or {}
            image_url = (
                params_block.get("image_url")
                or extra_body.get("image_url")
            )
            if image_url:
                if model_id in LIST_INPUT_I2V:
                    payload["images_list"] = [image_url]
                else:
                    payload["image_url"] = image_url

        # Forward any extra provider-specific params
        for k in ("parameters", "extra_body", "extra_headers", "user",
                  "duration", "aspect_ratio"):
            video_create_optional_request_params.pop(k, None)  # type: ignore[misc]

        return payload, None, api_base  # type: ignore[return-value]

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        try:
            data = raw_response.json()
        except Exception as exc:
            raise Exception(
                f"MuAPI: failed to parse create response: {exc}\n{raw_response.text}"
            )

        request_id = data.get("request_id")
        if not request_id:
            raise Exception(f"MuAPI did not return a request_id: {data}")

        video_obj = VideoObject(
            id=request_id,
            object="video",
            status="queued",
        )

        if custom_llm_provider:
            video_obj.id = encode_video_id_with_provider(
                request_id, custom_llm_provider, model
            )

        return video_obj

    # ------------------------------------------------------------------ #
    #  Status polling                                                      #
    # ------------------------------------------------------------------ #

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_id = extract_original_video_id(video_id)
        base = (api_base or BASE_URL).rstrip("/")
        url = f"{base}/predictions/{original_id}/result"
        return url, {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        try:
            data = raw_response.json()
        except Exception as exc:
            raise Exception(
                f"MuAPI: failed to parse status response: {exc}\n{raw_response.text}"
            )

        status = _muapi_status_to_litellm(data.get("status", ""))
        video_url = None
        if status == "completed":
            outputs = data.get("outputs", [])
            if outputs:
                video_url = outputs[0]

        video_obj = VideoObject(
            id=data.get("request_id", ""),
            object="video",
            status=status,
            error={"message": data.get("error")} if data.get("error") else None,
        )

        # Store the output URL in hidden params so the content handler can find it
        if video_url:
            video_obj._hidden_params["video_url"] = video_url  # type: ignore[index]

        return video_obj

    # ------------------------------------------------------------------ #
    #  Content download                                                    #
    # ------------------------------------------------------------------ #

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        """Return the polling URL; actual video URL comes from the status response."""
        original_id = extract_original_video_id(video_id)
        base = (api_base or BASE_URL).rstrip("/")
        url = f"{base}/predictions/{original_id}/result"
        return url, {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        """
        MuAPI returns a JSON status response, not a video binary.
        Extract the video URL then download the actual bytes.
        """
        try:
            data = raw_response.json()
        except Exception:
            return raw_response.content

        outputs = data.get("outputs", [])
        if not outputs:
            return b""

        video_url = outputs[0]
        video_resp = httpx.get(video_url, timeout=120, follow_redirects=True)
        if not video_resp.is_success:
            raise Exception(
                f"MuAPI: failed to download video from {video_url}: "
                f"{video_resp.status_code}"
            )
        return video_resp.content

    # ------------------------------------------------------------------ #
    #  Not-implemented stubs for list / delete / remix                     #
    # ------------------------------------------------------------------ #

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("MuAPI does not support video remix via LiteLLM")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        raise NotImplementedError("MuAPI does not support video remix via LiteLLM")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        extra_query: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("MuAPI does not support video listing via LiteLLM")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("MuAPI does not support video listing via LiteLLM")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("MuAPI does not support video deletion via LiteLLM")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("MuAPI does not support video deletion via LiteLLM")

    # ------------------------------------------------------------------ #
    #  Helper                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_model(model: str) -> Tuple[str, bool]:
        """
        Strip the 'muapi/' prefix and detect whether this is image-to-video.

        Convention: append '-i2v' suffix to the model ID for image-to-video,
        e.g. 'muapi/kling-master-i2v'.
        """
        model_id = model.replace("muapi/", "", 1)
        if model_id.endswith("-i2v"):
            return model_id[:-4], True
        return model_id, False

from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from litellm.constants import XAI_API_BASE
from litellm.exceptions import AuthenticationError
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.xai.common_utils import XAIModelInfo
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

_SIZE_TO_ASPECT_RATIO = {
    "1024x1024": "1:1",
    "1792x1024": "16:9",
    "1024x1792": "9:16",
    "1536x1024": "3:2",
    "1024x1536": "2:3",
    "1280x720": "16:9",
    "720x1280": "9:16",
    "1920x1080": "16:9",
    "1080x1920": "9:16",
}


class XAIImageGenerationConfig(BaseImageGenerationConfig):
    """xAI Imagine image generation (SuperGrok OAuth or API key)."""

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return ["n", "response_format", "size", "user"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = set(self.get_supported_openai_params(model))
        # xAI-native extras accepted via passthrough when present
        xai_native = {"aspect_ratio", "n"}
        for k, v in non_default_params.items():
            if k in optional_params:
                continue
            if k in supported_params or k in xai_native:
                optional_params[k] = v
            elif drop_params:
                pass
            else:
                raise ValueError(
                    f"Parameter {k} is not supported for model {model}. "
                    f"Supported parameters are {sorted(supported_params | xai_native)}. "
                    "Set drop_params=True to drop unsupported parameters."
                )

        size = optional_params.pop("size", None)
        if size and "aspect_ratio" not in optional_params:
            optional_params["aspect_ratio"] = _SIZE_TO_ASPECT_RATIO.get(str(size), "1:1")

        optional_params.pop("response_format", None)
        optional_params.pop("user", None)
        if "n" in optional_params and optional_params["n"] is not None:
            optional_params["n"] = int(optional_params["n"])
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
        from litellm.llms.xai.oauth import XAIOAuthAuthenticator, should_use_xai_oauth

        if should_use_xai_oauth(litellm_params) and not XAIModelInfo.get_api_key(api_key):
            token_file = litellm_params.get("xai_oauth_token_file")
            api_base = XAIOAuthAuthenticator(auth_file=token_file).get_api_base()
        else:
            api_base = (
                api_base
                or get_secret_str("XAI_API_BASE")
                or get_secret_str("XAI_OAUTH_API_BASE")
                or XAI_API_BASE
            )

        base = (api_base or XAI_API_BASE).rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/images/generations"
        return f"{base}/v1/images/generations"

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
        from litellm.llms.xai.oauth import (
            XAIOAuthAuthenticator,
            XAIOAuthError,
            should_use_xai_oauth,
        )

        dynamic_api_key = XAIModelInfo.get_api_key(api_key)
        if should_use_xai_oauth(litellm_params) and not dynamic_api_key:
            token_file = litellm_params.get("xai_oauth_token_file")
            try:
                headers["Authorization"] = (
                    f"Bearer {XAIOAuthAuthenticator(auth_file=token_file).get_access_token()}"
                )
            except XAIOAuthError as exc:
                raise AuthenticationError(
                    model=model,
                    llm_provider="xai",
                    message=str(exc),
                ) from exc
        else:
            if not dynamic_api_key:
                raise AuthenticationError(
                    model=model,
                    llm_provider="xai",
                    message=(
                        "Missing xAI credentials for image generation. "
                        "Pass api_key / XAI_API_KEY, or set use_xai_oauth=True."
                    ),
                )
            headers["Authorization"] = f"Bearer {dynamic_api_key}"

        if "content-type" not in headers and "Content-Type" not in headers:
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
        request: dict = {
            "model": XAIModelInfo.get_base_model(model) or model,
            "prompt": prompt,
        }
        if optional_params.get("aspect_ratio") is not None:
            request["aspect_ratio"] = optional_params["aspect_ratio"]
        if optional_params.get("n") is not None:
            request["n"] = int(optional_params["n"])
        return request

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
        except Exception:
            raise self.get_error_class(
                error_message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        logging_obj.post_call(
            input=request_data.get("prompt", ""),
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},
            original_response=response_data,
        )

        model_response.data = []
        for item in response_data.get("data") or []:
            if not isinstance(item, dict):
                continue
            model_response.data.append(
                ImageObject(
                    url=item.get("url"),
                    b64_json=item.get("b64_json") or item.get("b64"),
                )
            )

        if not model_response.data:
            raise self.get_error_class(
                error_message=f"xAI image generation returned no image data: {response_data}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        return model_response

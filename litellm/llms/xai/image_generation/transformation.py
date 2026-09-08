from typing import TYPE_CHECKING, Any, Final

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

_SIZE_TO_ASPECT_RATIO: Final = {
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
_XAI_NATIVE_PARAMS: Final = frozenset({"aspect_ratio", "n"})


class XAIImageGenerationConfig(BaseImageGenerationConfig):
    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return ["n", "response_format", "size", "user"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params: Final = frozenset(self.get_supported_openai_params(model))
        allowed: Final = supported_params | _XAI_NATIVE_PARAMS
        unknown: Final = tuple(
            key
            for key in non_default_params
            if key not in optional_params and key not in allowed
        )
        if unknown and not drop_params:
            raise ValueError(
                f"Parameter {unknown[0]} is not supported for model {model}. "
                f"Supported parameters are {sorted(allowed)}. "
                "Set drop_params=True to drop unsupported parameters."
            )

        merged: Final = {**optional_params, **{k: v for k, v in non_default_params.items() if k in allowed}}
        size: Final = merged.get("size")
        aspect_ratio: Final = merged.get("aspect_ratio") or (
            _SIZE_TO_ASPECT_RATIO.get(str(size), "1:1") if size else None
        )
        n: Final = merged.get("n")
        return {
            **({"aspect_ratio": aspect_ratio} if aspect_ratio is not None else {}),
            **({"n": int(n)} if n is not None else {}),
        }

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        from litellm.llms.xai.oauth import XAIOAuthAuthenticator, should_use_xai_oauth

        resolved_base: Final = (
            XAIOAuthAuthenticator().get_api_base()
            if should_use_xai_oauth(litellm_params) and not XAIModelInfo.get_api_key(api_key)
            else (
                api_base
                or get_secret_str("XAI_API_BASE")
                or get_secret_str("XAI_OAUTH_API_BASE")
                or XAI_API_BASE
            )
        )
        base: Final = (resolved_base or XAI_API_BASE).rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/images/generations"
        return f"{base}/v1/images/generations"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        from litellm.llms.xai.oauth import (
            XAIOAuthAuthenticator,
            XAIOAuthError,
            should_use_xai_oauth,
        )

        dynamic_api_key: Final = XAIModelInfo.get_api_key(api_key)
        if should_use_xai_oauth(litellm_params) and not dynamic_api_key:
            try:
                headers["Authorization"] = f"Bearer {XAIOAuthAuthenticator().get_access_token()}"
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
        n: Final = optional_params.get("n")
        return {
            "model": XAIModelInfo.get_base_model(model) or model,
            "prompt": prompt,
            **(
                {"aspect_ratio": optional_params["aspect_ratio"]}
                if optional_params.get("aspect_ratio") is not None
                else {}
            ),
            **({"n": int(n)} if n is not None else {}),
        }

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
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        try:
            response_data: Final = raw_response.json()
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

        images: Final = tuple(
            ImageObject(
                url=item.get("url"),
                b64_json=item.get("b64_json") or item.get("b64"),
            )
            for item in response_data.get("data") or ()
            if isinstance(item, dict)
        )
        if not images:
            raise self.get_error_class(
                error_message=f"xAI image generation returned no image data: {response_data}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        model_response.data = list(images)
        return model_response

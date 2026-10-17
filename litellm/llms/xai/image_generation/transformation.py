from typing import TYPE_CHECKING, Final, TypeAlias

import httpx

from litellm.constants import XAI_API_BASE
from litellm.exceptions import AuthenticationError
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.xai.common_utils import XAIModelInfo, xai_image_response
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

_Params: TypeAlias = dict[str, object]
_Headers: TypeAlias = dict[str, object]
_SupportedParams: TypeAlias = list[OpenAIImageGenerationOptionalParams]

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
    def get_supported_openai_params(self, model: str) -> _SupportedParams:
        return ["n", "response_format", "size", "user"]

    def map_openai_params(
        self,
        non_default_params: _Params,
        optional_params: _Params,
        model: str,
        drop_params: bool,
    ) -> _Params:
        supported_params: Final = frozenset(self.get_supported_openai_params(model))
        allowed: Final = supported_params | _XAI_NATIVE_PARAMS
        unknown: Final = tuple(key for key in non_default_params if key not in optional_params and key not in allowed)
        if unknown and not drop_params:
            raise ValueError(
                f"Parameter {unknown[0]} is not supported for model {model}. "
                f"Supported parameters are {sorted(allowed)}. "
                "Set drop_params=True to drop unsupported parameters."
            )

        pairs: Final = ((k, v) for k, v in non_default_params.items() if k in allowed)
        native: Final = dict(pairs)
        merged: Final = {**optional_params, **native}
        size: Final = merged.get("size")
        aspect_ratio: Final = merged.get("aspect_ratio") or (
            _SIZE_TO_ASPECT_RATIO.get(str(size), "1:1") if size else None
        )
        n: Final = merged.get("n")
        fields: Final = (("aspect_ratio", aspect_ratio), ("n", int(n) if isinstance(n, (int, float, str)) else None))
        return {key: value for key, value in fields if value is not None}

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: _Params,
        litellm_params: _Params,
        stream: bool | None = None,
    ) -> str:
        from litellm.llms.xai.oauth import XAIOAuthAuthenticator, should_use_xai_oauth

        resolved_base: Final = (
            XAIOAuthAuthenticator().get_api_base()
            if should_use_xai_oauth(litellm_params) and not XAIModelInfo.get_api_key(api_key)
            else (api_base or get_secret_str("XAI_API_BASE") or get_secret_str("XAI_OAUTH_API_BASE") or XAI_API_BASE)
        )
        base: Final = (resolved_base or XAI_API_BASE).rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/images/generations"
        return f"{base}/v1/images/generations"

    def _bearer_token(self, model: str, api_key: str | None, litellm_params: _Params) -> str:
        from litellm.llms.xai.oauth import (
            XAIOAuthAuthenticator,
            XAIOAuthError,
            should_use_xai_oauth,
        )

        dynamic_api_key: Final = XAIModelInfo.get_api_key(api_key)
        if dynamic_api_key:
            return dynamic_api_key
        if not should_use_xai_oauth(litellm_params):
            raise AuthenticationError(
                model=model,
                llm_provider="xai",
                message=(
                    "Missing xAI credentials for image generation. "
                    "Pass api_key / XAI_API_KEY, or set use_xai_oauth=True."
                ),
            )
        try:
            return XAIOAuthAuthenticator().get_access_token()
        except XAIOAuthError as exc:
            raise AuthenticationError(model=model, llm_provider="xai", message=str(exc)) from exc

    def validate_environment(
        self,
        headers: _Headers,
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: base-class signature
        optional_params: _Params,
        litellm_params: _Params,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> _Headers:
        has_content_type: Final = "content-type" in headers or "Content-Type" in headers
        return {
            **headers,
            "Authorization": f"Bearer {self._bearer_token(model, api_key, litellm_params)}",
            **({} if has_content_type else {"Content-Type": "application/json"}),
        }

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: _Params,
        litellm_params: _Params,
        headers: _Headers,
    ) -> _Params:
        n: Final = optional_params.get("n")
        requested_ratio: Final = optional_params.get("aspect_ratio")
        aspect: Final = {"aspect_ratio": requested_ratio} if requested_ratio is not None else None  # fmt: skip
        count: Final = {"n": int(n)} if isinstance(n, (int, float, str)) else None
        return {
            "model": XAIModelInfo.get_base_model(model) or model,
            "prompt": prompt,
            **(aspect or {}),
            **(count or {}),
        }

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: _Params,
        optional_params: _Params,
        litellm_params: _Params,
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        images: Final = xai_image_response(raw_response, "image generation")
        model_response.data = list(images)  # rebind-ok: the base contract fills the caller's ImageResponse
        return model_response

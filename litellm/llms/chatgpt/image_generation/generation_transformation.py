from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm.exceptions import AuthenticationError
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.openai.common_utils import OpenAIError
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse, ImageUsage

from ..authenticator import Authenticator
from ..common_utils import (
    CHATGPT_API_BASE,
    GetAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
    get_chatgpt_default_instructions,
)
from .response_parsing import (
    dedupe,
    extract_image_payloads,
    extract_image_usage,
    extract_images_from_nested_value,
    extract_images_from_payload,
    get_image_generation_usage,
    get_image_strings_from_dict,
    get_parsed_payloads,
    is_zero_image_usage,
    looks_like_sse,
    parse_sse_payloads,
    transform_image_usage,
)

GPT_IMAGE_MODEL_PREFIX = "gpt-image-"

ALLOWED_OUTPUT_FORMATS = {"png", "jpeg", "webp"}
INTERNAL_OPTIONAL_PARAMS = {"chatgpt_responses_model"}

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class ChatGPTImageGenerationConfig(BaseImageGenerationConfig):
    """
    Bridge OpenAI-style Images API calls to ChatGPT/Codex Responses image generation.
    """

    def __init__(self) -> None:
        self.authenticator = Authenticator()

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return [
            "output_format",
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
        for key, value in non_default_params.items():
            if key in optional_params:
                continue
            if key in supported_params:
                optional_params[key] = value
            elif drop_params:
                continue
            else:
                raise ValueError(
                    f"Parameter {key} is not supported for model {model}. "
                    f"Supported parameters are {supported_params}. "
                    "Set drop_params=True to drop unsupported parameters."
                )
        return optional_params

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
        try:
            access_token = self.authenticator.get_access_token()
        except GetAccessTokenError as e:
            raise AuthenticationError(
                model=model,
                llm_provider="chatgpt",
                message=str(e),
            )

        account_id = self.authenticator.get_account_id()
        session_id = ensure_chatgpt_session_id(litellm_params)
        default_headers = get_chatgpt_default_headers(
            access_token, account_id, session_id
        )
        return {**default_headers, **headers}

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        api_base = self.authenticator.get_api_base() or CHATGPT_API_BASE
        api_base = self._canonicalize_codex_api_base(api_base)
        return f"{api_base}/responses"

    @staticmethod
    def _canonicalize_codex_api_base(api_base: str) -> str:
        api_base = api_base.rstrip("/")
        if api_base.endswith("/responses"):
            api_base = api_base[: -len("/responses")]
        if api_base.endswith("/backend-api"):
            return f"{api_base}/codex"
        return api_base

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        self._validate_openai_image_generation_params(model, optional_params)

        return self._build_responses_image_request(
            model=model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

    def _build_responses_image_request(
        self,
        model: str,
        prompt: Optional[str],
        optional_params: dict,
        litellm_params: dict,
        input_images: Optional[List[Dict[str, Any]]] = None,
    ) -> dict:
        # Intentionally pinned fallback for ChatGPT image generation through the
        # Codex Responses API. Users can override this per request or via
        # litellm_params.
        responses_model = (
            optional_params.pop("chatgpt_responses_model", None)
            or litellm_params.get("chatgpt_responses_model")
            or "gpt-5.5"
        )
        content: List[Dict[str, Any]] = []
        if prompt:
            content.append({"type": "input_text", "text": prompt})
        if input_images:
            content.extend(input_images)

        request: Dict[str, Any] = {
            "model": responses_model,
            "input": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "instructions": get_chatgpt_default_instructions(),
            "tools": [{"type": "image_generation", "model": model}],
            "tool_choice": {"type": "image_generation"},
            "stream": True,
            "store": False,
        }

        image_tool = request["tools"][0]
        for key in (
            "output_format",
            "size",
        ):
            if optional_params.get(key) is not None:
                image_tool[key] = optional_params[key]

        return request

    def _validate_openai_image_generation_params(
        self, model: str, optional_params: dict
    ) -> None:
        if not model.startswith(GPT_IMAGE_MODEL_PREFIX):
            raise ValueError(
                "ChatGPT image generation requires a GPT Image model "
                "(for example gpt-image-1.5 or gpt-image-2)."
            )

        supported_params = set(self.get_supported_openai_params(model))
        unsupported_params = [
            key
            for key in optional_params
            if key not in supported_params and key not in INTERNAL_OPTIONAL_PARAMS
        ]
        if unsupported_params:
            raise ValueError(
                f"Parameters {unsupported_params} are not supported for model {model}. "
                f"Supported parameters are {sorted(supported_params)}."
            )

        output_format = optional_params.get("output_format")
        if output_format is not None and output_format not in ALLOWED_OUTPUT_FORMATS:
            raise ValueError("output_format must be one of png, jpeg, or webp")

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        logging_obj.post_call(
            input=request_data.get("input", ""),
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},
            original_response=raw_response.text,
        )

        image_payloads = self._extract_image_payloads(raw_response)
        if not image_payloads:
            raise OpenAIError(
                message="No image data found in ChatGPT image generation response",
                status_code=raw_response.status_code,
            )

        response = ImageResponse(
            data=[
                ImageObject(b64_json=image_payload) for image_payload in image_payloads
            ]
        )
        response.usage = None
        image_usage = self._extract_image_usage(raw_response)
        if image_usage is not None:
            response.usage = image_usage
        response.size = optional_params.get("size")
        response.output_format = optional_params.get("output_format")
        response._hidden_params["model"] = model
        return response

    def _extract_image_payloads(self, raw_response: httpx.Response) -> List[str]:
        return extract_image_payloads(raw_response)

    def _extract_image_usage(
        self, raw_response: httpx.Response
    ) -> Optional[ImageUsage]:
        return extract_image_usage(raw_response)

    def _get_parsed_payloads(self, raw_response: httpx.Response) -> List[dict]:
        return get_parsed_payloads(raw_response)

    @staticmethod
    def _transform_image_usage(usage: dict) -> ImageUsage:
        return transform_image_usage(usage)

    @staticmethod
    def _get_image_generation_usage(response_payload: Any) -> Optional[dict]:
        return get_image_generation_usage(response_payload)

    @staticmethod
    def _is_zero_image_usage(usage: dict) -> bool:
        return is_zero_image_usage(usage)

    @staticmethod
    def _looks_like_sse(body_text: str) -> bool:
        return looks_like_sse(body_text)

    @staticmethod
    def _parse_sse_payloads(body_text: str) -> List[dict]:
        return parse_sse_payloads(body_text)

    def _extract_images_from_payload(
        self, payload: dict
    ) -> Tuple[List[str], List[str]]:
        return extract_images_from_payload(payload)

    def _extract_images_from_nested_value(self, value: Any) -> List[str]:
        return extract_images_from_nested_value(value)

    @staticmethod
    def _get_image_strings_from_dict(value: dict) -> List[str]:
        return get_image_strings_from_dict(value)

    @staticmethod
    def _dedupe(values: List[str]) -> List[str]:
        return dedupe(values)

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> OpenAIError:
        return OpenAIError(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

import json
import base64
from io import BufferedReader, BytesIO
from os import PathLike
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx
from httpx._types import RequestFiles

from litellm.constants import STREAM_SSE_DONE_STRING
from litellm.exceptions import AuthenticationError
from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.llms.openai.common_utils import OpenAIError
from litellm.types.llms.openai import (
    AllMessageValues,
    FileTypes,
    OpenAIImageGenerationOptionalParams,
    ResponsesAPIStreamEvents,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import (
    ImageObject,
    ImageResponse,
    ImageUsage,
    ImageUsageInputTokensDetails,
)
from litellm.utils import CustomStreamWrapper

from ..authenticator import Authenticator
from ..common_utils import (
    CHATGPT_API_BASE,
    GetAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
    get_chatgpt_default_instructions,
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
        content_type = raw_response.headers.get("content-type", "")
        body_text = raw_response.text or ""
        parsed_payloads: List[dict] = []

        if "text/event-stream" in content_type.lower() or self._looks_like_sse(
            body_text
        ):
            parsed_payloads = self._parse_sse_payloads(body_text)
        else:
            try:
                response_json = raw_response.json()
            except Exception:
                response_json = {}
            if isinstance(response_json, dict):
                parsed_payloads = [response_json]

        images: List[str] = []
        partial_images: List[str] = []
        for payload in parsed_payloads:
            extracted_images, extracted_partial_images = (
                self._extract_images_from_payload(payload)
            )
            images.extend(extracted_images)
            partial_images.extend(extracted_partial_images)
        return self._dedupe(images) or self._dedupe(partial_images)

    def _extract_image_usage(
        self, raw_response: httpx.Response
    ) -> Optional[ImageUsage]:
        parsed_payloads = self._get_parsed_payloads(raw_response)

        for payload in parsed_payloads:
            if payload.get("type") != ResponsesAPIStreamEvents.RESPONSE_COMPLETED:
                continue
            image_gen_usage = self._get_image_generation_usage(payload)
            if image_gen_usage is not None:
                return self._transform_image_usage(image_gen_usage)

        for payload in reversed(parsed_payloads):
            image_gen_usage = self._get_image_generation_usage(payload)
            if image_gen_usage is not None and not self._is_zero_image_usage(
                image_gen_usage
            ):
                return self._transform_image_usage(image_gen_usage)
        return None

    def _get_parsed_payloads(self, raw_response: httpx.Response) -> List[dict]:
        content_type = raw_response.headers.get("content-type", "")
        body_text = raw_response.text or ""

        if "text/event-stream" in content_type.lower() or self._looks_like_sse(
            body_text
        ):
            return self._parse_sse_payloads(body_text)

        try:
            response_json = raw_response.json()
        except Exception:
            response_json = {}
        if isinstance(response_json, dict):
            return [response_json]
        return []

    @staticmethod
    def _transform_image_usage(usage: dict) -> ImageUsage:
        input_tokens_details = usage.get("input_tokens_details") or {}
        return ImageUsage(
            input_tokens=usage.get("input_tokens", 0),
            input_tokens_details=ImageUsageInputTokensDetails(
                image_tokens=input_tokens_details.get("image_tokens", 0),
                text_tokens=input_tokens_details.get("text_tokens", 0),
            ),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    @staticmethod
    def _get_image_generation_usage(response_payload: Any) -> Optional[dict]:
        if not isinstance(response_payload, dict):
            return None

        response = response_payload.get("response")
        if isinstance(response, dict):
            image_gen_usage = ChatGPTImageGenerationConfig._get_image_generation_usage(
                response
            )
            if image_gen_usage is not None:
                return image_gen_usage

        tool_usage = response_payload.get("tool_usage")
        if not isinstance(tool_usage, dict):
            return None

        image_gen_usage = tool_usage.get("image_gen")
        if not isinstance(image_gen_usage, dict):
            return None

        input_tokens = image_gen_usage.get("input_tokens")
        output_tokens = image_gen_usage.get("output_tokens")
        if input_tokens is None or output_tokens is None:
            return None

        normalized_usage = dict(image_gen_usage)
        if normalized_usage.get("total_tokens") is None:
            normalized_usage["total_tokens"] = input_tokens + output_tokens
        return normalized_usage

    @staticmethod
    def _is_zero_image_usage(usage: dict) -> bool:
        return (
            (usage.get("input_tokens") or 0) == 0
            and (usage.get("output_tokens") or 0) == 0
            and (usage.get("total_tokens") or 0) == 0
        )

    @staticmethod
    def _looks_like_sse(body_text: str) -> bool:
        trimmed_body = body_text.lstrip()
        return (
            trimmed_body.startswith("event:")
            or trimmed_body.startswith("data:")
            or "\nevent:" in body_text
            or "\ndata:" in body_text
        )

    @staticmethod
    def _parse_sse_payloads(body_text: str) -> List[dict]:
        payloads: List[dict] = []
        for line in body_text.splitlines():
            stripped_line = CustomStreamWrapper._strip_sse_data_from_chunk(line)
            if not stripped_line:
                continue
            stripped_line = stripped_line.strip()
            if not stripped_line or stripped_line == STREAM_SSE_DONE_STRING:
                continue
            try:
                parsed = json.loads(stripped_line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                payloads.append(parsed)
        return payloads

    def _extract_images_from_payload(
        self, payload: dict
    ) -> Tuple[List[str], List[str]]:
        event_type = payload.get("type")
        if event_type in (
            ResponsesAPIStreamEvents.RESPONSE_FAILED,
            ResponsesAPIStreamEvents.ERROR,
        ):
            error_obj = payload.get("error") or (payload.get("response") or {}).get(
                "error"
            )
            raise OpenAIError(message=str(error_obj or payload), status_code=400)

        partial_images: List[str] = []
        if event_type in (
            ResponsesAPIStreamEvents.IMAGE_GENERATION_PARTIAL_IMAGE,
            "response.image_generation_call.partial_image",
        ):
            partial_image_b64 = payload.get("partial_image_b64")
            b64_json = payload.get("b64_json")
            if isinstance(partial_image_b64, str):
                partial_images.append(partial_image_b64)
            if isinstance(b64_json, str):
                partial_images.append(b64_json)
            return [], partial_images

        candidates: List[str] = []
        if event_type == "image_generation.completed":
            b64_json = payload.get("b64_json")
            if isinstance(b64_json, str):
                candidates.append(b64_json)

        response_payload = payload.get("response")
        if isinstance(response_payload, dict):
            candidates.extend(self._extract_images_from_nested_value(response_payload))

        candidates.extend(self._extract_images_from_nested_value(payload))
        return self._dedupe(candidates), self._dedupe(partial_images)

    def _extract_images_from_nested_value(self, value: Any) -> List[str]:
        images: List[str] = []
        values_to_visit = [value]
        visited_container_ids = set()

        while values_to_visit:
            current_value = values_to_visit.pop()
            if isinstance(current_value, dict):
                container_id = id(current_value)
                if container_id in visited_container_ids:
                    continue
                visited_container_ids.add(container_id)

                value_type = current_value.get("type")
                if value_type in ("image_generation_call", "image_generation"):
                    images.extend(self._get_image_strings_from_dict(current_value))
                elif isinstance(current_value.get("b64_json"), str):
                    images.append(current_value["b64_json"])

                values_to_visit.extend(reversed(list(current_value.values())))
            elif isinstance(current_value, list):
                container_id = id(current_value)
                if container_id in visited_container_ids:
                    continue
                visited_container_ids.add(container_id)

                values_to_visit.extend(reversed(current_value))
        return self._dedupe(images)

    @staticmethod
    def _get_image_strings_from_dict(value: dict) -> List[str]:
        images: List[str] = []
        for key in ("result", "b64_json", "image"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                images.append(candidate)
            elif isinstance(candidate, list):
                images.extend(item for item in candidate if isinstance(item, str))
        return images

    @staticmethod
    def _dedupe(values: List[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> OpenAIError:
        return OpenAIError(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )


class ChatGPTImageEditConfig(BaseImageEditConfig):
    """
    Bridge OpenAI-style Images Edits calls to ChatGPT/Codex Responses image generation.
    """

    def __init__(self) -> None:
        self.image_generation_config = ChatGPTImageGenerationConfig()

    def get_supported_openai_params(self, model: str) -> List[str]:
        return ["size"]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict[str, Any]:
        supported_params = self.get_supported_openai_params(model)
        return {
            key: value
            for key, value in image_edit_optional_params.items()
            if key in supported_params
        }

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[dict] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return self.image_generation_config.validate_environment(
            headers=headers,
            model=model,
            messages=[],
            optional_params={},
            litellm_params=litellm_params or {},
            api_key=api_key,
            api_base=api_base,
        )

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        return self.image_generation_config.get_complete_url(
            api_base=api_base,
            api_key=litellm_params.get("api_key"),
            model=model,
            optional_params={},
            litellm_params=litellm_params,
        )

    def use_multipart_form_data(self) -> bool:
        return False

    def transform_image_edit_request(
        self,
        model: str,
        prompt: Optional[str],
        image: Optional[FileTypes],
        image_edit_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict[str, Any], RequestFiles]:
        optional_params = dict(image_edit_optional_request_params)
        self.image_generation_config._validate_openai_image_generation_params(
            model, optional_params
        )

        input_images = self._prepare_input_images(image)
        if not input_images:
            raise ValueError("ChatGPT image edit requires at least one image.")

        request = self.image_generation_config._build_responses_image_request(
            model=model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params=dict(litellm_params),
            input_images=input_images,
        )
        return request, []

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> ImageResponse:
        return self.image_generation_config.transform_image_generation_response(
            model=model,
            raw_response=raw_response,
            model_response=ImageResponse(),
            logging_obj=logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

    def _prepare_input_images(
        self, image: Optional[Union[FileTypes, List[FileTypes]]]
    ) -> List[Dict[str, Any]]:
        if image is None:
            return []

        images = image if isinstance(image, list) else [image]
        input_images: List[Dict[str, Any]] = []
        for img in images:
            if img is None:
                continue
            mime_type = ImageEditRequestUtils.get_image_content_type(img)
            image_bytes = self._read_image_bytes(img)
            b64_data = base64.b64encode(image_bytes).decode("utf-8")
            input_images.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{b64_data}",
                }
            )
        return input_images

    @staticmethod
    def _read_image_bytes(image: FileTypes) -> bytes:
        if isinstance(image, bytes):
            return image
        if isinstance(image, BytesIO):
            current_pos = image.tell()
            image.seek(0)
            data = image.read()
            image.seek(current_pos)
            return data
        if isinstance(image, BufferedReader):
            current_pos = image.tell()
            image.seek(0)
            data = image.read()
            image.seek(current_pos)
            return data
        if isinstance(image, tuple):
            return ChatGPTImageEditConfig._read_image_bytes(image[1])
        if isinstance(image, PathLike):
            with open(image, "rb") as image_file:
                return image_file.read()
        raise ValueError("Unsupported image type for ChatGPT image edit.")

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> OpenAIError:
        return self.image_generation_config.get_error_class(
            error_message=error_message,
            status_code=status_code,
            headers=headers,
        )

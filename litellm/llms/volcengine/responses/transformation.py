from collections.abc import Callable, Mapping, Sequence
from types import UnionType
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, Union, get_args, get_origin

import httpx
from pydantic import fields as pyd_fields

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _safe_convert_created_field,
)
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
    ResponsesAPIStreamingResponse,
)
from litellm.types.responses.main import DeleteResponseResult
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from ..common_utils import (
    VolcEngineError,
    get_volcengine_base_url,
    get_volcengine_headers,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class _EventModelClass(Protocol):
    @property
    def model_fields(self) -> Mapping[str, pyd_fields.FieldInfo]: ...

    def model_validate(self, obj: Mapping[str, object]) -> ResponsesAPIStreamingResponse: ...


class VolcEngineResponsesAPIConfig(OpenAIResponsesAPIConfig):
    _SUPPORTED_OPTIONAL_PARAMS: list[str] = [
        # Doc-listed knobs
        "instructions",
        "max_output_tokens",
        "previous_response_id",
        "store",
        "reasoning",
        "stream",
        "temperature",
        "top_p",
        "text",
        "tools",
        "tool_choice",
        "max_tool_calls",
        "thinking",
        "caching",
        "expire_at",
        "context_management",
        # LiteLLM-internal metadata (not sent to provider)
        "metadata",
        # Request plumbing helpers
        "extra_headers",
        "extra_query",
        "extra_body",
        "timeout",
    ]

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.VOLCENGINE

    def get_supported_openai_params(self, model: str) -> list:
        """
        Volcengine Responses API: only documented parameters are supported.
        """
        supported: Final = ["input", "model"] + list(self._SUPPORTED_OPTIONAL_PARAMS)
        # Do not advertise internal-only metadata to callers; we still accept and drop it before send.
        if "metadata" in supported:
            supported.remove("metadata")
        return supported

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> VolcEngineError:
        typed_headers: httpx.Headers = headers if isinstance(headers, httpx.Headers) else httpx.Headers(headers or {})
        return VolcEngineError(
            status_code=status_code,
            message=error_message,
            headers=typed_headers,
        )

    def validate_environment(self, headers: dict, model: str, litellm_params: GenericLiteLLMParams | None) -> dict:
        """
        Build auth headers for Volcengine Responses API.
        """
        if litellm_params is None:
            litellm_params = GenericLiteLLMParams()
        elif isinstance(litellm_params, dict):
            litellm_params = GenericLiteLLMParams.model_validate(litellm_params)

        api_key: Final = (
            litellm_params.api_key
            or litellm.api_key
            or get_secret_str("ARK_API_KEY")
            or get_secret_str("VOLCENGINE_API_KEY")
        )

        if api_key is None:
            raise ValueError("Volcengine API key is required. Set ARK_API_KEY / VOLCENGINE_API_KEY or pass api_key.")

        return get_volcengine_headers(api_key=api_key, extra_headers=headers)

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        """
        Construct Volcengine Responses API endpoint.
        """
        base_url = (
            api_base
            or litellm.api_base
            or get_secret_str("VOLCENGINE_API_BASE")
            or get_secret_str("ARK_API_BASE")
            or get_volcengine_base_url()
        )

        base_url = base_url.rstrip("/")

        if base_url.endswith("/responses"):
            return base_url
        if base_url.endswith("/api/v3"):
            return f"{base_url}/responses"
        return f"{base_url}/api/v3/responses"

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Volcengine Responses API aligns with OpenAI parameters.
        Remove parameters not supported by the public docs.
        """
        params: Final = {
            key: value
            for key, value in dict(response_api_optional_params).items()
            if key in self._SUPPORTED_OPTIONAL_PARAMS
        }

        # LiteLLM metadata is internal-only; don't send to provider
        params.pop("metadata", None)

        # Volcengine docs do not list parallel_tool_calls; drop it to avoid backend errors.
        if "parallel_tool_calls" in params:
            verbose_logger.debug("Volcengine Responses API: dropping unsupported 'parallel_tool_calls' param.")
            params.pop("parallel_tool_calls", None)

        return params

    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        """
        Volcengine rejects any undocumented fields (including extra_body). Fail fast
        with clear errors and re-filter with the documented whitelist before delegating
        to the OpenAI base transformer.
        """
        allowed: Final = set(self._SUPPORTED_OPTIONAL_PARAMS)

        sanitized_optional: Final = {k: v for k, v in response_api_optional_request_params.items() if k in allowed}
        # Ensure metadata never reaches provider
        sanitized_optional.pop("metadata", None)
        sanitized_optional.pop("parallel_tool_calls", None)

        # If extra_body is provided, filter its keys against the same allowlist to avoid
        # leaking unsupported params to the provider.
        if isinstance(sanitized_optional.get("extra_body"), dict):
            filtered_body: Final = {k: v for k, v in sanitized_optional["extra_body"].items() if k in allowed}
            if filtered_body:
                sanitized_optional["extra_body"] = filtered_body
            else:
                sanitized_optional.pop("extra_body", None)

        return super().transform_responses_api_request(
            model=model,
            input=input,
            response_api_optional_request_params=sanitized_optional,
            litellm_params=litellm_params,
            headers=headers,
        )

    def transform_streaming_response(
        self,
        model: str,
        parsed_chunk: Mapping[str, object],
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIStreamingResponse:
        """
        Volcengine may omit required fields; auto-fill them using event model defaults.
        """
        chunk = parsed_chunk

        # Patch missing response.output on response.* events
        if isinstance(chunk, dict):
            resp: Final = chunk.get("response")
            if isinstance(resp, dict) and "output" not in resp:
                resp_items: Final[Mapping[str, object]] = resp
                patched_chunk = dict(chunk)
                patched_resp: Final = dict(resp_items)
                patched_resp["output"] = []
                patched_chunk["response"] = patched_resp
                chunk = patched_chunk

        event_type: Final = str(chunk.get("type")) if isinstance(chunk, dict) else None
        event_pydantic_model: _EventModelClass = OpenAIResponsesAPIConfig.get_event_model_class(event_type=event_type)

        patched_chunk = self._fill_missing_fields(chunk, event_pydantic_model)

        return event_pydantic_model.model_validate(patched_chunk)

    def transform_response_api_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        try:
            logging_obj.post_call(
                original_response=raw_response.text,
                additional_args={"complete_input_dict": {}},
            )
            raw_response_json: Final = self._parsed_response_body(raw_response)
            if "created_at" in raw_response_json:
                raw_response_json["created_at"] = _safe_convert_created_field(raw_response_json["created_at"])
        except Exception:
            raise VolcEngineError(message=raw_response.text, status_code=raw_response.status_code)

        raw_response_headers: Final = dict(raw_response.headers)
        processed_headers: Final = process_response_headers(raw_response_headers)

        try:
            response = ResponsesAPIResponse.model_validate(raw_response_json)
        except Exception:
            verbose_logger.debug("Volcengine Responses API: falling back to model_construct for response parsing.")
            construct_response: Final[Callable[..., ResponsesAPIResponse]] = ResponsesAPIResponse.model_construct
            response = construct_response(**raw_response_json)

        response._hidden_params["additional_headers"] = processed_headers
        response._hidden_params["headers"] = raw_response_headers
        return response

    #########################################################
    ########## DELETE RESPONSE API TRANSFORMATION ##############
    #########################################################
    def transform_delete_response_api_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        encoded_response_id: Final = encode_url_path_segment(response_id, field_name="response_id")
        url: Final = f"{api_base}/{encoded_response_id}"
        data: Final[dict] = {}
        return url, data

    def transform_delete_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> DeleteResponseResult:
        try:
            raw_response_json: Final = self._parsed_response_body(raw_response)
        except Exception:
            raise VolcEngineError(message=raw_response.text, status_code=raw_response.status_code)
        try:
            return DeleteResponseResult.model_validate(raw_response_json)
        except Exception:
            verbose_logger.debug(
                "Volcengine Responses API: falling back to model_construct for delete response parsing."
            )
            construct_delete_result: Final[Callable[..., DeleteResponseResult]] = DeleteResponseResult.model_construct
            return construct_delete_result(**raw_response_json)

    #########################################################
    ########## GET RESPONSE API TRANSFORMATION ###############
    #########################################################
    def transform_get_response_api_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        encoded_response_id: Final = encode_url_path_segment(response_id, field_name="response_id")
        url: Final = f"{api_base}/{encoded_response_id}"
        data: Final[dict] = {}
        return url, data

    def transform_get_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        try:
            raw_response_json: Final = self._parsed_response_body(raw_response)
        except Exception:
            raise VolcEngineError(message=raw_response.text, status_code=raw_response.status_code)

        raw_response_headers: Final = dict(raw_response.headers)
        processed_headers: Final = process_response_headers(raw_response_headers)

        response: Final = ResponsesAPIResponse.model_validate(raw_response_json)
        response._hidden_params["additional_headers"] = processed_headers
        response._hidden_params["headers"] = raw_response_headers
        return response

    #########################################################
    ########## LIST INPUT ITEMS TRANSFORMATION #############
    #########################################################
    def transform_list_input_items_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: str | None = None,
        before: str | None = None,
        include: list[str] | None = None,
        limit: int = 20,
        order: Literal["asc", "desc"] = "desc",
    ) -> tuple[str, dict]:
        encoded_response_id: Final = encode_url_path_segment(response_id, field_name="response_id")
        url: Final = f"{api_base}/{encoded_response_id}/input_items"
        params: Final[dict[str, str | int]] = {}
        if after is not None:
            params["after"] = after
        if before is not None:
            params["before"] = before
        if include:
            params["include"] = ",".join(include)
        if limit is not None:
            params["limit"] = limit
        if order is not None:
            params["order"] = order
        return url, params

    def transform_list_input_items_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> dict:
        try:
            return self._parsed_response_body(raw_response)
        except Exception:
            raise VolcEngineError(message=raw_response.text, status_code=raw_response.status_code)

    #########################################################
    ########## CANCEL RESPONSE API TRANSFORMATION ##########
    #########################################################
    def transform_cancel_response_api_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        encoded_response_id: Final = encode_url_path_segment(response_id, field_name="response_id")
        url: Final = f"{api_base}/{encoded_response_id}/cancel"
        data: Final[dict] = {}
        return url, data

    def transform_cancel_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        try:
            raw_response_json: Final = self._parsed_response_body(raw_response)
        except Exception:
            raise VolcEngineError(message=raw_response.text, status_code=raw_response.status_code)

        raw_response_headers: Final = dict(raw_response.headers)
        processed_headers: Final = process_response_headers(raw_response_headers)

        response: Final = ResponsesAPIResponse.model_validate(raw_response_json)
        response._hidden_params["additional_headers"] = processed_headers
        response._hidden_params["headers"] = raw_response_headers
        return response

    def should_fake_stream(
        self,
        model: str | None,
        stream: bool | None,
        custom_llm_provider: str | None = None,
    ) -> bool:
        """
        Volcengine Responses API supports native streaming; never fall back to fake stream.
        """
        return False

    @staticmethod
    def _parsed_response_body(raw_response: httpx.Response) -> dict[str, object]:
        return raw_response.json()

    @staticmethod
    def _annotation_origin(annotation: object) -> object:
        return get_origin(annotation)

    @staticmethod
    def _annotation_args(annotation: object) -> tuple[object, ...]:
        return get_args(annotation)

    @staticmethod
    def _field_annotation(field: pyd_fields.FieldInfo) -> object:
        annotation: Final[object] = field.annotation
        return annotation

    @staticmethod
    def _fill_missing_fields(chunk: Mapping[str, object], event_model: object | None) -> Mapping[str, object]:
        """
        Heuristically fill missing required fields with safe defaults based on the
        event model's field annotations. This keeps parsing tolerant of providers that
        omit non-essential fields.
        """
        if not isinstance(chunk, dict) or event_model is None:
            return chunk

        patched: Final = dict(chunk)
        fields_map: Final[Mapping[str, pyd_fields.FieldInfo]] = getattr(event_model, "model_fields", {}) or {}

        for name, field in fields_map.items():
            if name in patched:
                patched[name] = VolcEngineResponsesAPIConfig._maybe_fill_nested(
                    patched[name], VolcEngineResponsesAPIConfig._field_annotation(field)
                )
                continue

            # Explicit default or factory
            field_default: object = field.default
            if field_default is not pyd_fields.PydanticUndefined and field_default is not None:
                patched[name] = field_default
                continue
            default_factory: Callable[..., object] | None = field.default_factory
            if default_factory is not None and default_factory is not pyd_fields.PydanticUndefined:
                patched[name] = default_factory()
                continue

            # Heuristic defaults for missing required fields
            patched[name] = VolcEngineResponsesAPIConfig._default_for_annotation(
                VolcEngineResponsesAPIConfig._field_annotation(field)
            )

        return patched

    @staticmethod
    def _default_for_annotation(annotation: object) -> object:
        origin: Final = VolcEngineResponsesAPIConfig._annotation_origin(annotation)
        args: Final = VolcEngineResponsesAPIConfig._annotation_args(annotation)

        if annotation is int:
            return 0
        if annotation is list or origin is list:
            return []
        if origin is Union or origin is UnionType:
            # Prefer empty list when any option is a list
            if any((arg is list or VolcEngineResponsesAPIConfig._annotation_origin(arg) is list) for arg in args):
                return []
            if type(None) in args:
                return None

        # Fallback to None when no safer guess exists
        return None

    @staticmethod
    def _maybe_fill_nested(value: object, annotation: object) -> object:
        """
        Recursively fill nested dict/list structures based on the annotated model.
        """
        model_cls: Final = VolcEngineResponsesAPIConfig._pick_model_class(annotation, value)
        args: Final = VolcEngineResponsesAPIConfig._annotation_args(annotation)

        if isinstance(value, dict) and model_cls is not None:
            nested_items: Final[Mapping[str, object]] = value
            return VolcEngineResponsesAPIConfig._fill_missing_fields(nested_items, model_cls)

        if isinstance(value, list):
            # Attempt to fill list elements if we know the element annotation
            elem_ann: Final[object] = args[0] if args else None
            if elem_ann is not None:
                nested_elements: Final[Sequence[object]] = value
                return [VolcEngineResponsesAPIConfig._maybe_fill_nested(v, elem_ann) for v in nested_elements]

        return value

    @staticmethod
    def _pick_model_class(annotation: object, value: object) -> object | None:
        """
        Choose the best-matching Pydantic model class for a nested dict.
        """
        origin: Final = VolcEngineResponsesAPIConfig._annotation_origin(annotation)
        union_args: Final = (
            VolcEngineResponsesAPIConfig._annotation_args(annotation) if origin is Union or origin is UnionType else ()
        )
        candidates = tuple(candidate for candidate in (annotation, *union_args) if hasattr(candidate, "model_fields"))

        if not candidates:
            return None

        # Try to match by literal "type" field when available
        if isinstance(value, dict):
            value_items: Final[Mapping[str, object]] = value
            v_type: Final = value_items.get("type")
            for candidate in candidates:
                try:
                    candidate_fields: Mapping[str, pyd_fields.FieldInfo] = getattr(candidate, "model_fields")
                    type_field = candidate_fields.get("type")
                    if type_field is None:
                        continue
                    literal_ann = VolcEngineResponsesAPIConfig._field_annotation(type_field)
                    if VolcEngineResponsesAPIConfig._annotation_origin(literal_ann) is Literal:
                        literal_values = VolcEngineResponsesAPIConfig._annotation_args(literal_ann)
                        if v_type in literal_values:
                            return candidate
                except Exception:
                    continue

        # Fall back to the first candidate
        return candidates[0]

    def supports_native_websocket(self) -> bool:
        """VolcEngine does not support native WebSocket for Responses API"""
        return False

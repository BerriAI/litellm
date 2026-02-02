from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
)

import httpx
from pydantic import fields as pyd_fields

import litellm
from litellm._logging import verbose_logger
from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIStreamingResponse
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _safe_convert_created_field,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
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


class VolcEngineResponsesAPIConfig(OpenAIResponsesAPIConfig):
    _SUPPORTED_OPTIONAL_PARAMS: List[str] = [
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
        supported = ["input", "model"] + list(self._SUPPORTED_OPTIONAL_PARAMS)
        # Do not advertise internal-only metadata to callers; we still accept and drop it before send.
        if "metadata" in supported:
            supported.remove("metadata")
        return supported

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> VolcEngineError:
        typed_headers: httpx.Headers = (
            headers if isinstance(headers, httpx.Headers) else httpx.Headers(headers or {})
        )
        return VolcEngineError(
            status_code=status_code,
            message=error_message,
            headers=typed_headers,
        )

    def validate_environment(
        self, headers: dict, model: str, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        """
        Build auth headers for Volcengine Responses API.
        """
        if litellm_params is None:
            litellm_params = GenericLiteLLMParams()
        elif isinstance(litellm_params, dict):
            litellm_params = GenericLiteLLMParams(**litellm_params)

        api_key = (
            litellm_params.api_key
            or litellm.api_key
            or get_secret_str("ARK_API_KEY")
            or get_secret_str("VOLCENGINE_API_KEY")
        )

        if api_key is None:
            raise ValueError(
                "Volcengine API key is required. Set ARK_API_KEY / VOLCENGINE_API_KEY or pass api_key."
            )

        return get_volcengine_headers(api_key=api_key, extra_headers=headers)

    def get_complete_url(
        self,
        api_base: Optional[str],
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
    ) -> Dict:
        """
        Volcengine Responses API aligns with OpenAI parameters.
        Remove parameters not supported by the public docs.
        """
        params = {
            key: value
            for key, value in dict(response_api_optional_params).items()
            if key in self._SUPPORTED_OPTIONAL_PARAMS
        }

        # LiteLLM metadata is internal-only; don't send to provider
        params.pop("metadata", None)

        # Volcengine docs do not list parallel_tool_calls; drop it to avoid backend errors.
        if "parallel_tool_calls" in params:
            verbose_logger.debug(
                "Volcengine Responses API: dropping unsupported 'parallel_tool_calls' param."
            )
            params.pop("parallel_tool_calls", None)

        return params

    def transform_responses_api_request(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        response_api_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """
        Volcengine rejects any undocumented fields (including extra_body). Fail fast
        with clear errors and re-filter with the documented whitelist before delegating
        to the OpenAI base transformer.
        """
        allowed = set(self._SUPPORTED_OPTIONAL_PARAMS)

        sanitized_optional = {
            k: v for k, v in response_api_optional_request_params.items() if k in allowed
        }
        # Ensure metadata never reaches provider
        sanitized_optional.pop("metadata", None)
        sanitized_optional.pop("parallel_tool_calls", None)

        # If extra_body is provided, filter its keys against the same allowlist to avoid
        # leaking unsupported params to the provider.
        if isinstance(sanitized_optional.get("extra_body"), dict):
            filtered_body = {
                k: v for k, v in sanitized_optional["extra_body"].items() if k in allowed
            }
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
        parsed_chunk: dict,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIStreamingResponse:
        """
        Volcengine may omit required fields; auto-fill them using event model defaults.
        """
        chunk = parsed_chunk

        # Patch missing response.output on response.* events
        if isinstance(chunk, dict):
            resp = chunk.get("response")
            if isinstance(resp, dict) and "output" not in resp:
                patched_chunk = dict(chunk)
                patched_resp = dict(resp)
                patched_resp["output"] = []
                patched_chunk["response"] = patched_resp
                chunk = patched_chunk

        event_type = str(chunk.get("type")) if isinstance(chunk, dict) else None
        event_pydantic_model = OpenAIResponsesAPIConfig.get_event_model_class(
            event_type=event_type
        )

        patched_chunk = self._fill_missing_fields(chunk, event_pydantic_model)

        return event_pydantic_model(**patched_chunk)

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
            raw_response_json = raw_response.json()
            if "created_at" in raw_response_json:
                raw_response_json["created_at"] = _safe_convert_created_field(
                    raw_response_json["created_at"]
                )
        except Exception:
            raise VolcEngineError(
                message=raw_response.text, status_code=raw_response.status_code
            )

        raw_response_headers = dict(raw_response.headers)
        processed_headers = process_response_headers(raw_response_headers)

        try:
            response = ResponsesAPIResponse(**raw_response_json)
        except Exception:
            verbose_logger.debug(
                "Volcengine Responses API: falling back to model_construct for response parsing."
            )
            response = ResponsesAPIResponse.model_construct(**raw_response_json)

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
    ) -> Tuple[str, Dict]:
        url = f"{api_base}/{response_id}"
        data: Dict = {}
        return url, data

    def transform_delete_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> DeleteResponseResult:
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise VolcEngineError(
                message=raw_response.text, status_code=raw_response.status_code
            )
        try:
            return DeleteResponseResult(**raw_response_json)
        except Exception:
            verbose_logger.debug(
                "Volcengine Responses API: falling back to model_construct for delete response parsing."
            )
            return DeleteResponseResult.model_construct(**raw_response_json)

    #########################################################
    ########## GET RESPONSE API TRANSFORMATION ###############
    #########################################################
    def transform_get_response_api_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        url = f"{api_base}/{response_id}"
        data: Dict = {}
        return url, data

    def transform_get_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise VolcEngineError(
                message=raw_response.text, status_code=raw_response.status_code
            )

        raw_response_headers = dict(raw_response.headers)
        processed_headers = process_response_headers(raw_response_headers)

        response = ResponsesAPIResponse(**raw_response_json)
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
        after: Optional[str] = None,
        before: Optional[str] = None,
        include: Optional[List[str]] = None,
        limit: int = 20,
        order: Literal["asc", "desc"] = "desc",
    ) -> Tuple[str, Dict]:
        url = f"{api_base}/{response_id}/input_items"
        params: Dict[str, Any] = {}
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
    ) -> Dict:
        try:
            return raw_response.json()
        except Exception:
            raise VolcEngineError(
                message=raw_response.text, status_code=raw_response.status_code
            )

    #########################################################
    ########## CANCEL RESPONSE API TRANSFORMATION ##########
    #########################################################
    def transform_cancel_response_api_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        url = f"{api_base}/{response_id}/cancel"
        data: Dict = {}
        return url, data

    def transform_cancel_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise VolcEngineError(
                message=raw_response.text, status_code=raw_response.status_code
            )

        raw_response_headers = dict(raw_response.headers)
        processed_headers = process_response_headers(raw_response_headers)

        response = ResponsesAPIResponse(**raw_response_json)
        response._hidden_params["additional_headers"] = processed_headers
        response._hidden_params["headers"] = raw_response_headers
        return response

    def should_fake_stream(
        self,
        model: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        """
        Volcengine Responses API supports native streaming; never fall back to fake stream.
        """
        return False

    @staticmethod
    def _fill_missing_fields(
        chunk: Any, event_model: Any
    ) -> Dict[str, Any]:
        """
        Heuristically fill missing required fields with safe defaults based on the
        event model's field annotations. This keeps parsing tolerant of providers that
        omit non-essential fields.
        """
        if not isinstance(chunk, dict) or event_model is None:
            return chunk

        patched: Dict[str, Any] = dict(chunk)
        fields_map = getattr(event_model, "model_fields", {}) or {}

        for name, field in fields_map.items():
            if name in patched:
                patched[name] = VolcEngineResponsesAPIConfig._maybe_fill_nested(
                    patched[name], field.annotation
                )
                continue

            # Explicit default or factory
            if field.default is not pyd_fields.PydanticUndefined and field.default is not None:
                patched[name] = field.default
                continue
            if (
                field.default_factory is not None
                and field.default_factory is not pyd_fields.PydanticUndefined
            ):
                patched[name] = field.default_factory()
                continue

            # Heuristic defaults for missing required fields
            patched[name] = VolcEngineResponsesAPIConfig._default_for_annotation(
                field.annotation
            )

        return patched

    @staticmethod
    def _default_for_annotation(annotation: Any) -> Any:
        origin = get_origin(annotation)
        args = get_args(annotation)

        if annotation is int:
            return 0
        if annotation is list or origin is list:
            return []
        if origin is Union:
            # Prefer empty list when any option is a list
            if any((arg is list or get_origin(arg) is list) for arg in args):
                return []
            if type(None) in args:
                return None
        if origin is Union and type(None) in args:
            return None

        # Fallback to None when no safer guess exists
        return None

    @staticmethod
    def _maybe_fill_nested(value: Any, annotation: Any) -> Any:
        """
        Recursively fill nested dict/list structures based on the annotated model.
        """
        model_cls = VolcEngineResponsesAPIConfig._pick_model_class(annotation, value)
        args = get_args(annotation)

        if isinstance(value, dict) and model_cls is not None:
            return VolcEngineResponsesAPIConfig._fill_missing_fields(value, model_cls)

        if isinstance(value, list):
            # Attempt to fill list elements if we know the element annotation
            elem_ann: Any = args[0] if args else None
            if elem_ann is not None:
                return [
                    VolcEngineResponsesAPIConfig._maybe_fill_nested(v, elem_ann)
                    for v in value
                ]

        return value

    @staticmethod
    def _pick_model_class(annotation: Any, value: Any) -> Optional[Any]:
        """
        Choose the best-matching Pydantic model class for a nested dict.
        """
        candidates: List[Any] = []
        origin = get_origin(annotation)

        if hasattr(annotation, "model_fields"):
            candidates.append(annotation)
        if origin is Union:
            for arg in get_args(annotation):
                if hasattr(arg, "model_fields"):
                    candidates.append(arg)

        if not candidates:
            return None

        # Try to match by literal "type" field when available
        if isinstance(value, dict):
            v_type = value.get("type")
            for candidate in candidates:
                try:
                    type_field = candidate.model_fields.get("type")
                    if type_field is None:
                        continue
                    literal_ann = type_field.annotation
                    if get_origin(literal_ann) is Literal:
                        literal_values = get_args(literal_ann)
                        if v_type in literal_values:
                            return candidate
                except Exception:
                    continue

        # Fall back to the first candidate
        return candidates[0]

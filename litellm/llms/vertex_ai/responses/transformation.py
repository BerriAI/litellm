import json
from collections.abc import Mapping
from typing import Final
from urllib.parse import urlparse

from pydantic import JsonValue, TypeAdapter

from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.openai.chat.gpt_reasoning_transformation import unsupported_reasoning_effort_message
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig, drops_params
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase, graft_default_vertex_path
from litellm.types.llms.openai import ResponseInputParam
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

_FIELDS: Final = TypeAdapter(dict[str, object])
_PARAM_NAMES: Final = TypeAdapter(tuple[str, ...])
_CREDENTIALS: Final[TypeAdapter[str | Mapping[str, JsonValue] | None]] = TypeAdapter(
    str | Mapping[str, JsonValue] | None
)
_AUTH: Final = VertexBase()
_MUSE_PREFIX: Final = "meta/muse-"
_STORED_RESPONSE_PARAMS: Final = frozenset({"background", "conversation", "previous_response_id"})
_MUSE_PUBLISHER_PATH: Final = "/publishers/meta/v1"
_MUSE_RESPONSES_PATH: Final = f"{_MUSE_PUBLISHER_PATH}/responses"
MUSE_CHAT_PARAMS: Final = (
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "stream_options",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "reasoning_effort",
    "user",
    "store",
    "service_tier",
    "safety_identifier",
    "prompt_cache_key",
    "prompt_cache_retention",
    "extra_headers",
)


def is_vertex_muse_model(model: str | None) -> bool:
    return (model or "").removeprefix("vertex_ai/").startswith(_MUSE_PREFIX)


def _reasoning_fields(reasoning: object) -> Mapping[str, object]:
    return _FIELDS.validate_python(reasoning) if isinstance(reasoning, Mapping) else {}


def _refused_reasoning_effort(model: str, reasoning: Mapping[str, object]) -> str | None:
    from litellm.utils import is_explicitly_disabled_factory

    effort: Final = reasoning.get("effort")
    if not isinstance(effort, str):
        return None
    disabled: Final = is_explicitly_disabled_factory(
        model=f"vertex_ai/{model.removeprefix('vertex_ai/')}",
        custom_llm_provider=None,
        key=f"supports_{effort}_reasoning_effort",
    )
    return effort if disabled else None


class VertexAIMuseResponsesAPIConfig(OpenAIResponsesAPIConfig):
    def __init__(self, vertex_auth: VertexBase | None = None) -> None:
        super().__init__()
        self._vertex_auth = vertex_auth or _AUTH

    @classmethod
    def for_model(cls, model: str | None) -> "VertexAIMuseResponsesAPIConfig | None":
        return cls() if is_vertex_muse_model(model) else None

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.VERTEX_AI

    def supports_native_file_search(self) -> bool:
        return False

    def supports_native_websocket(self) -> bool:
        return False

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: base API requires a list return
        names: Final = _PARAM_NAMES.validate_python(
            super().get_supported_openai_params(  # pyright: ignore[reportUnknownMemberType]  # legacy list is untyped
                model
            )
        )
        return [name for name in names if name not in _STORED_RESPONSE_PARAMS]

    def _access_token(self, params: Mapping[str, object]) -> tuple[str, str]:
        fields: Final = dict(params)
        credentials: Final = _CREDENTIALS.validate_python(
            VertexBase.safe_get_vertex_ai_credentials(  # pyright: ignore[reportUnknownMemberType]  # untyped bag
                fields
            )
        )
        return self._vertex_auth.ensure_access_token(
            credentials=json.dumps(dict(credentials)) if isinstance(credentials, Mapping) else credentials,
            project_id=VertexBase.safe_get_vertex_ai_project(  # pyright: ignore[reportUnknownMemberType]  # untyped bag
                fields
            ),
            custom_llm_provider="vertex_ai",
        )

    def validate_environment(
        self, headers: Mapping[str, str], model: str, litellm_params: GenericLiteLLMParams | None
    ) -> dict[str, str]:  # mutable-ok: Responses handler mutates the returned headers
        params: Final = litellm_params or GenericLiteLLMParams()
        token, _ = self._access_token(_FIELDS.validate_python(params.model_dump()))
        return {**headers, "Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    def get_complete_url(self, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        explicit_project: Final = VertexBase.safe_get_vertex_ai_project(  # pyright: ignore[reportUnknownMemberType]  # untyped bag
            dict(litellm_params)
        )
        project: Final = explicit_project or self._access_token(litellm_params)[1]
        default_url: Final = (
            f"https://aiplatform.googleapis.com/v1/projects/{encode_url_path_segment(project)}"
            "/locations/global/publishers/meta/v1/responses"
        )
        if not api_base:
            return default_url
        parsed_api_base: Final = urlparse(api_base)
        api_base_path: Final = parsed_api_base.path.rstrip("/")
        if api_base_path.endswith(_MUSE_RESPONSES_PATH):
            return api_base
        if api_base_path.endswith(_MUSE_PUBLISHER_PATH):
            return parsed_api_base._replace(path=api_base_path + "/responses").geturl()
        if api_base_path.endswith(("/v1", "/v1beta1")):
            return graft_default_vertex_path(api_base, default_url)
        return parsed_api_base._replace(path=api_base_path + urlparse(default_url).path).geturl()

    def transform_extra_body(
        self,
        extra_body: Mapping[str, object],
        request: Mapping[str, object],
        model: str,
        litellm_params: GenericLiteLLMParams,
    ) -> Mapping[str, object]:
        merged: Final = super().transform_extra_body(extra_body, request, model, litellm_params)
        return {key: value for key, value in merged.items() if key != "model"}

    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> dict[str, object]:  # mutable-ok: Responses handler mutates the returned request
        drops_unsupported: Final = drops_params(litellm_params)
        if response_api_optional_request_params.get("store") is True and not drops_unsupported:
            raise UnsupportedParamsError(
                status_code=400,
                message="Vertex Muse supports stateless Responses requests only. Set store=False or drop_params=True.",
            )
        location: Final = VertexBase.explicit_vertex_ai_location(_FIELDS.validate_python(litellm_params.model_dump()))
        if location not in (None, "global") and not drops_unsupported:
            raise UnsupportedParamsError(
                status_code=400,
                message="Vertex Muse is served from the global location only. Set vertex_location to global "
                "or drop_params=True.",
            )
        truncates: Final = response_api_optional_request_params.get("truncation") not in (None, "disabled")
        if truncates and not drops_unsupported:
            raise UnsupportedParamsError(
                status_code=400,
                message="Vertex Muse supports only truncation='disabled'. Set drop_params=True to omit it.",
            )
        tool_choice: Final = response_api_optional_request_params.get("tool_choice")
        if tool_choice is not None and tool_choice != "auto" and not drops_unsupported:
            raise UnsupportedParamsError(
                status_code=400,
                message="Vertex Muse supports only tool_choice='auto'. Set drop_params=True to normalize unsupported choices.",
            )
        reasoning: Final = _reasoning_fields(response_api_optional_request_params.get("reasoning"))
        refused_effort: Final = _refused_reasoning_effort(model, reasoning)
        if refused_effort is not None and not drops_unsupported:
            raise UnsupportedParamsError(
                status_code=400,
                message=unsupported_reasoning_effort_message("reasoning.effort", refused_effort, model),
            )
        kept_reasoning: Final = {key: value for key, value in reasoning.items() if key != "effort"}
        dropped_params: Final = (
            (_STORED_RESPONSE_PARAMS if drops_unsupported else frozenset[str]())
            | ({"tools"} if tool_choice == "none" else frozenset[str]())
            | ({"truncation"} if truncates else frozenset[str]())
            | ({"reasoning"} if refused_effort is not None else frozenset[str]())
        )
        tool_params: Final = {
            key: value for key, value in response_api_optional_request_params.items() if key not in dropped_params
        }
        normalized_params: Final = {
            **tool_params,
            **({"tool_choice": "auto"} if tool_choice is not None else {}),
            **({"reasoning": kept_reasoning} if refused_effort is not None and kept_reasoning else {}),
        }
        request: Final = _FIELDS.validate_python(
            super().transform_responses_api_request(  # pyright: ignore[reportUnknownMemberType]  # untyped bags
                model, input, normalized_params, litellm_params, dict(headers)
            )
        )
        return {
            **request,
            "model": model.removeprefix("vertex_ai/").removeprefix("meta/"),
            "store": False,
        }

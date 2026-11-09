import json
from collections.abc import Mapping
from typing import Final

from pydantic import JsonValue, TypeAdapter

from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
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
_MUSE_FAMILIES: Final = ("muse-wonder-", "muse-ember-", "muse-spark-")
_STORED_RESPONSE_PARAMS: Final = frozenset({"background", "conversation", "previous_response_id"})


class VertexAIMuseResponsesAPIConfig(OpenAIResponsesAPIConfig):
    def __init__(self, vertex_auth: VertexBase | None = None) -> None:
        super().__init__()
        self._vertex_auth = vertex_auth or _AUTH

    @classmethod
    def for_model(cls, model: str | None) -> "VertexAIMuseResponsesAPIConfig | None":
        name: Final = (model or "").removeprefix("vertex_ai/").removeprefix("meta/")
        return cls() if name.startswith(_MUSE_FAMILIES) else None

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.VERTEX_AI

    def supports_native_file_search(self) -> bool:
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
        _, project = self._access_token(litellm_params)
        default_url: Final = (
            f"https://aiplatform.googleapis.com/v1/projects/{encode_url_path_segment(project)}"
            "/locations/global/publishers/meta/v1/responses"
        )
        return graft_default_vertex_path(api_base, default_url) if api_base else default_url

    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> dict[str, object]:  # mutable-ok: Responses handler mutates the returned request
        if response_api_optional_request_params.get("store") is True:
            raise UnsupportedParamsError(
                status_code=400, message="Vertex Muse supports stateless Responses requests only. Set store=False."
            )
        request: Final = _FIELDS.validate_python(
            super().transform_responses_api_request(  # pyright: ignore[reportUnknownMemberType]  # untyped bags
                model, input, dict(response_api_optional_request_params), litellm_params, dict(headers)
            )
        )
        return {
            **request,
            "model": model.removeprefix("vertex_ai/").removeprefix("meta/"),
            "store": False,
        }

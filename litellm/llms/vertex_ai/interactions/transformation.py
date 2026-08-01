from typing import Protocol, TypeVar
from urllib.parse import urlparse

import httpx

from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.interactions.transformation import BaseInteractionsAPIConfig
from litellm.llms.gemini.interactions.transformation import (
    GoogleAIStudioInteractionsConfig,
    LiteLLMLoggingObj,
)
from litellm.llms.vertex_ai.common_utils import get_vertex_ai_lyria_model_info
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.interactions import (
    CancelInteractionResult,
    DeleteInteractionResult,
    InteractionInput,
    InteractionsAPIOptionalRequestParams,
    InteractionsAPIResponse,
    InteractionsAPIStreamingResponse,
)
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

_DEFAULT_VERTEX_AI_INTERACTIONS_API_BASE = "https://aiplatform.googleapis.com"
_TRUSTED_GOOGLE_API_HOST_SUFFIX = ".googleapis.com"
_InteractionsResponseT = TypeVar(
    "_InteractionsResponseT",
    InteractionsAPIResponse,
    InteractionsAPIStreamingResponse,
)


class VertexAIInteractionsAuth(Protocol):
    def get_access_token(
        self,
        credentials: VERTEX_CREDENTIALS_TYPES | None,
        project_id: str | None,
        _retry_reauth: bool = False,
    ) -> tuple[str, str]: ...


class VertexAIInteractionsConfig(BaseInteractionsAPIConfig):
    def __init__(
        self,
        vertex_auth: VertexAIInteractionsAuth | None = None,
        interactions_config: GoogleAIStudioInteractionsConfig | None = None,
    ) -> None:
        self._vertex_auth = vertex_auth or VertexBase()
        self._interactions_config = interactions_config or GoogleAIStudioInteractionsConfig()

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.VERTEX_AI

    @property
    def api_version(self) -> str:
        return "v1beta1"

    def get_supported_params(self, model: str) -> list[str]:
        return self._interactions_config.get_supported_params(model)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict:
        headers = headers or {}
        headers["Content-Type"] = "application/json"
        access_token, _ = self._get_access_token(litellm_params=litellm_params)
        headers["Authorization"] = f"Bearer {access_token}"
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        model: str | None,
        agent: str | None = None,
        litellm_params: dict | None = None,
        stream: bool | None = None,
    ) -> str:
        return self._get_interactions_url(
            api_base=api_base,
            litellm_params=litellm_params,
            stream=stream,
        )

    def transform_request(
        self,
        model: str | None,
        agent: str | None,
        input: InteractionInput | None,
        optional_params: InteractionsAPIOptionalRequestParams,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        return self._interactions_config.transform_request(
            model=self._get_base_model(model=model),
            agent=agent,
            input=input,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def transform_response(
        self,
        model: str | None,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> InteractionsAPIResponse:
        return self._apply_lyria_response_cost(
            model=model,
            response=self._interactions_config.transform_response(
                model=model,
                raw_response=raw_response,
                logging_obj=logging_obj,
            ),
        )

    def transform_streaming_response(
        self,
        model: str | None,
        parsed_chunk: dict,
        logging_obj: LiteLLMLoggingObj,
    ) -> InteractionsAPIStreamingResponse:
        return self._apply_lyria_response_cost(
            model=model,
            response=self._interactions_config.transform_streaming_response(
                model=model,
                parsed_chunk=parsed_chunk,
                logging_obj=logging_obj,
            ),
        )

    @staticmethod
    def _apply_lyria_response_cost(
        model: str | None,
        response: _InteractionsResponseT,
    ) -> _InteractionsResponseT:
        model_info = get_vertex_ai_lyria_model_info(model=model) if model is not None else None
        response_cost = (
            model_info.get("output_cost_per_image")
            if model_info is not None and model_info["vertex_ai_audio_api"] == "lyria_interactions"
            else None
        )
        if response_cost is None:
            return response
        response._hidden_params = {**response._hidden_params, "response_cost": response_cost}
        return response

    def transform_get_interaction_request(
        self,
        interaction_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        return (
            self._get_interactions_url(
                api_base=api_base,
                litellm_params=litellm_params,
                interaction_id=interaction_id,
            ),
            {},
        )

    def transform_get_interaction_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> InteractionsAPIResponse:
        return self._interactions_config.transform_get_interaction_response(
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

    def transform_delete_interaction_request(
        self,
        interaction_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        return (
            self._get_interactions_url(
                api_base=api_base,
                litellm_params=litellm_params,
                interaction_id=interaction_id,
            ),
            {},
        )

    def transform_delete_interaction_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        interaction_id: str,
    ) -> DeleteInteractionResult:
        return self._interactions_config.transform_delete_interaction_response(
            raw_response=raw_response,
            logging_obj=logging_obj,
            interaction_id=interaction_id,
        )

    def transform_cancel_interaction_request(
        self,
        interaction_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        return (
            self._get_interactions_url(
                api_base=api_base,
                litellm_params=litellm_params,
                interaction_id=interaction_id,
                cancel=True,
            ),
            {},
        )

    def transform_cancel_interaction_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> CancelInteractionResult:
        return self._interactions_config.transform_cancel_interaction_response(
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

    def _get_interactions_url(
        self,
        api_base: str | None,
        litellm_params: GenericLiteLLMParams | dict | None,
        interaction_id: str | None = None,
        cancel: bool = False,
        stream: bool | None = None,
    ) -> str:
        base_url = self._get_api_base(api_base=api_base)
        project_id = self._get_project_id(litellm_params=litellm_params)
        url = f"{base_url}/{self.api_version}/projects/{project_id}/locations/global/interactions"
        if interaction_id is not None:
            encoded_interaction_id = encode_url_path_segment(interaction_id, field_name="interaction_id")
            url = f"{url}/{encoded_interaction_id}"
        if cancel:
            url = f"{url}:cancel"
        if stream:
            url = f"{url}?alt=sse"
        return url

    @staticmethod
    def _get_api_base(api_base: str | None) -> str:
        if api_base is None or api_base == "":
            return _DEFAULT_VERTEX_AI_INTERACTIONS_API_BASE

        base_url = api_base.rstrip("/")
        parsed_api_base = urlparse(base_url)
        hostname = parsed_api_base.hostname
        if (
            parsed_api_base.scheme != "https"
            or hostname is None
            or not hostname.endswith(_TRUSTED_GOOGLE_API_HOST_SUFFIX)
        ):
            raise ValueError("Vertex AI interactions api_base must be a trusted Google API HTTPS endpoint")
        return base_url

    def _get_project_id(
        self,
        litellm_params: GenericLiteLLMParams | dict | None,
    ) -> str:
        params_dict = (
            litellm_params.model_dump(exclude_none=True)
            if isinstance(litellm_params, GenericLiteLLMParams)
            else litellm_params or {}
        )
        vertex_project = VertexBase.safe_get_vertex_ai_project(params_dict)
        if vertex_project is not None:
            return vertex_project

        _, project_id = self._get_access_token(litellm_params=litellm_params)
        return project_id

    def _get_access_token(
        self,
        litellm_params: GenericLiteLLMParams | dict | None,
    ) -> tuple[str, str]:
        params_dict = (
            litellm_params.model_dump(exclude_none=True)
            if isinstance(litellm_params, GenericLiteLLMParams)
            else litellm_params or {}
        )
        vertex_project = VertexBase.safe_get_vertex_ai_project(params_dict)
        vertex_credentials = VertexBase.safe_get_vertex_ai_credentials(params_dict)
        return self._vertex_auth.get_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
        )

    @staticmethod
    def _get_base_model(model: str | None) -> str | None:
        if model is None:
            return None
        return model.replace("vertex_ai/", "", 1)

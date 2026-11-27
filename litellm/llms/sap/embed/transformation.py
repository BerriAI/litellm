"""
Translates from OpenAI's `/v1/embeddings` to IBM's `/text/embeddings` route.
"""

from collections.abc import Mapping
from functools import cached_property
from typing import Final, Literal

import httpx
from pydantic import Field, TypeAdapter

from litellm.litellm_core_utils.core_helpers import iter_items, require_get, subscript
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.embedding.transformation import (
    BaseEmbeddingConfig,
    LiteLLMLoggingObj,
)
from litellm.llms.sap.chat.models import MaskingModuleConfig
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.openai import AllEmbeddingInputValues
from litellm.types.utils import EmbeddingResponse

from ..chat.handler import GenAIHubOrchestrationError
from ..credentials import get_token_creator

_DECODED_JSON: Final = TypeAdapter(object)


class Usage(LiteLLMBaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingItem(LiteLLMBaseModel):
    object: Literal["embedding"]
    embedding: list[float] = Field(..., description="Vector of floats (length varies by model).")
    index: int


class FinalResult(LiteLLMBaseModel):
    object: Literal["list"]
    data: list[EmbeddingItem]
    model: str
    usage: Usage


class EmbeddingsResponse(LiteLLMBaseModel):
    request_id: str
    final_result: FinalResult


class EmbeddingModel(LiteLLMBaseModel):
    name: str
    version: str = "latest"
    params: dict = Field(default_factory=dict)
    timeout: int | None = Field(default=None, ge=1, le=600)
    max_retries: int | None = Field(default=None, ge=0, le=5)


class EmbeddingsModelConfig(LiteLLMBaseModel):
    model: EmbeddingModel


class EmbeddingsModules(LiteLLMBaseModel):
    embeddings: EmbeddingsModelConfig
    masking: MaskingModuleConfig | None = None


class EmbeddingInput(LiteLLMBaseModel):
    text: str | list[str]
    type: Literal["text", "document", "query"] | None = None


class EmbeddingConfig(LiteLLMBaseModel):
    modules: EmbeddingsModules


class EmbeddingRequest(LiteLLMBaseModel):
    config: EmbeddingConfig
    input: EmbeddingInput


def validate_dict(data: Mapping[str, object], model: type[LiteLLMBaseModel]) -> dict[str, object]:
    return model(**data).model_dump(exclude_unset=True, by_alias=True)


class GenAIHubEmbeddingConfig(BaseEmbeddingConfig):
    def __init__(self):
        super().__init__()
        self._access_token_data = {}
        self.token_creator, self.base_url, self.resource_group = get_token_creator()
        self._request_authorization: str | None = None

    @property
    def headers(self) -> dict[str, str]:
        access_token: Final = self.token_creator()
        # headers for completions and embeddings requests
        headers: Final = {
            "Authorization": access_token,
            "AI-Resource-Group": self.resource_group,
            "Content-Type": "application/json",
            "AI-Client-Type": "LiteLLM",
        }
        return headers

    def _get_with_current_token(self, client: httpx.Client, path: str) -> httpx.Response:
        request_headers: Final = self.headers
        response: Final = client.get(join_api_base(self.base_url, path), headers=request_headers)
        if response.status_code == 401:
            self.token_creator.invalidate(request_headers["Authorization"])
        return response

    def _get_from_ai_core(self, client: httpx.Client, path: str) -> httpx.Response:
        first_try: Final = self._get_with_current_token(client, path)
        response: Final = self._get_with_current_token(client, path) if first_try.status_code == 401 else first_try
        if response.status_code == 401:
            raise GenAIHubOrchestrationError(status_code=401, message=response.text)
        return response

    @cached_property
    def deployment_url(self) -> str:
        with httpx.Client(timeout=30) as client:
            valid_deployments: Final = []
            deployments: Final = require_get(
                _DECODED_JSON.validate_python(self._get_from_ai_core(client, "lm/deployments").json())
            )
            for deployment in iter_items(deployments.get("resources", [])):
                if subscript(deployment, "scenarioId") == "orchestration":
                    config_details = _DECODED_JSON.validate_python(
                        self._get_from_ai_core(
                            client, f"lm/configurations/{subscript(deployment, 'configurationId')}"
                        ).json()
                    )
                    if subscript(config_details, "executableId") == "orchestration":
                        valid_deployments.append(
                            (subscript(deployment, "deploymentUrl"), subscript(deployment, "createdAt"))
                        )
            return sorted(valid_deployments, key=lambda x: x[1], reverse=True)[0][0]

    def get_error_class(self, error_message: str, status_code: int, headers: dict[str, object] | httpx.Headers):
        if status_code == 401 and self._request_authorization is not None:
            self.token_creator.invalidate(self._request_authorization)
        return GenAIHubOrchestrationError(status_code, error_message)

    def get_supported_openai_params(self, model: str) -> list[str]:
        if "text-embedding-3" in model:
            return ["encoding_format", "dimensions"]
        else:
            return [
                "encoding_format",
            ]

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: dict[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        return optional_params

    def validate_environment(self, headers: Mapping[str, object], *args: object, **kwargs: object) -> dict[str, str]:
        return self.headers

    def sign_request(
        self,
        headers: Mapping[str, object],
        optional_params: Mapping[str, object],
        request_data: Mapping[str, object],
        api_base: str,
        api_key: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        fake_stream: bool | None = None,
    ) -> tuple[dict[str, object], bytes | None]:  # mutable-ok: BaseConfig.sign_request returns a dict
        authorization: Final = self.headers["Authorization"]
        self._request_authorization = authorization
        return {**headers, "Authorization": authorization}, None

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        url: Final = join_api_base(self.deployment_url, "v2/embeddings")
        return url

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: Mapping[str, object],
        headers: Mapping[str, object],
    ) -> dict[str, object]:
        model_dict: Final[dict[str, object]] = {}
        model_dict["name"] = model
        model_dict["version"] = optional_params.get("version", "latest")
        model_dict["params"] = optional_params.get("parameters", {})
        timeout: Final = optional_params.get("timeout", None)
        if timeout is not None:
            model_dict["timeout"] = timeout
        max_retries: Final = optional_params.get("max_retries", None)
        if max_retries is not None:
            model_dict["max_retries"] = max_retries
        input_dict: Final[dict[str, object]] = {"text": input}
        input_type: Final = optional_params.get("type")
        if input_type is not None:
            input_dict["type"] = input_type
        masking = optional_params.get("masking")
        masking = {"masking": masking} if masking is not None else {}
        body = {
            "config": {"modules": {"embeddings": {"model": model_dict}, **masking}},
            "input": input_dict,
        }
        body = validate_dict(body, EmbeddingRequest)
        return body

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None,
        request_data: Mapping[str, object],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> EmbeddingResponse:
        return EmbeddingResponse.model_validate(raw_response.json()["final_result"])

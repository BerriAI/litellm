"""
Translates from OpenAI's `/v1/embeddings` to IBM's `/text/embeddings` route.
"""

from functools import cached_property
from typing import Final, Literal

import httpx
from pydantic import BaseModel, Field

from litellm.llms.base_llm.embedding.transformation import (
    BaseEmbeddingConfig,
    LiteLLMLoggingObj,
)
from litellm.llms.sap.chat.models import MaskingModuleConfig
from litellm.types.llms.openai import AllEmbeddingInputValues
from litellm.types.utils import EmbeddingResponse

from ..chat.handler import GenAIHubOrchestrationError
from ..credentials import get_token_creator


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingItem(BaseModel):
    object: Literal["embedding"]
    embedding: list[float] = Field(..., description="Vector of floats (length varies by model).")
    index: int


class FinalResult(BaseModel):
    object: Literal["list"]
    data: list[EmbeddingItem]
    model: str
    usage: Usage


class EmbeddingsResponse(BaseModel):
    request_id: str
    final_result: FinalResult


class EmbeddingModel(BaseModel):
    name: str
    version: str = "latest"
    params: dict = Field(default_factory=dict)
    timeout: int | None = Field(default=None, ge=1, le=600)
    max_retries: int | None = Field(default=None, ge=0, le=5)


class EmbeddingsModelConfig(BaseModel):
    model: EmbeddingModel


class EmbeddingsModules(BaseModel):
    embeddings: EmbeddingsModelConfig
    masking: MaskingModuleConfig | None = None


class EmbeddingInput(BaseModel):
    text: str | list[str]
    type: Literal["text", "document", "query"] | None = None


class EmbeddingConfig(BaseModel):
    modules: EmbeddingsModules


class EmbeddingRequest(BaseModel):
    config: EmbeddingConfig
    input: EmbeddingInput


def validate_dict(data: dict, model) -> dict:
    return model(**data).model_dump(exclude_unset=True, by_alias=True)


class GenAIHubEmbeddingConfig(BaseEmbeddingConfig):
    def __init__(self):
        super().__init__()
        self._access_token_data = {}
        self.token_creator, self.base_url, self.resource_group = get_token_creator()

    @property
    def headers(self) -> dict:
        access_token: Final = self.token_creator()
        # headers for completions and embeddings requests
        headers: Final = {
            "Authorization": access_token,
            "AI-Resource-Group": self.resource_group,
            "Content-Type": "application/json",
            "AI-Client-Type": "LiteLLM",
        }
        return headers

    @cached_property
    def deployment_url(self) -> str:
        with httpx.Client(timeout=30) as client:
            valid_deployments: Final = []
            deployments: Final = client.get(self.base_url + "/lm/deployments", headers=self.headers).json()
            for deployment in deployments.get("resources", []):
                if deployment["scenarioId"] == "orchestration":
                    config_details = client.get(
                        self.base_url + f"/lm/configurations/{deployment['configurationId']}",
                        headers=self.headers,
                    ).json()
                    if config_details["executableId"] == "orchestration":
                        valid_deployments.append((deployment["deploymentUrl"], deployment["createdAt"]))
            return sorted(valid_deployments, key=lambda x: x[1], reverse=True)[0][0]

    def get_error_class(self, error_message, status_code, headers):
        return GenAIHubOrchestrationError(status_code, error_message)

    def get_supported_openai_params(self, model: str) -> list:
        if "text-embedding-3" in model:
            return ["encoding_format", "dimensions"]
        else:
            return [
                "encoding_format",
            ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return optional_params

    def validate_environment(self, headers: dict, *args, **kwargs) -> dict:
        return self.headers

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        url: Final = self.deployment_url.rstrip("/") + "/v2/embeddings"
        return url

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        model_dict: Final = {}
        model_dict["name"] = model
        model_dict["version"] = optional_params.get("version", "latest")
        model_dict["params"] = optional_params.get("parameters", {})
        timeout: Final = optional_params.get("timeout", None)
        if timeout is not None:
            model_dict["timeout"] = timeout
        max_retries: Final = optional_params.get("max_retries", None)
        if max_retries is not None:
            model_dict["max_retries"] = max_retries
        input_dict: Final = {"text": input}
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
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
    ) -> EmbeddingResponse:
        return EmbeddingResponse.model_validate(raw_response.json()["final_result"])

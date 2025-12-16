"""
Translates from OpenAI's `/v1/embeddings` to IBM's `/text/embeddings` route.
"""

from typing import Optional, List, Dict, Literal, Union
from pydantic import BaseModel, Field
from functools import cached_property

import httpx

from litellm.llms.base_llm.embedding.transformation import (
    BaseEmbeddingConfig,
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import AllEmbeddingInputValues
from litellm.types.utils import EmbeddingResponse

from ..chat.handler import GenAIHubOrchestrationError
from ..credentials import get_token_creator


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingItem(BaseModel):
    object: Literal["embedding"]
    embedding: List[float] = Field(
        ..., description="Vector of floats (length varies by model)."
    )
    index: int


class FinalResult(BaseModel):
    object: Literal["list"]
    data: List[EmbeddingItem]
    model: str
    usage: Usage


class EmbeddingsResponse(BaseModel):
    request_id: str
    final_result: FinalResult


class EmbeddingModel(BaseModel):
    name: str
    version: str = "latest"
    params: dict = Field(default_factory=dict, validation_alias="parameters")


class EmbeddingsModules(BaseModel):
    embeddings: EmbeddingModel


class EmbeddingInput(BaseModel):
    text: Union[str, List[str]]
    type: Literal["text", "document", "query"] = "text"


class EmbeddingRequest(BaseModel):
    config: EmbeddingsModules
    input: EmbeddingInput


def validate_dict(data: dict, model) -> dict:
    return model(**data).model_dump()


class GenAIHubEmbeddingConfig(BaseEmbeddingConfig):
    def __init__(self):
        super().__init__()
        self._access_token_data = {}
        self.token_creator, self.base_url, self.resource_group = get_token_creator()

    @property
    def headers(self) -> Dict:
        access_token = self.token_creator()
        # headers for completions and embeddings requests
        headers = {
            "Authorization": access_token,
            "AI-Resource-Group": self.resource_group,
            "Content-Type": "application/json",
        }
        return headers

    @cached_property
    def deployment_url(self) -> str:
        with httpx.Client(timeout=30) as client:
            valid_deployments = []
            deployments = client.get(
                self.base_url + "/lm/deployments", headers=self.headers
            ).json()
            for deployment in deployments.get("resources", []):
                if deployment["scenarioId"] == "orchestration":
                    config_details = client.get(
                        self.base_url
                        + f'/lm/configurations/{deployment["configurationId"]}',
                        headers=self.headers,
                    ).json()
                    if config_details["executableId"] == "orchestration":
                        valid_deployments.append(
                            (deployment["deploymentUrl"], deployment["createdAt"])
                        )
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
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        url = self.deployment_url.rstrip("/") + "/v2/embeddings"
        return url

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        model_dict = {}
        model_dict["name"] = model
        model_dict["version"] = optional_params.get("version", "latest")
        model_dict["params"] = optional_params.get("parameters", {})
        input_dict = {"text": input}
        body = {
            "config": {
                "modules": {
                    "embeddings": {"model": validate_dict(model_dict, EmbeddingModel)}
                }
            },
            "input": validate_dict(input_dict, EmbeddingInput),
        }
        return body

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
    ) -> EmbeddingResponse:
        return EmbeddingResponse.model_validate(raw_response.json()["final_result"])

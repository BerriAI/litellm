from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Final
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, ConfigDict, Field

from litellm.litellm_core_utils.aws_partition import get_aws_dns_suffix
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM, bedrock_bearer_token
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.llms.bedrock import AwsAuthParams
from litellm.types.router import LiteLLM_Params

MODEL_PREFIX: Final = "bedrock/"
LIST_MODELS_TIMEOUT: Final = 10.0
INFERENCE_PROFILES_PAGE_SIZE: Final = 1000

QueryPairs = tuple[tuple[str, str], ...]


class FoundationModelSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(alias="modelId")


class ListFoundationModelsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summaries: tuple[FoundationModelSummary, ...] = Field(default=(), alias="modelSummaries")


class InferenceProfileSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(alias="inferenceProfileId")
    status: str


class ListInferenceProfilesResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summaries: tuple[InferenceProfileSummary, ...] = Field(default=(), alias="inferenceProfileSummaries")
    next_token: str | None = Field(default=None, alias="nextToken")


class BedrockModelLister(BaseAWSLLM):
    def __init__(self, deployment: LiteLLM_Params, client: HTTPHandler) -> None:
        super().__init__()
        self._aws_params: Final = deployment.model_dump(exclude_none=True)
        self._aws_region_name: Final = self._get_aws_region_name(optional_params=self._aws_params)
        self._bearer_token: Final = bedrock_bearer_token(deployment.api_key)
        self._client: Final = client

    def invocable_model_ids(self) -> frozenset[str]:
        model_ids: Final = frozenset(self._active_inference_profile_ids(None)) | frozenset(
            self._on_demand_foundation_model_ids()
        )
        return frozenset(MODEL_PREFIX + model_id for model_id in model_ids)

    def _on_demand_foundation_model_ids(self) -> Iterator[str]:
        response: Final = ListFoundationModelsResponse.model_validate(
            self._get_json("/foundation-models", (("byInferenceType", "ON_DEMAND"),))
        )
        return (summary.id for summary in response.summaries)

    def _active_inference_profile_ids(self, next_token: str | None) -> Iterator[str]:
        continuation: Final[QueryPairs] = (("nextToken", next_token),) if next_token is not None else ()
        page: Final = ListInferenceProfilesResponse.model_validate(
            self._get_json(
                "/inference-profiles",
                (("maxResults", str(INFERENCE_PROFILES_PAGE_SIZE)), ("typeEquals", "SYSTEM_DEFINED"), *continuation),
            )
        )
        yield from (summary.id for summary in page.summaries if summary.status == "ACTIVE")
        if page.next_token is not None:
            yield from self._active_inference_profile_ids(page.next_token)

    def _get_json(self, path: str, query: QueryPairs) -> object:
        url: Final = f"https://bedrock.{self._aws_region_name}.{get_aws_dns_suffix(self._aws_region_name)}{path}?{urlencode(query)}"
        response: Final = self._client.get(url=url, headers=self._auth_headers(url), timeout=LIST_MODELS_TIMEOUT)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            raise Exception(
                f"Failed to list Bedrock models in {self._aws_region_name}. "
                f"Status code: {response.status_code}, Response: {response.text}"
            )
        return response.json()

    def _auth_headers(self, url: str) -> Mapping[str, str]:
        if self._bearer_token is not None:
            return MappingProxyType({"Authorization": f"Bearer {self._bearer_token}"})
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        credentials: Final = self.resolve_credentials(
            AwsAuthParams.model_validate(self._aws_params), self._aws_region_name
        )
        request: Final = AWSRequest(method="GET", url=url, data="")
        SigV4Auth(credentials, "bedrock", self._aws_region_name).add_auth(request)
        return MappingProxyType({name: str(value) for name, value in request.prepare().headers.items()})

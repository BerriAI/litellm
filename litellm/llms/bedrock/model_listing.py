from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Final, TypeAlias
from urllib.parse import quote, urlencode

import httpx
from pydantic import BaseModel, ConfigDict, Field

from litellm.litellm_core_utils.aws_partition import get_aws_dns_suffix
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM, bedrock_bearer_token
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.llms.bedrock import AwsAuthParams

LIST_MODELS_TIMEOUT: Final = 10.0
INFERENCE_PROFILES_PAGE_SIZE: Final = 1000
INFERENCE_PROFILES_PAGE_CAP: Final = 20

QueryPairs: TypeAlias = tuple[tuple[str, str], ...]


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
    def __init__(self, deployment: Mapping[str, object], client: HTTPHandler) -> None:
        super().__init__()
        self._aws_params: Final = dict(deployment)
        self._aws_region_name: Final = self._get_aws_region_name(optional_params=self._aws_params)
        api_key: Final = deployment.get("api_key")
        self._bearer_token: Final = bedrock_bearer_token(api_key if isinstance(api_key, str) else None)
        self._client: Final = client

    def invocable_model_ids(self) -> frozenset[str]:
        return frozenset(self._active_inference_profile_ids()) | frozenset(self._on_demand_foundation_model_ids())

    def _on_demand_foundation_model_ids(self) -> Iterator[str]:
        response: Final = ListFoundationModelsResponse.model_validate(
            self._get_json("/foundation-models", (("byInferenceType", "ON_DEMAND"),))
        )
        return (summary.id for summary in response.summaries)

    def _active_inference_profile_ids(self) -> tuple[str, ...]:
        collected: tuple[str, ...] = ()  # rebind-ok: accumulates one page of ids per iteration
        next_token: str | None = None  # rebind-ok: advances to each page's nextToken
        for _ in range(INFERENCE_PROFILES_PAGE_CAP):
            page = self._inference_profiles_page(next_token)
            collected += tuple(summary.id for summary in page.summaries if summary.status == "ACTIVE")
            if page.next_token is None:
                return collected
            next_token = page.next_token
        raise BedrockError(
            status_code=500,
            message=(
                f"Bedrock inference profile listing in {self._aws_region_name} did not end within "
                f"{INFERENCE_PROFILES_PAGE_CAP} pages."
            ),
        )

    def _inference_profiles_page(self, next_token: str | None) -> ListInferenceProfilesResponse:
        continuation: Final[QueryPairs] = (("nextToken", next_token),) if next_token is not None else ()
        return ListInferenceProfilesResponse.model_validate(
            self._get_json(
                "/inference-profiles",
                (("maxResults", str(INFERENCE_PROFILES_PAGE_SIZE)), ("typeEquals", "SYSTEM_DEFINED"), *continuation),
            )
        )

    def _get_json(self, path: str, query: QueryPairs) -> object:
        host: Final = f"bedrock.{self._aws_region_name}.{get_aws_dns_suffix(self._aws_region_name)}"
        url: Final = f"https://{host}{path}?{urlencode(query, quote_via=quote)}"
        response: Final = self._client.get(url=url, headers=self._auth_headers(url), timeout=LIST_MODELS_TIMEOUT)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            raise BedrockError(
                status_code=response.status_code,
                message=(
                    f"Failed to list Bedrock models in {self._aws_region_name}. "
                    f"Status code: {response.status_code}, Response: {response.text}"
                ),
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

import hashlib
import json
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias
from urllib.parse import quote, urlencode

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from litellm._logging import verbose_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM, bedrock_bearer_token
from litellm.llms.bedrock.common_utils import BedrockError, bedrock_control_plane_base_url
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.llms.bedrock import AwsAuthParams

LIST_MODELS_TIMEOUT: Final = 10.0
INFERENCE_PROFILES_PAGE_SIZE: Final = 1000
INFERENCE_PROFILES_PAGE_CAP: Final = 20
CONTROL_PLANE_ENDPOINT_RETRY_SECONDS: Final = 3600
CONTROL_PLANE_ENDPOINT_FAILURES: Final = InMemoryCache(default_ttl=CONTROL_PLANE_ENDPOINT_RETRY_SECONDS)
_DECODED_JSON: Final = TypeAdapter(object)

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


class EndpointFailureMemory(Protocol):
    def get_cache(self, key: str) -> object: ...

    def set_cache(self, key: str, value: object) -> None: ...

    def delete_cache(self, key: str) -> None: ...


class BedrockModelLister(BaseAWSLLM):
    def __init__(
        self, deployment: Mapping[str, object], client: HTTPHandler, endpoint_failures: EndpointFailureMemory
    ) -> None:
        super().__init__()
        self._aws_params: Final = dict(deployment)
        self._aws_region_name: Final = self._get_aws_region_name(optional_params=self._aws_params)
        api_key: Final = deployment.get("api_key")
        self._bearer_token: Final = bedrock_bearer_token(api_key if isinstance(api_key, str) else None)
        self._regional_base_url: Final = bedrock_control_plane_base_url(None, self._aws_region_name)
        configured_base_url: Final = bedrock_control_plane_base_url(
            deployment.get("aws_bedrock_control_plane_endpoint"), self._aws_region_name
        )
        own_endpoint: Final = None if configured_base_url == self._regional_base_url else configured_base_url
        self._sigv4_endpoint_base_url: Final = None if self._bearer_token is not None else own_endpoint
        self._endpoint_skipped_for_api_key: Final = own_endpoint if self._bearer_token is not None else None
        self._client: Final = client
        self._endpoint_failures: Final = endpoint_failures
        self._deployment_key: Final = hashlib.sha256(
            json.dumps(self._aws_params, sort_keys=True, default=str).encode()
        ).hexdigest()

    def invocable_model_ids(self) -> frozenset[str]:
        endpoint: Final = self._sigv4_endpoint_base_url
        if endpoint is None:
            return self._regional_model_ids()
        if self._endpoint_failures.get_cache(self._deployment_key) is None:
            return self._list_through(endpoint, self._regional_base_url)
        return self._list_through(self._regional_base_url, endpoint)

    def _regional_model_ids(self) -> frozenset[str]:
        skipped: Final = self._endpoint_skipped_for_api_key
        if skipped is None:
            return self._invocable_model_ids(self._regional_base_url)
        try:
            return self._invocable_model_ids(self._regional_base_url)
        except BedrockError as error:
            raise BedrockError(
                status_code=error.status_code,
                message=(
                    f"{error.message} A deployment with a Bedrock API key lists through "
                    f"{httpx.URL(self._regional_base_url).host}, never through the configured control-plane "
                    f"endpoint {httpx.URL(skipped).host}."
                ),
            ) from error

    def _list_through(self, first_base_url: str, fallback_base_url: str) -> frozenset[str]:
        try:
            listed: Final = self._invocable_model_ids(first_base_url)
        except (BedrockError, ValueError) as error:
            return self._list_through_fallback(first_base_url, fallback_base_url, error)
        if listed:
            return listed
        return self._list_through_fallback(first_base_url, fallback_base_url, "it listed no models")

    def _list_through_fallback(self, failed_base_url: str, fallback_base_url: str, reason: object) -> frozenset[str]:
        verbose_logger.warning(
            "Listing Bedrock models through %s failed, so listing through %s instead: %s",
            httpx.URL(failed_base_url).host,
            httpx.URL(fallback_base_url).host,
            reason,
        )
        listed: Final = self._invocable_model_ids(fallback_base_url)
        if failed_base_url == self._regional_base_url:
            self._endpoint_failures.delete_cache(self._deployment_key)
        else:
            self._endpoint_failures.set_cache(self._deployment_key, True)
        return listed

    def _invocable_model_ids(self, base_url: str) -> frozenset[str]:
        return frozenset(self._active_inference_profile_ids(base_url)) | frozenset(
            self._on_demand_foundation_model_ids(base_url)
        )

    def _on_demand_foundation_model_ids(self, base_url: str) -> Iterator[str]:
        response: Final = ListFoundationModelsResponse.model_validate(
            self._get_json(base_url, "/foundation-models", (("byInferenceType", "ON_DEMAND"),))
        )
        return (summary.id for summary in response.summaries)

    def _active_inference_profile_ids(self, base_url: str) -> tuple[str, ...]:
        collected: tuple[str, ...] = ()  # rebind-ok: accumulates one page of ids per iteration
        next_token: str | None = None  # rebind-ok: advances to each page's nextToken
        for _ in range(INFERENCE_PROFILES_PAGE_CAP):
            page = self._inference_profiles_page(base_url, next_token)
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

    def _inference_profiles_page(self, base_url: str, next_token: str | None) -> ListInferenceProfilesResponse:
        continuation: Final[QueryPairs] = (("nextToken", next_token),) if next_token is not None else ()
        return ListInferenceProfilesResponse.model_validate(
            self._get_json(
                base_url,
                "/inference-profiles",
                (("maxResults", str(INFERENCE_PROFILES_PAGE_SIZE)), ("typeEquals", "SYSTEM_DEFINED"), *continuation),
            )
        )

    def _get_json(self, base_url: str, path: str, query: QueryPairs) -> object:
        url: Final = f"{base_url}{path}?{urlencode(query, quote_via=quote)}"
        host: Final = httpx.URL(url).host
        try:
            response: Final = self._client.get(url=url, headers=self._auth_headers(url), timeout=LIST_MODELS_TIMEOUT)
        except httpx.TransportError as error:
            raise BedrockError(
                status_code=500,
                message=f"Failed to reach {host} to list Bedrock models in {self._aws_region_name}: {error!r}",
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            raise BedrockError(
                status_code=response.status_code,
                message=(
                    f"Failed to list Bedrock models from {host} in {self._aws_region_name}. "
                    f"Status code: {response.status_code}, Response: {response.text}"
                ),
            )
        return _DECODED_JSON.validate_python(response.json())

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

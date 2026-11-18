from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias
from urllib.parse import quote

import httpx
from fastapi import HTTPException, Request
from pydantic import TypeAdapter

from litellm.llms.opensearch.vector_stores.index_passthrough import (
    IndexPath,
    IndexRefusal,
    index_operation,
    prepare_index_request,
)
from litellm.llms.opensearch.vector_stores.transformation import OpenSearchIndexAWS, OpenSearchVectorStoreConfig
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.utils import LlmProviders
from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
from litellm.utils import ProviderConfigManager, with_named_credential
from litellm.vector_stores.fallbacks import (
    Attempt,
    Failed,
    Served,
    VectorStoreFallbackPlan,
    VectorStoreTarget,
    classify_error,
    classify_status,
    execute_with_vector_store_fallbacks,
    native_store_id,
    served_headers,
)

_AUTH: Final = TypeAdapter(Mapping[str, Mapping[str, str]])
PASSTHROUGH_FALLBACK_STATE: Final = "_litellm_vector_store_passthrough_plan"
NativeSearchOperation: TypeAlias = Literal["azure_ai", "milvus", "opensearch", "vertex_ai"]
_AUTH_HEADERS: Final = frozenset(
    {
        "authorization",
        "api-key",
        "x-api-key",
        "host",
        "content-length",
        "x-amz-date",
        "x-amz-security-token",
        "x-amz-content-sha256",
    }
)


@dataclass(frozen=True, slots=True)
class VectorStorePassthroughPlan:
    fallback_plan: VectorStoreFallbackPlan
    operation: NativeSearchOperation
    endpoint: str
    primary_resource_id: str


def _is_search(operation: NativeSearchOperation, method: str, endpoint: str) -> bool:
    path: Final = endpoint.strip("/")
    match operation:
        case "azure_ai":
            return (method == "POST" and re.fullmatch(r"indexes/[^/]+/docs/search", path) is not None) or (
                method == "GET" and re.fullmatch(r"indexes/[^/]+/docs", path) is not None
            )
        case "milvus":
            return method == "POST" and path == "v2/vectordb/entities/search"
        case "opensearch":
            return method in ("GET", "POST") and re.fullmatch(r"[^/]+/_search", path) is not None
        case "vertex_ai":
            return (
                method == "POST"
                and re.fullmatch(
                    r"v1(?:beta|alpha)?/projects/[^/]+/locations/[^/]+/collections/[^/]+/dataStores/[^/]+/servingConfigs/[^/]+:search",
                    path,
                )
                is not None
            )


async def set_passthrough_fallback_plan(
    request: Request,
    store: LiteLLM_ManagedVectorStore,
    caller: UserAPIKeyAuth,
    *,
    operation: NativeSearchOperation,
    endpoint: str,
    primary_resource_id: str,
) -> None:
    if not store.get("is_config") or not _is_search(operation, request.method, endpoint):
        return
    from litellm.proxy.vector_store_endpoints.utils import resolve_vector_store_fallback_plan

    plan: Final = await resolve_vector_store_fallback_plan(store, caller)
    if plan is None:
        return
    if any(target.provider != plan.targets[0].provider for target in plan.targets) or any(
        target.provider.split("/", 1)[0] != operation for target in plan.targets
    ):
        raise HTTPException(400, "Native vector store fallbacks require the same provider and API type")
    if operation == "vertex_ai" and any(
        target.provider != "vertex_ai/search_api" or target.params.get("vertex_engine_id") for target in plan.targets
    ):
        raise HTTPException(400, "Native Vertex fallbacks require search_api data stores")
    setattr(
        request.state,
        PASSTHROUGH_FALLBACK_STATE,
        VectorStorePassthroughPlan(plan, operation, endpoint, primary_resource_id),
    )


def _string(params: Mapping[str, object], key: str) -> str | None:
    value: Final = params.get(key)
    return value if isinstance(value, str) else None


async def _target_request(
    request: httpx.Request, plan: VectorStorePassthroughPlan, target: VectorStoreTarget
) -> httpx.Request | IndexRefusal:
    params: Final = with_named_credential(target.params)
    headers: Final = {key: value for key, value in request.headers.items() if key.lower() not in _AUTH_HEADERS}
    if plan.operation == "opensearch":
        path: Final = IndexPath(plan.primary_resource_id, ("_search",))
        prepared: Final = await prepare_index_request(
            config=OpenSearchVectorStoreConfig(aws=OpenSearchIndexAWS()),
            method=request.method,
            path=path,
            operation=index_operation(request.method, path.api),
            provider_index=plan.primary_resource_id,
            content_type=_string(request.headers, "content-type"),
            query=tuple(request.url.params.multi_items()),
            body=request.content,
            store_params=params,
            can_read=True,
        )
        if isinstance(prepared, IndexRefusal):
            return prepared
        return httpx.Request(
            request.method, prepared.url, headers=prepared.headers, content=prepared.body, extensions=request.extensions
        )
    if plan.operation == "vertex_ai":
        from litellm.llms.vertex_ai.common_utils import get_vertex_location_from_url, get_vertex_project_id_from_url

        vertex: Final = VertexBase()
        configured_project: Final = _string(params, "vertex_project")
        configured_location: Final = _string(params, "vertex_location")
        token, project = await vertex.ensure_access_token_async(
            credentials=_string(params, "vertex_credentials"),
            project_id=configured_project or get_vertex_project_id_from_url(str(request.url)),
            custom_llm_provider="vertex_ai",
        )
        location: Final = configured_location or get_vertex_location_from_url(str(request.url))
        replacements: Final = {2: project, 4: location, 8: native_store_id(target)}
        parts: Final = request.url.path.strip("/").split("/")
        path_parts: Final = tuple(
            quote(str(replacements[index]), safe="") if index in replacements else part
            for index, part in enumerate(parts)
        )
        vertex_url: Final = httpx.URL("https://discoveryengine.googleapis.com/" + "/".join(path_parts)).copy_with(
            query=request.url.query
        )
        return httpx.Request(
            request.method,
            vertex_url,
            headers={**headers, "Authorization": f"Bearer {token}"},
            content=request.content,
            extensions=request.extensions,
        )
    config: Final = ProviderConfigManager.get_provider_vector_stores_config(LlmProviders(plan.operation))
    if config is None:
        return IndexRefusal(400, "Unsupported native vector store provider")
    base: Final = config.get_complete_url(_string(params, "api_base"), dict(params))  # pyright: ignore[reportUnknownMemberType]  # provider interface still accepts an untyped parameter dict
    url: Final = httpx.URL(base.rstrip("/") + "/" + plan.endpoint.lstrip("/")).copy_with(query=request.url.query)
    credentials: Final = _AUTH.validate_python(config.get_auth_credentials(dict(params)))  # pyright: ignore[reportUnknownMemberType]  # validate the legacy provider credential boundary
    auth: Final = credentials.get("headers", {})
    return httpx.Request(
        request.method, url, headers={**headers, **auth}, content=request.content, extensions=request.extensions
    )


async def send_with_vector_store_fallbacks(
    client: httpx.AsyncClient,
    request: httpx.Request,
    plan: VectorStorePassthroughPlan,
) -> httpx.Response:
    async def attempt(target: VectorStoreTarget) -> Attempt[httpx.Response]:
        try:
            prepared: Final = await _target_request(request, plan, target)
        except Exception as error:  # noqa: BLE001  # credential preparation failures are terminal attempt values
            return Failed(target, error, False)
        if isinstance(prepared, IndexRefusal):
            return Failed(target, HTTPException(prepared.status_code, prepared.message), False)
        try:
            response: Final = await client.send(prepared, stream=True, follow_redirects=False)
            if response.is_success:
                return Served(target, response)
            try:
                await response.aread()
            finally:
                await response.aclose()
            return Failed(target, None, classify_status(response.status_code), response)
        except Exception as error:  # noqa: BLE001  # classify provider transport failures for the shared executor
            return classify_error(target, error)

    outcome: Final = await execute_with_vector_store_fallbacks(plan.fallback_plan, attempt)
    if isinstance(outcome, Served):
        outcome.value.headers.update(served_headers(outcome))
        return outcome.value
    if outcome.value is not None:
        return outcome.value
    if outcome.error is not None:
        raise outcome.error
    raise HTTPException(503, "No vector store could serve the request")

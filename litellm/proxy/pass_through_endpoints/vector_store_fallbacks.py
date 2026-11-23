from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import httpx
from fastapi import HTTPException, Request

from litellm.exceptions import ServiceUnavailableError
from litellm.llms.opensearch.vector_stores.index_passthrough import IndexRefusal
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
from litellm.vector_stores.fallbacks import (
    Attempt,
    Failed,
    Served,
    VectorStoreFallbackPlan,
    VectorStoreTarget,
    classify_error,
    classify_status,
    execute_with_vector_store_fallbacks,
    served_headers,
)

PASSTHROUGH_FALLBACK_STATE: Final = "_litellm_vector_store_passthrough_plan"
NativeSearchProvider: TypeAlias = Literal["azure_ai", "milvus", "opensearch", "vertex_ai"]
TargetRequestBuilder: TypeAlias = Callable[[VectorStoreTarget, httpx.Request], Awaitable[httpx.Request | IndexRefusal]]
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
    build_request: TargetRequestBuilder
    client_query: str


async def set_passthrough_fallback_plan(
    request: Request,
    store: LiteLLM_ManagedVectorStore,
    caller: UserAPIKeyAuth,
    *,
    provider: NativeSearchProvider,
    build_request: TargetRequestBuilder,
) -> None:
    if not store.get("is_config"):
        return
    from litellm.proxy.vector_store_endpoints.utils import resolve_vector_store_fallback_plan

    plan: Final = await resolve_vector_store_fallback_plan(store, caller)
    if plan is None:
        return
    primary_provider: Final = plan.targets[0].provider
    if primary_provider.split("/", 1)[0] != provider or any(
        target.provider != primary_provider for target in plan.targets
    ):
        raise HTTPException(400, "Native vector store fallbacks require the same provider and API type")
    if provider == "vertex_ai" and any(
        target.provider != "vertex_ai/search_api" or target.params.get("vertex_engine_id") for target in plan.targets
    ):
        raise HTTPException(400, "Native Vertex fallbacks require search_api data stores")
    setattr(
        request.state,
        PASSTHROUGH_FALLBACK_STATE,
        VectorStorePassthroughPlan(plan, build_request, request.url.query),
    )


def retargeted_request(request: httpx.Request, url: httpx.URL | str, auth_headers: Mapping[str, str]) -> httpx.Request:
    forwarded: Final = {key: value for key, value in request.headers.items() if key.lower() not in _AUTH_HEADERS}
    return httpx.Request(
        request.method,
        httpx.URL(url).copy_merge_params(request.url.params),
        headers={**forwarded, **auth_headers},
        content=request.content,
        extensions=request.extensions,
    )


async def send_with_vector_store_fallbacks(
    client: httpx.AsyncClient,
    request: httpx.Request,
    plan: VectorStorePassthroughPlan,
) -> httpx.Response:
    primary: Final = plan.fallback_plan.targets[0]
    client_request: Final = httpx.Request(
        request.method,
        request.url.copy_with(query=plan.client_query.encode() or None),
        headers=request.headers,
        content=request.content,
        extensions=request.extensions,
    )

    async def attempt(target: VectorStoreTarget) -> Attempt[httpx.Response]:
        try:
            prepared: Final = request if target is primary else await plan.build_request(target, client_request)
        except Exception as error:  # noqa: BLE001  # credential preparation failures are terminal attempt values
            return Failed(target, error, False)
        if isinstance(prepared, IndexRefusal):
            return Failed(target, HTTPException(prepared.status_code, prepared.message), False)
        try:
            response: Final = await client.send(prepared, stream=True, follow_redirects=False)
            try:
                await response.aread()
            except httpx.DecodingError as error:
                return Failed(target, error, classify_status(response.status_code))
            finally:
                await response.aclose()
            if response.is_success:
                return Served(target, response)
            return Failed(target, None, classify_status(response.status_code), response)
        except Exception as error:  # noqa: BLE001  # classify provider transport failures for the shared executor
            return classify_error(target, error)

    outcome: Final = await execute_with_vector_store_fallbacks(plan.fallback_plan, attempt)
    if isinstance(outcome, Served):
        outcome.value.headers.update(served_headers(outcome))
        return outcome.value
    if outcome.value is not None:
        return outcome.value
    if isinstance(outcome.error, ServiceUnavailableError):
        return httpx.Response(
            503,
            headers=outcome.error.response.headers,
            json={"error": {"message": outcome.error.message, "type": "service_unavailable"}},
            request=request,
        )
    if outcome.error is not None:
        raise outcome.error
    raise HTTPException(503, "No vector store could serve the request")

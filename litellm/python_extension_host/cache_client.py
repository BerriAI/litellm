# pyright: reportAny=false, reportMissingModuleSource=false
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import json
from typing import Final

import grpc

from litellm.python_extension.generated.v1 import extension_host_pb2 as pb
from litellm.python_extension.generated.v1 import extension_host_pb2_grpc as pb_grpc

from .constants import TOKEN_METADATA_KEY


class CacheClient:
    """Invocation-scoped subset of DualCache exposed to customer plugins."""

    def __init__(
        self,
        stub: pb_grpc.GatewayServicesStub,
        cache_ref: pb.CacheRef,
        token: str,
    ) -> None:
        self._stub: Final = stub
        self._cache_ref: Final = cache_ref
        self._metadata: Final = ((TOKEN_METADATA_KEY, token),)

    async def async_get_cache(
        self,
        key: str,
        local_only: bool = False,
        **_: object,  # kwargs-ok: LiteLLM callback compatibility
    ) -> object | None:
        response: Final = await self._stub.CacheGet(
            pb.CacheGetRequest(cache=self._cache_ref, key=key, local_only=local_only),
            metadata=self._metadata,
        )
        if response.operation is None or not response.operation.ok:
            return None
        return json.loads(response.value_json) if response.HasField("value_json") else None

    async def async_set_cache(
        self,
        key: str,
        value: object,
        ttl: float | None = None,
        local_only: bool = False,
        **_: object,  # kwargs-ok: LiteLLM callback compatibility
    ) -> None:
        request: Final = pb.CacheSetRequest(
            cache=self._cache_ref,
            key=key,
            value_json=_json_bytes(value),
            local_only=local_only,
        )
        if ttl is not None:
            request.ttl_seconds = ttl
        await self._stub.CacheSet(request, metadata=self._metadata)

    def get_cache(
        self,
        *_: object,
        **__: object,  # kwargs-ok: LiteLLM callback compatibility
    ) -> object:
        raise RuntimeError("CacheClient supports async_get_cache() only")

    def set_cache(
        self,
        *_: object,
        **__: object,  # kwargs-ok: LiteLLM callback compatibility
    ) -> object:
        raise RuntimeError("CacheClient supports async_set_cache() only")


def create_gateway_channel(endpoint: str | None) -> grpc.aio.Channel | None:
    if not endpoint:
        return None
    return grpc.aio.insecure_channel(grpc_target(endpoint))


def grpc_target(endpoint: str) -> str:
    return endpoint.removeprefix("http://").removeprefix("https://")


def _json_bytes(value: object) -> bytes:
    model_dump: Final = getattr(value, "model_dump", None)
    if callable(model_dump):
        value = model_dump(mode="json")  # rebind-ok: invocation-scoped RPC state
    return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str).encode()

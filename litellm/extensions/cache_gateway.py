# pyright: reportMissingModuleSource=false, reportMissingTypeArgument=false
# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false
from __future__ import annotations

import json
import secrets
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Protocol, cast  # noqa: TID251  # validated cache/protobuf boundaries below

import grpc

from litellm.python_extension.generated.v1 import extension_host_pb2 as pb
from litellm.python_extension.generated.v1 import extension_host_pb2_grpc as pb_grpc
from litellm.python_extension_host.constants import TOKEN_METADATA_KEY


class _AsyncCache(Protocol):
    async def async_get_cache(
        self,
        key: str,
        local_only: bool = False,
        **kwargs: object,  # kwargs-ok: LiteLLM callback compatibility
    ) -> object | None: ...

    async def async_set_cache(
        self,
        key: str,
        value: object,
        **kwargs: object,  # kwargs-ok: LiteLLM callback compatibility
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _CacheBinding:
    invocation_id: str
    cache: _AsyncCache


class InvocationCacheRegistry:
    def __init__(self) -> None:
        self._bindings: dict[str, _CacheBinding] = {}  # mutable-ok: LiteLLM compatibility payload

    def register(self, invocation_id: str, cache: object | None) -> pb.CacheRef | None:
        if cache is None:
            return None
        handle: Final = secrets.token_urlsafe(24)
        self._bindings[handle] = _CacheBinding(
            invocation_id,
            cast(_AsyncCache, cache),  # cast-ok: validated protobuf boundary
        )
        return pb.CacheRef(invocation_id=invocation_id, opaque_handle=handle)

    def resolve(self, cache_ref: pb.CacheRef) -> _AsyncCache | None:
        binding: Final = self._bindings.get(cache_ref.opaque_handle)
        if binding is None or binding.invocation_id != cache_ref.invocation_id:
            return None
        return binding.cache

    def revoke(self, cache_ref: pb.CacheRef | None) -> None:
        if cache_ref is not None:
            self._bindings.pop(cache_ref.opaque_handle, None)


class GatewayServices(pb_grpc.GatewayServicesServicer):
    def __init__(self, token: str, registry: InvocationCacheRegistry) -> None:
        self._token: Final = token
        self._registry: Final = registry

    async def CacheGet(self, request: pb.CacheGetRequest, context: grpc.aio.ServicerContext) -> pb.CacheGetResponse:
        await self._authenticate(context)
        cache = self._registry.resolve(request.cache)  # rebind-ok: invocation-scoped RPC state
        if cache is None:
            return pb.CacheGetResponse(operation=_error(pb.ERROR_CODE_NOT_FOUND, "cache reference is invalid"))
        try:
            value = await cache.async_get_cache(  # rebind-ok: invocation-scoped RPC state
                key=request.key, local_only=request.local_only
            )
            response: Final = pb.CacheGetResponse(operation=pb.OperationResult(ok=True))
            if value is not None:
                response.value_json = json.dumps(value, separators=(",", ":"), default=str).encode()
        except Exception as error:  # noqa: BLE001  # cache backends may raise arbitrary exceptions
            return pb.CacheGetResponse(operation=_error(pb.ERROR_CODE_EXTENSION_FAILED, str(error)))
        else:
            return response

    async def CacheSet(self, request: pb.CacheSetRequest, context: grpc.aio.ServicerContext) -> pb.OperationResult:
        await self._authenticate(context)
        cache = self._registry.resolve(request.cache)  # rebind-ok: invocation-scoped RPC state
        if cache is None:
            return _error(pb.ERROR_CODE_NOT_FOUND, "cache reference is invalid")
        try:
            kwargs: dict[str, object] = {  # mutable-ok: cache kwargs # rebind-ok: cache kwargs
                "local_only": request.local_only
            }
            if request.HasField("ttl_seconds"):
                kwargs["ttl"] = request.ttl_seconds
            value = cast(  # cast-ok: protobuf value # rebind-ok: decoded value
                object, json.loads(request.value_json)
            )
            await cache.async_set_cache(request.key, value, **kwargs)
            return pb.OperationResult(ok=True)
        except Exception as error:  # noqa: BLE001  # cache backends may raise arbitrary exceptions
            return _error(pb.ERROR_CODE_EXTENSION_FAILED, str(error))

    async def _authenticate(self, context: grpc.aio.ServicerContext) -> None:
        metadata: Final = cast(  # cast-ok: validated protobuf boundary
            Iterable[tuple[str, str]], context.invocation_metadata() or ()
        )  # cast-ok: validated protobuf boundary
        if dict(metadata).get(TOKEN_METADATA_KEY) != self._token:  # mutable-ok: LiteLLM compatibility payload
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid extension host token")


def _error(code: pb.ErrorCode, message: str) -> pb.OperationResult:
    return pb.OperationResult(ok=False, error_code=code, error_message=message)

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final
from urllib.parse import quote

from fastapi import HTTPException, Request
from pydantic import TypeAdapter
from starlette.responses import JSONResponse, Response

from litellm import NotFoundError
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.memory.continuation import MemoryContinuations
from litellm.proxy.memory.store import MemoryStore
from litellm.proxy.memory.transport import RoundExecutor, gateway_round

_OBJECT: Final = TypeAdapter(dict[str, object])


async def memory_response_operation(
    data: Mapping[str, object], request: Request, auth: UserAPIKeyAuth, route: str, execute: RoundExecutor
) -> Response | None:
    response_id: Final = data.get("response_id")
    if not isinstance(response_id, str) or not response_id.startswith("resp_litellm_memory_"):
        return None
    from litellm.proxy.memory.gateway import gateway_memory_store

    store: Final = await gateway_memory_store(auth)
    if store is None:
        raise HTTPException(status_code=404, detail="Memory response not found or expired")
    return await serve_memory_response(response_id, request, route, store, execute, auth)


async def serve_memory_response(
    response_id: str, request: Request, route: str, store: MemoryStore, execute: RoundExecutor, auth: UserAPIKeyAuth
) -> Response:
    continuations: Final = MemoryContinuations(store)
    patch: Final = await continuations.load_response(response_id)
    if patch is None or patch.response is None or not patch.upstream_ids:
        raise HTTPException(status_code=404, detail="Memory response not found or expired")
    if route == "aget_responses":
        return JSONResponse(patch.response)
    if route == "alist_input_items":
        raise HTTPException(
            status_code=501,
            detail="Input history is unavailable for gateway memory responses; retain the original client input",
        )
    await store.authorize_namespace(write=True, require_active=False)

    async def dispatch(identifier: str) -> Mapping[str, object]:
        path: Final = "/v1/responses/" + identifier
        raw_path: Final = ("/v1/responses/" + quote(identifier, safe="")).encode()
        inner: Final = Request(
            {  # mutable-ok: Native ASGI or JSON payload.
                **request.scope,
                "path": path,
                "raw_path": raw_path,
            }
        )
        try:
            async with gateway_round(
                execute,
                inner,
                MappingProxyType({"response_id": identifier}),
                auth.model_copy(update=MappingProxyType({"budget_reservation": None})),
            ) as call:
                start: Final = await call.started
                if start.status >= 400:
                    if start.status == 404:
                        return {  # mutable-ok: Native ASGI or JSON payload.
                        }
                    raise HTTPException(status_code=start.status, detail="The upstream response operation failed")
                content: Final = await call.read()
                return _OBJECT.validate_json(content)
        except (HTTPException, NotFoundError) as exc:
            if exc.status_code != 404:
                raise
            return MappingProxyType({})

    for identifier in patch.upstream_ids:
        await dispatch(identifier)
    await continuations.delete_response(response_id)
    return JSONResponse(
        {  # mutable-ok: Native provider JSON containers.
            "id": response_id,
            "object": "response.deleted",
            "deleted": True,
        }
    )

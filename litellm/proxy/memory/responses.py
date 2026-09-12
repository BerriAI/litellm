from collections.abc import Mapping
from typing import Final
from urllib.parse import quote

from fastapi import HTTPException, Request
from pydantic import TypeAdapter
from starlette.responses import JSONResponse, Response

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.memory.continuation import MemoryContinuations
from litellm.proxy.memory.transport import gateway_round

_OBJECT: Final = TypeAdapter(dict[str, object])


async def memory_response_operation(
    data: Mapping[str, object], request: Request, auth: UserAPIKeyAuth, route: str
) -> Response | None:
    response_id: Final = data.get("response_id")
    if not isinstance(response_id, str) or not response_id.startswith("resp_litellm_memory_"):
        return None
    from litellm.proxy.memory.gateway import gateway_memory_store
    from litellm.proxy.proxy_server import app

    store: Final = await gateway_memory_store(auth)
    if store is None:
        raise HTTPException(status_code=404, detail="Memory response not found or expired")
    continuations: Final = MemoryContinuations(store, "aresponses")
    patch: Final = await continuations.load_response(response_id)
    if patch is None or patch.response is None or not patch.upstream_ids:
        raise HTTPException(status_code=404, detail="Memory response not found or expired")
    if route == "aget_responses":
        return JSONResponse(patch.response)
    if route == "adelete_responses":
        await store.authorize_namespace(write=True)
    ids: Final = patch.upstream_ids if route == "adelete_responses" else patch.upstream_ids[-1:]

    async def dispatch(identifier: str) -> Mapping[str, object]:
        suffix: Final = "/input_items" if route == "alist_input_items" else ""
        path: Final = "/v1/responses/" + identifier + suffix
        raw_path: Final = ("/v1/responses/" + quote(identifier, safe="") + suffix).encode()
        inner: Final = Request(
            {  # mutable-ok: Native ASGI or JSON payload.
                **request.scope,
                "path": path,
                "raw_path": raw_path,
            }
        )
        async with gateway_round(
            app,
            inner,
            {  # mutable-ok: Native ASGI or JSON payload.
            },
        ) as call:
            start: Final = await call.started
            if start.status >= 400:
                if route == "adelete_responses" and start.status == 404:
                    return {  # mutable-ok: Native ASGI or JSON payload.
                    }
                raise HTTPException(status_code=start.status, detail="The upstream response operation failed")
            content: Final = await call.read()
            return _OBJECT.validate_json(content)

    results: Final = tuple([await dispatch(identifier) for identifier in ids])
    if route == "alist_input_items":
        return JSONResponse(results[-1])
    await continuations.delete_response(response_id, patch)
    return JSONResponse(
        {  # mutable-ok: Native provider JSON containers.
            "id": response_id,
            "object": "response.deleted",
            "deleted": True,
        }
    )

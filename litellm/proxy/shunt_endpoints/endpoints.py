"""
`/v1/bulk_read` and `/v1/code_write`: the worker endpoints shunt's generated Bash commands call.

Each request names the auto-router marker it belongs to (`router=<model_alias>`), so the
worker model is whatever that marker's `auto_router_shunt_bulk_read_model` /
`auto_router_shunt_code_write_model` configures, not one the caller chooses. Auth is the
short-lived capability token minted at rewrite time (`shunt_capability_token.py`), not a
normal virtual key: only a shunt-generated command should ever call these routes.
"""

from collections.abc import Sequence
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Request, UploadFile

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.env_utils import get_env_int
from litellm.proxy._types import LitellmUserRoles, ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.guardrails.auto_router_shunt import ShuntConfig, shunt_config_for_model
from litellm.proxy.guardrails.shunt_capability_token import open_shunt_capability_token
from litellm.proxy.route_llm_request import route_request
from litellm.proxy.shunt_endpoints.worker import (
    BULK_READ_SYSTEM_PROMPT,
    CODE_WRITE_SYSTEM_PROMPT,
    WORKER_TEMPERATURE,
    build_bulk_read_message,
    build_code_write_message,
    strip_code_fences,
)
from litellm.types.llms.openai import ChatCompletionSystemMessage, ChatCompletionUserMessage
from litellm.types.utils import Choices, ModelResponse

if TYPE_CHECKING:
    from litellm.router import Router

router: Final = APIRouter()

# Both endpoints are registered twice, at the `/v1`-prefixed path the generated commands call
# and at the bare path, matching how the rest of the proxy exposes its native routes.
_SHUNT_TAGS: Final[list[str | Enum]] = ["shunt"]  # mutable-ok: FastAPI's `tags=` takes an invariant list

_BEARER_PREFIX: Final = "Bearer "


async def _caller_from_capability_token(authorization: Annotated[str | None, Header()] = None) -> UserAPIKeyAuth:
    """Resolve the request's caller from its shunt capability token, or reject the request.

    Never falls through to the proxy's own key/DB lookup: a request that reaches these routes
    without a valid token is rejected outright, since a shunt-generated command is the only
    thing that should ever call them.
    """
    if authorization is None or not authorization.startswith(_BEARER_PREFIX):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")
    grant: Final = open_shunt_capability_token(authorization[len(_BEARER_PREFIX) :])
    if grant is None:
        raise HTTPException(status_code=401, detail="Invalid or expired shunt capability token")

    if grant.master_key is not None:
        from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS
        from litellm.proxy.proxy_server import master_key

        if master_key is None or grant.master_key != master_key:
            raise HTTPException(status_code=401, detail="Invalid or expired shunt capability token")
        # The alias substitutes for the real master key here for the same reason normal
        # master-key auth substitutes it (user_api_key_auth.py): neither the key nor a
        # reversible derivation of it should reach spend logs, Prometheus labels, or any raw-
        # metadata logging callback the worker call's own metadata is later forwarded to.
        return UserAPIKeyAuth(api_key=LITELLM_PROXY_MASTER_KEY_ALIAS, user_role=LitellmUserRoles.PROXY_ADMIN)

    if grant.key_hash is None:
        raise HTTPException(status_code=401, detail="Invalid or expired shunt capability token")

    from litellm.proxy.auth.auth_checks import get_key_object
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    try:
        return await get_key_object(
            hashed_token=grant.key_hash,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired shunt capability token") from e


_AUTH_DEPENDENCIES: Final = [
    Depends(_caller_from_capability_token)
]  # mutable-ok: FastAPI's `dependencies=` takes a list


def _worker_config(
    model_alias: str, user_api_key_dict: UserAPIKeyAuth, request_tags: Sequence[str]
) -> tuple["Router", ShuntConfig]:
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        raise ProxyException(
            message="LLM Router not found", type=ProxyErrorTypes.internal_server_error, param=None, code=500
        )
    # `request_tags` comes from the query string the rewrite generated, so a marker armed only
    # under a tag resolves here the same way it did when the original request was rewritten.
    config: Final = shunt_config_for_model(
        llm_router=llm_router,
        model_alias=model_alias,
        team_id=user_api_key_dict.team_id,
        request_tags=request_tags,
    )
    if config is None:
        raise ProxyException(
            message=f"'{model_alias}' is not a shunt-armed auto router",
            type=ProxyErrorTypes.bad_request_error,
            param="router",
            code=400,
        )
    return llm_router, config


async def _worker_text(
    request: Request,
    llm_router: "Router",
    *,
    model: str,
    system_prompt: str,
    message: str,
    user_api_key_dict: UserAPIKeyAuth,
    label: str,
) -> str:
    """The worker model's reply text, or a 502 if it produced none.

    Goes through the same `common_processing_pre_call_logic` + `route_request` pipeline
    `/chat/completions` and every other LLM-calling route uses, rather than calling
    `llm_router.acompletion` directly: that pipeline is what actually applies model-level
    guardrails, budget/rate-limit enforcement, and fallbacks to the model being called, and a
    hand-rolled call here would have to re-derive each of those separately and correctly.
    Skips only the HTTP response/streaming shaping half of that pipeline, since a worker call
    is never itself an HTTP response and is never streamed.
    """
    from litellm.proxy.proxy_server import general_settings, proxy_config, proxy_logging_obj

    system: Final = ChatCompletionSystemMessage(role="system", content=system_prompt)
    user: Final = ChatCompletionUserMessage(role="user", content=message)
    processor: Final = ProxyBaseLLMRequestProcessing(
        data={"model": model, "messages": [system, user], "temperature": WORKER_TEMPERATURE, "stream": False}
    )
    try:
        data, _logging_obj = await processor.common_processing_pre_call_logic(  # pyright: ignore[reportUnknownVariableType]  # common_processing_pre_call_logic's own dict/Logging return is unrefined at this call shape
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            proxy_config=proxy_config,
            route_type="acompletion",
            llm_router=llm_router,
        )
        response: Final = await route_request(  # pyright: ignore[reportUnknownVariableType]  # route_request's own return type is intentionally an untyped union (see its ANN202 suppression)
            data=data,
            route_type="acompletion",
            llm_router=llm_router,
            user_model=None,
            user_api_key_dict=user_api_key_dict,
        )
        processed: Final = await proxy_logging_obj.post_call_success_hook(
            data=data,
            user_api_key_dict=user_api_key_dict,
            response=response,  # pyright: ignore[reportArgumentType]  # response is the same real ModelResponse a router acompletion call returns; the hook's own signature just can't narrow it here
        )
    except Exception as e:  # noqa: BLE001  # _handle_llm_api_exception must see every failure mode a real request can hit, same as proxy_server.py's own catch-all here
        raise await processor._handle_llm_api_exception(  # pyright: ignore[reportPrivateUsage]  # same cross-module call proxy_server.py's own /chat/completions and /embeddings routes already make
            e=e, user_api_key_dict=user_api_key_dict, proxy_logging_obj=proxy_logging_obj
        )
    if not isinstance(processed, ModelResponse):
        raise HTTPException(status_code=502, detail=f"shunt {label}: worker model returned no completion")
    choice: Final = processed.choices[0] if processed.choices else None
    text: Final = choice.message.content if isinstance(choice, Choices) else None
    if not isinstance(text, str):
        raise HTTPException(status_code=502, detail=f"shunt {label}: worker model returned no text")
    return text


# A global request-size limit exists (RequestSizeLimitMiddleware) but is opt-in and
# premium-gated, so these endpoints cannot rely on it: they accept arbitrary caller-supplied
# multipart uploads specifically to hand their contents to a worker model, an authenticated
# caller with a valid capability token could otherwise upload enough data to exhaust a proxy
# worker's memory before the size limit ever runs.
_MAX_UPLOAD_BYTES_PER_FILE: Final = get_env_int("LITELLM_SHUNT_MAX_UPLOAD_BYTES_PER_FILE", 1024 * 1024)
_MAX_UPLOAD_BYTES_TOTAL: Final = get_env_int("LITELLM_SHUNT_MAX_UPLOAD_BYTES_TOTAL", 8 * 1024 * 1024)
_MAX_UPLOAD_FILE_COUNT: Final = get_env_int("LITELLM_SHUNT_MAX_UPLOAD_FILE_COUNT", 20)


async def _read_upload_text(upload: UploadFile, *, remaining_total_bytes: int) -> str:
    """`upload`'s content as UTF-8 text, reading at most the smaller of the per-file and
    remaining-total byte budgets -- never the whole file, so a caller can't force this endpoint
    to buffer more than that regardless of how large the real upload is.

    `remaining_total_bytes <= 0` is checked explicitly rather than left to `min()` +
    `.read(limit + 1)`: a non-positive `remaining_total_bytes` would make `limit` zero or
    negative, and `UploadFile.read` treats a negative size as "read the whole file", which
    would silently defeat this budget for any caller of this function that ever passes one.
    `_read_upload_texts` below never actually produces a negative value (each read is already
    bounded by what was left when it started), so this is the function's own contract holding
    regardless of caller, not a path reachable through that call site today.
    """
    if remaining_total_bytes <= 0:
        raise ProxyException(
            message=f"uploads exceed the {_MAX_UPLOAD_BYTES_TOTAL}-byte total limit for this call",
            type=ProxyErrorTypes.bad_request_error,
            param="paths",
            code=400,
        )
    limit: Final = min(_MAX_UPLOAD_BYTES_PER_FILE, remaining_total_bytes)
    content: Final = await upload.read(limit + 1)
    if len(content) > limit:
        raise ProxyException(
            message=f"'{upload.filename}' exceeds the {limit}-byte upload limit for this call",
            type=ProxyErrorTypes.bad_request_error,
            param="paths" if upload.filename else "file",
            code=400,
        )
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ProxyException(
            message=f"'{upload.filename}' is not valid UTF-8 text",
            type=ProxyErrorTypes.bad_request_error,
            param="paths" if upload.filename else "file",
            code=400,
        ) from e


async def _read_upload_texts(uploads: Sequence[UploadFile]) -> tuple[str, ...]:
    """Every upload's text, in order, enforcing the aggregate byte and file-count budgets
    across the whole request rather than per file.

    A plain accumulator, not a comprehension: each read's byte budget is whatever the
    aggregate limit has left after every earlier file in this same request, so the reads are
    inherently sequential and the running total has to be rebound as they complete.
    """
    if len(uploads) > _MAX_UPLOAD_FILE_COUNT:
        raise ProxyException(
            message=f"at most {_MAX_UPLOAD_FILE_COUNT} files are allowed per call",
            type=ProxyErrorTypes.bad_request_error,
            param="paths",
            code=400,
        )
    texts: tuple[str, ...] = ()  # rebind-ok: sequential running total, see docstring
    remaining_total_bytes: int = _MAX_UPLOAD_BYTES_TOTAL  # rebind-ok: same
    for upload in uploads:
        text = await _read_upload_text(upload, remaining_total_bytes=remaining_total_bytes)
        texts = (*texts, text)
        remaining_total_bytes -= len(text.encode("utf-8"))
    return texts


@router.post(
    "/v1/bulk_read",
    dependencies=_AUTH_DEPENDENCIES,
    tags=_SHUNT_TAGS,
)
@router.post(
    "/bulk_read",
    dependencies=_AUTH_DEPENDENCIES,
    tags=_SHUNT_TAGS,
)
async def bulk_read(
    request: Request,
    router_name: Annotated[str, Query(alias="router")],
    question: Annotated[str, Form()],
    paths: Annotated[list[UploadFile], File()],  # mutable-ok: FastAPI requires a list for a repeated file field
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(_caller_from_capability_token)],
    tags: Annotated[list[str] | None, Query()] = None,  # mutable-ok: FastAPI requires a list for a repeated query param
) -> str:
    """Summarize or answer a question about one or more files via a cheap worker model.

    The paths shunt's generated command uploads are read here as plain UTF-8 text and never
    written to disk; only the text and the question reach the worker model.
    """
    llm_router, config = _worker_config(router_name, user_api_key_dict, tags or ())

    texts: Final = await _read_upload_texts(paths)
    files: Final = MappingProxyType(
        {upload.filename or "unnamed": text for upload, text in zip(paths, texts, strict=True)}
    )
    message: Final = build_bulk_read_message(question=question, files=files)

    verbose_proxy_logger.debug("shunt bulk_read: %s file(s) via %s", len(files), config.bulk_read_model)
    return await _worker_text(
        request,
        llm_router,
        model=config.bulk_read_model,
        system_prompt=BULK_READ_SYSTEM_PROMPT,
        message=message,
        user_api_key_dict=user_api_key_dict,
        label="bulk_read",
    )


@router.post(
    "/v1/code_write",
    dependencies=_AUTH_DEPENDENCIES,
    tags=_SHUNT_TAGS,
)
@router.post(
    "/code_write",
    dependencies=_AUTH_DEPENDENCIES,
    tags=_SHUNT_TAGS,
)
async def code_write(
    request: Request,
    router_name: Annotated[str, Query(alias="router")],
    spec: Annotated[str, Form()],
    reference: Annotated[UploadFile, File()],
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(_caller_from_capability_token)],
    tags: Annotated[list[str] | None, Query()] = None,  # mutable-ok: FastAPI requires a list for a repeated query param
) -> str:
    """Generate boilerplate code matching a reference file's patterns, via a cheap worker model.

    Returns the generated code as plain text. shunt's own script writes straight to disk since
    it runs on the file's own machine; this has no local filesystem, so the client writes the
    returned text itself (the `Bash` rewrite this backs redirects the curl output to `target`).
    """
    llm_router, config = _worker_config(router_name, user_api_key_dict, tags or ())

    reference_content: Final = await _read_upload_text(reference, remaining_total_bytes=_MAX_UPLOAD_BYTES_TOTAL)
    message: Final = build_code_write_message(
        spec=spec, reference_path=reference.filename or "", reference_content=reference_content
    )

    verbose_proxy_logger.debug("shunt code_write: reference=%s via %s", reference.filename, config.code_write_model)
    return strip_code_fences(
        await _worker_text(
            request,
            llm_router,
            model=config.code_write_model,
            system_prompt=CODE_WRITE_SYSTEM_PROMPT,
            message=message,
            user_api_key_dict=user_api_key_dict,
            label="code_write",
        )
    )

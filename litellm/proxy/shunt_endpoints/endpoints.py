"""
`/v1/bulk_read` and `/v1/code_write`: the worker endpoints shunt's generated Bash commands
call. See `auto_router_shunt.py`'s module docstring for the source this ports.

Each request names the auto-router marker it belongs to (`router=<model_alias>`, the same
value the client originally called), so the worker model is the one configured on that
marker's `auto_router_shunt_bulk_read_model` / `auto_router_shunt_code_write_model`, not a
model chosen by the caller. The call goes through `llm_router.acompletion`, not
`litellm.acompletion` directly, so worker-model spend is tracked and budgeted against the
caller's key/team exactly like any other request.
"""

from collections.abc import Sequence
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.guardrails.auto_router_shunt import ShuntConfig, shunt_config_for_model
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.proxy.shunt_endpoints.worker import (
    BULK_READ_SYSTEM_PROMPT,
    CODE_WRITE_SYSTEM_PROMPT,
    WORKER_TEMPERATURE,
    build_bulk_read_message,
    build_code_write_message,
    strip_code_fences,
)
from litellm.types.llms.openai import ChatCompletionSystemMessage, ChatCompletionUserMessage

if TYPE_CHECKING:
    from litellm.router import Router

router: Final = APIRouter()

# Both endpoints are registered twice, at the `/v1`-prefixed path the generated commands call
# and at the bare path, matching how the rest of the proxy exposes its native routes.
_AUTH_DEPENDENCIES: Final = [Depends(user_api_key_auth)]  # mutable-ok: FastAPI's `dependencies=` takes a list
_SHUNT_TAGS: Final[list[str | Enum]] = ["shunt"]  # mutable-ok: FastAPI's `tags=` takes an invariant list


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
    llm_router: "Router",
    *,
    model: str,
    system_prompt: str,
    message: str,
    user_api_key_dict: UserAPIKeyAuth,
    label: str,
) -> str:
    """The worker model's reply text, or a 502 if it produced none.

    One call site for both endpoints: they differ only in model, system prompt, and message, so
    the `acompletion` shape (temperature, non-streaming, caller attribution) lives here once.

    Attribution reuses the proxy's own key-metadata builder rather than hand-picking a couple of
    fields, so the worker call is billed and budgeted against the calling key, user, team, and
    org exactly like a normal request instead of only carrying a team id.
    """
    system: Final = ChatCompletionSystemMessage(role="system", content=system_prompt)
    user: Final = ChatCompletionUserMessage(role="user", content=message)
    key_metadata: Final = LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(
        user_api_key_dict=user_api_key_dict
    )
    response: Final = await llm_router.acompletion(
        model=model,
        messages=[system, user],  # mutable-ok: acompletion's own signature takes a concrete list
        temperature=WORKER_TEMPERATURE,
        stream=False,
        metadata={  # mutable-ok: same
            **key_metadata,
            "user_api_key": user_api_key_dict.api_key,
        },
    )
    text: Final = response.choices[0].message.content
    if not isinstance(text, str):
        raise HTTPException(status_code=502, detail=f"shunt {label}: worker model returned no text")
    return text


async def _read_upload_text(upload: UploadFile) -> str:
    content: Final = await upload.read()
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ProxyException(
            message=f"'{upload.filename}' is not valid UTF-8 text",
            type=ProxyErrorTypes.bad_request_error,
            param="paths" if upload.filename else "file",
            code=400,
        ) from e


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
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    tags: Annotated[list[str] | None, Query()] = None,
) -> str:
    """Summarize or answer a question about one or more files via a cheap worker model.

    The paths shunt's generated command uploads are read here as plain UTF-8 text and never
    written to disk; only the text and the question reach the worker model.
    """
    llm_router, config = _worker_config(router_name, user_api_key_dict, tags or ())

    files: Final = MappingProxyType({upload.filename or "unnamed": await _read_upload_text(upload) for upload in paths})
    message: Final = build_bulk_read_message(question=question, files=files)

    verbose_proxy_logger.debug("shunt bulk_read: %s file(s) via %s", len(files), config.bulk_read_model)
    return await _worker_text(
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
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    tags: Annotated[list[str] | None, Query()] = None,
) -> str:
    """Generate boilerplate code matching a reference file's patterns, via a cheap worker model.

    Returns the generated code as plain text. shunt's own script writes straight to disk since
    it runs on the file's own machine; this has no local filesystem, so the client writes the
    returned text itself (the `Bash` rewrite this backs redirects the curl output to `target`).
    """
    llm_router, config = _worker_config(router_name, user_api_key_dict, tags or ())

    reference_content: Final = await _read_upload_text(reference)
    message: Final = build_code_write_message(
        spec=spec, reference_path=reference.filename or "", reference_content=reference_content
    )

    verbose_proxy_logger.debug("shunt code_write: reference=%s via %s", reference.filename, config.code_write_model)
    return strip_code_fences(
        await _worker_text(
            llm_router,
            model=config.code_write_model,
            system_prompt=CODE_WRITE_SYSTEM_PROMPT,
            message=message,
            user_api_key_dict=user_api_key_dict,
            label="code_write",
        )
    )

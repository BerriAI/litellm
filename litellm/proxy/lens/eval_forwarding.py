import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Annotated, Final, TypeAlias
from urllib.parse import quote

import httpx
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.lens.dataset_endpoints import dataset_store, eval_cases
from litellm.proxy.lens.dataset_repository import DatasetStore
from litellm.proxy.lens.endpoints import Auth
from litellm.tracing.remote import LensConnection, bounded_response

CONTRACT_HEADER: Final = "x-lens-contract"
IDENTITY_LIFETIME_SECONDS: Final = 60
MAX_EVAL_RESPONSE_BYTES: Final = 16 * 1024 * 1024
_FORWARDED_HEADERS: Final = frozenset({"content-type", "idempotency-key", CONTRACT_HEADER})
_RETURNED_HEADERS: Final = frozenset({"content-type", "retry-after"})
_TIMEOUT: Final = httpx.Timeout(45, connect=3)

router: Final = APIRouter(tags=["Lens"])


def _epoch_seconds() -> int:
    return int(time.time())


@dataclass(frozen=True, slots=True, repr=False)
class EvalUpstream:
    url: str
    secret: str
    client: httpx.AsyncClient
    now: Callable[[], int] = field(default=_epoch_seconds)


def upstream_from(environ: Mapping[str, str]) -> EvalUpstream | None:
    try:
        connection: Final = LensConnection.from_env(environ)
    except ValueError:
        return None
    secret: Final = environ.get("LITELLM_LENS_GATEWAY_SECRET", "")
    if len(secret) < 32:
        return None
    return EvalUpstream(url=connection.url, secret=secret, client=connection.control_client())


def eval_upstream() -> EvalUpstream | None:
    return upstream_from(os.environ)


def dataset_store_factory() -> Callable[[], DatasetStore]:
    return dataset_store


Upstream: TypeAlias = Annotated[EvalUpstream | None, Depends(eval_upstream)]
StoreFactory: TypeAlias = Annotated[Callable[[], DatasetStore], Depends(dataset_store_factory)]


def gateway_identity(auth: UserAPIKeyAuth, secret: str, now: int) -> str | None:
    subject: Final = auth.user_id or auth.token
    if not subject:
        return None
    return jwt.encode(
        {
            "iss": "litellm",
            "aud": "litellm-lens",
            "sub": subject,
            "iat": now,
            "exp": now + IDENTITY_LIFETIME_SECONDS,
            "identity": {
                "user_role": auth.user_role.value if auth.user_role else "internal_user",
                "user_id": auth.user_id,
                "team_id": auth.team_id,
                "org_id": auth.org_id,
                "token": auth.token,
                "log_team_ids": [auth.team_id] if auth.team_id else [],
            },
        },
        secret,
        algorithm="HS256",
    )


async def forward(request: Request, path: str, auth: UserAPIKeyAuth, upstream: EvalUpstream | None) -> Response:
    if upstream is None:
        raise HTTPException(
            501, "Lens evals are not enabled. Set LITELLM_LENS_URL and LITELLM_LENS_GATEWAY_SECRET to match Lens"
        )
    token: Final = gateway_identity(auth, upstream.secret, upstream.now())
    if token is None:
        raise HTTPException(401, "Lens evals need a key tied to a user or a hashed token")
    headers: Final = {
        **{name: value for name, value in request.headers.items() if name.lower() in _FORWARDED_HEADERS},
        "authorization": f"Bearer {token}",
    }
    query: Final = f"?{request.url.query}" if request.url.query else ""
    try:
        async with upstream.client.stream(
            request.method,
            f"{upstream.url}{path}{query}",
            headers=headers,
            content=await request.body() or None,
            timeout=_TIMEOUT,
        ) as response:
            body: Final = await bounded_response(response, MAX_EVAL_RESPONSE_BYTES)
            return Response(
                body,
                status_code=response.status_code,
                headers={name: value for name, value in response.headers.items() if name.lower() in _RETURNED_HEADERS},
            )
    except (httpx.HTTPError, RuntimeError) as error:
        raise HTTPException(502, "Lens service is unavailable") from error


@router.get("/lens/evals", include_in_schema=False)
async def list_evals(request: Request, auth: Auth, upstream: Upstream) -> Response:
    return await forward(request, "/lens/evals", auth, upstream)


@router.api_route("/lens/evals/{path:path}", methods=["GET", "POST", "PUT"], include_in_schema=False)
async def forward_evals(path: str, request: Request, auth: Auth, upstream: Upstream) -> Response:
    return await forward(request, f"/lens/evals/{quote(path, safe='/')}", auth, upstream)


@router.get("/lens/datasets/resolve", include_in_schema=False)
async def resolve_dataset(request: Request, auth: Auth, upstream: Upstream) -> Response:
    return await forward(request, "/lens/datasets/resolve", auth, upstream)


@router.get("/lens/datasets/{dataset_id}/revisions/{revision}/cases", include_in_schema=False)
async def contract_cases(
    dataset_id: str, revision: int, request: Request, auth: Auth, upstream: Upstream, store: StoreFactory
) -> Response:
    if CONTRACT_HEADER in request.headers:
        return await forward(
            request, f"/lens/datasets/{quote(dataset_id, safe='')}/revisions/{revision}/cases", auth, upstream
        )
    cases: Final = await eval_cases(dataset_id, revision, auth, store())
    return JSONResponse(cases.model_dump(mode="json"))

from collections.abc import Callable
from typing import Final

import httpx
import jwt
import pytest
from fastapi import FastAPI
from jwt.types import Options

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.lens.dataset_endpoints import router as dataset_router
from litellm.proxy.lens.dataset_endpoints import save_revision
from litellm.proxy.lens.eval_forwarding import (
    EvalUpstream,
    dataset_store_factory,
    eval_upstream,
    gateway_identity,
    upstream_from,
)
from litellm.proxy.lens.eval_forwarding import router as eval_router
from litellm.proxy.lens.models import RevisionSave
from tests.unit.proxy.lens.test_dataset_endpoints import ADMIN, MemoryStore, case, create_dataset_named

SECRET: Final = "s" * 32
NOW: Final = 1_800_000_000
FIXED_CLOCK: Final[Options] = {"verify_exp": False, "verify_iat": False}
CI_KEY: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.TEAM, token="hashed-ci-key", team_id="team-a", models=["gpt"])

Handler = Callable[[httpx.Request], httpx.Response]


def app_with(auth: UserAPIKeyAuth, upstream: EvalUpstream | None, store: MemoryStore | None = None) -> FastAPI:
    app: Final = FastAPI()
    app.include_router(eval_router)
    app.include_router(dataset_router)
    app.dependency_overrides[user_api_key_auth] = lambda: auth
    app.dependency_overrides[eval_upstream] = lambda: upstream
    app.dependency_overrides[dataset_store_factory] = lambda: lambda: store
    return app


def lens(handler: Handler) -> EvalUpstream:
    return EvalUpstream(
        url="http://lens:4318",
        secret=SECRET,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        now=lambda: NOW,
    )


def client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway")


def claims(request: httpx.Request) -> dict[str, object]:
    token: Final = request.headers["authorization"].removeprefix("Bearer ")
    return jwt.decode(
        token, SECRET, algorithms=["HS256"], audience="litellm-lens", issuer="litellm", options=FIXED_CLOCK
    )


@pytest.mark.asyncio
async def test_eval_writes_reach_lens_with_a_signed_identity_for_the_calling_key_and_no_caller_credentials() -> None:
    seen: Final[list[httpx.Request]] = []  # mutable-ok: captures what the fake Lens received

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201, json={"id": "run-1"}, headers={"retry-after": "2", "x-internal": "1"})

    async with client(app_with(CI_KEY, lens(handler))) as gateway:
        response: Final = await gateway.post(
            "/lens/evals/runs?wait=30",
            content=b'{"eval":"e"}',
            headers={
                "authorization": "Bearer sk-ci",
                "cookie": "session=1",
                "x-lens-contract": "1",
                "idempotency-key": "k1",
                "content-type": "application/json",
            },
        )

    assert (response.status_code, response.json(), response.headers.get("retry-after")) == (201, {"id": "run-1"}, "2")
    assert "x-internal" not in response.headers
    (sent,) = seen
    assert (sent.method, str(sent.url), sent.content) == (
        "POST",
        "http://lens:4318/lens/evals/runs?wait=30",
        b'{"eval":"e"}',
    )
    assert (sent.headers["x-lens-contract"], sent.headers["idempotency-key"]) == ("1", "k1")
    assert "cookie" not in sent.headers
    assert claims(sent) == {
        "iss": "litellm",
        "aud": "litellm-lens",
        "sub": "hashed-ci-key",
        "iat": NOW,
        "exp": NOW + 60,
        "identity": {
            "user_role": "team",
            "user_id": None,
            "team_id": "team-a",
            "org_id": None,
            "token": "hashed-ci-key",
            "log_team_ids": ["team-a"],
        },
    }


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("PUT", "/lens/evals/runs/run-1/results/case-1/0"),
        ("GET", "/lens/evals/runs/run-1/cases/case-1"),
        ("GET", "/lens/datasets/resolve?name=regressions&revision=7"),
    ],
)
@pytest.mark.asyncio
async def test_every_eval_route_is_forwarded_to_the_same_path_on_lens(method: str, path: str) -> None:
    seen: Final[list[str]] = []  # mutable-ok: captures what the fake Lens received

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.raw_path.decode()}")
        return httpx.Response(200, json={})

    async with client(app_with(CI_KEY, lens(handler))) as gateway:
        response: Final = await gateway.request(method, path)

    assert (response.status_code, seen) == (200, [f"{method} {path}"])


@pytest.mark.asyncio
async def test_dataset_cases_go_to_lens_for_the_eval_contract_and_stay_on_the_proxy_otherwise() -> None:
    store: Final = MemoryStore()
    created: Final = await create_dataset_named(store)
    await save_revision(created.id, RevisionSave(base_revision=0, cases=(case("local"),)), ADMIN, store)
    seen: Final[list[str]] = []  # mutable-ok: captures what the fake Lens received

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"from": "lens"})

    async with client(app_with(ADMIN, lens(handler), store)) as gateway:
        contract: Final = await gateway.get(
            f"/lens/datasets/{created.id}/revisions/1/cases", headers={"x-lens-contract": "1"}
        )
        local: Final = await gateway.get(f"/lens/datasets/{created.id}/revisions/1/cases")

    assert contract.json() == {"from": "lens"}
    assert seen == [f"/lens/datasets/{created.id}/revisions/1/cases"]
    assert local.json()["cases"][0]["messages"][0]["content"] == "local"


@pytest.mark.asyncio
async def test_lens_errors_pass_through_with_their_status_and_body() -> None:
    body: Final = {"error": {"code": "gate_failed"}}

    async with client(app_with(CI_KEY, lens(lambda _: httpx.Response(409, json=body)))) as gateway:
        response: Final = await gateway.post("/lens/evals/runs/run-1/finish")

    assert (response.status_code, response.json()) == (409, body)


@pytest.mark.asyncio
async def test_an_unreachable_lens_is_a_bad_gateway() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async with client(app_with(CI_KEY, lens(handler))) as gateway:
        response: Final = await gateway.get("/lens/evals/runs")

    assert response.status_code == 502


@pytest.mark.asyncio
async def test_eval_routes_are_not_implemented_until_lens_is_configured() -> None:
    async with client(app_with(CI_KEY, None)) as gateway:
        response: Final = await gateway.get("/lens/evals/runs")

    assert response.status_code == 501


@pytest.mark.parametrize(
    ("environ", "configured"),
    [
        ({"LITELLM_LENS_URL": "http://lens:4318", "LITELLM_LENS_SERVICE_TOKEN": "t" * 32}, False),
        (
            {
                "LITELLM_LENS_URL": "http://lens:4318",
                "LITELLM_LENS_SERVICE_TOKEN": "t" * 32,
                "LITELLM_LENS_GATEWAY_SECRET": "short",
            },
            False,
        ),
        ({"LITELLM_LENS_GATEWAY_SECRET": SECRET}, False),
        (
            {
                "LITELLM_LENS_URL": "http://lens:4318",
                "LITELLM_LENS_SERVICE_TOKEN": "t" * 32,
                "LITELLM_LENS_GATEWAY_SECRET": SECRET,
            },
            True,
        ),
    ],
)
def test_forwarding_needs_the_lens_url_and_a_full_length_gateway_secret(
    environ: dict[str, str], configured: bool
) -> None:
    assert (upstream_from(environ) is not None) == configured


def test_a_user_key_is_identified_by_its_user_and_a_key_with_neither_user_nor_token_is_refused() -> None:
    user: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="u1", token="hashed")
    token: Final = gateway_identity(user, SECRET, NOW)
    assert token is not None
    assert jwt.decode(token, SECRET, algorithms=["HS256"], audience="litellm-lens", options=FIXED_CLOCK)["sub"] == "u1"
    assert gateway_identity(UserAPIKeyAuth(), SECRET, NOW) is None


@pytest.mark.asyncio
async def test_a_key_lens_cannot_identify_is_unauthorized_without_calling_lens() -> None:
    seen: Final[list[httpx.Request]] = []  # mutable-ok: captures what the fake Lens received

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    async with client(app_with(UserAPIKeyAuth(), lens(handler))) as gateway:
        response: Final = await gateway.get("/lens/evals/runs")

    assert (response.status_code, seen) == (401, [])

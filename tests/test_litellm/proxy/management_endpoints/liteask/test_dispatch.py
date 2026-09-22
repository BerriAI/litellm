import asyncio
from collections.abc import Mapping

import httpx2
import pytest
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse, Response, StreamingResponse

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.liteask.dispatch import credential_fingerprint, credential_secrets, dispatch
from litellm.proxy.middleware.admission_control_middleware import (
    AdmissionControlMiddleware,
    AdmissionControlSettings,
    AdmissionControlState,
)
from litellm.proxy.middleware.billable_request_metrics_middleware import (
    BillableCategory,
    BillableRequestMetricsMiddleware,
)
from litellm.proxy.middleware.budget_reservation_release_middleware import BudgetReservationReleaseMiddleware


def outer_request(app: FastAPI, headers: tuple[tuple[bytes, bytes], ...] = (), root_path: str = "") -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": root_path + "/management/v1/liteask/chat",
            "root_path": root_path,
            "headers": headers,
            "client": ("192.0.2.10", 1234),
            "server": ("gateway.test", 443),
            "scheme": "https",
            "app": app,
            "state": {"principal": "must-not-be-reused", "budget_reservation": {"amount": 99}},
        }
    )


@pytest.mark.asyncio
async def test_child_resolves_auth_from_raw_custom_header_and_preserves_network_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.proxy_server import general_settings

    monkeypatch.setitem(general_settings, "litellm_key_header_name", "X-Company-Key")
    app = FastAPI()

    async def identity(request: Request) -> UserAPIKeyAuth:
        assert request.headers["x-company-key"] == "Bearer personal"
        assert request.headers["x-litellm-api-key"] == "Bearer secondary"
        assert request.headers["authorization"] == "Bearer original"
        assert "x-litellm-user-id" not in request.headers
        assert "litellm-changed-by" not in request.headers
        assert not hasattr(request.state, "principal")
        assert not hasattr(request.state, "budget_reservation")
        assert request.app is app
        assert request.client is not None and request.client.host == "192.0.2.10"
        assert request.scope["server"] == ("gateway.test", 443)
        assert request.scope["root_path"] == "/gateway"
        assert request.url.hostname == "external-gateway.test"
        return UserAPIKeyAuth(user_id="signed-in-admin", user_role=LitellmUserRoles.PROXY_ADMIN)

    app.dependency_overrides[user_api_key_auth] = identity

    @app.get("/key/list")
    async def keys(request: Request, caller: UserAPIKeyAuth = Depends(user_api_key_auth)) -> dict[str, object]:
        return {"user_id": caller.user_id, "ids": request.query_params.getlist("id")}

    request = outer_request(
        app,
        (
            (b"x-company-key", b"Bearer personal"),
            (b"x-litellm-api-key", b"Bearer secondary"),
            (b"authorization", b"Bearer original"),
            (b"x-litellm-user-id", b"spoofed-admin"),
            (b"litellm-changed-by", b"spoofed-audit"),
            (b"host", b"external-gateway.test"),
        ),
        root_path="/gateway",
    )
    result = await dispatch(
        request,
        "GET",
        "/key/list",
        query=(("id", "first"), ("id", "second")),
        allowed_routes=frozenset({("GET", "/key/list")}),
    )
    assert result.status_code == 200
    assert result.data == {"user_id": "signed-in-admin", "ids": ["first", "second"]}


@pytest.mark.asyncio
async def test_non_catalog_route_never_reaches_endpoint() -> None:
    app = FastAPI()

    @app.post("/key/delete")
    async def forbidden() -> None:
        pytest.fail("Non-catalog write reached its handler")

    result = await dispatch(outer_request(app), "POST", "/key/delete", allowed_routes=frozenset())
    assert result.status_code == 403


@pytest.mark.asyncio
async def test_componentized_backend_can_resolve_standard_chat_auth_without_public_chat_route() -> None:
    app = FastAPI()

    async def reject(request: Request) -> UserAPIKeyAuth:
        assert request.scope["route"].endpoint.__name__ == "chat_completion"
        assert (await request.json())["model"] == "example-model"
        raise HTTPException(401, "caller credential rejected")

    app.dependency_overrides[user_api_key_auth] = reject
    result = await dispatch(
        outer_request(app),
        "POST",
        "/chat/completions",
        {"model": "example-model", "messages": []},
        allowed_routes=frozenset({("POST", "/chat/completions")}),
    )
    assert result.status_code == 401
    assert result.data == {"detail": "caller credential rejected"}


class Sink:
    def __init__(self) -> None:
        self.calls: tuple[tuple[BillableCategory, str, int], ...] = ()
        self.factory_calls = 0

    def factory(self) -> "Sink":
        self.factory_calls += 1
        return self

    def record(self, *, category: BillableCategory, route: str, status_code: int) -> None:
        self.calls += ((category, route, status_code),)


@pytest.mark.asyncio
async def test_child_cleans_reservations_and_records_usage_without_reentering_admission() -> None:
    app = FastAPI()
    sink = Sink()
    releases: asyncio.Queue[Mapping[str, object]] = asyncio.Queue()
    state = AdmissionControlState(metrics_factory=lambda: None)

    async def release(reservation: Mapping[str, object]) -> None:
        await releases.put(reservation)

    app.add_middleware(BillableRequestMetricsMiddleware, sink_factory=sink.factory)
    app.add_middleware(BudgetReservationReleaseMiddleware, release=release)
    app.add_middleware(
        AdmissionControlMiddleware,
        get_settings=lambda: AdmissionControlSettings(1, 1, 10),
        state=state,
    )

    @app.post("/chat/completions")
    async def completion(request: Request) -> None:
        request.state.budget_reservation = {"amount": 1}
        raise HTTPException(429, "model budget exceeded")

    @app.post("/management/v1/liteask/test")
    async def ask(request: Request) -> JSONResponse:
        result = await dispatch(
            request,
            "POST",
            "/chat/completions",
            {"model": "example-model"},
            allowed_routes=frozenset({("POST", "/chat/completions")}),
        )
        return JSONResponse(result.data, status_code=result.status_code)

    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app), base_url="http://gateway.test") as client:
        for _ in range(2):
            result = await asyncio.wait_for(client.post("/management/v1/liteask/test"), timeout=1)
            assert result.status_code == 429
            assert releases.get_nowait() == {"amount": 1}
    assert releases.empty()
    assert sink.calls == ((BillableCategory.LLM, "/chat/completions", 429),) * 2
    assert sink.factory_calls == 1


@pytest.mark.asyncio
async def test_original_exception_handlers_render_child_failures() -> None:
    app = FastAPI()

    async def handle_error(request: Request, error: Exception) -> JSONResponse:
        assert request.app is app
        return JSONResponse({"error": "handled"}, status_code=422)

    app.add_exception_handler(ValueError, handle_error)

    @app.get("/key/list")
    async def broken() -> None:
        raise ValueError("do not expose this")

    result = await dispatch(
        outer_request(app),
        "GET",
        "/key/list",
        allowed_routes=frozenset({("GET", "/key/list")}),
    )
    assert result.status_code == 422
    assert result.data == {"error": "handled"}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["oversized", "non-json", "redirect"])
async def test_unsupported_response_never_reaches_agent(kind: str) -> None:
    app = FastAPI()

    @app.get("/key/list")
    async def response() -> Response:
        if kind == "non-json":
            return PlainTextResponse("private-error")
        if kind == "redirect":
            return RedirectResponse("https://untrusted.test")

        async def chunks():
            for _ in range(9):
                yield b" " * (16 * 1024)

        return StreamingResponse(chunks(), media_type="application/json")

    result = await dispatch(
        outer_request(app),
        "GET",
        "/key/list",
        allowed_routes=frozenset({("GET", "/key/list")}),
    )
    assert result.status_code == 502
    assert "private-error" not in str(result.data)
    assert "untrusted.test" not in str(result.data)


def test_fingerprint_binds_actual_credentials_and_secret_values_share_header_owner() -> None:
    app = FastAPI()
    first = outer_request(app, ((b"authorization", b"Bearer first"),))
    second = outer_request(app, ((b"authorization", b"Bearer second"),))
    assert credential_fingerprint(first) != credential_fingerprint(second)
    assert credential_fingerprint(first) == credential_fingerprint(first)
    assert credential_fingerprint(outer_request(app)) is None
    assert credential_secrets(first) == ("Bearer first", "first")

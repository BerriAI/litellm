import asyncio
import threading
from datetime import timedelta
from typing import Annotated, Final

import httpx
from fastapi import Depends, FastAPI, Security
from fastapi.testclient import TestClient

from litellm.proxy.control_plane.auth.dependencies import get_principal, get_token_verifier, require_scopes
from litellm.proxy.control_plane.auth.jwt import Rejected, SessionTokenCodec, TokenVerifier
from litellm.proxy.control_plane.auth.rbac import ADMIN_ROLE, VIEWER_ROLE, Principal, build_principal, default_policy
from litellm.proxy.control_plane.common.exceptions import install_exception_handlers

MASTER_KEY: Final = "sk-master"
POLICY: Final = default_policy()


def _app() -> tuple[TestClient, SessionTokenCodec]:
    codec: Final = SessionTokenCodec(MASTER_KEY, "cp", timedelta(hours=1), POLICY)
    app: Final = FastAPI()
    verifier: Final = TokenVerifier(MASTER_KEY, codec, None, POLICY)
    app.dependency_overrides[get_token_verifier] = lambda: verifier
    install_exception_handlers(app)

    @app.get("/usage")
    async def usage(principal: Annotated[Principal, Security(require_scopes, scopes=["usage:read"])]) -> str:
        return principal.subject

    @app.post("/dataplanes")
    async def write(principal: Annotated[Principal, Security(require_scopes, scopes=["dataplanes:write"])]) -> str:
        return principal.subject

    return TestClient(app), codec


def _token(codec: SessionTokenCodec, role: str) -> dict[str, str]:
    principal: Final = build_principal(POLICY, "u", None, frozenset({role}), None, frozenset(), "session")
    return {"Authorization": f"Bearer {codec.mint(principal)}"}


def test_viewer_reads_usage_but_cannot_write_dataplanes() -> None:
    client, codec = _app()
    assert client.get("/usage", headers=_token(codec, VIEWER_ROLE)).status_code == 200
    denied = client.post("/dataplanes", headers=_token(codec, VIEWER_ROLE))
    assert denied.status_code == 403
    assert denied.json()["detail"] == "Missing scope: dataplanes:write."
    assert denied.headers["content-type"] == "application/problem+json"


def test_admin_passes_both_scope_gates() -> None:
    client, codec = _app()
    assert client.post("/dataplanes", headers=_token(codec, ADMIN_ROLE)).status_code == 200


class _GatedVerifier:
    """Blocks the slow credential on an Event and records whether that Event was set before the wait gave up."""

    def __init__(self, gate: threading.Event) -> None:
        self._gate = gate
        self.slow_saw_the_gate_open: bool | None = None

    def verify(self, credential: str) -> Principal | Rejected:
        if credential == "slow":
            self.slow_saw_the_gate_open = self._gate.wait(timeout=5)
        return build_principal(POLICY, credential, None, frozenset({VIEWER_ROLE}), None, frozenset(), "session")


async def test_a_slow_token_verification_does_not_block_other_requests() -> None:
    gate: Final = threading.Event()
    verifier: Final = _GatedVerifier(gate)
    app: Final = FastAPI()
    app.dependency_overrides[get_token_verifier] = lambda: verifier

    async def me(principal: Annotated[Principal, Depends(get_principal)]) -> str:
        return principal.subject

    app.add_api_route("/me", me)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://cp") as client:
        slow: Final = asyncio.create_task(client.get("/me", headers={"Authorization": "Bearer slow"}))
        await asyncio.sleep(0)
        fast: Final = await client.get("/me", headers={"Authorization": "Bearer fast"})
        gate.set()
        assert (await slow).json() == "slow"
    assert fast.json() == "fast"
    assert verifier.slow_saw_the_gate_open is True, "the fast request ran while the slow JWKS lookup was still waiting"


def test_security_scopes_show_up_in_openapi() -> None:
    client, _ = _app()
    security: Final = client.get("/openapi.json").json()["paths"]["/usage"]["get"]["security"]
    assert security == [{"ControlPlaneBearer": ["usage:read"]}]

from collections.abc import Mapping
from typing import Final

import httpx
import jwt
import pytest
from fastapi import FastAPI, Header, HTTPException

from litellm.integrations.clickhouse.context import is_lens_analysis
from litellm.litellm_core_utils.initialize_dynamic_callback_params import initialize_standard_callback_dynamic_params
from litellm.proxy.lens.internal import LensInternalMiddleware, verified

SECRET: Final = "gateway-internal-identity-fixture-key-32"
NOW: Final = 1_800_000_000


def claims(**changes: object) -> Mapping[str, object]:
    return {
        "iss": "litellm-lens",
        "aud": "litellm",
        "sub": "lens-internal",
        "purpose": "analysis",
        "iat": NOW,
        "exp": NOW + 30,
        **changes,
    }


@pytest.mark.parametrize(
    "changes,valid",
    (
        ({}, True),
        ({"purpose": "signals"}, True),
        ({"purpose": "billing"}, False),
        ({"iss": "unknown"}, False),
        ({"aud": "unknown"}, False),
        ({"sub": "admin"}, False),
        ({"exp": NOW}, False),
        ({"iat": NOW + 1, "exp": NOW + 31}, True),
        ({"iat": NOW + 5, "exp": NOW + 35}, True),
        ({"iat": NOW + 6, "exp": NOW + 36}, False),
        ({"iat": NOW + 5, "exp": NOW + 5}, False),
        ({"iat": NOW + 5, "exp": NOW + 4}, False),
        ({"exp": NOW + 61}, False),
        ({"iat": True}, False),
        ({"exp": str(NOW + 30)}, False),
        ({"unknown": "value"}, False),
        ({"iat": NOW - 59, "exp": NOW + 1}, True),
    ),
    ids=(
        "analysis",
        "signals",
        "purpose",
        "issuer",
        "audience",
        "subject",
        "expired",
        "small_clock_skew",
        "clock_skew_boundary",
        "excessive_clock_skew",
        "zero_lifetime",
        "negative_lifetime",
        "ttl",
        "bool_timestamp",
        "string_timestamp",
        "unknown_claim",
        "maximum_ttl",
    ),
)
def test_only_bounded_internal_identity_is_accepted(changes: Mapping[str, object], valid: bool) -> None:
    token: Final = jwt.encode(dict(claims(**changes)), SECRET, algorithm="HS256")
    assert verified(token, SECRET, NOW) is valid


@pytest.mark.parametrize("secret", ("", "short", SECRET + "wrong"), ids=("unset", "short", "wrong"))
def test_marker_does_not_authorize_itself(secret: str) -> None:
    token: Final = jwt.encode(dict(claims()), SECRET, algorithm="HS256")
    assert not verified(token, secret, NOW)


@pytest.mark.asyncio
@pytest.mark.parametrize("authenticated", (True, False), ids=("authenticated", "unauthenticated"))
async def test_internal_marker_preserves_authentication_and_is_private(authenticated: bool) -> None:
    app: Final = FastAPI()

    @app.post("/chat/completions")
    async def completion(
        authorization: str | None = Header(default=None), x_lens_internal: str | None = Header(default=None)
    ) -> Mapping[str, object]:
        if authorization != "Bearer gateway-key":
            raise HTTPException(401, "Key required")
        return {
            "internal": is_lens_analysis(),
            "marker": x_lens_internal,
            "privacy": initialize_standard_callback_dynamic_params({}).get("turn_off_message_logging"),
        }

    app.add_middleware(LensInternalMiddleware, environ={"LENS_GATEWAY_SECRET": SECRET}, now=lambda: NOW)
    token: Final = jwt.encode(dict(claims()), SECRET, algorithm="HS256")
    headers: Final = {"X-Lens-Internal": token, **({"Authorization": "Bearer gateway-key"} if authenticated else {})}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.test") as client:
        response: Final = await client.post("/chat/completions", headers=headers)
        ordinary: Final = await client.post("/chat/completions", headers={"Authorization": "Bearer gateway-key"})
    assert response.status_code == (200 if authenticated else 401)
    if authenticated:
        assert response.json() == {"internal": True, "marker": None, "privacy": True}
    assert ordinary.json() == {"internal": False, "marker": None, "privacy": None}
    assert not is_lens_analysis()


@pytest.mark.asyncio
async def test_duplicate_markers_are_rejected_before_inference() -> None:
    app: Final = FastAPI()

    @app.post("/chat/completions")
    async def completion() -> Mapping[str, object]:
        pytest.fail("Invalid duplicate marker reached inference")

    app.add_middleware(LensInternalMiddleware, environ={"LENS_GATEWAY_SECRET": SECRET}, now=lambda: NOW)
    token: Final = jwt.encode(dict(claims()), SECRET, algorithm="HS256")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.test") as client:
        response: Final = await client.post(
            "/chat/completions", headers=[("X-Lens-Internal", token), ("X-Lens-Internal", "forged")]
        )
    assert response.status_code == 401

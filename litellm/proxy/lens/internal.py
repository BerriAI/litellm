import os
import time
from collections.abc import Callable, Mapping
from typing import Final, Literal

import jwt
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from litellm.integrations.clickhouse.context import lens_analysis
from litellm.litellm_core_utils.initialize_dynamic_callback_params import inherit_message_logging_privacy

CLOCK_SKEW_SECONDS: Final = 5


class Claims(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    iss: Literal["litellm-lens"]
    aud: Literal["litellm"]
    sub: Literal["lens-internal"]
    purpose: Literal["analysis", "signals"]
    iat: int
    exp: int


def verified(token: str, secret: str, now: int) -> bool:
    if not 32 <= len(secret.encode()) <= 512:
        return False
    try:
        claims: Final = Claims.model_validate(
            jwt.decode(
                token,
                secret,
                algorithms=["HS256"],
                issuer="litellm-lens",
                audience="litellm",
                options={
                    "require": ["iss", "aud", "sub", "iat", "exp"],
                    "verify_exp": False,
                    "verify_iat": False,
                    "verify_nbf": False,
                },
            )
        )
    except (jwt.InvalidTokenError, ValidationError):
        return False
    return claims.iat <= now + CLOCK_SKEW_SECONDS and now < claims.exp and 0 < claims.exp - claims.iat <= 60


class LensInternalMiddleware:
    def __init__(
        self, app: ASGIApp, environ: Mapping[str, str] = os.environ, now: Callable[[], float] = time.time
    ) -> None:
        self.app: Final = app
        self.environ: Final = environ
        self.now: Final = now

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        tokens: Final = Headers(scope=scope).getlist("x-lens-internal")
        if not tokens:
            await self.app(scope, receive, send)
            return
        if len(tokens) != 1 or not verified(tokens[0], self.environ.get("LENS_GATEWAY_SECRET", ""), int(self.now())):
            await JSONResponse({"detail": "Invalid Lens internal identity"}, status_code=401)(scope, receive, send)
            return
        sanitized: Final[Scope] = {
            **scope,
            "headers": tuple(
                (name, value) for name, value in Headers(scope=scope).raw if name.lower() != b"x-lens-internal"
            ),
        }
        with lens_analysis(), inherit_message_logging_privacy(True):
            await self.app(sanitized, receive, send)

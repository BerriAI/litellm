import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from pydantic import TypeAdapter
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

_STRING_SEQUENCE: Final = TypeAdapter(tuple[str, ...])
_BOOLEAN: Final = TypeAdapter(bool)


def get_cors_config(
    cors_origins_env: str | None = None,
    cors_credentials_env: str | None = None,
    *,
    settings: Mapping[str, object] | None = None,
) -> tuple[tuple[str, ...], bool]:
    current_settings: Final[Mapping[str, object]] = settings if settings is not None else {}
    origins_raw: Final = cors_origins_env if cors_origins_env is not None else os.getenv("LITELLM_CORS_ORIGINS")
    origins: Final = (
        _STRING_SEQUENCE.validate_python(current_settings.get("cors_allow_origins", ("*",)))
        if origins_raw is None
        else tuple(origin.strip() for origin in origins_raw.split(",") if origin.strip())
        if origins_raw.strip()
        else ("*",)
    )
    credentials_raw: Final = (
        cors_credentials_env if cors_credentials_env is not None else os.getenv("LITELLM_CORS_ALLOW_CREDENTIALS")
    )
    if credentials_raw is not None:
        return origins, credentials_raw.strip().lower() == "true"
    credentials: Final = current_settings.get("cors_allow_credentials")
    if credentials is not None:
        return origins, _BOOLEAN.validate_python(credentials, strict=True) and "*" not in origins
    return origins, "*" not in origins


@dataclass(frozen=True)
class _CorsOptions:
    origins: tuple[str, ...]
    credentials: bool
    methods: tuple[str, ...]
    headers: tuple[str, ...]


class ConfigurableCORSMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        get_settings: Callable[[], Mapping[str, object]],
        expose_headers: Sequence[str],
    ) -> None:
        self.app: Final[ASGIApp] = app
        self.get_settings: Final[Callable[[], Mapping[str, object]]] = get_settings
        self.expose_headers: Final[tuple[str, ...]] = tuple(expose_headers)
        self._options: _CorsOptions | None = None
        self._middleware: CORSMiddleware | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        settings: Final = self.get_settings()
        origins, credentials = get_cors_config(settings=settings)
        options: Final = _CorsOptions(
            origins=origins,
            credentials=credentials,
            methods=_STRING_SEQUENCE.validate_python(settings.get("cors_allow_methods", ("*",))),
            headers=_STRING_SEQUENCE.validate_python(settings.get("cors_allow_headers", ("*",))),
        )
        if options != self._options or self._middleware is None:
            self._middleware = CORSMiddleware(
                self.app,
                allow_origins=options.origins,
                allow_credentials=options.credentials,
                allow_methods=options.methods,
                allow_headers=options.headers,
                expose_headers=self.expose_headers,
            )
            self._options = options
        middleware: Final = self._middleware
        await middleware(scope, receive, send)

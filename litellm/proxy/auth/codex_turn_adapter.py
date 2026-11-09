from __future__ import annotations

import hmac
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Final

from fastapi import Request

from litellm._auth.codex_turn import (
    CodexAccessFacts,
    CodexCredentials,
    CodexKeyFacts,
    CodexTurnFacts,
    JwtRead,
    codex_access,
    codex_jwt_token,
    codex_key,
    codex_turn,
)
from litellm._auth.failure import NativeAuthFailure, NativeAuthRuntimeError
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.codex_model_adapter import read_codex_model_facts
from litellm.proxy.auth.route_enable_adapter import read_route_enable_facts
from litellm.proxy.auth.route_policy_adapter import read_virtual_route_facts
from litellm.proxy.auth.route_scope_adapter import read_route_scope_facts
from litellm.router_api import RegistryReader

if TYPE_CHECKING:
    from litellm.proxy.pass_through_endpoints.codex_passthrough_endpoints import (
        _SharedCache,  # pyright: ignore[reportPrivateUsage]  # existing account-catalog IO protocol
    )

RESPONSES_PATH: Final = "/v1/responses"


def read_codex_key_facts(token: UserAPIKeyAuth, facts: CodexKeyFacts = CodexKeyFacts()) -> CodexKeyFacts:
    decision: Final = codex_key(facts)
    match decision.kind:
        case "Complete":
            return facts
        case "Failure":
            raise TypeError(decision.message)
        case "ReadProxy":
            return read_codex_key_facts(token, replace(facts, proxy=read_route_scope_facts(RESPONSES_PATH, "Proxy")))
        case "ReadEnabled":
            return read_codex_key_facts(token, replace(facts, enabled=read_route_enable_facts(RESPONSES_PATH)))
        case "ReadVirtual":
            return read_codex_key_facts(
                token, replace(facts, virtual_route=read_virtual_route_facts(RESPONSES_PATH, token, None))
            )
        case "ReadRole":
            return read_codex_key_facts(token, replace(facts, role_loaded=True, role=token.user_role))
        case "ReadAdmin":
            return read_codex_key_facts(token, replace(facts, admin=read_route_scope_facts(RESPONSES_PATH, "Admin")))
        case _:
            raise NativeAuthRuntimeError("invalid Codex key read stage")


async def read_jwt_result(gateway_token: str, model: str, headers: Mapping[str, str]) -> JwtRead:
    from litellm.proxy.auth.handle_jwt import JWTAuthManager
    from litellm.proxy.proxy_server import (
        general_settings_view,
        jwt_handler,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    try:
        _ = await JWTAuthManager.auth_builder(  # pyright: ignore[reportUnknownMemberType]  # unchanged validator accepts legacy dictionaries; its return value is not interpreted here
            api_key=gateway_token,
            jwt_handler=jwt_handler,
            request_data={"model": model},
            general_settings=dict(general_settings_view()),
            route=RESPONSES_PATH,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=proxy_logging_obj,
            request_headers=headers,
            request_method="POST",
        )
    except NativeAuthFailure:
        raise
    except Exception:  # noqa: BLE001  # the unchanged JWT validator reports refusals by raising
        return "Exception"
    return "Returned"


@dataclass(frozen=True, slots=True)
class CodexAccess:
    key: CodexKeyFacts
    jwt_token: str | None = field(repr=False)
    request_headers: Mapping[str, str] = field(repr=False)

    @property
    def key_admits(self) -> bool:
        return codex_key(self.key).allowed

    async def admits(self, model: str) -> bool:
        return codex_access(await read_codex_access_facts(self, model)).allowed


def read_codex_access(token: UserAPIKeyAuth, request: Request) -> CodexAccess:
    from litellm.proxy.pass_through_endpoints.codex_passthrough_endpoints import (
        _gateway_token,  # pyright: ignore[reportPrivateUsage]  # raw credential extraction, not auth policy
    )

    selection: Final = codex_jwt_token(claims_present=bool(token.jwt_claims), token_present=bool(token.token))
    key: Final = read_codex_key_facts(token)
    return CodexAccess(key, _gateway_token(request) if selection.allowed else None, request.headers)


async def read_codex_access_facts(source: CodexAccess, model: str) -> CodexAccessFacts:
    facts: Final = CodexAccessFacts(source.key, source.jwt_token is not None)
    decision: Final = codex_access(facts)
    match decision.kind:
        case "Complete":
            return facts
        case "ReadJwt":
            if source.jwt_token is None:
                raise NativeAuthRuntimeError("Codex JWT read without credential")
            return replace(facts, jwt=await read_jwt_result(source.jwt_token, model, source.request_headers))
        case _:
            raise NativeAuthRuntimeError("invalid Codex access read stage")


@dataclass(frozen=True, slots=True)
class _Source:
    router: RegistryReader | None
    token: UserAPIKeyAuth
    request: Request | None = None
    access: CodexAccess | None = None
    caller: str | None = field(default=None, repr=False)
    headers: Mapping[str, str] | None = field(default=None, repr=False)
    cache: _SharedCache | None = None
    gateway_token: str | None = field(default=None, repr=False)


async def _read(facts: CodexTurnFacts, source: _Source) -> CodexTurnFacts:
    from litellm.proxy.pass_through_endpoints.codex_passthrough_endpoints import (
        _account_slugs,  # pyright: ignore[reportPrivateUsage]  # unchanged catalog IO
        _chatgpt_bearer,  # pyright: ignore[reportPrivateUsage]  # raw credential parsing
        _forwarded_headers,  # pyright: ignore[reportPrivateUsage]  # transport header projection
        _gateway_token,  # pyright: ignore[reportPrivateUsage]  # raw credential parsing
        _models_etag_caller,  # pyright: ignore[reportPrivateUsage]  # cache identity projection
        codex_shared_cache,
    )

    decision: Final = codex_turn(facts)
    match decision.kind:
        case "Complete":
            return facts
        case "ReadCredentials":
            if source.request is None:
                raise NativeAuthRuntimeError("native Codex turn without request")
            bearer: Final = _chatgpt_bearer(source.request)
            gateway: Final = _gateway_token(source.request)
            credentials: Final = CodexCredentials(
                bool(bearer), bool(gateway), hmac.compare_digest(bearer.encode(), gateway.encode())
            )
            return await _read(replace(facts, credentials=credentials), replace(source, gateway_token=gateway))
        case "ReadAccess":
            if source.request is None or source.gateway_token is None:
                raise NativeAuthRuntimeError("native Codex access without request credential")
            access: Final = read_codex_access(source.token, source.request)
            caller: Final = _models_etag_caller(source.token, source.gateway_token)
            headers: Final = _forwarded_headers(source.request)
            cache: Final = codex_shared_cache()
            return await _read(
                replace(facts, access=CodexAccessFacts(access.key, access.jwt_token is not None)),
                replace(source, access=access, caller=caller, headers=headers, cache=cache),
            )
        case "ReadRouting":
            routing: Final = read_codex_model_facts(facts.model, source.router, source.token.team_id)
            return await _read(replace(facts, routing=routing), source)
        case "ReadJwt":
            if source.access is None or facts.model is None:
                raise NativeAuthRuntimeError("Codex JWT read without access context")
            return await _read(replace(facts, access=await read_codex_access_facts(source.access, facts.model)), source)
        case "ReadCatalog":
            if source.headers is None or source.caller is None or source.cache is None:
                raise NativeAuthRuntimeError("Codex catalog read without IO context")
            slugs: Final = await _account_slugs(source.headers, source.caller, source.cache)
            return await _read(
                replace(facts, catalog_loaded=True, slugs=tuple(slugs) if slugs is not None else None), source
            )
        case _:
            raise NativeAuthRuntimeError("invalid Codex turn read stage")


async def read_native_turn_facts(
    model: str, router: RegistryReader, request: Request, token: UserAPIKeyAuth
) -> CodexTurnFacts:
    return await _read(CodexTurnFacts(model, native=True), _Source(router, token, request=request))


async def native_turn_serves_through_litellm(
    model: str, llm_router: RegistryReader, request: Request, user_api_key_dict: UserAPIKeyAuth
) -> bool:
    return codex_turn(await read_native_turn_facts(model, llm_router, request, user_api_key_dict)).allowed


async def serves_through_litellm(
    model: object,
    llm_router: RegistryReader | None,
    user_api_key_dict: UserAPIKeyAuth,
    access: CodexAccess,
    caller: str,
    headers: Mapping[str, str],
    cache: _SharedCache,
) -> bool:
    facts: Final = CodexTurnFacts(
        model if isinstance(model, str) else None, access=CodexAccessFacts(access.key, access.jwt_token is not None)
    )
    source: Final = _Source(llm_router, user_api_key_dict, access=access, caller=caller, headers=headers, cache=cache)
    return codex_turn(await _read(facts, source)).allowed

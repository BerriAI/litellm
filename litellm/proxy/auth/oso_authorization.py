import re
from collections.abc import Mapping, Sequence
from typing import Final, Protocol

import httpx
from fastapi import status
from pydantic import BaseModel, ConfigDict, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy._types import (
    OsoAuthorizationConfig,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.types.llms.custom_http import httpxSpecialProvider

_HASHED_TOKEN_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_MAX_MODELS_PER_REQUEST: Final = 10
_MULTI_MODEL_ROUTES: Final = frozenset({"/cost/predict-cache"})
_MODEL_REQUIRED_ROUTES: Final = frozenset(
    {
        "/chat/completions",
        "/v1/chat/completions",
        "/completions",
        "/v1/completions",
        "/embeddings",
        "/v1/embeddings",
        "/images/generations",
        "/v1/images/generations",
        "/images/edits",
        "/v1/images/edits",
        "/moderations",
        "/v1/moderations",
        "/audio/speech",
        "/v1/audio/speech",
        "/audio/transcriptions",
        "/v1/audio/transcriptions",
        "/responses",
        "/v1/responses",
        "/openai/v1/responses",
        "/v1/messages",
    }
)


class OsoValue(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    id: str


class OsoContextFact(BaseModel):
    model_config = ConfigDict(frozen=True)

    predicate: str
    args: tuple[OsoValue | str, ...]


class OsoAuthorizeRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    actor_type: str
    actor_id: str
    action: str
    resource_type: str
    resource_id: str
    context_facts: tuple[OsoContextFact, ...]


class OsoAuthorizeResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed: bool


class OsoHTTPClient(Protocol):
    async def post(
        self,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
        timeout: float,
    ) -> httpx.Response: ...


class OsoAuthorizer(Protocol):
    async def authorize(self, request: OsoAuthorizeRequest) -> bool: ...


class OsoCloudAuthorizer:
    def __init__(self, config: OsoAuthorizationConfig, http_client: OsoHTTPClient | None = None) -> None:
        self._config: Final = config
        self._http_client: Final = http_client or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.OsoAuthorization,
            params={
                "timeout": config.timeout,
                "client_alias": "oso_authorization",
            },  # mutable-ok: client API requires dict
        )

    async def authorize(self, request: OsoAuthorizeRequest) -> bool:
        api_key: Final = self._config.api_key
        if api_key is None:
            raise ValueError("Oso API key is not configured")
        response: Final = await self._http_client.post(
            f"{str(self._config.url).rstrip('/')}/api/authorize",
            json=request.model_dump(mode="json"),
            headers={  # mutable-ok: HTTP client protocol requires a concrete header dictionary
                "Authorization": f"Bearer {api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            timeout=self._config.timeout,
        )
        return OsoAuthorizeResponse.model_validate_json(response.content).allowed


def _safe_key_id(valid_token: UserAPIKeyAuth) -> str | None:
    candidate: Final = valid_token.token or valid_token.api_key
    if not candidate:
        return None
    if _HASHED_TOKEN_PATTERN.fullmatch(candidate) or candidate.startswith("hashed-jwt-"):
        return candidate
    return None


def _actor(valid_token: UserAPIKeyAuth, key_id: str | None) -> OsoValue | None:
    if valid_token.user_id:
        return OsoValue(type="User", id=valid_token.user_id)
    if valid_token.team_id:
        return OsoValue(type="Team", id=valid_token.team_id)
    if key_id:
        return OsoValue(type="ApiKey", id=key_id)
    return None


def _relation(actor: OsoValue, name: str, related: OsoValue) -> OsoContextFact:
    return OsoContextFact(predicate="has_relation", args=(actor, name, related))


def _context_facts(
    actor: OsoValue,
    valid_token: UserAPIKeyAuth,
    key_id: str | None,
) -> tuple[OsoContextFact, ...]:
    relations: Final = (
        ("team", OsoValue(type="Team", id=valid_token.team_id)) if valid_token.team_id else None,
        ("organization", OsoValue(type="Organization", id=valid_token.org_id)) if valid_token.org_id else None,
        ("project", OsoValue(type="Project", id=valid_token.project_id)) if valid_token.project_id else None,
        ("api_key", OsoValue(type="ApiKey", id=key_id)) if key_id else None,
    )
    return tuple(
        _relation(actor, name, related)
        for relation in relations
        if relation is not None
        for name, related in (relation,)
        if related != actor
    )


def _model_names(model: str | Sequence[str] | None) -> tuple[str, ...]:
    if isinstance(model, str):
        return (model,) if model else ()
    if isinstance(model, Sequence):
        return tuple(dict.fromkeys(item for item in model if isinstance(item, str) and item))
    return ()


def _config(general_settings: Mapping[str, object]) -> OsoAuthorizationConfig | None:
    raw_config: Final = general_settings.get("oso_authorization")
    if raw_config is None:
        return None
    try:
        config: Final = OsoAuthorizationConfig.model_validate(raw_config)
    except ValidationError as exc:
        raise ProxyException(
            message="Oso authorization is enabled but its configuration is invalid",
            type=ProxyErrorTypes.auth_provider_unavailable,
            param=None,
            code=status.HTTP_503_SERVICE_UNAVAILABLE,
        ) from exc
    return config if config.enabled else None


def _denied(model: str | Sequence[str] | None) -> ProxyException:
    return ProxyException(
        message=f"Oso authorization denied access to model '{model}'",
        type=ProxyErrorTypes.key_model_access_denied,
        param="model",
        code=status.HTTP_403_FORBIDDEN,
    )


def _invalid_model(message: str) -> ProxyException:
    return ProxyException(
        message=message,
        type="bad_request",
        param="model",
        code=status.HTTP_400_BAD_REQUEST,
    )


async def _authorize_all(
    authorizer: OsoAuthorizer,
    requests: tuple[OsoAuthorizeRequest, ...],
) -> bool:
    for request in requests:
        if not await authorizer.authorize(request):
            return False
    return True


async def enforce_oso_model_authorization(
    *,
    general_settings: Mapping[str, object],
    valid_token: UserAPIKeyAuth,
    model: str | Sequence[str] | None,
    route: str,
    request_method: str,
    authorizer: OsoAuthorizer | None = None,
) -> None:
    config: Final = _config(general_settings)
    if config is None or request_method.upper() != "POST":
        return

    if isinstance(model, Sequence) and not isinstance(model, str) and route not in _MULTI_MODEL_ROUTES:
        raise _invalid_model("The model field must be a string for this endpoint")
    models: Final = _model_names(model)
    if len(models) > _MAX_MODELS_PER_REQUEST:
        raise _invalid_model(f"At most {_MAX_MODELS_PER_REQUEST} models may be authorized per request")
    if not models:
        if route in _MODEL_REQUIRED_ROUTES or route.endswith(("/chat/completions", "/embeddings", "/messages")):
            raise _denied(model)
        return

    key_id: Final = _safe_key_id(valid_token)
    actor: Final = _actor(valid_token, key_id)
    if actor is None:
        raise _denied(model)

    oso_authorizer: Final = authorizer or OsoCloudAuthorizer(config=config)
    context_facts: Final = _context_facts(actor=actor, valid_token=valid_token, key_id=key_id)
    authorization_requests: Final = tuple(
        OsoAuthorizeRequest(
            actor_type=actor.type,
            actor_id=actor.id,
            action="invoke",
            resource_type="Model",
            resource_id=model_name,
            context_facts=context_facts,
        )
        for model_name in models
    )
    try:
        allowed: Final = await _authorize_all(oso_authorizer, authorization_requests)
    except ProxyException:
        raise
    except Exception as exc:
        verbose_proxy_logger.warning("Oso authorization provider unavailable: %s", exc)
        raise ProxyException(
            message="Oso authorization provider is unavailable; request denied",
            type=ProxyErrorTypes.auth_provider_unavailable,
            param="model",
            code=status.HTTP_503_SERVICE_UNAVAILABLE,
        ) from exc

    if not allowed:
        raise _denied(model)

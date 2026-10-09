"""Per-user GitHub Copilot OAuth: exchange a caller's stored GitHub token for a
short-lived Copilot token, plus the device-flow helpers the proxy endpoints use
to establish that connection. The shared device login (Authenticator) is never
touched on the per-user path.
"""

import asyncio
import hashlib
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import (
    Final,
    Literal,
    Protocol,
    TypeAlias,
    cast,  # noqa: TID251  # casts pin the untyped litellm clients/caches to the local Protocols
)
from urllib.parse import urlparse

import httpx
from pydantic import ConfigDict, TypeAdapter, ValidationError, with_config
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import (
    GITHUB_COPILOT_AUTH_TYPE_KEY,
    GITHUB_COPILOT_PER_USER_AUTH_TYPE,
    GITHUB_COPILOT_USER_TOKEN_SAFETY_MARGIN_SECONDS,
)
from litellm.exceptions import (
    APIConnectionError,
    BadRequestError,
    CallerCredentialAuthenticationError,
    CallerCredentialRateLimitError,
    ServiceUnavailableError,
)

from .authenticator import (
    DEFAULT_GITHUB_ACCESS_TOKEN_URL,
    DEFAULT_GITHUB_API_KEY_URL,
    DEFAULT_GITHUB_CLIENT_ID,
    DEFAULT_GITHUB_DEVICE_CODE_URL,
    github_api_headers,
)
from .common_utils import DEFAULT_GITHUB_COPILOT_API_BASE

GITHUB_COPILOT_USER_SESSION_KWARG_KEY: Final = "github_copilot_user_session"
_LLM_PROVIDER: Final = "github_copilot"
_DEVICE_FLOW_SCOPE: Final = "read:user"
_SESSION_CACHE_MAX_SIZE: Final = 1000
_SYNC_LOCK_COUNT: Final = 64


class _SessionCache(Protocol):
    def get_cache(self, key: str) -> object: ...

    def set_cache(self, key: str, value: object, ttl: float | None = None) -> object: ...

    def delete_cache(self, key: str) -> object: ...


class _SyncGetClient(Protocol):
    def get(
        self,
        url: str,
        params: dict[str, str] | None = None,  # mutable-ok: mirrors the untyped client's dict parameters
        headers: dict[str, str] | None = None,  # mutable-ok: mirrors the untyped client's dict parameters
    ) -> httpx.Response: ...


class _AsyncGitHubClient(Protocol):
    async def get(
        self,
        url: str,
        params: dict[str, str] | None = None,  # mutable-ok: mirrors the untyped client's dict parameters
        headers: dict[str, str] | None = None,  # mutable-ok: mirrors the untyped client's dict parameters
    ) -> httpx.Response: ...

    async def post(
        self,
        url: str,
        data: dict[str, str] | str | bytes | None = None,  # mutable-ok: mirrors the untyped client's dict parameters
        json: dict[str, str] | None = None,  # mutable-ok: mirrors the untyped client's dict parameters
        params: dict[str, str] | None = None,  # mutable-ok: mirrors the untyped client's dict parameters
        headers: dict[str, str] | None = None,  # mutable-ok: mirrors the untyped client's dict parameters
    ) -> httpx.Response: ...


_SESSION_CACHE: Final = cast(  # cast-ok: pins the untyped module cache to _SessionCache
    _SessionCache,
    InMemoryCache(
        max_size_in_memory=_SESSION_CACHE_MAX_SIZE,
        default_ttl=GITHUB_COPILOT_USER_TOKEN_SAFETY_MARGIN_SECONDS,
    ),
)
_EXCHANGE_LOCKS: Final = tuple(threading.Lock() for _ in range(_SYNC_LOCK_COUNT))
_IN_FLIGHT: Final[  # mutable-ok: single-flight dedup registry mutated under its own locking
    dict[str, "asyncio.Future[GithubCopilotUserSession]"]
] = {}


@with_config(ConfigDict(extra="allow", strict=True))
class _CopilotTokenPayload(TypedDict):
    token: ReadOnly[str]
    expires_at: ReadOnly[NotRequired[object]]
    endpoints: ReadOnly[NotRequired[object]]


@with_config(ConfigDict(extra="allow", strict=True))
class _DeviceCodePayload(TypedDict):
    device_code: ReadOnly[str]
    user_code: ReadOnly[str]
    verification_uri: ReadOnly[str]
    expires_in: ReadOnly[object]
    interval: ReadOnly[object]


@with_config(ConfigDict(extra="allow", strict=True))
class _AccessTokenPollPayload(TypedDict):
    access_token: ReadOnly[NotRequired[str]]
    error: ReadOnly[NotRequired[object]]
    error_description: ReadOnly[NotRequired[object]]
    interval: ReadOnly[NotRequired[object]]


@with_config(ConfigDict(extra="allow", strict=True))
class _CopilotEndpointsPayload(TypedDict):
    api: ReadOnly[NotRequired[object]]


@with_config(ConfigDict(extra="allow", strict=True))
class _GithubUserPayload(TypedDict):
    login: ReadOnly[str]


_COPILOT_TOKEN: Final = TypeAdapter(_CopilotTokenPayload)
_DEVICE_CODE: Final = TypeAdapter(_DeviceCodePayload)
_ACCESS_TOKEN_POLL: Final = TypeAdapter(_AccessTokenPollPayload)
_GITHUB_USER: Final = TypeAdapter(_GithubUserPayload)
_COPILOT_ENDPOINTS: Final = TypeAdapter(_CopilotEndpointsPayload)
_CACHED_SESSION_ENTRY: Final = TypeAdapter(tuple[str, str, float])


@dataclass(frozen=True, slots=True, repr=False)
class GithubCopilotUserSession:
    """A resolved per-user Copilot token plus the validated host it may be sent
    to. Only this module constructs it, so ``isinstance`` checks downstream are
    unforgeable by request JSON."""

    token: str = field(repr=False)
    api_base: str

    def __repr__(self) -> str:
        return f"GithubCopilotUserSession(api_base={self.api_base!r}, token=REDACTED)"

    __str__ = __repr__


@dataclass(frozen=True, slots=True)
class GithubCopilotDeviceFlowStart:
    device_code: str = field(repr=False)
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int


DeviceFlowPollStatus: TypeAlias = Literal["pending", "slow_down", "expired", "denied", "connected"]


@dataclass(frozen=True, slots=True)
class GithubCopilotDeviceFlowPoll:
    status: DeviceFlowPollStatus
    interval: int | None = None
    access_token: str | None = field(default=None, repr=False)


def validated_copilot_api_base(endpoints_api: object) -> str:
    """Only a genuine *.githubcopilot.com HTTPS endpoint may receive a Copilot
    token; anything else falls back to the default base."""
    if not isinstance(endpoints_api, str):
        return DEFAULT_GITHUB_COPILOT_API_BASE
    parsed: Final = urlparse(endpoints_api)
    hostname: Final = (parsed.hostname or "").lower()
    if (
        parsed.scheme.lower() != "https"
        or not (hostname == "githubcopilot.com" or hostname.endswith(".githubcopilot.com"))
        or parsed.username is not None
        or parsed.password is not None
    ):
        return DEFAULT_GITHUB_COPILOT_API_BASE
    try:
        port: Final = parsed.port
    except ValueError:
        return DEFAULT_GITHUB_COPILOT_API_BASE
    if port not in (None, 443):
        return DEFAULT_GITHUB_COPILOT_API_BASE
    return endpoints_api.rstrip("/")


def _session_cache_key(user_id: str, github_token: str) -> str:
    return hashlib.sha256(f"{user_id}\x00{github_token}".encode()).hexdigest()


def evict_copilot_user_session(user_id: str, github_token: str) -> None:
    _SESSION_CACHE.delete_cache(_session_cache_key(user_id, github_token))


def _cached_session(cache_key: str) -> GithubCopilotUserSession | None:
    cached: Final[object] = _SESSION_CACHE.get_cache(cache_key)
    try:
        token, api_base, expires_at = _CACHED_SESSION_ENTRY.validate_python(cached)
    except ValidationError:
        return None
    if expires_at - GITHUB_COPILOT_USER_TOKEN_SAFETY_MARGIN_SECONDS <= time.time():
        return None
    return GithubCopilotUserSession(token=token, api_base=api_base)


def _cache_session(cache_key: str, session: GithubCopilotUserSession, expires_at: float) -> None:
    ttl: Final = expires_at - GITHUB_COPILOT_USER_TOKEN_SAFETY_MARGIN_SECONDS - time.time()
    if ttl <= 0:
        return
    _SESSION_CACHE.set_cache(cache_key, (session.token, session.api_base, expires_at), ttl=ttl)


def _reject_connection(credential_name: str) -> CallerCredentialAuthenticationError:
    return CallerCredentialAuthenticationError(
        message=(
            f"GitHub rejected the GitHub Copilot connection for credential '{credential_name}'. "
            f"Reconnect GitHub Copilot for credential '{credential_name}' in the LiteLLM UI (LLM Credentials)"
        ),
        llm_provider=_LLM_PROVIDER,
        model="",
    )


def _session_from_response(response: httpx.Response, credential_name: str, cache_key: str) -> GithubCopilotUserSession:
    if response.status_code in (401, 403, 404):
        _SESSION_CACHE.delete_cache(cache_key)
        raise _reject_connection(credential_name)
    if response.status_code == 429:
        raise CallerCredentialRateLimitError(
            message="GitHub rate limited the GitHub Copilot token exchange",
            llm_provider=_LLM_PROVIDER,
            model="",
        )
    if not response.is_success:
        raise ServiceUnavailableError(
            message=f"GitHub Copilot token exchange failed with status {response.status_code}",
            llm_provider=_LLM_PROVIDER,
            model="",
        )
    try:
        payload: Final = _COPILOT_TOKEN.validate_python(response.json())
    except Exception as e:
        raise ServiceUnavailableError(
            message="GitHub Copilot token exchange returned an invalid response",
            llm_provider=_LLM_PROVIDER,
            model="",
        ) from e

    endpoints: Final[object] = payload.get("endpoints")
    endpoints_api: Final[object] = (
        _COPILOT_ENDPOINTS.validate_python(endpoints).get("api") if isinstance(endpoints, Mapping) else None
    )
    api_base: Final = validated_copilot_api_base(endpoints_api)
    session: Final = GithubCopilotUserSession(token=payload["token"], api_base=api_base)
    expires_at: Final = _expires_at_seconds(payload.get("expires_at"))
    if expires_at > 0:
        _cache_session(cache_key, session, expires_at)
    return session


def _to_int(raw: object) -> int:
    if isinstance(raw, bool):
        return 0
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return 0
    return 0


def _expires_at_seconds(raw: object) -> float:
    if isinstance(raw, bool):
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw)
        except ValueError:
            return 0.0
    return 0.0


def _do_exchange(github_token: str, credential_name: str, cache_key: str) -> GithubCopilotUserSession:
    import litellm

    try:
        client: Final = cast(  # cast-ok: pins the untyped litellm client to the local Protocol
            _SyncGetClient, litellm.module_level_client
        )
        response: Final = client.get(
            DEFAULT_GITHUB_API_KEY_URL,
            headers=github_api_headers(github_token),
        )
    except Exception as e:
        raise APIConnectionError(
            message="GitHub Copilot token exchange request failed",
            llm_provider=_LLM_PROVIDER,
            model="",
        ) from e
    return _session_from_response(response, credential_name, cache_key)


def _new_async_github_client() -> _AsyncGitHubClient:
    from litellm.llms.custom_httpx import http_handler

    return cast(  # cast-ok: pins the untyped litellm client factory to the local Protocol
        Callable[[str], _AsyncGitHubClient],
        getattr(http_handler, "get_async_httpx_client"),  # noqa: B009  # untyped factory
    )(_LLM_PROVIDER)


async def _do_aexchange(github_token: str, credential_name: str, cache_key: str) -> GithubCopilotUserSession:
    client: Final = _new_async_github_client()
    try:
        response: Final = await client.get(
            DEFAULT_GITHUB_API_KEY_URL,
            headers=github_api_headers(github_token),
        )
    except Exception as e:
        raise APIConnectionError(
            message="GitHub Copilot token exchange request failed",
            llm_provider=_LLM_PROVIDER,
            model="",
        ) from e
    return _session_from_response(response, credential_name, cache_key)


def exchange_github_token(
    user_id: str,
    github_token: str,
    credential_name: str,
) -> GithubCopilotUserSession:
    cache_key: Final = _session_cache_key(user_id, github_token)
    cached: Final = _cached_session(cache_key)
    if cached is not None:
        return cached
    lock: Final = _EXCHANGE_LOCKS[int(cache_key, 16) % _SYNC_LOCK_COUNT]
    with lock:
        rechecked: Final = _cached_session(cache_key)
        if rechecked is not None:
            return rechecked
        return _do_exchange(github_token, credential_name, cache_key)


async def aexchange_github_token(
    user_id: str,
    github_token: str,
    credential_name: str,
) -> GithubCopilotUserSession:
    cache_key: Final = _session_cache_key(user_id, github_token)
    cached: Final = _cached_session(cache_key)
    if cached is not None:
        return cached
    loop: Final = asyncio.get_running_loop()
    in_flight: Final = _IN_FLIGHT.get(cache_key)
    if in_flight is not None and not in_flight.done() and in_flight.get_loop() is loop:
        return await asyncio.shield(in_flight)
    future: Final = loop.create_future()
    _IN_FLIGHT[cache_key] = future
    try:
        session: Final = await _do_aexchange(github_token, credential_name, cache_key)
    except BaseException as e:
        if not future.done():
            future.set_exception(e)
            future.exception()  # consume so unawaited waiters do not log "exception was never retrieved"
        raise
    else:
        if not future.done():
            future.set_result(session)
        return session
    finally:
        if _IN_FLIGHT.get(cache_key) is future:
            _IN_FLIGHT.pop(cache_key, None)


async def astart_device_flow() -> GithubCopilotDeviceFlowStart:
    client: Final = _new_async_github_client()
    device_code_url: Final = os.getenv("GITHUB_COPILOT_DEVICE_CODE_URL", DEFAULT_GITHUB_DEVICE_CODE_URL)
    client_id: Final = os.getenv("GITHUB_COPILOT_CLIENT_ID", DEFAULT_GITHUB_CLIENT_ID)
    try:
        response: Final = await client.post(
            device_code_url,
            headers=github_api_headers(),
            json={"client_id": client_id, "scope": _DEVICE_FLOW_SCOPE},
        )
    except httpx.HTTPStatusError as e:
        raise ServiceUnavailableError(
            message=f"GitHub device flow start failed with status {e.response.status_code}",
            llm_provider=_LLM_PROVIDER,
            model="",
        ) from e
    except Exception as e:
        raise APIConnectionError(
            message="GitHub device flow start request failed",
            llm_provider=_LLM_PROVIDER,
            model="",
        ) from e
    try:
        payload: Final = _DEVICE_CODE.validate_python(response.json())
    except Exception as e:
        raise ServiceUnavailableError(
            message="GitHub device flow start returned an invalid response",
            llm_provider=_LLM_PROVIDER,
            model="",
        ) from e
    return GithubCopilotDeviceFlowStart(
        device_code=payload["device_code"],
        user_code=payload["user_code"],
        verification_uri=payload["verification_uri"],
        expires_in=_to_int(payload["expires_in"]),
        interval=_to_int(payload["interval"]),
    )


async def apoll_device_flow(device_code: str) -> GithubCopilotDeviceFlowPoll:
    client: Final = _new_async_github_client()
    access_token_url: Final = os.getenv("GITHUB_COPILOT_ACCESS_TOKEN_URL", DEFAULT_GITHUB_ACCESS_TOKEN_URL)
    client_id: Final = os.getenv("GITHUB_COPILOT_CLIENT_ID", DEFAULT_GITHUB_CLIENT_ID)
    try:
        response: Final = await client.post(
            access_token_url,
            headers=github_api_headers(),
            json={
                "client_id": client_id,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
        )
    except httpx.HTTPStatusError as e:
        return _poll_payload_from_response(e.response)
    except Exception as e:
        raise APIConnectionError(
            message="GitHub device flow poll request failed",
            llm_provider=_LLM_PROVIDER,
            model="",
        ) from e
    return _poll_payload_from_response(response)


def _poll_payload_from_response(response: httpx.Response) -> GithubCopilotDeviceFlowPoll:
    if not response.is_success:
        raise ServiceUnavailableError(
            message=f"GitHub device flow poll failed with status {response.status_code}",
            llm_provider=_LLM_PROVIDER,
            model="",
        )
    try:
        payload: Final = _ACCESS_TOKEN_POLL.validate_python(response.json())
    except Exception as e:
        raise ServiceUnavailableError(
            message="GitHub device flow poll returned an invalid response",
            llm_provider=_LLM_PROVIDER,
            model="",
        ) from e

    access_token: Final = payload.get("access_token")
    if isinstance(access_token, str) and access_token:
        return GithubCopilotDeviceFlowPoll(status="connected", access_token=access_token)

    error: Final = payload.get("error")
    raw_interval: Final = payload.get("interval")
    interval: Final = _to_int(raw_interval) or None
    match error:
        case "authorization_pending":
            return GithubCopilotDeviceFlowPoll(status="pending", interval=interval)
        case "slow_down":
            return GithubCopilotDeviceFlowPoll(status="slow_down", interval=interval)
        case "expired_token":
            return GithubCopilotDeviceFlowPoll(status="expired", interval=interval)
        case "access_denied":
            return GithubCopilotDeviceFlowPoll(status="denied", interval=interval)
        case _:
            raise ServiceUnavailableError(
                message="GitHub device flow poll returned an unrecognized error",
                llm_provider=_LLM_PROVIDER,
                model="",
            )


async def afetch_github_login(github_token: str) -> str:
    client: Final = _new_async_github_client()
    try:
        response: Final = await client.get(
            "https://api.github.com/user",
            headers=github_api_headers(github_token),
        )
    except Exception as e:
        raise APIConnectionError(
            message="GitHub user lookup request failed",
            llm_provider=_LLM_PROVIDER,
            model="",
        ) from e
    if not response.is_success:
        raise ServiceUnavailableError(
            message=f"GitHub user lookup failed with status {response.status_code}",
            llm_provider=_LLM_PROVIDER,
            model="",
        )
    try:
        payload: Final = _GITHUB_USER.validate_python(response.json())
    except Exception as e:
        raise ServiceUnavailableError(
            message="GitHub user lookup returned an invalid response",
            llm_provider=_LLM_PROVIDER,
            model="",
        ) from e
    return payload["login"]


async def acheck_copilot_seat(user_id: str, github_token: str, credential_name: str) -> GithubCopilotUserSession:
    return await aexchange_github_token(
        user_id=user_id,
        github_token=github_token,
        credential_name=credential_name,
    )


def github_copilot_auth_mode(litellm_credential_name: object, auth_type_value: object) -> bool:
    """Whether this call runs in per-user mode. A resolved credential's stored
    ``credential_values`` are the only authority once a credential is named and
    found; the request's own ``github_copilot_auth_type`` can never flip the
    mode either way on a named credential."""
    if isinstance(litellm_credential_name, str) and litellm_credential_name:
        from litellm.litellm_core_utils.credential_accessor import CredentialAccessor

        credential: Final = CredentialAccessor.find_credential(litellm_credential_name)
        if credential is not None:
            values: Final[object] = getattr(credential, "credential_values", None)
            return isinstance(values, Mapping) and (
                cast(  # cast-ok: value is Mapping-checked or dict-shaped at runtime
                    Mapping[object, object], values
                ).get(GITHUB_COPILOT_AUTH_TYPE_KEY)
                == GITHUB_COPILOT_PER_USER_AUTH_TYPE
            )
        return auth_type_value == GITHUB_COPILOT_PER_USER_AUTH_TYPE
    if auth_type_value == GITHUB_COPILOT_PER_USER_AUTH_TYPE:
        raise BadRequestError(
            message=f"{GITHUB_COPILOT_AUTH_TYPE_KEY} {GITHUB_COPILOT_PER_USER_AUTH_TYPE} requires litellm_credential_name",
            llm_provider=_LLM_PROVIDER,
            model="",
        )
    return False


def _connect_error(credential_name: str) -> CallerCredentialAuthenticationError:
    return CallerCredentialAuthenticationError(
        message=(f"Connect GitHub Copilot for credential '{credential_name}' in the LiteLLM UI (LLM Credentials)"),
        llm_provider=_LLM_PROVIDER,
        model="",
    )


def _caller_connection(kwargs: Mapping[str, object], credential_name: str) -> tuple[str, str]:
    secret_fields_raw: Final[object] = kwargs.get("secret_fields")
    secret_fields: Final = (
        cast(Mapping[object, object], secret_fields_raw)  # cast-ok: value is Mapping-checked or dict-shaped at runtime
        if isinstance(secret_fields_raw, Mapping)
        else None
    )
    user_id: Final[object] = (
        secret_fields.get("user_provider_credentials_user_id") if secret_fields is not None else None
    )
    credentials_raw: Final[object] = (
        secret_fields.get("user_provider_credentials") if secret_fields is not None else None
    )
    credentials: Final = (
        cast(Mapping[object, object], credentials_raw)  # cast-ok: value is Mapping-checked or dict-shaped at runtime
        if isinstance(credentials_raw, Mapping)
        else None
    )
    token: Final[object] = credentials.get(credential_name) if credentials is not None else None
    if not isinstance(user_id, str) or not user_id or not isinstance(token, str) or not token:
        raise _connect_error(credential_name)
    return user_id, token


def _params_value(source: object, key: str) -> object:
    if isinstance(source, Mapping):
        return cast(Mapping[object, object], source).get(key)  # cast-ok: isinstance above, Mapping values are object
    return getattr(source, key, None)


def github_copilot_per_user_credential_name(litellm_params: object) -> str | None:
    """The credential name iff these litellm_params select per-user mode."""
    credential_name: Final[object] = _params_value(litellm_params, "litellm_credential_name")
    auth_type: Final[object] = _params_value(litellm_params, GITHUB_COPILOT_AUTH_TYPE_KEY)
    if not github_copilot_auth_mode(credential_name, auth_type):
        return None
    return credential_name if isinstance(credential_name, str) else ""


def github_copilot_user_session_from(source: object) -> GithubCopilotUserSession | None:
    if source is None:
        return None
    candidate: Final = (
        cast(Mapping[object, object], source).get(  # cast-ok: value is Mapping-checked or dict-shaped at runtime
            GITHUB_COPILOT_USER_SESSION_KWARG_KEY
        )
        if isinstance(source, Mapping)
        else getattr(source, GITHUB_COPILOT_USER_SESSION_KWARG_KEY, None)
    )
    return candidate if isinstance(candidate, GithubCopilotUserSession) else None


def require_github_copilot_user_session(litellm_params: object) -> GithubCopilotUserSession | None:
    """The attached session, or None for shared mode. Per-user params with no
    attached session only reach here via paths that skipped the attach step, so
    they get the same not-connected 401 instead of the shared device login."""
    session: Final = github_copilot_user_session_from(litellm_params)
    if session is not None:
        return session
    credential_name: Final = github_copilot_per_user_credential_name(litellm_params)
    if credential_name is not None:
        raise _connect_error(credential_name)
    return None


def is_github_copilot_per_user_request(kwargs: Mapping[str, object]) -> bool:
    """True when this call carries a resolved per-user Copilot session, meaning it
    must bypass the shared response cache (cache keys ignore caller identity).
    The session kwarg is attached in ``utils.wrapper``/``wrapper_async`` after
    ``load_credentials_from_list``, before the cache lookup runs."""
    return isinstance(kwargs.get(GITHUB_COPILOT_USER_SESSION_KWARG_KEY), GithubCopilotUserSession)


def _without_authorization(headers: Mapping[str, object]) -> dict[str, object]:
    return {k: v for k, v in headers.items() if k.lower() != "authorization"}


def _str_keyed_mapping(value: object) -> Mapping[str, object] | None:
    return (
        cast("Mapping[str, object]", value)  # cast-ok: request mappings are str-keyed
        if isinstance(value, Mapping)
        else None
    )


def _strip_caller_authorization(
    kwargs: dict[str, object],  # mutable-ok: kwargs is the request mutation channel
) -> None:
    """Once a per-user session is attached, its token owns Authorization on the
    wire. Caller-supplied bearer headers would be re-merged by the HTTP handler
    after the transformations run, so they are rebuilt here without any
    Authorization key; the caller's mappings are left untouched."""
    for key in ("extra_headers", "headers"):
        value = kwargs.get(key)
        if isinstance(value, Mapping):
            stripped = _without_authorization(
                cast("Mapping[str, object]", value)  # cast-ok: Mapping-checked request header dict
            )
            if len(stripped) != len(
                cast("Mapping[object, object]", value)  # cast-ok: Mapping-checked request header dict
            ):
                kwargs[key] = stripped  # rebind-ok: kwargs is the request mutation channel
    optional_params: Final = kwargs.get("optional_params")
    if isinstance(optional_params, dict):
        eh: Final = _str_keyed_mapping(
            cast("dict[str, object]", optional_params).get(  # cast-ok: optional_params is a plain str-keyed dict
                "extra_headers"
            )
        )
        if eh is not None:
            kwargs["optional_params"] = {  # rebind-ok: kwargs is the request mutation channel
                **optional_params,
                "extra_headers": _without_authorization(eh),
            }
    litellm_params: Final = kwargs.get("litellm_params")
    if isinstance(litellm_params, dict):
        eh2: Final = _str_keyed_mapping(
            cast("dict[str, object]", litellm_params).get(  # cast-ok: litellm_params is a plain str-keyed dict
                "extra_headers"
            )
        )
        if eh2 is not None:
            kwargs["litellm_params"] = {  # rebind-ok: kwargs is the request mutation channel
                **litellm_params,
                "extra_headers": _without_authorization(eh2),
            }


def attach_github_copilot_user_session(
    kwargs: dict[str, object],  # mutable-ok: kwargs is the request mutation channel
) -> None:
    credential_name_obj: Final[object] = kwargs.get("litellm_credential_name")
    auth_type_obj: Final[object] = kwargs.get(GITHUB_COPILOT_AUTH_TYPE_KEY)
    if not github_copilot_auth_mode(credential_name_obj, auth_type_obj):
        return
    if isinstance(kwargs.get(GITHUB_COPILOT_USER_SESSION_KWARG_KEY), GithubCopilotUserSession):
        return
    credential_name: Final = credential_name_obj if isinstance(credential_name_obj, str) else ""
    user_id, github_token = _caller_connection(kwargs, credential_name)
    session: Final = exchange_github_token(
        user_id=user_id,
        github_token=github_token,
        credential_name=credential_name,
    )
    kwargs[GITHUB_COPILOT_USER_SESSION_KWARG_KEY] = session  # rebind-ok: kwargs is the request mutation channel
    _strip_caller_authorization(kwargs)


async def aattach_github_copilot_user_session(
    kwargs: dict[str, object],  # mutable-ok: kwargs is the request mutation channel
) -> None:
    credential_name_obj: Final[object] = kwargs.get("litellm_credential_name")
    auth_type_obj: Final[object] = kwargs.get(GITHUB_COPILOT_AUTH_TYPE_KEY)
    if not github_copilot_auth_mode(credential_name_obj, auth_type_obj):
        return
    if isinstance(kwargs.get(GITHUB_COPILOT_USER_SESSION_KWARG_KEY), GithubCopilotUserSession):
        return
    credential_name: Final = credential_name_obj if isinstance(credential_name_obj, str) else ""
    user_id, github_token = _caller_connection(kwargs, credential_name)
    session: Final = await aexchange_github_token(
        user_id=user_id,
        github_token=github_token,
        credential_name=credential_name,
    )
    kwargs[GITHUB_COPILOT_USER_SESSION_KWARG_KEY] = session  # rebind-ok: kwargs is the request mutation channel
    _strip_caller_authorization(kwargs)

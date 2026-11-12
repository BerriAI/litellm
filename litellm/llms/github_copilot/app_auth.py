import asyncio
import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final, NoReturn, cast  # noqa: TID251  # runtime Mapping keys and values are objects

import httpx
from pydantic import AwareDatetime, BaseModel, ConfigDict, SecretStr, ValidationError

from litellm.exceptions import AuthenticationError, BadRequestError, RateLimitError, ServiceUnavailableError
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.secret_managers.main import get_secret_str

from .github_host import github_host_endpoints, normalized_github_host
from .per_user_auth import GithubCopilotUserSession

APP_AUTH_TYPE: Final = "github_app"
APP_FIELDS: Final = (
    "github_copilot_app_id",
    "github_copilot_app_installation_id",
    "github_copilot_app_private_key_ref",
)
_REFRESH_MARGIN: Final = 120
_CACHE_SIZE: Final = 128


@dataclass(frozen=True, slots=True)
class AppConfig:
    app_id: str
    installation_id: int
    private_key_ref: str = field(repr=False)
    host: str = "github.com"

    @property
    def token_url(self) -> str:
        api_host: Final = "api.github.com" if self.host == "github.com" else f"api.{self.host}"
        return f"https://{api_host}/app/installations/{self.installation_id}/access_tokens"

    @property
    def api_base(self) -> str:
        return github_host_endpoints(self.host).default_copilot_api_base


def _value(source: object, key: str) -> object:
    if isinstance(source, Mapping):
        return cast(Mapping[object, object], source).get(key)  # cast-ok: Mapping values are objects
    return getattr(source, key, None)


def app_config(source: object) -> AppConfig | None:
    name: Final = _value(source, "litellm_credential_name")
    credential: Final = CredentialAccessor.find_credential(name) if isinstance(name, str) else None
    values: Final = credential.credential_values if credential is not None else source
    mode: Final = _value(values, "github_copilot_auth_type")
    configured: Final = any(_value(values, key) is not None for key in APP_FIELDS)
    if mode != APP_AUTH_TYPE and not configured:
        return None
    app_id: Final = _value(values, APP_FIELDS[0])
    installation: Final = _value(values, APP_FIELDS[1])
    key_ref: Final = _value(values, APP_FIELDS[2])
    host: Final = normalized_github_host(_value(values, "github_copilot_host"))
    installation_text: Final = str(installation) if isinstance(installation, (str, int)) else ""
    if (
        mode != APP_AUTH_TYPE
        or not isinstance(app_id, str)
        or not app_id.strip()
        or not installation_text.isascii()
        or not installation_text.isdecimal()
        or int(installation_text) <= 0
        or not isinstance(key_ref, str)
        or not key_ref.strip()
        or "-----BEGIN" in key_ref
        or any(c.isspace() for c in key_ref)
        or (key_ref.startswith("oidc/") and not key_ref.startswith("oidc/file/"))
        or host is None
    ):
        raise BadRequestError(
            message="GitHub App authentication requires github_app auth type, an App/client ID, a positive "
            "installation ID, a private-key secret reference, and a supported GitHub host.",
            llm_provider="github_copilot",
            model="",
        )
    return AppConfig(app_id.strip(), int(installation_text), key_ref, host)


class _InstallationToken(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    token: SecretStr
    expires_at: AwareDatetime
    permissions: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _Token:
    value: str = field(repr=False)
    expires_at: float
    deadline: float


@dataclass(frozen=True, slots=True)
class _Failure:
    message: str
    status: int
    deadline: float


def _raise_failure(failure: _Failure, now: float) -> NoReturn:
    if failure.status == 429:
        raise RateLimitError(
            message=failure.message,
            llm_provider="github_copilot",
            model="",
            headers={"retry-after": str(max(1, math.ceil(failure.deadline - now)))},
        )
    if failure.status in (401, 403, 404):
        raise AuthenticationError(message=failure.message, llm_provider="github_copilot", model="")
    raise ServiceUnavailableError(message=failure.message, llm_provider="github_copilot", model="")


def _new_client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(15, connect=5), follow_redirects=False)


class AppTokenManager:
    def __init__(
        self,
        client_factory: Callable[[], httpx.Client] = _new_client,
        secret_reader: Callable[[str], str | None] = get_secret_str,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client_factory = client_factory
        self._secret_reader = secret_reader
        self._clock = clock
        self._monotonic = monotonic
        self._cache: OrderedDict[str, _Token | _Failure] = OrderedDict()  # mutable-ok: locked token cache
        self._cache_lock = threading.Lock()
        self._locks = tuple(threading.Lock() for _ in range(64))

    def resolve(self, config: AppConfig) -> GithubCopilotUserSession:
        try:
            private_key: Final = self._secret_reader(config.private_key_ref)
        except Exception:  # noqa: BLE001  # secret backends can raise arbitrary errors containing credentials
            raise AuthenticationError(
                message="Unable to read the GitHub App private-key secret.", llm_provider="github_copilot", model=""
            ) from None
        if not private_key:
            raise AuthenticationError(
                message="GitHub App private-key secret is empty or unavailable.",
                llm_provider="github_copilot",
                model="",
            )
        cache_key: Final = hashlib.sha256(
            json.dumps(
                (config.host, config.app_id, config.installation_id, config.private_key_ref, private_key)
            ).encode()
        ).hexdigest()
        with self._locks[int(cache_key[:8], 16) % len(self._locks)]:
            cached: Final = self._get(cache_key)
            result: Final = cached if cached is not None else self._mint(config, private_key)
            if cached is None:
                self._put(cache_key, result)
            if isinstance(result, _Failure):
                _raise_failure(result, self._monotonic())
            return GithubCopilotUserSession(token=result.value, api_base=config.api_base, is_app=True)

    async def aresolve(self, config: AppConfig) -> GithubCopilotUserSession:
        return await asyncio.to_thread(self.resolve, config)

    def _get(self, key: str) -> _Token | _Failure | None:
        with self._cache_lock:
            cached: Final = self._cache.get(key)
            if cached is None:
                return None
            if cached.deadline <= self._monotonic() or (
                isinstance(cached, _Token) and cached.expires_at - _REFRESH_MARGIN <= self._clock()
            ):
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return cached

    def _put(self, key: str, value: _Token | _Failure) -> None:
        with self._cache_lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > _CACHE_SIZE:
                self._cache.popitem(last=False)

    def _failure(self, message: str, status: int, cooldown: float = 5) -> _Failure:
        return _Failure(message, status, self._monotonic() + cooldown)

    def _rate_limit_cooldown(self, response: httpx.Response) -> float:
        retry_after: Final = next(iter(response.headers.get_list("retry-after")), "")
        reset: Final = next(iter(response.headers.get_list("x-ratelimit-reset")), "")
        try:
            if retry_after.isascii() and retry_after.isdecimal():
                return max(1, min(3600, float(retry_after)))
            if reset.isascii() and reset.isdecimal():
                return max(1, min(3600, float(reset) - self._clock()))
        except (ValueError, OverflowError):
            return 60
        return 60

    def _mint(self, config: AppConfig, private_key: str) -> _Token | _Failure:
        try:
            import jwt
        except ImportError:
            return self._failure("Install PyJWT and cryptography to use GitHub App authentication.", 401)
        now: Final = self._clock()
        started: Final = self._monotonic()
        try:
            assertion: Final = jwt.encode(
                {"iat": int(now) - 60, "exp": int(now) + 540, "iss": config.app_id},
                private_key,
                algorithm="RS256",
            )
        except (ValueError, TypeError, jwt.PyJWTError, NotImplementedError):
            return self._failure("Unable to sign a GitHub App JWT. Check the RSA private key.", 401)
        try:
            with self._client_factory() as client:
                response: Final = client.post(
                    config.token_url,
                    headers={
                        "Authorization": f"Bearer {assertion}",
                        "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2026-03-10",
                    },
                    json={"permissions": {"organization_copilot_requests": "read"}},
                    follow_redirects=False,
                )
        except (httpx.HTTPError, ValueError):
            return self._failure("GitHub App installation token request failed.", 503)
        if response.status_code == 429 or (
            response.status_code == 403
            and ("retry-after" in response.headers or response.headers.get("x-ratelimit-remaining") == "0")
        ):
            return self._failure(
                "GitHub rate limited installation token creation. Retry later.",
                429,
                self._rate_limit_cooldown(response),
            )
        if response.status_code != 201:
            return self._failure(
                f"GitHub App installation token request returned HTTP {response.status_code}. "
                "Check the App installation and Copilot Requests permission.",
                response.status_code,
            )
        try:
            payload: Final = _InstallationToken.model_validate_json(response.content)
        except ValidationError:
            return self._failure("GitHub returned an invalid installation token response.", 503)
        expires_at: Final = min(payload.expires_at.timestamp(), now + 3600)
        token: Final = payload.token.get_secret_value()
        if (
            not token
            or any(c.isspace() for c in token)
            or expires_at <= self._clock() + _REFRESH_MARGIN
            or payload.permissions.get("organization_copilot_requests") != "read"
        ):
            return self._failure("GitHub returned an expired token or missing Copilot Requests permission.", 401)
        return _Token(token, expires_at, started + expires_at - now - _REFRESH_MARGIN)


APP_TOKEN_MANAGER: Final = AppTokenManager()

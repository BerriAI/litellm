import asyncio
import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Final, NoReturn, cast  # noqa: TID251  # runtime Mapping keys and values are objects

import httpx
from pydantic import AwareDatetime, BaseModel, ConfigDict, SecretStr, ValidationError

from litellm.exceptions import (
    BadRequestError,
    DeploymentCredentialAuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
)
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.llms.custom_httpx.http_handler import get_ssl_configuration
from litellm.secret_managers.main import get_secret_str

from .github_host import github_host_endpoints, normalized_github_host
from .per_user_auth import GithubCopilotUserSession

APP_AUTH_TYPE: Final = "github_app"
APP_PRIVATE_KEY: Final = "github_copilot_app_private_key"
APP_FIELDS: Final = (
    "github_copilot_app_id",
    "github_copilot_app_installation_id",
    "github_copilot_app_private_key_ref",
    APP_PRIVATE_KEY,
)
_REFRESH_MARGIN: Final = 120
_CACHE_SIZE: Final = 128


@dataclass(frozen=True, slots=True)
class AppConfig:
    app_id: str
    installation_id: int
    private_key_ref: str = field(repr=False)
    host: str = "github.com"
    credential_name: str | None = field(default=None, repr=False)

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


def app_config(source: object, *, validate_stored_key: bool = False) -> AppConfig | None:
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
    stored_key: Final = _value(values, APP_PRIVATE_KEY)
    stored_key_selected: Final = isinstance(stored_key, str) and bool(stored_key)
    valid_key_source: Final = (
        stored_key_selected and not key_ref and (credential is not None or validate_stored_key)
    ) or (
        stored_key is None
        and isinstance(key_ref, str)
        and bool(key_ref.strip())
        and "-----BEGIN" not in key_ref
        and not any(c.isspace() for c in key_ref)
        and (not key_ref.startswith("oidc/") or key_ref.startswith("oidc/file/"))
    )
    host: Final = normalized_github_host(_value(values, "github_copilot_host"))
    app_id_text: Final = _id_text(app_id).strip()
    installation_text: Final = _id_text(installation)
    if (
        mode != APP_AUTH_TYPE
        or not app_id_text
        or not installation_text.isascii()
        or not installation_text.isdecimal()
        or int(installation_text) <= 0
        or not valid_key_source
        or host is None
    ):
        raise BadRequestError(
            message="GitHub App authentication requires github_app auth type, an App/client ID, a positive "
            "installation ID, a stored credential private key or secret reference, and a supported GitHub host.",
            llm_provider="github_copilot",
            model="",
        )
    if stored_key_selected and validate_stored_key:
        validate_private_key(stored_key)
    return AppConfig(
        app_id_text,
        int(installation_text),
        key_ref if isinstance(key_ref, str) else "",
        host,
        credential.credential_name if stored_key_selected and credential is not None else None,
    )


def _id_text(value: object) -> str:
    if isinstance(value, bool):
        return ""
    return str(value) if isinstance(value, (str, int)) else ""


def validate_private_key(value: object) -> None:
    """Validate uploaded keys without echoing their contents in errors."""
    from cryptography.exceptions import UnsupportedAlgorithm
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    try:
        if not isinstance(value, str) or len(value) > 16384:
            raise ValueError("Invalid key size")
        key: Final = load_pem_private_key(value.encode(), password=None)
        if not isinstance(key, RSAPrivateKey) or key.key_size < 2048:
            raise ValueError("Unsupported key")
    except (ValueError, TypeError, UnsupportedAlgorithm):
        raise BadRequestError(
            message="Upload an unencrypted RSA PEM private key of at least 2048 bits, no larger than 16 KiB.",
            llm_provider="github_copilot",
            model="",
        ) from None


def _stored_private_key(name: str) -> str | None:
    credential: Final = CredentialAccessor.find_credential(name)
    value: Final = credential.credential_values.get(APP_PRIVATE_KEY) if credential is not None else None
    return value if isinstance(value, str) else None


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
        raise DeploymentCredentialAuthenticationError(message=failure.message, llm_provider="github_copilot", model="")
    if failure.status == 422:
        raise BadRequestError(message=failure.message, llm_provider="github_copilot", model="")
    raise ServiceUnavailableError(message=failure.message, llm_provider="github_copilot", model="")


def _session(config: AppConfig, result: _Token | _Failure, now: float) -> GithubCopilotUserSession:
    if isinstance(result, _Failure):
        _raise_failure(result, now)
    return GithubCopilotUserSession(token=result.value, api_base=config.api_base, is_app=True)


def _new_client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(15, connect=5), follow_redirects=False, verify=get_ssl_configuration())


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
        self._in_flight: dict[str, Future[_Token | _Failure]] = {}  # mutable-ok: locked single-flight registry
        self._cache_lock = threading.Lock()

    def resolve(self, config: AppConfig) -> GithubCopilotUserSession:
        key: Final = self._cache_key(config)
        cached: Final = self._get(key)
        if cached is not None:
            return _session(config, cached, self._monotonic())
        future, owner = self._claim(key)
        if owner:
            self._fill(key, config, future)
        return _session(config, future.result(), self._monotonic())

    async def aresolve(self, config: AppConfig) -> GithubCopilotUserSession:
        key: Final = self._cache_key(config)
        cached: Final = self._get(key)
        if cached is not None:
            return _session(config, cached, self._monotonic())
        future, owner = self._claim(key)
        if owner:
            _ = asyncio.get_running_loop().run_in_executor(None, self._fill, key, config, future)
        return _session(config, await asyncio.wrap_future(future), self._monotonic())

    def _cache_key(self, config: AppConfig) -> str:
        stored_key: Final = _stored_private_key(config.credential_name) if config.credential_name is not None else None
        return hashlib.sha256(
            json.dumps(
                (
                    config.host,
                    config.app_id,
                    config.installation_id,
                    config.private_key_ref,
                    config.credential_name,
                    stored_key,
                )
            ).encode()
        ).hexdigest()

    def _claim(self, key: str) -> tuple[Future[_Token | _Failure], bool]:
        with self._cache_lock:
            existing: Final = self._in_flight.get(key)
            if existing is not None:
                return existing, False
            future: Final = Future[_Token | _Failure]()
            _ = future.set_running_or_notify_cancel()
            self._in_flight[key] = future
            return future, True

    def _fill(self, key: str, config: AppConfig, future: Future[_Token | _Failure]) -> None:
        try:
            cached: Final = self._get(key)
            result: Final = cached if cached is not None else self._fetch(config)
        except BaseException as error:  # noqa: BLE001  # every waiter receives the error through the future
            self._settle(key, None)
            future.set_exception(error)
            return
        self._settle(key, result)
        future.set_result(result)

    def _fetch(self, config: AppConfig) -> _Token | _Failure:
        try:
            private_key: Final = (
                _stored_private_key(config.credential_name)
                if config.credential_name is not None
                else self._secret_reader(config.private_key_ref)
            )
        except Exception:  # noqa: BLE001  # secret backends can raise arbitrary errors containing credentials
            return self._failure("Unable to read the GitHub App private-key secret.", 401, cooldown=0)
        if not private_key:
            return self._failure("GitHub App private-key secret is empty or unavailable.", 401, cooldown=0)
        return self._mint(config, private_key)

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

    def _settle(self, key: str, value: _Token | _Failure | None) -> None:
        with self._cache_lock:
            _ = self._in_flight.pop(key, None)
            if value is None:
                return
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > _CACHE_SIZE:
                _ = self._cache.popitem(last=False)

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

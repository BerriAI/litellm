from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Final

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import HTTPHandler, _get_httpx_client

AUTH_ENDPOINT_SUFFIX: Final = "/oauth/token"

CONFIG_FILE_ENV_VAR: Final = "AICORE_CONFIG"
HOME_PATH_ENV_VAR: Final = "AICORE_HOME"
PROFILE_ENV_VAR: Final = "AICORE_PROFILE"

VCAP_SERVICES_ENV_VAR: Final = "VCAP_SERVICES"
VCAP_AICORE_SERVICE_NAME: Final = "aicore"
SERVICE_KEY_ENV_VAR: Final = "AICORE_SERVICE_KEY"

DEFAULT_HOME_PATH: Final = os.path.join(os.path.expanduser("~"), ".aicore")


def _get_home() -> str:
    return os.getenv(HOME_PATH_ENV_VAR, DEFAULT_HOME_PATH)


def _get_nested(d: dict[str, Any] | str, path: Sequence[str]) -> Any:
    cur: Any = d
    if isinstance(cur, str):
        # This shouldn't happen if service keys are pre-parsed correctly
        try:
            cur = json.loads(cur)
        except json.JSONDecodeError:
            verbose_logger.warning("SAP service key or VCAP service is a string but not valid JSON.")
            return None
    for k in path:
        if not isinstance(cur, dict):
            verbose_logger.warning(
                "SAP service key or VCAP service traversal hit non-dict type '%s' at key '%s'.", type(cur).__name__, k
            )
            return None
        if k not in cur:
            return None
        cur = cur[k]
    return cur


def _load_json_env(var_name: str) -> dict[str, Any] | None:
    raw: Final = os.environ.get(var_name)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _str_or_none(value) -> str | None:
    try:
        return str(value) if value is not None else None
    except Exception:
        return None


def _load_vcap() -> dict[str, Any]:
    return _load_json_env(VCAP_SERVICES_ENV_VAR) or {}


def _get_vcap_service(label: str) -> dict[str, Any] | None:
    for services in _load_vcap().values():
        for svc in services:
            if svc.get("label") == label:
                return svc
    return None


@dataclass
class Source:
    name: str
    get: Callable[[CredentialsValue], str | None]


@dataclass(frozen=True)
class CredentialsValue:
    name: str
    vcap_key: tuple[str, ...] | None = None
    default: str | None = None
    transform_fn: Callable[[str], str] | None = None


CREDENTIAL_VALUES: Final[list[CredentialsValue]] = [
    CredentialsValue("client_id", ("clientid",)),
    CredentialsValue("client_secret", ("clientsecret",)),
    CredentialsValue(
        "auth_url",
        ("url",),
        transform_fn=lambda url: url.rstrip("/") + ("" if url.endswith(AUTH_ENDPOINT_SUFFIX) else AUTH_ENDPOINT_SUFFIX),
    ),
    CredentialsValue(
        "base_url",
        ("serviceurls", "AI_API_URL"),
        transform_fn=lambda url: url.rstrip("/") + ("" if url.endswith("/v2") else "/v2"),
    ),
    CredentialsValue(
        "cert_url",
        ("certurl",),
        transform_fn=lambda url: url.rstrip("/") + ("" if url.endswith(AUTH_ENDPOINT_SUFFIX) else AUTH_ENDPOINT_SUFFIX),
    ),
    # file paths (kept for config compatibility)
    CredentialsValue("cert_file_path"),
    CredentialsValue("key_file_path"),
    # inline PEMs from VCAP
    CredentialsValue("cert_str", ("certificate",), transform_fn=lambda s: s.replace("\\n", "\n")),
    CredentialsValue("key_str", ("key",), transform_fn=lambda s: s.replace("\\n", "\n")),
]


def init_conf(profile: str | None = None) -> dict[str, Any]:
    """
    Loads config JSON from:
      1) $AICORE_CONFIG if set, otherwise
      2) $AICORE_HOME/config.json (or config_<profile>.json when profile is given/not default)
    Returns {} when nothing is found.
    """
    home: Final = Path(_get_home())
    profile = profile or os.environ.get(PROFILE_ENV_VAR)
    cfg_env: Final = os.getenv(CONFIG_FILE_ENV_VAR)
    cfg_path: Final = (
        Path(cfg_env)
        if cfg_env
        else (home / ("config.json" if profile in (None, "", "default") else f"config_{profile}.json"))
    )

    if cfg_path and cfg_path.exists():
        try:
            with cfg_path.open(encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise KeyError(f"{cfg_path} is not valid JSON. Please fix or remove it!")

    # If an explicit non-default profile was requested but not found, raise.
    if cfg_env or (profile not in (None, "", "default")):
        raise FileNotFoundError(f"Unable to locate profile config file at '{cfg_path}' in AICORE_HOME '{home}'")

    return {}


def _env_name(name: str) -> str:
    return f"AICORE_{name.upper()}"


def extract_credentials(source: Source) -> dict[str, str]:
    """Extract all credentials from a source."""
    credentials: Final = {}
    for cv in CREDENTIAL_VALUES:
        value = source.get(cv)
        if value is not None:
            credentials[cv.name] = cv.transform_fn(value) if cv.transform_fn else value
    return credentials


def resolve_credentials(sources: list[Source]) -> dict[str, str]:
    """Extract credentials from the first source that has any defined."""
    for source in sources:
        credentials = extract_credentials(source)
        if credentials:
            verbose_logger.debug("Resolved SAP credentials from source %s", source.name)
            return credentials
    raise ValueError("No credentials found in any source")


def resolve_resource_group(sources: list[Source]) -> str | None:
    """Find resource_group from the first source that defines it."""
    rg_cred: Final = CredentialsValue("resource_group", default="default")
    for source in sources:
        value = source.get(rg_cred)
        if value is not None:
            verbose_logger.debug("Resolved GEN AI Hub resource_group from source %s", source.name)
            return value
    return rg_cred.default


def _parse_service_key_once(
    service_key: str | dict | None,
) -> dict[str, Any] | None:
    """
    Pre-parse service_key if it's a string to avoid repeated JSON parsing.

    Returns None if parsing fails (other credential sources may still work).
    """
    if service_key is None:
        return None
    if isinstance(service_key, dict):
        return service_key
    if isinstance(service_key, str):
        try:
            return json.loads(service_key)
        except json.JSONDecodeError:
            verbose_logger.warning("SAP service key is a string but not valid JSON. Skipping this source.")
            return None
    verbose_logger.warning(
        "SAP service key has unexpected type '%s'. Expected str or dict. Ignoring.", type(service_key).__name__
    )
    return None


def _resolve_credential_from_service_key(service_key: str | dict | None, cv: CredentialsValue) -> str | None:
    if service_key is None:
        return None
    val: Final = _str_or_none(_get_nested(service_key, (("credentials",) + cv.vcap_key) if cv.vcap_key else (cv.name,)))
    if val is None:
        return _str_or_none(_get_nested(service_key, cv.vcap_key if cv.vcap_key else (cv.name,)))
    return val


def fetch_credentials(
    service_key: str | dict | None = None,
    profile: str | None = None,
    **kwargs,
) -> dict[str, str]:
    """
    Resolution order (first-source-wins):

    Sources are checked in this order:
      kwargs
      > service key
      > env (AICORE_<NAME>)
      > config (AICORE_<NAME> or plain <name>)
      > vcap service key
      > default

    Important:
      - Credentials are extracted from the FIRST source that provides any credential value.
      - Values are NOT merged per key across sources. Except resource_group, which is merged.

    Warning:
      - This function does NOT validate the returned credentials just parsed it from the sources.
      - Callers MUST explicitly call validate_credentials() on the returned dict
    """
    config: Final = init_conf(profile)

    service_key = _parse_service_key_once(service_key or litellm.sap_service_key or os.environ.get(SERVICE_KEY_ENV_VAR))
    vcap_service: Final = _get_vcap_service(VCAP_AICORE_SERVICE_NAME)

    sources: Final = [
        Source("kwargs", lambda cv: _str_or_none(kwargs.get(cv.name))),
        Source(
            "service key",
            lambda cv: _resolve_credential_from_service_key(service_key, cv),
        ),
        Source(
            "environment variables",
            lambda cv: _str_or_none(os.environ.get(f"AICORE_{cv.name.upper()}")),
        ),
        Source(
            "config file",
            lambda cv: _str_or_none(
                config.get(f"AICORE_{cv.name.upper()}")
                if config.get(f"AICORE_{cv.name.upper()}") is not None
                else config.get(cv.name)
            ),
        ),
        Source(
            "VCAP service",
            lambda cv: (
                _str_or_none(
                    _get_nested(
                        vcap_service,
                        (("credentials",) + cv.vcap_key) if cv.vcap_key else (cv.name,),
                    )
                )
                if vcap_service
                else None
            ),
        ),
    ]

    credentials: Final = resolve_credentials(sources)

    resource_group: Final = resolve_resource_group(sources)
    if resource_group is not None:
        credentials["resource_group"] = resource_group

    if "cert_url" in credentials:
        credentials["auth_url"] = credentials.pop("cert_url")
    return credentials


def validate_credentials(
    auth_url: str | None = None,
    base_url: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    cert_str: str | None = None,
    key_str: str | None = None,
    cert_file_path: str | None = None,
    key_file_path: str | None = None,
) -> None:
    """
    Validate SAP AI Core credentials for completeness and consistency.

    Args:
        auth_url: OAuth2 token endpoint URL (required)
        base_url: SAP AI Core API base URL (required)
        client_id: OAuth2 client ID (required)
        client_secret: OAuth2 client secret (for secret-based auth)
        cert_str: PEM-encoded certificate string (for cert-based auth)
        key_str: PEM-encoded private key string (for cert-based auth)
        cert_file_path: Path to certificate file (for file-based cert auth)
        key_file_path: Path to private key file (for file-based cert auth)

    Raises:
        ValueError: If required fields are missing or authentication mode is ambiguous.

    Note:
        - This function does NOT validate resource_group (resolved separately).
        - Exactly one authentication method must be provided:
          * client_secret, OR
          * (cert_str AND key_str), OR
          * (cert_file_path AND key_file_path)
    """
    if not auth_url or not client_id or not base_url:
        raise ValueError(
            "SAP AI Core credentials not found. "
            "Please provide credentials by setting appropriate environment variables "
            "(e.g. AICORE_CLIENT_ID, AICORE_CLIENT_SECRET, etc.)"
        )

    modes: Final = [
        bool(client_secret),
        bool(cert_str) and bool(key_str),
        bool(cert_file_path) and bool(key_file_path),
    ]
    if sum(bool(m) for m in modes) != 1:
        raise ValueError(
            "SAP AI Core credentials are incomplete. "
            "Invalid credentials: provide exactly one of client_secret, "
            "(cert_str & key_str), or (cert_file_path & key_file_path)."
        )


def _request_token(
    client_id: str, auth_url: str, timeout: float, cert_pair=None, client_secret=None
) -> tuple[str, datetime]:
    data: Final = {"grant_type": "client_credentials", "client_id": client_id}
    if client_secret:
        data["client_secret"] = client_secret

    resp: httpx.Response | None = None
    try:
        if cert_pair:
            with httpx.Client(cert=cert_pair) as raw_client:
                handler = HTTPHandler(client=raw_client)
                resp = handler.post(auth_url, data=data, timeout=timeout)
                payload = resp.json()
        else:
            handler = _get_httpx_client()
            resp = handler.post(auth_url, data=data, timeout=timeout)
            payload = resp.json()
        access_token: Final = payload["access_token"]
        expires_in: Final = int(payload.get("expires_in", 3600))
        expiry_date: Final = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        return f"Bearer {access_token}", expiry_date
    except Exception as e:
        msg: Final = resp.text if resp is not None else getattr(e, "text", str(e))
        raise RuntimeError(f"Token request failed: {msg}") from e


def get_token_creator(
    service_key: str | dict | None = None,
    profile: str | None = None,
    *,
    timeout: float = 30.0,
    expiry_buffer_minutes: int = 60,
    **overrides,
) -> tuple[Callable[[], str], str, str]:
    """
    Creates a callable that fetches and caches an OAuth2 bearer token
    using credentials from `fetch_credentials()`.

    The callable:
      - Automatically loads credentials via fetch_credentials(profile, **overrides)
      - Fetches a new token only if expired or near expiry
      - Caches token thread-safely with a configurable refresh buffer

    Args:
        profile: Optional AICore profile name
        timeout: Timeout for HTTP requests
        expiry_buffer_minutes: Refresh the token this many minutes before expiry
        overrides: Any explicit credential overrides (client_id, client_secret, etc.)

    Returns:
        Callable[[], str]: function returning a valid "Bearer <token>" string.
    """

    # Resolve credentials using your helper
    credentials: Final[dict[str, str]] = fetch_credentials(service_key=service_key, profile=profile, **overrides)

    auth_url: Final = credentials.get("auth_url")
    base_url: Final = credentials.get("base_url")
    client_id: Final = credentials.get("client_id")
    client_secret: Final = credentials.get("client_secret")
    cert_str: Final = credentials.get("cert_str")
    key_str: Final = credentials.get("key_str")
    cert_file_path: Final = credentials.get("cert_file_path")
    key_file_path: Final = credentials.get("key_file_path")

    # Sanity check
    validate_credentials(
        auth_url,
        base_url,
        client_id,
        client_secret,
        cert_str,
        key_str,
        cert_file_path,
        key_file_path,
    )

    lock: Final = Lock()
    token: str | None = None
    token_expiry: datetime | None = None

    def _fetch_token() -> tuple[str, datetime]:
        # Case 1: secret-based auth
        if client_secret:
            return _request_token(
                auth_url=auth_url,
                client_id=client_id,
                timeout=timeout,
                client_secret=client_secret,
            )
        # Case 2: cert/key strings
        if cert_str and key_str:
            cert_str_fixed: Final = cert_str.replace("\\n", "\n")
            key_str_fixed: Final = key_str.replace("\\n", "\n")
            with tempfile.TemporaryDirectory() as tmp:
                cert_path: Final = os.path.join(tmp, "cert.pem")
                key_path: Final = os.path.join(tmp, "key.pem")
                with open(cert_path, "w") as f:
                    f.write(cert_str_fixed)
                with open(key_path, "w") as f:
                    f.write(key_str_fixed)
                return _request_token(
                    auth_url=auth_url,
                    client_id=client_id,
                    timeout=timeout,
                    cert_pair=(cert_path, key_path),
                )
        # Case 3: file-based cert/key
        if cert_file_path is not None and key_file_path is not None:
            return _request_token(
                auth_url=auth_url,
                client_id=client_id,
                timeout=timeout,
                cert_pair=(cert_file_path, key_file_path),
            )
        # Defensive guard: should never reach here due to validate_credentials()
        raise ValueError("Invalid authentication configuration: no valid credentials found. ")

    def get_token() -> str:
        nonlocal token, token_expiry
        with lock:
            now: Final = datetime.now(timezone.utc)
            if token is None or token_expiry is None or token_expiry - now < timedelta(minutes=expiry_buffer_minutes):
                token, token_expiry = _fetch_token()
            return token

    return get_token, credentials["base_url"], credentials["resource_group"]

from __future__ import annotations
from typing import Any, Callable, Dict, Final, List, Optional, Sequence, Tuple
from datetime import datetime, timedelta, timezone
from threading import Lock
from pathlib import Path
from dataclasses import dataclass
import json
import os
import tempfile

from litellm import sap_service_key
from litellm.llms.custom_httpx.http_handler import _get_httpx_client

AUTH_ENDPOINT_SUFFIX = "/oauth/token"

CONFIG_FILE_ENV_VAR = "AICORE_CONFIG"
HOME_PATH_ENV_VAR = "AICORE_HOME"
PROFILE_ENV_VAR = "AICORE_PROFILE"

VCAP_SERVICES_ENV_VAR = "VCAP_SERVICES"
VCAP_AICORE_SERVICE_NAME = "aicore"
SERVICE_KEY_ENV_VAR = "AICORE_SERVICE_KEY"

DEFAULT_HOME_PATH = os.path.join(os.path.expanduser("~"), ".aicore")


def _get_home() -> str:
    return os.getenv(HOME_PATH_ENV_VAR, DEFAULT_HOME_PATH)


def _get_nested(d: Dict[str, Any], path: Sequence[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(".".join(path))
        cur = cur[k]
    return cur


def _load_json_env(var_name: str) -> Optional[Dict[str, Any]]:
    raw = os.environ.get(var_name)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _load_vcap() -> Dict[str, Any]:
    return _load_json_env(VCAP_SERVICES_ENV_VAR) or {}


def _get_vcap_service(label: str) -> Optional[Dict[str, Any]]:
    for services in _load_vcap().values():
        for svc in services:
            if svc.get("label") == label:
                return svc
    return None


@dataclass(frozen=True)
class CredentialsValue:
    name: str
    vcap_key: Optional[Tuple[str, ...]] = None
    default: Optional[str] = None
    transform_fn: Optional[Callable[[str], str]] = None


CREDENTIAL_VALUES: Final[List[CredentialsValue]] = [
    CredentialsValue("client_id", ("clientid",)),
    CredentialsValue("client_secret", ("clientsecret",)),
    CredentialsValue(
        "auth_url",
        ("url",),
        transform_fn=lambda url: url.rstrip("/")
        + ("" if url.endswith(AUTH_ENDPOINT_SUFFIX) else AUTH_ENDPOINT_SUFFIX),
    ),
    CredentialsValue(
        "base_url",
        ("serviceurls", "AI_API_URL"),
        transform_fn=lambda url: url.rstrip("/")
        + ("" if url.endswith("/v2") else "/v2"),
    ),
    CredentialsValue("resource_group", default="default"),
    CredentialsValue(
        "cert_url",
        ("certurl",),
        transform_fn=lambda url: url.rstrip("/")
        + ("" if url.endswith(AUTH_ENDPOINT_SUFFIX) else AUTH_ENDPOINT_SUFFIX),
    ),
    # file paths (kept for config compatibility)
    CredentialsValue("cert_file_path"),
    CredentialsValue("key_file_path"),
    # inline PEMs from VCAP
    CredentialsValue(
        "cert_str", ("certificate",), transform_fn=lambda s: s.replace("\\n", "\n")
    ),
    CredentialsValue(
        "key_str", ("key",), transform_fn=lambda s: s.replace("\\n", "\n")
    ),
]


def init_conf(profile: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads config JSON from:
      1) $AICORE_CONFIG if set, otherwise
      2) $AICORE_HOME/config.json (or config_<profile>.json when profile is given/not default)
    Returns {} when nothing is found.
    """
    home = Path(_get_home())
    profile = profile or os.environ.get(PROFILE_ENV_VAR)
    cfg_env = os.getenv(CONFIG_FILE_ENV_VAR)
    cfg_path = (
        Path(cfg_env)
        if cfg_env
        else (
            home
            / (
                "config.json"
                if profile in (None, "", "default")
                else f"config_{profile}.json"
            )
        )
    )

    if cfg_path and cfg_path.exists():
        try:
            with cfg_path.open(encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise KeyError(f"{cfg_path} is not valid JSON. Please fix or remove it!")

    # If an explicit non-default profile was requested but not found, raise.
    if cfg_env or (profile not in (None, "", "default")):
        raise FileNotFoundError(
            f"Unable to locate profile config file at '{cfg_path}' in AICORE_HOME '{home}'"
        )

    return {}


def _env_name(name: str) -> str:
    return f"AICORE_{name.upper()}"


def _resolve_value(
    cred: CredentialsValue,
    *,
    kwargs: Dict[str, Any],
    env: Dict[str, str],
    config: Dict[str, Any],
    service_like: Optional[Dict[str, Any]],
) -> Optional[str]:
    # 1) explicit kwargs
    if cred.name in kwargs and kwargs[cred.name] is not None:
        return kwargs[cred.name]

    # 2) environment variables (primary name)
    env_key = _env_name(cred.name)
    if env_key in env and env[env_key] is not None:
        return env[env_key]

    # 3) config file (accept both prefixed and plain keys)
    for key in (env_key, cred.name):
        if key in config and config[key] is not None:
            return config[key]

    # 4) service-like source (AICORE_SERVICE_KEY first, else VCAP)
    if service_like and cred.vcap_key:
        try:
            val = _get_nested(service_like, ("credentials",) + cred.vcap_key)
            if val is not None:
                return val
        except KeyError:
            pass

    # 5) default
    return cred.default


def fetch_credentials(service_key: Optional[str] = None, profile: Optional[str] = None, **kwargs) -> Dict[str, str]:
    """
    Resolution order per key:
      kwargs
      > env (AICORE_<NAME>)
      > config (AICORE_<NAME> or plain <name>)
      > service-like source from JSON in $AICORE_SERVICE_KEY (same structure as a VCAP service object)
        falling back to service entry in $VCAP_SERVICES with label 'aicore'
      > default
    """
    config = init_conf(profile)
    env = os.environ  # snapshot for testability
    service_like = None

    if not config:
        # Prefer AICORE_SERVICE_KEY if present; otherwise fall back to the VCAP service.
        service_like = service_key or sap_service_key or _load_json_env(SERVICE_KEY_ENV_VAR) or _get_vcap_service(
            VCAP_AICORE_SERVICE_NAME
        )

    out: Dict[str, str] = {}
    for cred in CREDENTIAL_VALUES:
        value = _resolve_value(cred, kwargs=kwargs, env=env, config=config, service_like=service_like)  # type: ignore
        if value is None:
            continue
        if cred.transform_fn:
            value = cred.transform_fn(value)
        out[cred.name] = value
    if "cert_url" in out.keys():
        out["auth_url"] = out.pop("cert_url")
    return out


def get_token_creator(
    service_key: Optional[str] = None,
    profile: Optional[str] = None,
    *,
    timeout: float = 30.0,
    expiry_buffer_minutes: int = 60,
    **overrides,
) -> Tuple[Callable[[], str], str, str]:
    """
    Creates a callable that fetches and caches an OAuth2 bearer token
    using credentials from `fetch_credentials()`.

    The callable:
      - Automatically loads credentials via fetch_credentials(profile, **overrides)
      - Fetches a new token only if expired or near expiry
      - Caches token thread-safely with a configurable refresh buffer

    Args:
        profile: Optional AICore profile name
        timeout: HTTP request timeout in seconds (default 30s)
        expiry_buffer_minutes: Refresh the token this many minutes before expiry
        overrides: Any explicit credential overrides (client_id, client_secret, etc.)

    Returns:
        Callable[[], str]: function returning a valid "Bearer <token>" string.
    """

    # Resolve credentials using your helper
    credentials: Dict[str, str] = fetch_credentials(service_key=service_key, profile=profile, **overrides)

    auth_url = credentials.get("auth_url")
    client_id = credentials.get("client_id")
    client_secret = credentials.get("client_secret")
    cert_str = credentials.get("cert_str")
    key_str = credentials.get("key_str")
    cert_file_path = credentials.get("cert_file_path")
    key_file_path = credentials.get("key_file_path")

    # Sanity check
    if not auth_url or not client_id:
        raise ValueError(
            "fetch_credentials did not return valid 'auth_url' or 'client_id'"
        )

    modes = [
        client_secret is not None,
        (cert_str is not None and key_str is not None),
        (cert_file_path is not None and key_file_path is not None),
    ]
    if sum(bool(m) for m in modes) != 1:
        raise ValueError(
            "Invalid credentials: provide exactly one of client_secret, "
            "(cert_str & key_str), or (cert_file_path & key_file_path)."
        )

    lock = Lock()
    token: Optional[str] = None
    token_expiry: Optional[datetime] = None

    def _request_token(cert_pair=None) -> tuple[str, datetime]:
        data = {"grant_type": "client_credentials", "client_id": client_id}
        if client_secret:
            data["client_secret"] = client_secret

        client = _get_httpx_client()
        # with httpx.Client(cert=cert_pair, timeout=timeout) as client:
        resp = client.post(auth_url, data=data)
        try:
            resp.raise_for_status()
            payload = resp.json()
            access_token = payload["access_token"]
            expires_in = int(payload.get("expires_in", 3600))
            expiry_date = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
            return f"Bearer {access_token}", expiry_date
        except Exception as e:
            msg = getattr(resp, "text", str(e))
            raise RuntimeError(f"Token request failed: {msg}") from e

    def _fetch_token() -> tuple[str, datetime]:
        # Case 1: secret-based auth
        if client_secret:
            return _request_token()
        # Case 2: cert/key strings
        if cert_str and key_str:
            cert_str_fixed = cert_str.replace("\\n", "\n")
            key_str_fixed = key_str.replace("\\n", "\n")
            with tempfile.TemporaryDirectory() as tmp:
                cert_path = os.path.join(tmp, "cert.pem")
                key_path = os.path.join(tmp, "key.pem")
                with open(cert_path, "w") as f:
                    f.write(cert_str_fixed)
                with open(key_path, "w") as f:
                    f.write(key_str_fixed)
                return _request_token(cert_pair=(cert_path, key_path))
        # Case 3: file-based cert/key
        return _request_token(cert_pair=(cert_file_path, key_file_path))

    def get_token() -> str:
        nonlocal token, token_expiry
        with lock:
            now = datetime.now(timezone.utc)
            if (
                token is None
                or token_expiry is None
                or token_expiry - now < timedelta(minutes=expiry_buffer_minutes)
            ):
                token, token_expiry = _fetch_token()
            return token

    return get_token, credentials["base_url"], credentials["resource_group"]

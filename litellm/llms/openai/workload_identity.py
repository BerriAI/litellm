from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Final
from urllib.parse import urlparse

import httpx
from openai import NOT_GIVEN, AsyncOpenAI, NotGiven, OpenAI

import litellm
from litellm.constants import DEFAULT_MAX_RETRIES
from litellm.litellm_core_utils.get_litellm_params import OPENAI_WIF_KWARGS_KEYS
from litellm.secret_managers.main import get_secret_str, normalize_nonempty_secret_str

from .common_utils import BaseOpenAILLM, OpenAIError, is_openai_backed_api_base

if TYPE_CHECKING:
    from collections.abc import Callable

    from openai.auth import SubjectTokenProvider, WorkloadIdentity, WorkloadIdentityAuth

OPENAI_WIF_CLIENT_ID: Final = "litellm"
_SDK_UPGRADE_MESSAGE: Final = (
    "OpenAI workload identity federation requires openai>=2.32.0. "
    "Upgrade the installed openai package to use OPENAI_IDENTITY_PROVIDER_ID / "
    "OPENAI_SERVICE_ACCOUNT_ID / OPENAI_IDENTITY_TOKEN_FILE."
)


@dataclass(frozen=True, slots=True)
class OpenAIWorkloadIdentityConfig:
    identity_provider_id: str
    service_account_id: str
    token_file: str

    def to_sdk_workload_identity(self) -> WorkloadIdentity:
        k8s_token_provider: Final = _load_sdk_k8s_token_provider()
        workload_identity: Final[WorkloadIdentity] = {
            "client_id": OPENAI_WIF_CLIENT_ID,
            "identity_provider_id": self.identity_provider_id,
            "service_account_id": self.service_account_id,
            "provider": k8s_token_provider(self.token_file),
        }
        return workload_identity


def resolve_openai_workload_identity_config(
    api_key: str | None,
    api_base: str | None,
    litellm_params: Mapping[str, object] | None = None,
) -> OpenAIWorkloadIdentityConfig | None:
    static_api_key: Final = normalize_nonempty_secret_str(api_key) or normalize_nonempty_secret_str(
        get_secret_str("OPENAI_API_KEY")
    )
    if static_api_key is not None and not _deployment_identity_outranks(static_api_key, litellm_params):
        return None
    effective_api_base: Final = (
        api_base or litellm.api_base or get_secret_str("OPENAI_BASE_URL") or get_secret_str("OPENAI_API_BASE")
    )
    if not _targets_openai_api(effective_api_base):
        return None
    identity_provider_id: Final = _config_value(
        litellm_params, "openai_identity_provider_id", "OPENAI_IDENTITY_PROVIDER_ID"
    )
    service_account_id: Final = _config_value(litellm_params, "openai_service_account_id", "OPENAI_SERVICE_ACCOUNT_ID")
    token_file: Final = _config_value(litellm_params, "openai_identity_token_file", "OPENAI_IDENTITY_TOKEN_FILE")
    if identity_provider_id is None or service_account_id is None or token_file is None:
        return None
    return OpenAIWorkloadIdentityConfig(
        identity_provider_id=identity_provider_id,
        service_account_id=service_account_id,
        token_file=token_file,
    )


def get_workload_identity_bearer_token(config: OpenAIWorkloadIdentityConfig) -> str:
    return _workload_identity_auth(config).get_token()


def _deployment_identity_outranks(static_api_key: str, litellm_params: Mapping[str, object] | None) -> bool:
    if litellm_params is None:
        return False
    carries_identity: Final = any(_param_str(litellm_params, key) is not None for key in OPENAI_WIF_KWARGS_KEYS)
    return carries_identity and static_api_key in _process_wide_static_keys()


def _process_wide_static_keys() -> frozenset[str]:
    candidates: Final = (get_secret_str("OPENAI_API_KEY"), litellm.api_key, litellm.openai_key)
    return frozenset(key for key in map(normalize_nonempty_secret_str, candidates) if key is not None)


def _param_str(litellm_params: Mapping[str, object], key: str) -> str | None:
    value: Final = litellm_params.get(key)
    return value if isinstance(value, str) and value else None


def _config_value(litellm_params: Mapping[str, object] | None, param_key: str, env_name: str) -> str | None:
    param_value: Final = _param_str(litellm_params, param_key) if litellm_params is not None else None
    return param_value or normalize_nonempty_secret_str(get_secret_str(env_name))


def build_openai_client(
    api_key: str | None,
    api_base: str | None,
    timeout: float | httpx.Timeout | NotGiven = NOT_GIVEN,
    max_retries: int | None = None,
    organization: str | None = None,
    litellm_params: Mapping[str, object] | None = None,
    http_client: httpx.Client | None = None,
) -> OpenAI:
    workload_identity_config: Final = resolve_openai_workload_identity_config(
        api_key=api_key, api_base=api_base, litellm_params=litellm_params
    )
    retries: Final = _max_retries_or_default(max_retries)
    if workload_identity_config is None:
        return OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            max_retries=retries,
            organization=organization,
            http_client=http_client,
        )
    cache_params: Final = _client_cache_params(
        is_async=False,
        workload_identity_config=workload_identity_config,
        api_base=api_base,
        timeout=timeout,
        max_retries=retries,
        organization=organization,
    )
    cached: Final = BaseOpenAILLM.get_cached_openai_client(
        client_initialization_params=cache_params, client_type="openai"
    )
    if isinstance(cached, OpenAI):
        return cached
    client: Final = OpenAI(
        workload_identity=workload_identity_config.to_sdk_workload_identity(),
        base_url=api_base,
        timeout=timeout,
        max_retries=retries,
        organization=organization,
        http_client=http_client,
    )
    BaseOpenAILLM.set_cached_openai_client(
        openai_client=client,
        client_type="openai",
        client_initialization_params=cache_params,
        litellm_owned_client=BaseOpenAILLM.owns_wrapped_http_client(http_client),
    )
    return client


def build_async_openai_client(
    api_key: str | None,
    api_base: str | None,
    timeout: float | httpx.Timeout | NotGiven = NOT_GIVEN,
    max_retries: int | None = None,
    organization: str | None = None,
    litellm_params: Mapping[str, object] | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> AsyncOpenAI:
    workload_identity_config: Final = resolve_openai_workload_identity_config(
        api_key=api_key, api_base=api_base, litellm_params=litellm_params
    )
    retries: Final = _max_retries_or_default(max_retries)
    if workload_identity_config is None:
        return AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            max_retries=retries,
            organization=organization,
            http_client=http_client,
        )
    cache_params: Final = _client_cache_params(
        is_async=True,
        workload_identity_config=workload_identity_config,
        api_base=api_base,
        timeout=timeout,
        max_retries=retries,
        organization=organization,
    )
    cached: Final = BaseOpenAILLM.get_cached_openai_client(
        client_initialization_params=cache_params, client_type="openai"
    )
    if isinstance(cached, AsyncOpenAI):
        return cached
    client: Final = AsyncOpenAI(
        workload_identity=workload_identity_config.to_sdk_workload_identity(),
        base_url=api_base,
        timeout=timeout,
        max_retries=retries,
        organization=organization,
        http_client=http_client,
    )
    BaseOpenAILLM.set_cached_openai_client(
        openai_client=client,
        client_type="openai",
        client_initialization_params=cache_params,
        litellm_owned_client=BaseOpenAILLM.owns_wrapped_http_client(http_client),
    )
    return client


def _client_cache_params(
    is_async: bool,
    workload_identity_config: OpenAIWorkloadIdentityConfig,
    api_base: str | None,
    timeout: float | httpx.Timeout | NotGiven,
    max_retries: int,
    organization: str | None,
) -> dict[str, object]:
    return {
        "api_key": None,
        "is_async": is_async,
        "workload_identity_config": workload_identity_config,
        "api_base": api_base,
        "timeout": timeout,
        "max_retries": max_retries,
        "organization": organization,
    }


def _max_retries_or_default(max_retries: int | None) -> int:
    return DEFAULT_MAX_RETRIES if max_retries is None else max_retries


def _targets_openai_api(api_base: str | None) -> bool:
    if api_base is None:
        return True
    parsed: Final = urlparse(api_base)
    return parsed.scheme == "https" and is_openai_backed_api_base(api_base)


@lru_cache(maxsize=16)
def _workload_identity_auth(config: OpenAIWorkloadIdentityConfig) -> WorkloadIdentityAuth:
    sdk_workload_identity_auth: Final = _load_sdk_workload_identity_auth()
    return sdk_workload_identity_auth(workload_identity=config.to_sdk_workload_identity())


def _load_sdk_workload_identity_auth() -> type[WorkloadIdentityAuth]:
    try:
        from openai.auth import WorkloadIdentityAuth as sdk_workload_identity_auth
    except ImportError as e:
        raise OpenAIError(status_code=500, message=_SDK_UPGRADE_MESSAGE) from e
    return sdk_workload_identity_auth


def _load_sdk_k8s_token_provider() -> Callable[[str], SubjectTokenProvider]:
    try:
        from openai.auth import k8s_service_account_token_provider
    except ImportError as e:
        raise OpenAIError(status_code=500, message=_SDK_UPGRADE_MESSAGE) from e
    return k8s_service_account_token_provider

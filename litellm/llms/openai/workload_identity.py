from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Final
from urllib.parse import urlparse

import litellm
from litellm.secret_managers.main import get_secret_str

from .common_utils import OpenAIError

if TYPE_CHECKING:
    from collections.abc import Callable

    from openai.auth import SubjectTokenProvider, WorkloadIdentity, WorkloadIdentityAuth

OPENAI_WIF_CLIENT_ID: Final = "litellm"
_OPENAI_API_HOST: Final = "api.openai.com"
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
) -> OpenAIWorkloadIdentityConfig | None:
    if api_key is not None or get_secret_str("OPENAI_API_KEY") is not None:
        return None
    effective_api_base: Final = (
        api_base or litellm.api_base or get_secret_str("OPENAI_BASE_URL") or get_secret_str("OPENAI_API_BASE")
    )
    if not _targets_openai_api(effective_api_base):
        return None
    identity_provider_id: Final = get_secret_str("OPENAI_IDENTITY_PROVIDER_ID")
    service_account_id: Final = get_secret_str("OPENAI_SERVICE_ACCOUNT_ID")
    token_file: Final = get_secret_str("OPENAI_IDENTITY_TOKEN_FILE")
    if not identity_provider_id or not service_account_id or not token_file:
        return None
    return OpenAIWorkloadIdentityConfig(
        identity_provider_id=identity_provider_id,
        service_account_id=service_account_id,
        token_file=token_file,
    )


def get_workload_identity_bearer_token(config: OpenAIWorkloadIdentityConfig) -> str:
    return _workload_identity_auth(config).get_token()


def _targets_openai_api(api_base: str | None) -> bool:
    if api_base is None:
        return True
    parsed: Final = urlparse(api_base)
    return parsed.scheme == "https" and parsed.hostname == _OPENAI_API_HOST


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

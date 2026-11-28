import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, Field

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.single_flight import SingleFlight, single_flight_lru_cache
from litellm.llms.azure.common_utils import cached_azure_ad_token_refresh_provider
from litellm.secret_managers.main import get_secret_str

_RESOURCE_ID: Final = re.compile(
    r"/subscriptions/[0-9a-f-]{36}/resourceGroups/[A-Za-z0-9_.()-]+"
    r"/providers/Microsoft\.CognitiveServices/accounts/(?P<account>[A-Za-z0-9][A-Za-z0-9-]{1,63})",
    re.IGNORECASE | re.ASCII,
)
_DEPLOYMENT_NAME: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*", re.ASCII)
_ARM_ORIGIN: Final = "https://management.azure.com"
_ARM_API_VERSION: Final = "2024-10-01"
_DEFAULT_TTL_SECONDS: Final = 300
_MAX_PAGES: Final = 32


class _Deployment(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str


class _Page(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: tuple[_Deployment, ...]
    next_link: str | None = Field(default=None, alias="nextLink")


def _deployment_names(
    url: httpx.URL,
    read_page: Callable[[httpx.URL], httpx.Response],
    remaining_pages: int = _MAX_PAGES,
) -> tuple[str, ...]:
    response: Final = read_page(url)
    response.raise_for_status()
    page: Final = _Page.model_validate_json(response.content)
    names: Final = tuple(item.name for item in page.value if _DEPLOYMENT_NAME.fullmatch(item.name))
    if not page.next_link:
        return names
    next_url: Final = httpx.URL(page.next_link)
    if (
        remaining_pages <= 1
        or next_url.scheme != url.scheme
        or next_url.netloc != url.netloc
        or next_url.raw_path.partition(b"?")[0] != url.raw_path.partition(b"?")[0]
        or next_url.userinfo
        or next_url.fragment
    ):
        raise ValueError("Invalid Azure deployment listing continuation")
    return names + _deployment_names(next_url, read_page, remaining_pages - 1)


def _read_arm_deployments(resource_id: str) -> tuple[str, ...]:
    token_provider: Final = cached_azure_ad_token_refresh_provider(f"{_ARM_ORIGIN}/.default")
    headers: Final = {"Authorization": f"Bearer {token_provider()}"}
    url: Final = httpx.URL(f"{_ARM_ORIGIN}{resource_id}/deployments", params={"api-version": _ARM_API_VERSION})
    with httpx.Client(timeout=5.0, follow_redirects=False, headers=headers) as client:
        return tuple(dict.fromkeys(_deployment_names(url, client.get)))


@dataclass(frozen=True, slots=True)
class _Snapshot:
    names: tuple[str, ...] | None
    refresh_at: float


class _DeploymentCache:
    def __init__(self, resource_id: str, ttl_seconds: int) -> None:
        self._resource_id = resource_id
        self._ttl_seconds = ttl_seconds
        self._snapshot = _Snapshot(None, 0.0)
        self._refresh: Final[SingleFlight[tuple[str, ...] | None]] = SingleFlight()

    def get(self) -> tuple[str, ...] | None:
        snapshot: Final = self._snapshot
        if monotonic() < snapshot.refresh_at:
            return snapshot.names
        return self._refresh.run(self._refresh_if_expired, while_busy=snapshot.names)

    def _refresh_if_expired(self) -> tuple[str, ...] | None:
        previous: Final = self._snapshot
        if monotonic() < previous.refresh_at:
            return previous.names
        snapshot: Final = self._fetch(previous)
        self._snapshot = snapshot
        return snapshot.names

    def _fetch(self, previous: _Snapshot) -> _Snapshot:
        try:
            names: Final = _read_arm_deployments(self._resource_id)
        except Exception as exc:  # noqa: BLE001  # Azure authentication, transport, and response failures are optional here
            verbose_proxy_logger.warning("Azure deployment discovery failed (%s)", type(exc).__name__)
            return _Snapshot(previous.names, monotonic() + min(30, self._ttl_seconds))
        return _Snapshot(names, monotonic() + self._ttl_seconds)


@single_flight_lru_cache
def _discovery_cache(resource_id: str, ttl_seconds: int) -> _DeploymentCache:
    return _DeploymentCache(resource_id, ttl_seconds)


def discover_azure_openai_deployments(api_base: str | None) -> tuple[str, ...] | None:
    resource_id: Final = os.getenv("AZURE_OPENAI_DEPLOYMENT_DISCOVERY_RESOURCE_ID")
    if not resource_id:
        return None
    resource: Final = _RESOURCE_ID.fullmatch(resource_id)
    if resource is None:
        return None

    endpoint: Final = api_base or litellm.api_base or get_secret_str("AZURE_API_BASE")
    if not endpoint:
        return None
    resolved: Final = get_secret_str(endpoint) if endpoint.startswith("os.environ/") else endpoint
    if not resolved:
        return None
    try:
        url: Final = httpx.URL(resolved)
        ttl: Final = int(os.getenv("AZURE_OPENAI_DEPLOYMENT_DISCOVERY_TTL_SECONDS", str(_DEFAULT_TTL_SECONDS)))
    except (ValueError, httpx.InvalidURL):
        return None
    if (
        url.scheme != "https"
        or url.host != f"{resource['account'].lower()}.openai.azure.com"
        or url.port not in (None, 443)
        or url.userinfo
        or ttl <= 0
    ):
        return None
    return _discovery_cache(resource_id, ttl).get()

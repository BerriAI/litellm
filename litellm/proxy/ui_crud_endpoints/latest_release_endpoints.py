import re
from collections import Counter
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Final, Literal, Protocol

import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router: Final = APIRouter()

LATEST_RELEASE_URL: Final = "https://api.github.com/repos/BerriAI/litellm/releases/latest"
LATEST_RELEASE_FETCH_TIMEOUT_SECONDS: Final = 5
LATEST_RELEASE_CACHE_TTL_SECONDS: Final = 60 * 60
LATEST_RELEASE_UNAVAILABLE_CACHE_TTL_SECONDS: Final = 5 * 60
LATEST_RELEASE_CACHE_KEY: Final = "latest_release_info"

_RELEASE_BULLET_PATTERN: Final = re.compile(r"^\*\s+([A-Za-z]+)(\([^)]*\))?!?:\s")

_Bucket = Literal["new_features", "bug_fixes", "other_updates"]
_PREFIX_BUCKETS: Final[Mapping[str, _Bucket]] = MappingProxyType({"feat": "new_features", "fix": "bug_fixes"})


class LatestReleaseInfo(BaseModel):
    version: str
    new_features: int
    bug_fixes: int
    other_updates: int
    release_url: str


@dataclass(frozen=True, slots=True)
class LatestReleaseUnavailable:
    reason: str


class _GitHubRelease(BaseModel):
    tag_name: str
    html_url: str
    body: str


class _AsyncGetClient(Protocol):
    def get(self, url: str, *, timeout: float | None = None) -> Awaitable[httpx.Response]: ...


_latest_release_cache: Final = InMemoryCache(max_size_in_memory=1, default_ttl=LATEST_RELEASE_CACHE_TTL_SECONDS)


def _default_client() -> _AsyncGetClient:
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
    from litellm.types.llms.custom_http import httpxSpecialProvider

    return get_async_httpx_client(llm_provider=httpxSpecialProvider.UI)


def _default_cache() -> InMemoryCache:
    return _latest_release_cache


def count_release_bullets(body: str) -> Counter[_Bucket]:
    """
    Bucket a release body's ``* type(scope): title by @user in <pr-url>`` bullets by conventional-commit type.
    Lines without that shape (headings, "New Contributors" entries) are skipped, not counted as other.
    """
    return Counter(
        _PREFIX_BUCKETS.get(match.group(1).lower(), "other_updates")
        for line in body.splitlines()
        if (match := _RELEASE_BULLET_PATTERN.match(line)) is not None
    )


def parse_latest_release(response: httpx.Response) -> LatestReleaseInfo | LatestReleaseUnavailable:
    if response.status_code != 200:
        return LatestReleaseUnavailable(reason=f"GitHub responded with status {response.status_code}")
    try:
        release: Final = _GitHubRelease.model_validate_json(response.content)
    except ValidationError as e:
        return LatestReleaseUnavailable(reason=f"GitHub release payload was not the expected shape: {e}")
    counts: Final = count_release_bullets(release.body)
    return LatestReleaseInfo(
        version=release.tag_name.removeprefix("v"),
        new_features=counts["new_features"],
        bug_fixes=counts["bug_fixes"],
        other_updates=counts["other_updates"],
        release_url=release.html_url,
    )


async def fetch_latest_release(client: _AsyncGetClient) -> LatestReleaseInfo | LatestReleaseUnavailable:
    try:
        response: Final = await client.get(LATEST_RELEASE_URL, timeout=LATEST_RELEASE_FETCH_TIMEOUT_SECONDS)
    except httpx.HTTPError as e:
        return LatestReleaseUnavailable(reason=f"{type(e).__name__}: {e}")
    return parse_latest_release(response)


async def get_latest_release_info(
    client: _AsyncGetClient, cache: InMemoryCache
) -> LatestReleaseInfo | LatestReleaseUnavailable:
    cached: Final = cache.get_cache(LATEST_RELEASE_CACHE_KEY)
    if isinstance(cached, (LatestReleaseInfo, LatestReleaseUnavailable)):
        return cached
    result: Final = await fetch_latest_release(client)
    ttl: Final = (
        LATEST_RELEASE_UNAVAILABLE_CACHE_TTL_SECONDS
        if isinstance(result, LatestReleaseUnavailable)
        else LATEST_RELEASE_CACHE_TTL_SECONDS
    )
    cache.set_cache(LATEST_RELEASE_CACHE_KEY, result, ttl=ttl)
    return result


@router.get(
    "/get/latest_release_info",
    tags=["UI Settings"],  # mutable-ok: FastAPI's route decorator only accepts a list
    dependencies=[Depends(user_api_key_auth)],  # mutable-ok: FastAPI's route decorator only accepts a list
    response_model=LatestReleaseInfo | None,
)
async def latest_release_info(
    client: Annotated[_AsyncGetClient, Depends(_default_client)],
    cache: Annotated[InMemoryCache, Depends(_default_cache)],
) -> LatestReleaseInfo | None:
    """
    Latest stable LiteLLM GitHub release with its PR count split into new features, bug fixes and other updates.
    Returns null when GitHub can't be reached so the dashboard upgrade banner simply doesn't render.
    """
    result: Final = await get_latest_release_info(client=client, cache=cache)
    if isinstance(result, LatestReleaseUnavailable):
        verbose_proxy_logger.warning("LiteLLM: latest release info unavailable: %s", result.reason)
        return None
    return result

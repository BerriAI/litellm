import asyncio
import re
from collections import Counter
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Final, Literal, Protocol, TypeAlias

import httpx
from fastapi import APIRouter, Depends
from pydantic import ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.llms.base import LiteLLMBaseModel

router: Final = APIRouter()

LATEST_RELEASE_URL: Final = "https://api.github.com/repos/BerriAI/litellm/releases/latest"
RELEASE_BY_TAG_URL: Final = "https://api.github.com/repos/BerriAI/litellm/releases/tags/{tag}"
LATEST_RELEASE_FETCH_TIMEOUT_SECONDS: Final = 5
LATEST_RELEASE_CACHE_TTL_SECONDS: Final = 60 * 60
LATEST_RELEASE_UNAVAILABLE_CACHE_TTL_SECONDS: Final = 5 * 60
LATEST_RELEASE_CACHE_KEY: Final = "latest_release_info"

_RELEASE_BULLET_PATTERN: Final = re.compile(r"^\*\s+(?:([A-Za-z]+)(?:\([^)]*\))?!?:\s)?\S")
_NEW_CONTRIBUTOR_PATTERN: Final = re.compile(r"^\*\s+@\S+ made their first contribution\b")
_STABLE_TAG_PATTERN: Final = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")

_Bucket: TypeAlias = Literal["new_features", "bug_fixes", "other_updates"]
_PREFIX_BUCKETS: Final[Mapping[str, _Bucket]] = MappingProxyType({"feat": "new_features", "fix": "bug_fixes"})


class LatestReleaseInfo(LiteLLMBaseModel):
    version: str
    new_features: int
    bug_fixes: int
    other_updates: int
    release_url: str


@dataclass(frozen=True, slots=True)
class LatestReleaseUnavailable:
    reason: str


class _GitHubRelease(LiteLLMBaseModel):
    tag_name: str
    html_url: str
    body: str | None
    prerelease: bool = False
    draft: bool = False


class _AsyncGetClient(Protocol):
    def get(self, url: str, *, timeout: float | None = None) -> Awaitable[httpx.Response]: ...


_latest_release_cache: Final = InMemoryCache(max_size_in_memory=1, default_ttl=LATEST_RELEASE_CACHE_TTL_SECONDS)
_latest_release_fetch_lock: Final = asyncio.Lock()


def _default_client() -> _AsyncGetClient:
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
    from litellm.types.llms.custom_http import httpxSpecialProvider

    return get_async_httpx_client(llm_provider=httpxSpecialProvider.UI)


def _default_cache() -> InMemoryCache:
    return _latest_release_cache


def _default_fetch_lock() -> asyncio.Lock:
    return _latest_release_fetch_lock


def _bucket_for(line: str) -> _Bucket | None:
    if _NEW_CONTRIBUTOR_PATTERN.match(line) is not None:
        return None
    match: Final = _RELEASE_BULLET_PATTERN.match(line)
    if match is None:
        return None
    prefix: Final = match.group(1)
    return "other_updates" if prefix is None else _PREFIX_BUCKETS.get(prefix.lower(), "other_updates")


def count_release_bullets(body: str) -> Mapping[_Bucket, int]:
    """Bucket release-note bullets by conventional-commit type or ``other_updates``."""
    return MappingProxyType(Counter(bucket for line in body.splitlines() if (bucket := _bucket_for(line)) is not None))


def _stable_version(tag_name: str) -> tuple[int, int, int] | None:
    match: Final = _STABLE_TAG_PATTERN.match(tag_name)
    return None if match is None else (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def earlier_patch_tags(version: tuple[int, int, int]) -> tuple[str, ...]:
    """Tags of the patches that precede ``version`` on its minor line (v1.105.0 for v1.105.1)."""
    major, minor, patch = version
    return tuple(f"v{major}.{minor}.{earlier}" for earlier in range(patch))


def summarize_release_line(latest: _GitHubRelease, earlier: tuple[_GitHubRelease, ...]) -> LatestReleaseInfo:
    """A patch release's notes only list its backports, so count the whole minor line it belongs to."""
    counts: Final = sum(
        (Counter[_Bucket](count_release_bullets(release.body or "")) for release in (latest, *earlier)),
        Counter[_Bucket](),
    )
    return LatestReleaseInfo(
        version=latest.tag_name.removeprefix("v"),
        new_features=counts.get("new_features", 0),
        bug_fixes=counts.get("bug_fixes", 0),
        other_updates=counts.get("other_updates", 0),
        release_url=latest.html_url,
    )


def parse_github_release(response: httpx.Response) -> _GitHubRelease | LatestReleaseUnavailable:
    if response.status_code != 200:
        return LatestReleaseUnavailable(reason=f"GitHub responded with status {response.status_code}")
    try:
        return _GitHubRelease.model_validate_json(response.content)
    except ValidationError as e:
        return LatestReleaseUnavailable(reason=f"GitHub release payload was not the expected shape: {e}")


async def _fetch_earlier_release(client: _AsyncGetClient, tag: str) -> _GitHubRelease | LatestReleaseUnavailable | None:
    """The stable release for ``tag``, or None when that patch was never published as one."""
    response: Final = await client.get(RELEASE_BY_TAG_URL.format(tag=tag), timeout=LATEST_RELEASE_FETCH_TIMEOUT_SECONDS)
    if response.status_code == 404:
        return None
    release: Final = parse_github_release(response)
    if isinstance(release, _GitHubRelease) and (release.prerelease or release.draft):
        return None
    return release


async def fetch_latest_release(client: _AsyncGetClient) -> LatestReleaseInfo | LatestReleaseUnavailable:
    try:
        latest: Final = parse_github_release(
            await client.get(LATEST_RELEASE_URL, timeout=LATEST_RELEASE_FETCH_TIMEOUT_SECONDS)
        )
        if isinstance(latest, LatestReleaseUnavailable):
            return latest
        version: Final = _stable_version(latest.tag_name)
        if version is None:
            return summarize_release_line(latest, ())
        earlier: Final = await asyncio.gather(
            *(_fetch_earlier_release(client, tag) for tag in earlier_patch_tags(version))
        )
    except httpx.HTTPError as e:
        return LatestReleaseUnavailable(reason=f"{type(e).__name__}: {e}")
    unavailable: Final = next((release for release in earlier if isinstance(release, LatestReleaseUnavailable)), None)
    if unavailable is not None:
        return unavailable
    return summarize_release_line(latest, tuple(release for release in earlier if isinstance(release, _GitHubRelease)))


async def get_latest_release_info(
    client: _AsyncGetClient, cache: InMemoryCache, fetch_lock: asyncio.Lock
) -> LatestReleaseInfo | LatestReleaseUnavailable:
    cached: Final = cache.get_cache(LATEST_RELEASE_CACHE_KEY)
    if isinstance(cached, (LatestReleaseInfo, LatestReleaseUnavailable)):
        return cached
    async with fetch_lock:
        cached_after_lock: Final = cache.get_cache(LATEST_RELEASE_CACHE_KEY)
        if isinstance(cached_after_lock, (LatestReleaseInfo, LatestReleaseUnavailable)):
            return cached_after_lock
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
    tags=["UI Settings"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LatestReleaseInfo | None,
)
async def latest_release_info(
    client: Annotated[_AsyncGetClient, Depends(_default_client)],
    cache: Annotated[InMemoryCache, Depends(_default_cache)],
    fetch_lock: Annotated[asyncio.Lock, Depends(_default_fetch_lock)],
) -> LatestReleaseInfo | None:
    """
    Latest stable LiteLLM GitHub release with the PR count of its whole minor line (e.g. v1.105.0 + v1.105.1)
    split into new features, bug fixes and other updates.
    Returns null when GitHub can't be reached so the dashboard upgrade banner simply doesn't render.
    """
    result: Final = await get_latest_release_info(client=client, cache=cache, fetch_lock=fetch_lock)
    if isinstance(result, LatestReleaseUnavailable):
        verbose_proxy_logger.warning("LiteLLM: latest release info unavailable: %s", result.reason)
        return None
    return result

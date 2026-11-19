from datetime import date
from typing import Final, Protocol
from urllib.parse import urlsplit

import httpx

from litellm.proxy.roi_calculator.github import GitHub, GitHubPullListItem
from litellm.types.roi_calculator import ROIPullEvidence, ROISettings


class RepositorySource(Protocol):
    async def repositories(self, query: str = "", page: int = 1) -> tuple[tuple[tuple[str, str, bool], ...], bool]: ...
    async def test_repositories(self, repos: tuple[str, ...]) -> None: ...
    async def pulls(self, repo: str, start: date, end: date) -> tuple[GitHubPullListItem, ...]: ...
    async def evidence(self, repo: str, pull: GitHubPullListItem) -> ROIPullEvidence: ...
    async def profile_email(self, login: str, *, fallback: str = "") -> str: ...
    async def close(self) -> None: ...


def repository_tag(settings: ROISettings, repo: str) -> str:
    parsed: Final = urlsplit(settings.source_api_url)
    host: Final = "github.com" if parsed.netloc == "api.github.com" else parsed.netloc
    prefix: Final = parsed.path.removesuffix("/api/v4").removesuffix("/api/v3").rstrip("/")
    value: Final = host + prefix + "/" + repo
    return value.casefold() if settings.source_provider == "github" else value


def create_source(settings: ROISettings, transport: httpx.AsyncBaseTransport | None = None) -> RepositorySource:
    if settings.source_provider == "gitlab":
        from litellm.proxy.roi_calculator.gitlab import GitLab

        return GitLab(settings, transport)
    return GitHub(settings, transport)

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import httpx

from sdk_watch.changelog import Release


@dataclass(frozen=True, slots=True)
class Sdk:
    name: str
    repo: str

    @property
    def changelog_url(self) -> str:
        return f"https://raw.githubusercontent.com/{self.repo}/HEAD/CHANGELOG.md"

    def release_url(self, version: str) -> str:
        return f"https://github.com/{self.repo}/releases/tag/v{version}"


@dataclass(frozen=True, slots=True)
class Candidate:
    sdk: Sdk
    release: Release


WATCHED_SDKS: Final = (
    Sdk(name="openai", repo="openai/openai-python"),
    Sdk(name="anthropic", repo="anthropics/anthropic-sdk-python"),
    Sdk(name="google-genai", repo="googleapis/python-genai"),
)


def http_fetcher(client: httpx.Client) -> Callable[[str], str]:
    def fetch(url: str) -> str:
        response: Final = client.get(url, follow_redirects=True)
        response.raise_for_status()
        return response.text

    return fetch

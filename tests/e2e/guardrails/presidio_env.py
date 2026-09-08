"""Where the Presidio analyzer/anonymizer endpoints come from, and how they stay
out of test output.

The endpoints are an internal deployment, so they are handled at the same
secrecy as a provider API key: never checked in, supplied only through
PRESIDIO_ANALYZER_API_BASE / PRESIDIO_ANONYMIZER_API_BASE in the runner's
environment, and hard-failed when absent rather than skipped.

`scrub` exists because a failing e2e test prints proxy response bodies into CI
logs, which are read by more people than the endpoints are meant for. Every
failure message in the presidio suites that interpolates a body goes through it,
so a body that ever echoes an endpoint back cannot publish it.
"""

from __future__ import annotations

import os
from functools import cache, reduce
from urllib.parse import urlparse

import pytest

ANALYZER_ENV_VAR = "PRESIDIO_ANALYZER_API_BASE"
ANONYMIZER_ENV_VAR = "PRESIDIO_ANONYMIZER_API_BASE"

REDACTED = "<presidio-endpoint>"


@cache
def presidio_bases() -> tuple[str, str]:
    analyzer = os.environ.get(ANALYZER_ENV_VAR, "").strip()
    anonymizer = os.environ.get(ANONYMIZER_ENV_VAR, "").strip()
    if not analyzer or not anonymizer:
        pytest.fail(
            f"Presidio e2e requires {ANALYZER_ENV_VAR} and {ANONYMIZER_ENV_VAR} "
            "(the running Presidio analyzer/anonymizer services); missing env is a hard failure, not a skip"
        )
    return analyzer, anonymizer


def secret_fragments(urls: tuple[str, ...]) -> tuple[str, ...]:
    """Every substring that would identify the deployment: each URL as given,
    without its trailing slash, and its bare host, so a connection error naming
    only the host is caught as well as a message quoting the whole URL. Longest
    first, so a full URL is replaced whole instead of leaving its path behind."""
    hosts = tuple(netloc for netloc in (urlparse(url).netloc for url in urls) if netloc)
    return tuple(sorted({*urls, *(url.rstrip("/") for url in urls), *hosts}, key=len, reverse=True))


def redact(text: str, fragments: tuple[str, ...]) -> str:
    return reduce(lambda carried, fragment: carried.replace(fragment, REDACTED), fragments, text)


def scrub(text: str) -> str:
    """Replace any trace of the configured Presidio deployment in text bound for a log."""
    return redact(text, secret_fragments(presidio_bases()))

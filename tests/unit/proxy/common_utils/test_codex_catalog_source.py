import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final

import httpx
import pytest

from litellm.proxy.common_utils.codex_catalog_source import (
    CODEX_RELEASE_FILE_URL,
    CodexCatalogSource,
    bundled_codex_stock,
)
from litellm.proxy.common_utils.codex_model_catalog import bundled_codex_models

_TTL: Final = 3600.0


def _release_files(slug: str, prompt: str) -> Mapping[str, bytes]:
    entry: Final = {**bundled_codex_models()["gpt-5.5"].model_dump(mode="json", exclude_unset=True), "slug": slug}
    return {"models.json": json.dumps({"models": [entry]}).encode(), "prompt.md": prompt.encode()}


@dataclass
class _GitHub:
    """Serves each tagged release's files, 404 for any other tag, and records every URL asked for."""

    releases: Mapping[str, Mapping[str, bytes]]
    fail: Callable[[httpx.Request], httpx.Response] | None = None
    requested: list[str] = field(default_factory=list)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requested.append(str(request.url))
        if self.fail is not None:
            return self.fail(request)
        files: Final = {
            CODEX_RELEASE_FILE_URL.format(version=version, file=name): body
            for version, release in self.releases.items()
            for name, body in release.items()  # comprehension-ok: flattens release -> file in one lookup table
        }
        body: Final = files.get(str(request.url))
        return httpx.Response(404, text="404: Not Found") if body is None else httpx.Response(200, content=body)


@dataclass
class _Clock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now


def _source(
    github: _GitHub,
    clock: _Clock | None = None,
    *,
    failure_ttl: float = _TTL,
    max_versions: int = 8,
    fetches_per_minute: int = 100,
) -> CodexCatalogSource:
    client: Final = httpx.AsyncClient(transport=httpx.MockTransport(github.handle))
    return CodexCatalogSource(
        lambda: client,
        offline=lambda: False,
        clock=clock or _Clock(),
        failure_ttl=failure_ttl,
        max_versions=max_versions,
        fetches_per_minute=fetches_per_minute,
    )


async def test_release_catalog_and_prompt_are_served_for_the_asking_codex_version() -> None:
    github: Final = _GitHub({"0.170.0": _release_files("gpt-from-0.170.0", "prompt of 0.170.0")})

    stock: Final = await _source(github).stock("0.170.0")

    assert (list(stock.models), stock.instructions) == (["gpt-from-0.170.0"], "prompt of 0.170.0")
    assert stock.models["gpt-from-0.170.0"].display_name == bundled_codex_models()["gpt-5.5"].display_name


async def test_each_codex_version_gets_its_own_release_and_a_missing_tag_does_not_spill_over() -> None:
    github: Final = _GitHub(
        {
            "0.166.0": _release_files("gpt-from-0.166.0", "prompt of 0.166.0"),
            "0.171.0-alpha.2": _release_files("gpt-from-alpha", "prompt of alpha"),
        }
    )
    source: Final = _source(github)

    served: Final = {version: await source.stock(version) for version in ("0.166.0", "0.171.0-alpha.2", "0.0.0")}

    assert {version: (list(stock.models), stock.instructions) for version, stock in served.items()} == {
        "0.166.0": (["gpt-from-0.166.0"], "prompt of 0.166.0"),
        "0.171.0-alpha.2": (["gpt-from-alpha"], "prompt of alpha"),
        "0.0.0": (list(bundled_codex_models()), bundled_codex_stock().instructions),
    }


def _unreachable(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("egress blocked", request=request)


def _server_error(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, request=request)


@pytest.mark.parametrize("fail", [_unreachable, _server_error], ids=["unreachable", "server-error"])
async def test_a_failed_fetch_serves_the_bundled_catalog(fail: Callable[[httpx.Request], httpx.Response]) -> None:
    github: Final = _GitHub({"0.170.0": _release_files("gpt-from-0.170.0", "p")}, fail=fail)

    assert await _source(github).stock("0.170.0") is bundled_codex_stock()


async def test_a_release_that_never_answers_serves_the_bundled_catalog_within_the_timeout() -> None:
    async def hang(_: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    client: Final = httpx.AsyncClient(transport=httpx.MockTransport(hang))
    source: Final = CodexCatalogSource(lambda: client, offline=lambda: False, clock=_Clock(), timeout=0.01)

    assert await asyncio.wait_for(source.stock("0.170.0"), timeout=30) is bundled_codex_stock()


@pytest.mark.parametrize(
    "files",
    [
        {"models.json": b'{"models": [{"slug": "gpt-x"}]}', "prompt.md": b"p"},
        {"models.json": b"<html>rate limited</html>", "prompt.md": b"p"},
        {"models.json": _release_files("gpt-x", "p")["models.json"], "prompt.md": b"\xff\xfe"},
        {"models.json": _release_files("gpt-x", "p")["models.json"]},
    ],
    ids=["entry-missing-fields", "not-json", "prompt-not-utf8", "prompt-missing"],
)
async def test_a_release_whose_files_do_not_validate_serves_the_bundled_catalog(files: Mapping[str, bytes]) -> None:
    assert await _source(_GitHub({"0.170.0": files})).stock("0.170.0") is bundled_codex_stock()


async def test_a_failed_fetch_is_not_retried_until_the_failure_expires() -> None:
    github: Final = _GitHub({}, fail=_unreachable)
    clock: Final = _Clock()
    source: Final = _source(github, clock, failure_ttl=_TTL)

    first: Final = await source.stock("0.170.0")
    after_first: Final = len(github.requested)
    clock.now = _TTL - 1
    within_ttl: Final = await source.stock("0.170.0")
    after_within: Final = len(github.requested)
    clock.now = _TTL + 1
    await source.stock("0.170.0")

    assert first is within_ttl is bundled_codex_stock()
    assert (after_first > 0, after_within == after_first, len(github.requested) > after_within) == (True, True, True)


async def test_a_fetched_release_is_not_fetched_again() -> None:
    github: Final = _GitHub({"0.170.0": _release_files("gpt-from-0.170.0", "p")})
    clock: Final = _Clock()
    source: Final = _source(github, clock)

    first: Final = await source.stock("0.170.0")
    fetches: Final = len(github.requested)
    clock.now = _TTL * 100

    assert await source.stock("0.170.0") is first
    assert len(github.requested) == fetches


async def test_only_the_most_recent_versions_are_kept() -> None:
    github: Final = _GitHub({version: _release_files(version, "p") for version in ("0.1.0", "0.2.0", "0.3.0")})
    source: Final = _source(github, max_versions=2)
    for version in ("0.1.0", "0.2.0", "0.3.0"):
        await source.stock(version)
    fetches: Final = len(github.requested)

    await source.stock("0.3.0")
    kept_was_cached: Final = len(github.requested) == fetches
    await source.stock("0.1.0")

    assert (kept_was_cached, len(github.requested) > fetches) == (True, True)


async def test_versions_that_fail_never_push_out_a_fetched_one() -> None:
    github: Final = _GitHub({"0.170.0": _release_files("gpt-from-0.170.0", "p")})
    source: Final = _source(github, max_versions=2)
    fetched: Final = await source.stock("0.170.0")
    for made_up in ("9.9.1", "9.9.2", "9.9.3"):
        await source.stock(made_up)
    fetches: Final = len(github.requested)

    assert await source.stock("0.170.0") is fetched
    assert len(github.requested) == fetches


async def test_a_failed_version_keeps_its_cooldown_past_the_fetched_catalog_cap() -> None:
    github: Final = _GitHub({}, fail=_unreachable)
    source: Final = _source(github, max_versions=2)
    for made_up in ("9.9.1", "9.9.2", "9.9.3", "9.9.4"):
        await source.stock(made_up)
    fetches: Final = len(github.requested)

    await source.stock("9.9.1")

    assert len(github.requested) == fetches, github.requested


async def test_a_worker_starts_at_most_its_fetch_budget_each_minute() -> None:
    github: Final = _GitHub({})
    clock: Final = _Clock()
    source: Final = _source(github, clock, fetches_per_minute=2)
    for made_up in ("9.9.1", "9.9.2", "9.9.3"):
        await source.stock(made_up)
    within_minute: Final = len(github.requested)
    clock.now = 61
    await source.stock("9.9.3")

    assert (within_minute, len(github.requested)) == (4, 6), github.requested


async def test_requests_for_a_version_being_fetched_share_that_fetch() -> None:
    answer: Final = asyncio.Event()
    files: Final = _release_files("gpt-from-0.170.0", "p")
    requested: Final[list[str]] = []

    async def slow_github(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        await answer.wait()
        return httpx.Response(200, content=files[str(request.url).rsplit("/", 1)[1]])

    client: Final = httpx.AsyncClient(transport=httpx.MockTransport(slow_github))
    source: Final = CodexCatalogSource(lambda: client, offline=lambda: False, clock=_Clock())
    waiting: Final = [asyncio.ensure_future(source.stock("0.170.0")) for _ in range(5)]
    await asyncio.sleep(0)
    answer.set()
    served: Final = await asyncio.gather(*waiting)

    assert [list(stock.models) for stock in served] == [["gpt-from-0.170.0"]] * 5
    assert len(requested) == 2, requested


async def test_offline_mode_never_fetches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_OFFLINE_MODE", "true")
    github: Final = _GitHub({"0.170.0": _release_files("gpt-from-0.170.0", "p")})
    client: Final = httpx.AsyncClient(transport=httpx.MockTransport(github.handle))

    stock: Final = await CodexCatalogSource(lambda: client, clock=_Clock()).stock("0.170.0")

    assert (stock, github.requested) == (bundled_codex_stock(), [])


@pytest.mark.parametrize("client_version", ["../../main", "0.170.0/../../x", "latest", "", "\u0660.\u0661.\u0662"])
async def test_a_version_that_is_no_release_tag_is_never_fetched(client_version: str) -> None:
    github: Final = _GitHub({})

    assert await _source(github).stock(client_version) is bundled_codex_stock()
    assert github.requested == []

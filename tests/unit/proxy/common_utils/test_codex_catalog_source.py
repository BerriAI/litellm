import asyncio
import json
from collections import ChainMap
from collections.abc import Awaitable, Callable, Mapping
from typing import Final

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPResponseLimitError
from litellm.proxy.common_utils.codex_catalog_source import (
    CODEX_CATALOG_BASE_URL,
    CODEX_CATALOG_MAX_BYTES,
    CODEX_CATALOG_OPENAI_URL,
    CodexCatalogSource,
    codex_release_tag,
    vendored_codex_stock,
)
from litellm.proxy.common_utils.codex_model_catalog import (
    CodexCatalogRow,
    CodexStockModel,
    codex_base_instructions,
    codex_models_response_json,
)

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
RELEASE: Final = "0.166.0"
FRESH_SLUG: Final = "gpt-fresh"
FRESH_ENTRY: Final = {
    "slug": FRESH_SLUG,
    "display_name": "Fresh",
    "priority": 0,
    "visibility": "list",
    "shell_type": "shell_command",
    "apply_patch_tool_type": "freeform",
    "supports_parallel_tool_calls": True,
}
FRESH_CATALOG: Final = json.dumps({"models": [FRESH_ENTRY]}).encode()
FRESH_PROMPT: Final = b"# Fresh base prompt"


def _tree(version: str, catalog: bytes = FRESH_CATALOG, prompt: bytes = FRESH_PROMPT) -> dict[str, tuple[int, bytes]]:
    base: Final = f"{CODEX_CATALOG_BASE_URL}/rust-v{version}/codex-rs/models-manager"
    return {f"{base}/models.json": (200, catalog), f"{base}/prompt.md": (200, prompt)}


class _RawTree:
    def __init__(self, replies: Mapping[str, tuple[int, bytes]]) -> None:
        self.replies = replies
        self.calls: list[str] = []
        self.headers: list[dict[str, str] | None] = []
        self.byte_limits: list[int | None] = []

    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        max_response_bytes: int | None = None,
    ) -> httpx.Response:
        self.calls.append(url)
        self.headers.append(headers)
        self.byte_limits.append(max_response_bytes)
        status, body = self.replies.get(url, (404, b"404: Not Found"))
        return httpx.Response(status, content=body)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _fetching_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_LOCAL_CODEX_CATALOG", raising=False)
    monkeypatch.delenv("LITELLM_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("LITELLM_CODEX_CATALOG_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_CODEX_CATALOG_OPENAI_URL", raising=False)


def _source(
    tree: _RawTree,
    clock: _Clock | None = None,
    *,
    retry_seconds: float = 300.0,
    refresh_seconds: float = 3600.0,
    cached_versions: int = 16,
) -> CodexCatalogSource:
    return CodexCatalogSource(
        client=tree,
        clock=clock if clock is not None else _Clock(),
        retry_seconds=retry_seconds,
        refresh_seconds=refresh_seconds,
        cached_versions=cached_versions,
    )


def object_value(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict), value
    return value


def _entry(stock_models: Mapping[str, CodexStockModel], instructions: str, slug: str) -> dict[str, JsonValue]:
    body: Final = codex_models_response_json(
        (CodexCatalogRow(id=slug, mode="responses"),), stock=stock_models, instructions=instructions
    )
    entries: Final = JSON_OBJECT.validate_json(body.json)["models"]
    assert isinstance(entries, list)
    return object_value(entries[0])


def test_the_requesting_release_catalog_serves_its_entry_for_a_model_the_vendored_pair_lacks() -> None:
    assert FRESH_SLUG not in vendored_codex_stock().models
    tree: Final = _RawTree(_tree(RELEASE))
    stock: Final = asyncio.run(_source(tree).stock(RELEASE))
    entry: Final = _entry(stock.models, stock.instructions, FRESH_SLUG)
    assert stock.source == f"github rust-v{RELEASE}"
    assert entry["shell_type"] == "shell_command" and entry["apply_patch_tool_type"] == "freeform", entry
    assert entry["supports_parallel_tool_calls"] is True, entry
    assert sorted(tree.calls) == sorted(_tree(RELEASE)), tree.calls


def test_a_release_is_fetched_once_and_concurrent_requests_share_the_fetch() -> None:
    tree: Final = _RawTree(_tree(RELEASE))
    source: Final = _source(tree)

    async def burst() -> list[str]:
        stocks: Final = await asyncio.gather(*(source.stock(RELEASE) for _ in range(5)))
        again: Final = await source.stock(RELEASE)
        return [stock.source for stock in (*stocks, again)]

    assert asyncio.run(burst()) == [f"github rust-v{RELEASE}"] * 6
    assert len(tree.calls) == 2, tree.calls


def test_a_release_without_a_tag_serves_the_vendored_pair_and_is_fetched_again_behind_it_after_the_window() -> None:
    tree: Final = _RawTree({})
    clock: Final = _Clock()
    source: Final = _source(tree, clock, retry_seconds=300)
    vendored: Final = vendored_codex_stock().source

    async def scenario() -> tuple[str, ...]:
        first: Final = await source.stock(RELEASE)
        assert first.models == vendored_codex_stock().models
        assert _entry(first.models, first.instructions, FRESH_SLUG)["shell_type"] == "unified_exec"
        calls_after_first: Final = len(tree.calls)
        clock.now += 299
        inside_window: Final = await source.stock(RELEASE)
        assert len(tree.calls) == calls_after_first, "a failed fetch was retried inside its window"
        clock.now += 2
        tree.replies = _tree(RELEASE)
        past_window: Final = await source.stock(RELEASE)
        await asyncio.gather(*(asyncio.all_tasks() - {asyncio.current_task()}))
        refetched: Final = await source.stock(RELEASE)
        return first.source, inside_window.source, past_window.source, refetched.source

    assert asyncio.run(asyncio.wait_for(scenario(), timeout=30)) == (
        vendored,
        vendored,
        vendored,
        f"github rust-v{RELEASE}",
    )


def test_fetches_past_the_budget_inside_one_window_serve_the_vendored_pair_without_a_fetch() -> None:
    tree: Final = _RawTree({})
    clock: Final = _Clock()
    source: Final = _source(tree, clock, retry_seconds=300, cached_versions=2)

    async def scenario() -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        await source.stock("0.150.0")
        await source.stock("0.151.0")
        over_budget: Final = await source.stock("0.152.0")
        calls_inside_window: Final = tuple(tree.calls)
        clock.now += 301
        await source.stock("0.152.0")
        return over_budget.source, calls_inside_window, tuple(tree.calls)

    over_budget, calls_inside_window, calls = asyncio.run(asyncio.wait_for(scenario(), timeout=30))
    assert over_budget == vendored_codex_stock().source
    assert not any("0.152.0" in url for url in calls_inside_window), calls_inside_window
    assert any("0.152.0" in url for url in calls), calls


@pytest.mark.parametrize(
    "replies",
    [
        pytest.param(_tree(RELEASE, catalog=b"<html>rate limited</html>"), id="catalog-not-json"),
        pytest.param(_tree(RELEASE, catalog=b'{"models": []}'), id="catalog-empty"),
        pytest.param(_tree(RELEASE, catalog=b'{"models": [{"slug": "x"}]}'), id="catalog-missing-fields"),
        pytest.param(_tree(RELEASE, prompt=b"   "), id="prompt-blank"),
        pytest.param(
            {**_tree(RELEASE), **{k: (500, b"") for k in _tree(RELEASE) if k.endswith("prompt.md")}}, id="prompt-500"
        ),
    ],
)
def test_a_release_whose_pair_does_not_parse_serves_the_vendored_pair(replies: Mapping[str, tuple[int, bytes]]) -> None:
    stock: Final = asyncio.run(_source(_RawTree(replies)).stock(RELEASE))
    assert stock.source == vendored_codex_stock().source
    assert FRESH_SLUG not in stock.models


@pytest.mark.parametrize(
    "error",
    [
        lambda url: httpx.ConnectError("offline", request=httpx.Request("GET", url)),
        lambda url: httpx.InvalidURL(f"Invalid non-printable ASCII character in URL {url!r}"),
    ],
)
def test_a_transport_error_or_an_unusable_url_serves_the_vendored_pair(error: Callable[[str], Exception]) -> None:
    class _Down:
        async def get(
            self,
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: float | None = None,
            max_response_bytes: int | None = None,
        ) -> httpx.Response:
            raise error(url)

    stock: Final = asyncio.run(CodexCatalogSource(client=_Down(), clock=_Clock()).stock(RELEASE))
    assert stock.source == vendored_codex_stock().source


@pytest.mark.parametrize("client_version", [None, "", "codex", "1.0", "0.166", "0.166.0 ", "0.166.0\n", "v0.166.0"])
def test_a_value_that_is_no_release_version_serves_the_vendored_pair_without_a_fetch(
    client_version: str | None,
) -> None:
    tree: Final = _RawTree(_tree(RELEASE))
    assert codex_release_tag(client_version) is None
    assert asyncio.run(_source(tree).stock(client_version)).source == vendored_codex_stock().source
    assert tree.calls == []


@pytest.mark.parametrize("client_version", ["0.166.0", "0.167.0-alpha.4", "10.0.1"])
def test_a_release_version_names_its_rust_tag(client_version: str) -> None:
    assert codex_release_tag(client_version) == f"rust-v{client_version}"


@pytest.mark.parametrize("variable", ["LITELLM_LOCAL_CODEX_CATALOG", "LITELLM_OFFLINE_MODE"])
def test_local_and_offline_settings_keep_the_vendored_pair_without_a_fetch(
    monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.setenv(variable, "true")
    tree: Final = _RawTree(_tree(RELEASE))
    assert asyncio.run(_source(tree).stock(RELEASE)).source == vendored_codex_stock().source
    assert tree.calls == []


def test_the_base_url_setting_points_the_fetch_at_a_mirror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_CODEX_CATALOG_BASE_URL", "http://mirror.internal/codex/")
    mirrored: Final = {
        url.replace(CODEX_CATALOG_BASE_URL, "http://mirror.internal/codex"): reply
        for url, reply in _tree(RELEASE).items()
    }
    tree: Final = _RawTree(mirrored)
    assert asyncio.run(_source(tree).stock(RELEASE)).source == f"github rust-v{RELEASE}"
    assert sorted(tree.calls) == sorted(mirrored), tree.calls


def test_the_cache_keeps_only_the_newest_releases() -> None:
    versions: Final = ("0.164.0", "0.165.0", "0.166.0")
    tree: Final = _RawTree(ChainMap(*map(_tree, versions)))
    clock: Final = _Clock()
    source: Final = _source(tree, clock, retry_seconds=300, cached_versions=2)

    async def walk() -> None:
        for version in (*versions, versions[-1], versions[0]):
            clock.now += 301
            await source.stock(version)

    asyncio.run(walk())
    oldest_models_url: Final = next(url for url in _tree(versions[0]) if url.endswith("models.json"))
    newest_models_url: Final = next(url for url in _tree(versions[-1]) if url.endswith("models.json"))
    assert tree.calls.count(oldest_models_url) == 2, tree.calls
    assert tree.calls.count(newest_models_url) == 1, tree.calls


def test_releases_no_catalog_answered_leave_the_cache_before_a_fetched_one() -> None:
    tree: Final = _RawTree(_tree(RELEASE))
    clock: Final = _Clock()
    source: Final = _source(tree, clock, retry_seconds=300, cached_versions=2)

    async def walk() -> None:
        await source.stock(RELEASE)
        for minor in range(201, 205):
            clock.now += 301
            await source.stock(f"0.{minor}.0")
        clock.now += 301
        await source.stock(RELEASE)

    asyncio.run(walk())
    models_url: Final = next(url for url in _tree(RELEASE) if url.endswith("models.json"))
    assert tree.calls.count(models_url) == 1, tree.calls


def test_a_release_whose_fetch_just_failed_keeps_its_retry_time_when_the_cache_is_full_of_fetched_ones() -> None:
    older: Final = "0.165.0"
    tree: Final = _RawTree({**_tree(older), **_tree(RELEASE)})
    clock: Final = _Clock()
    source: Final = _source(tree, clock, retry_seconds=300, cached_versions=2)
    missing: Final = "0.204.0"

    async def walk() -> None:
        await source.stock(older)
        clock.now += 301
        await source.stock(RELEASE)
        clock.now += 301
        await source.stock(missing)
        clock.now += 100
        await source.stock(missing)

    asyncio.run(walk())
    missing_url: Final = next(url for url in tree.calls if f"rust-v{missing}/" in url and url.endswith("models.json"))
    assert tree.calls.count(missing_url) == 1, tree.calls


def test_a_catalog_url_that_redirects_to_itself_serves_the_vendored_pair_and_waits_out_the_retry_window() -> None:
    hops: Final[list[str]] = []

    def redirect_to_itself(request: httpx.Request) -> httpx.Response:
        hops.append(str(request.url))
        return httpx.Response(302, headers={"location": str(request.url)})

    source: Final = CodexCatalogSource(
        client=AsyncHTTPHandler(transport=httpx.MockTransport(redirect_to_itself)), clock=_Clock()
    )

    async def ask_twice() -> tuple[str, int, str]:
        first: Final = await source.stock(RELEASE)
        hops_for_first: Final = len(hops)
        second: Final = await source.stock(RELEASE)
        return first.source, hops_for_first, second.source

    first_source, hops_for_first, second_source = asyncio.run(ask_twice())
    assert first_source == second_source == vendored_codex_stock().source
    assert hops_for_first > 0
    assert len(hops) == hops_for_first


OPENAI_KEY: Final = "sk-unit-openai-key"
OPENAI_ENTRY: Final = {**FRESH_ENTRY, "shell_type": "unified_exec", "base_instructions": "# OpenAI live prompt"}
OPENAI_CATALOG: Final = json.dumps({"models": [OPENAI_ENTRY]}).encode()


def _openai_url(version: str) -> str:
    return f"{CODEX_CATALOG_OPENAI_URL}?client_version={version}"


def test_an_openai_key_fetches_the_live_catalog_from_openai_with_that_key_and_skips_github() -> None:
    tree: Final = _RawTree({**_tree(RELEASE), _openai_url(RELEASE): (200, OPENAI_CATALOG)})
    stock: Final = asyncio.run(_source(tree).stock(RELEASE, openai_api_key=OPENAI_KEY))
    entry: Final = _entry(stock.models, stock.instructions, FRESH_SLUG)
    assert stock.source == f"openai {RELEASE}" and stock.instructions == codex_base_instructions(), stock.source
    assert entry["shell_type"] == "unified_exec" and entry["base_instructions"] == "# OpenAI live prompt", entry
    assert tree.calls == [_openai_url(RELEASE)] and tree.headers == [{"Authorization": f"Bearer {OPENAI_KEY}"}]


@pytest.mark.parametrize("status", [401, 500])
def test_a_refused_openai_fetch_falls_back_to_the_release_tag_on_github(status: int) -> None:
    tree: Final = _RawTree({**_tree(RELEASE), _openai_url(RELEASE): (status, b"{}")})
    stock: Final = asyncio.run(_source(tree).stock(RELEASE, openai_api_key=OPENAI_KEY))
    assert stock.source == f"github rust-v{RELEASE}" and FRESH_SLUG in stock.models
    assert tree.calls[0] == _openai_url(RELEASE) and sorted(tree.calls[1:]) == sorted(_tree(RELEASE)), tree.calls


def test_an_openai_answer_that_never_finishes_falls_back_to_the_release_tag_within_the_fetch_timeout() -> None:
    class _NeverFinishingOpenAI(_RawTree):
        async def get(
            self,
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: float | None = None,
            max_response_bytes: int | None = None,
        ) -> httpx.Response:
            if url == _openai_url(RELEASE):
                self.calls.append(url)
                await asyncio.Event().wait()
            return await super().get(url, headers=headers, timeout=timeout, max_response_bytes=max_response_bytes)

    tree: Final = _NeverFinishingOpenAI(_tree(RELEASE))
    source: Final = CodexCatalogSource(client=tree, clock=_Clock(), timeout=0.05, response_deadline=30)
    stock: Final = asyncio.run(asyncio.wait_for(source.stock(RELEASE, openai_api_key=OPENAI_KEY), timeout=5))
    assert stock.source == f"github rust-v{RELEASE}" and FRESH_SLUG in stock.models


def test_an_empty_openai_url_setting_keeps_the_key_home_and_fetches_from_github(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_CODEX_CATALOG_OPENAI_URL", "")
    tree: Final = _RawTree({**_tree(RELEASE), _openai_url(RELEASE): (200, OPENAI_CATALOG)})
    stock: Final = asyncio.run(_source(tree).stock(RELEASE, openai_api_key=OPENAI_KEY))
    assert stock.source == f"github rust-v{RELEASE}"
    assert sorted(tree.calls) == sorted(_tree(RELEASE)) and tree.headers == [None, None], tree.calls


def test_without_an_openai_key_openai_is_never_asked() -> None:
    tree: Final = _RawTree({**_tree(RELEASE), _openai_url(RELEASE): (200, OPENAI_CATALOG)})
    stock: Final = asyncio.run(_source(tree).stock(RELEASE))
    assert stock.source == f"github rust-v{RELEASE}"
    assert _openai_url(RELEASE) not in tree.calls and tree.headers == [None, None], tree.calls


def test_every_fetch_caps_the_download_at_the_catalog_byte_limit_while_it_streams() -> None:
    tree: Final = _RawTree(_tree(RELEASE))
    asyncio.run(_source(tree).stock(RELEASE))
    assert tree.byte_limits == [CODEX_CATALOG_MAX_BYTES] * len(tree.calls) and tree.calls, tree.byte_limits


def test_a_release_asked_for_while_the_in_flight_fetches_are_at_the_cap_gets_the_vendored_pair_without_a_fetch() -> (
    None
):
    class _Hanging(_RawTree):
        async def get(
            self,
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: float | None = None,
            max_response_bytes: int | None = None,
        ) -> httpx.Response:
            self.calls.append(url)
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def scenario() -> tuple[str, list[str]]:
        tree: Final = _Hanging({})
        source: Final = _source(tree, cached_versions=2)
        hung: Final = [asyncio.ensure_future(source.stock(f"0.{n}.0")) for n in (150, 151)]
        await asyncio.sleep(0)
        over_cap: Final = await source.stock("0.152.0")
        for task in hung:
            task.cancel()
        return over_cap.source, list(tree.calls)

    source_label, calls = asyncio.run(scenario())
    assert source_label == vendored_codex_stock().source and not any("0.152.0" in url for url in calls), calls


def test_a_fetch_that_outlasts_the_response_deadline_serves_the_vendored_pair_then_the_release_once_it_lands() -> None:
    class _Gated(_RawTree):
        def __init__(self, replies: Mapping[str, tuple[int, bytes]]) -> None:
            super().__init__(replies)
            self.gate = asyncio.Event()

        async def get(
            self,
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: float | None = None,
            max_response_bytes: int | None = None,
        ) -> httpx.Response:
            self.calls.append(url)
            await self.gate.wait()
            status, body = self.replies.get(url, (404, b"404: Not Found"))
            return httpx.Response(status, content=body)

    async def scenario() -> tuple[str, str, int]:
        tree: Final = _Gated(_tree(RELEASE))
        source: Final = CodexCatalogSource(client=tree, clock=_Clock(), response_deadline=0.01)
        early: Final = await source.stock(RELEASE)
        tree.gate.set()
        await asyncio.gather(*(asyncio.all_tasks() - {asyncio.current_task()}))
        late: Final = await source.stock(RELEASE)
        return early.source, late.source, len(tree.calls)

    assert asyncio.run(asyncio.wait_for(scenario(), timeout=30)) == (
        vendored_codex_stock().source,
        f"github rust-v{RELEASE}",
        2,
    )


def test_a_download_the_client_cuts_at_the_byte_limit_falls_through_to_the_next_source() -> None:
    class _CuttingTree(_RawTree):
        async def get(
            self,
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: float | None = None,
            max_response_bytes: int | None = None,
        ) -> httpx.Response:
            if url == _openai_url(RELEASE):
                self.calls.append(url)
                raise HTTPResponseLimitError(f"{url} exceeded {max_response_bytes} bytes")
            return await super().get(url, headers=headers, timeout=timeout, max_response_bytes=max_response_bytes)

    tree: Final = _CuttingTree(_tree(RELEASE))
    stock: Final = asyncio.run(_source(tree).stock(RELEASE, openai_api_key=OPENAI_KEY))
    assert stock.source == f"github rust-v{RELEASE}" and tree.calls[0] == _openai_url(RELEASE), tree.calls


def test_catalogs_fetched_with_different_openai_keys_are_cached_apart() -> None:
    tree: Final = _RawTree({**_tree(RELEASE), _openai_url(RELEASE): (200, OPENAI_CATALOG)})

    async def scenario() -> None:
        source: Final = _source(tree)
        await source.stock(RELEASE, openai_api_key=OPENAI_KEY)
        await source.stock(RELEASE, openai_api_key=OPENAI_KEY)
        await source.stock(RELEASE, openai_api_key="sk-unit-other-key")

    asyncio.run(scenario())
    assert tree.headers == [
        {"Authorization": f"Bearer {OPENAI_KEY}"},
        {"Authorization": "Bearer sk-unit-other-key"},
    ], tree.headers


@pytest.mark.parametrize(
    ("openai_api_key", "source"), [(None, f"github rust-v{RELEASE}"), (OPENAI_KEY, f"openai {RELEASE}")]
)
def test_a_fetched_catalog_wins_for_the_models_it_lists_and_the_vendored_entry_stays_for_the_rest(
    openai_api_key: str | None, source: str
) -> None:
    vendored: Final = vendored_codex_stock().models
    listed, missing = tuple(vendored)[:2]
    catalog: Final = json.dumps({"models": [{**FRESH_ENTRY, "slug": listed, "display_name": "Release"}]}).encode()
    tree: Final = _RawTree({**_tree(RELEASE, catalog=catalog), _openai_url(RELEASE): (200, catalog)})
    stock: Final = asyncio.run(_source(tree).stock(RELEASE, openai_api_key=openai_api_key))
    assert stock.source == source
    assert stock.models[listed].display_name == "Release" != vendored[listed].display_name
    assert stock.models[missing] == vendored[missing]


def _settled() -> Awaitable[list[object]]:
    return asyncio.gather(*(asyncio.all_tasks() - {asyncio.current_task()}))


def _openai_catalog(display_name: str) -> bytes:
    return json.dumps({"models": [{**OPENAI_ENTRY, "display_name": display_name}]}).encode()


def test_openai_catalog_is_refetched_after_the_refresh_window_and_a_failed_refresh_keeps_it_over_the_release_tag() -> (
    None
):
    tree: Final = _RawTree({_openai_url(RELEASE): (200, _openai_catalog("First"))})
    clock: Final = _Clock()
    source: Final = _source(tree, clock, retry_seconds=300, refresh_seconds=3600)

    async def names() -> list[str]:
        await _settled()
        return [(await source.stock(RELEASE, openai_api_key=OPENAI_KEY)).models[FRESH_SLUG].display_name]

    async def scenario() -> tuple[list[str], ...]:
        first: Final = await names()
        clock.now += 3599
        tree.replies = {_openai_url(RELEASE): (200, _openai_catalog("Second"))}
        inside_window: Final = await names()
        clock.now += 2
        await source.stock(RELEASE, openai_api_key=OPENAI_KEY)
        refreshed: Final = await names()
        clock.now += 3601
        tree.replies = _tree(RELEASE)
        await source.stock(RELEASE, openai_api_key=OPENAI_KEY)
        refresh_failed: Final = await names()
        clock.now += 301
        await source.stock(RELEASE, openai_api_key=OPENAI_KEY)
        retry_failed: Final = await names()
        clock.now += 301
        tree.replies = {_openai_url(RELEASE): (200, _openai_catalog("Third"))}
        await source.stock(RELEASE, openai_api_key=OPENAI_KEY)
        retried: Final = await names()
        return first, inside_window, refreshed, refresh_failed, retry_failed, retried

    assert asyncio.run(asyncio.wait_for(scenario(), timeout=30)) == (
        ["First"],
        ["First"],
        ["Second"],
        ["Second"],
        ["Second"],
        ["Third"],
    )


def test_a_github_answer_after_openai_failed_asks_openai_again_after_the_retry_window() -> None:
    tree: Final = _RawTree({**_tree(RELEASE), _openai_url(RELEASE): (500, b"{}")})
    clock: Final = _Clock()
    source: Final = _source(tree, clock, retry_seconds=300, refresh_seconds=3600)

    async def scenario() -> tuple[str, str, str]:
        first: Final = await source.stock(RELEASE, openai_api_key=OPENAI_KEY)
        clock.now += 299
        inside_window: Final = await source.stock(RELEASE, openai_api_key=OPENAI_KEY)
        clock.now += 2
        tree.replies = {_openai_url(RELEASE): (200, OPENAI_CATALOG)}
        await source.stock(RELEASE, openai_api_key=OPENAI_KEY)
        await _settled()
        recovered: Final = await source.stock(RELEASE, openai_api_key=OPENAI_KEY)
        return first.source, inside_window.source, recovered.source

    assert asyncio.run(asyncio.wait_for(scenario(), timeout=30)) == (
        f"github rust-v{RELEASE}",
        f"github rust-v{RELEASE}",
        f"openai {RELEASE}",
    )
    assert tree.calls.count(_openai_url(RELEASE)) == 2, tree.calls


def test_a_release_tag_fetched_without_an_openai_key_is_never_fetched_again() -> None:
    tree: Final = _RawTree(_tree(RELEASE))
    clock: Final = _Clock()
    source: Final = _source(tree, clock, retry_seconds=300, refresh_seconds=3600)

    async def scenario() -> str:
        await source.stock(RELEASE)
        clock.now += 10 * 3600
        await source.stock(RELEASE)
        await _settled()
        return (await source.stock(RELEASE)).source

    assert asyncio.run(asyncio.wait_for(scenario(), timeout=30)) == f"github rust-v{RELEASE}"
    assert len(tree.calls) == 2, tree.calls


def test_openai_entries_drop_the_prompt_copy_of_their_template_so_a_codex_capped_at_1_mib_lists_every_model() -> None:
    prompt: Final = "p" * 21_000
    slugs: Final = tuple(f"gpt-big-{index}" for index in range(34))
    catalog: Final = json.dumps(
        {
            "models": [
                {
                    **OPENAI_ENTRY,
                    "slug": slug,
                    "base_instructions": prompt,
                    "model_messages": {"instructions_template": prompt},
                }
                for slug in slugs
            ]
        }
    ).encode()
    tree: Final = _RawTree({_openai_url("0.165.0"): (200, catalog)})
    stock: Final = asyncio.run(_source(tree).stock("0.165.0", openai_api_key=OPENAI_KEY))
    body: Final = codex_models_response_json(
        tuple(CodexCatalogRow(id=slug, mode="responses") for slug in slugs),
        stock=stock.models,
        instructions=stock.instructions,
        client_version="0.165.0",
    )
    entries: Final = JSON_OBJECT.validate_json(body.json)["models"]
    assert isinstance(entries, list)
    listed: Final = tuple(map(object_value, entries))
    assert tuple(entry["slug"] for entry in listed) == slugs
    assert all(entry["base_instructions"] == "" for entry in listed), listed[0]["base_instructions"]
    assert all(object_value(entry["model_messages"])["instructions_template"] == prompt for entry in listed)

import asyncio
import json
from collections.abc import Mapping
from typing import Final

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.common_utils.codex_catalog_source import (
    CODEX_CATALOG_BASE_URL,
    CodexCatalogSource,
    codex_release_tag,
    vendored_codex_stock,
)
from litellm.proxy.common_utils.codex_model_catalog import (
    CodexCatalogRow,
    CodexStockModel,
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

    async def get(self, url: str, *, timeout: float | None = None) -> httpx.Response:
        self.calls.append(url)
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


def _source(
    tree: _RawTree, clock: _Clock | None = None, *, retry_seconds: float = 300.0, cached_versions: int = 16
) -> CodexCatalogSource:
    return CodexCatalogSource(
        client=tree,
        clock=clock if clock is not None else _Clock(),
        retry_seconds=retry_seconds,
        cached_versions=cached_versions,
    )


def object_value(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict), value
    return value


def _entry(stock_models: Mapping[str, CodexStockModel], instructions: str, slug: str) -> dict[str, JsonValue]:
    body = codex_models_response_json(
        (CodexCatalogRow(id=slug, mode="responses"),), stock=stock_models, instructions=instructions
    )
    entries: Final = JSON_OBJECT.validate_json(body.json)["models"]
    assert isinstance(entries, list)
    return object_value(entries[0])


def test_the_requesting_release_catalog_serves_its_entry_for_a_model_the_vendored_pair_lacks() -> None:
    assert FRESH_SLUG not in vendored_codex_stock().models
    tree = _RawTree(_tree(RELEASE))
    stock = asyncio.run(_source(tree).stock(RELEASE))
    entry = _entry(stock.models, stock.instructions, FRESH_SLUG)
    assert stock.source == f"rust-v{RELEASE}"
    assert entry["shell_type"] == "shell_command" and entry["apply_patch_tool_type"] == "freeform", entry
    assert entry["supports_parallel_tool_calls"] is True, entry
    assert sorted(tree.calls) == sorted(_tree(RELEASE)), tree.calls


def test_a_release_is_fetched_once_and_concurrent_requests_share_the_fetch() -> None:
    tree = _RawTree(_tree(RELEASE))
    source = _source(tree)

    async def burst() -> list[str]:
        stocks = await asyncio.gather(*(source.stock(RELEASE) for _ in range(5)))
        again = await source.stock(RELEASE)
        return [stock.source for stock in (*stocks, again)]

    assert asyncio.run(burst()) == [f"rust-v{RELEASE}"] * 6
    assert len(tree.calls) == 2, tree.calls


def test_a_release_without_a_tag_serves_the_vendored_pair_and_is_retried_after_the_window() -> None:
    tree = _RawTree({})
    clock = _Clock()
    source = _source(tree, clock, retry_seconds=300)
    first = asyncio.run(source.stock(RELEASE))
    assert first.source == vendored_codex_stock().source and first.models == vendored_codex_stock().models
    assert _entry(first.models, first.instructions, FRESH_SLUG)["shell_type"] == "unified_exec"
    calls_after_first = len(tree.calls)
    clock.now += 299
    assert asyncio.run(source.stock(RELEASE)).source == first.source
    assert len(tree.calls) == calls_after_first, "a failed fetch was retried inside its window"
    clock.now += 2
    tree.replies = _tree(RELEASE)
    assert asyncio.run(source.stock(RELEASE)).source == f"rust-v{RELEASE}"
    assert len(tree.calls) > calls_after_first, "a failed fetch was never retried"


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
    stock = asyncio.run(_source(_RawTree(replies)).stock(RELEASE))
    assert stock.source == vendored_codex_stock().source
    assert FRESH_SLUG not in stock.models


def test_a_transport_error_serves_the_vendored_pair() -> None:
    class _Down:
        async def get(self, url: str, *, timeout: float | None = None) -> httpx.Response:
            raise httpx.ConnectError("offline", request=httpx.Request("GET", url))

    stock = asyncio.run(CodexCatalogSource(client=_Down(), clock=_Clock()).stock(RELEASE))
    assert stock.source == vendored_codex_stock().source


@pytest.mark.parametrize("client_version", [None, "", "codex", "1.0", "0.166", "0.166.0 ", "v0.166.0"])
def test_a_value_that_is_no_release_version_serves_the_vendored_pair_without_a_fetch(
    client_version: str | None,
) -> None:
    tree = _RawTree(_tree(RELEASE))
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
    tree = _RawTree(_tree(RELEASE))
    assert asyncio.run(_source(tree).stock(RELEASE)).source == vendored_codex_stock().source
    assert tree.calls == []


def test_the_base_url_setting_points_the_fetch_at_a_mirror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_CODEX_CATALOG_BASE_URL", "http://mirror.internal/codex/")
    mirrored = {
        url.replace(CODEX_CATALOG_BASE_URL, "http://mirror.internal/codex"): reply
        for url, reply in _tree(RELEASE).items()
    }
    tree = _RawTree(mirrored)
    assert asyncio.run(_source(tree).stock(RELEASE)).source == f"rust-v{RELEASE}"
    assert sorted(tree.calls) == sorted(mirrored), tree.calls


def test_the_cache_keeps_only_the_newest_releases() -> None:
    versions = ("0.164.0", "0.165.0", "0.166.0")
    tree = _RawTree({url: reply for version in versions for url, reply in _tree(version).items()})
    source = _source(tree, cached_versions=2)

    async def walk() -> None:
        for version in versions:
            await source.stock(version)
        await source.stock(versions[-1])
        await source.stock(versions[0])

    asyncio.run(walk())
    oldest_models_url = next(url for url in _tree(versions[0]) if url.endswith("models.json"))
    newest_models_url = next(url for url in _tree(versions[-1]) if url.endswith("models.json"))
    assert tree.calls.count(oldest_models_url) == 2, tree.calls
    assert tree.calls.count(newest_models_url) == 1, tree.calls

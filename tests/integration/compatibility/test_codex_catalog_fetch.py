"""`/v1/models?client_version=` serves the requesting Codex release's own catalog, fetched from the
openai/codex raw tree (a wire peer standing in for raw.githubusercontent.com here), and the vendored pair
for a release that tree has no tag for."""

import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import JsonValue

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server

pytestmark: Final = pytest.mark.shares_worker_state

ROOT: Final = Path(os.environ.get("INTEGRATION_PROXY_ROOT") or Path(__file__).resolve().parents[3])
BUNDLED_STOCK_PATH: Final = ROOT / "litellm" / "proxy" / "common_utils" / "codex_bundled_models_0.166.0.json"
BASE_INSTRUCTIONS_PATH: Final = ROOT / "litellm" / "proxy" / "common_utils" / "codex_base_instructions.md"
PUBLISHED_VERSION: Final = "0.999.0"
UNPUBLISHED_VERSION: Final = "0.998.0"
FRESH_SLUG: Final = "gpt-fresh-release"
FRESH_PROMPT: Final = "# Base prompt of the published release"
MODELS_PATH: Final = "codex-rs/models-manager/models.json"
PROMPT_PATH: Final = "codex-rs/models-manager/prompt.md"
FRESH_CONNECTION: Final = {"Connection": "close"}
CONVERGENCE_SECONDS: Final = 60


def _vendored_models() -> tuple[dict[str, JsonValue], ...]:
    models: Final = JSON_OBJECT.validate_json(BUNDLED_STOCK_PATH.read_bytes())["models"]
    assert isinstance(models, list)
    return tuple(object_value(model) for model in models)


def _vendored_slugs() -> frozenset[JsonValue]:
    return frozenset(model["slug"] for model in _vendored_models())


def _vendored_entry(slug: str) -> dict[str, JsonValue]:
    return next(model for model in _vendored_models() if model["slug"] == slug)


def _published_entry() -> dict[str, JsonValue]:
    return {**_vendored_entry("gpt-5.5"), "slug": FRESH_SLUG, "display_name": "Fresh Release", "upgrade": None}


def _raw_tree(request: Request) -> Reply:
    if request.target == f"/rust-v{PUBLISHED_VERSION}/{MODELS_PATH}":
        return Reply(body=json.dumps({"models": [_published_entry()]}).encode())
    if request.target == f"/rust-v{PUBLISHED_VERSION}/{PROMPT_PATH}":
        return Reply(body=FRESH_PROMPT.encode(), content_type="text/plain; charset=utf-8")
    return Reply(status=404, body=b"404: Not Found", content_type="text/plain; charset=utf-8")


def _config_without_model_list(directory: Path) -> Path:
    """The shared config minus its models, so the catalog's 1 MiB cut never drops the models a test adds."""
    shared_config: Final = object_value(yaml.safe_load((ROOT / "tests/integration/proxy_config.yaml").read_text()))
    path: Final = directory / "proxy_config_without_model_list.yaml"
    path.write_text(yaml.safe_dump({name: value for name, value in shared_config.items() if name != "model_list"}))
    return path


@pytest.fixture(scope="module")
def raw_tree() -> Iterator[Wire]:
    with wire_server(_raw_tree) as wire:
        yield wire


@pytest.fixture(scope="module")
def gateway(tmp_path_factory: pytest.TempPathFactory, raw_tree: Wire) -> Iterator[Gateway]:
    with gateway_from_environment() as parent_gateway:
        directory: Final = tmp_path_factory.mktemp("codex_catalog_fetch")
        overrides: Final = {"LITELLM_LOCAL_CODEX_CATALOG": "False", "LITELLM_CODEX_CATALOG_BASE_URL": raw_tree.url}
        config: Final = _config_without_model_list(directory)
        with owned_proxy(parent_gateway, directory, overrides, config=config) as isolated_gateway:
            yield isolated_gateway


def _catalog_listing(gateway: Gateway, client_version: str, *names: str) -> dict[str, dict[str, JsonValue]]:
    """The catalog once every model in `names` is listed, since a `/model/new` reaches the listing asynchronously."""
    return eventually(
        lambda: _catalog(gateway, client_version),
        lambda value: all(name in value for name in names),
        seconds=CONVERGENCE_SECONDS,
    )


def _catalog(gateway: Gateway, client_version: str) -> dict[str, dict[str, JsonValue]]:
    response: Final = gateway.request(
        "GET", "/v1/models", params={"client_version": client_version}, headers=FRESH_CONNECTION
    )
    assert response.status_code == 200, response.text
    models: Final = JSON_OBJECT.validate_json(response.content)["models"]
    assert isinstance(models, list), response.text
    return {string_value(object_value(entry)["slug"]): object_value(entry) for entry in models}


def test_a_published_release_gets_its_own_entry_for_a_model_the_vendored_pair_lacks(
    gateway: Gateway, raw_tree: Wire
) -> None:
    assert FRESH_SLUG not in _vendored_slugs()
    with gateway.scenario() as scenario:
        scenario.model(name=FRESH_SLUG)
        first: Final = _catalog_listing(gateway, PUBLISHED_VERSION, FRESH_SLUG)
        second: Final = _catalog(gateway, PUBLISHED_VERSION)
        published: Final = _published_entry()
        expected: Final = {
            **published,
            "priority": first[FRESH_SLUG]["priority"],
            "visibility": "list",
            "supported_in_api": True,
            "upgrade": None,
        }
        assert first[FRESH_SLUG] == expected, first[FRESH_SLUG]
        assert second[FRESH_SLUG] == expected, second[FRESH_SLUG]
    targets: Final = [request.target for request in raw_tree.drain()]
    assert targets.count(f"/rust-v{PUBLISHED_VERSION}/{MODELS_PATH}") == gateway.workers, targets
    assert targets.count(f"/rust-v{PUBLISHED_VERSION}/{PROMPT_PATH}") == gateway.workers, targets


def test_an_unpublished_release_gets_the_vendored_pair(gateway: Gateway, raw_tree: Wire) -> None:
    with gateway.scenario() as scenario:
        unknown_name: Final = scenario.model()
        stock_name: Final = scenario.model(name=f"codex-stock-{uuid.uuid4().hex}", model="openai/gpt-5.5")
        catalog: Final = _catalog_listing(gateway, UNPUBLISHED_VERSION, unknown_name, stock_name)
        fresh: Final = catalog[unknown_name]
        assert fresh["shell_type"] == "unified_exec" and fresh["apply_patch_tool_type"] is None, fresh
        assert fresh["base_instructions"] == BASE_INSTRUCTIONS_PATH.read_text(encoding="utf-8"), fresh
        vendored: Final = _vendored_entry("gpt-5.5")
        assert catalog[stock_name]["shell_type"] == vendored["shell_type"], catalog[stock_name]
        assert catalog[stock_name]["model_messages"] == vendored["model_messages"], catalog[stock_name]
    assert f"/rust-v{UNPUBLISHED_VERSION}/{MODELS_PATH}" in [request.target for request in raw_tree.drain()]

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import (
    Gateway,
    eventually,
    gateway_from_environment,
    list_value,
    object_value,
    string_value,
)
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._cache_control_marks_support import owned_config
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_API_KEY: Final = "integration-openrouter-key"
_CATALOG_ROUTE: Final = "/api/v1/models/user"
_CATALOG: Final = {
    "data": [
        {"id": "vendor/capped", "context_length": 4096, "top_provider": {"max_completion_tokens": 1024}},
        {"id": "vendor/uncapped", "context_length": 8192, "top_provider": {"max_completion_tokens": None}},
    ]
}
_NAMED_MODEL: Final = "gpt-mini-named"
_EXPECTED_LIMITS: Final = {
    _NAMED_MODEL: (4096, 1024),
    "openrouter/vendor/capped": (4096, 1024),
    "openrouter/vendor/uncapped": (8192, None),
}


@dataclass(frozen=True, slots=True)
class _CatalogProxy:
    gateway: Gateway
    wire: Wire


def _catalog_reply(_request: Request) -> Reply:
    return Reply(body=json.dumps(_CATALOG).encode())


def _deployment(model_name: str, model: str, wire: Wire) -> JsonValue:
    return {
        "model_name": model_name,
        "litellm_params": {"model": model, "api_key": _API_KEY, "api_base": f"{wire.url}/api/v1"},
    }


@pytest.fixture(scope="module")
def catalog_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_CatalogProxy]:
    directory: Final[Path] = tmp_path_factory.mktemp("openrouter-catalog-limits")
    with gateway_from_environment() as gateway, wire_server(_catalog_reply) as wire:
        config: Final = owned_config(
            directory,
            [
                _deployment(_NAMED_MODEL, "openrouter/vendor/capped", wire),
                _deployment("openrouter/*", "openrouter/*", wire),
            ],
            litellm_settings={"check_provider_endpoint": True},
            general_settings={"disable_model_info_refresh": False},
        )
        with owned_proxy_process(gateway, directory, {}, config=config) as owned:
            yield _CatalogProxy(gateway=owned.gateway, wire=wire)


def _listed_limits(
    gateway: Gateway, path: str, name_key: str, row_limits: str | None
) -> Mapping[str, tuple[JsonValue, JsonValue]]:
    rows: Final = tuple(object_value(row) for row in list_value(gateway.get(path)["data"]))
    named_rows: Final = {string_value(row[name_key]): row for row in rows}
    limit_rows: Final = {
        name: object_value(row[row_limits]) if row_limits is not None else row for name, row in named_rows.items()
    }
    return {
        name: (limit_rows[name].get("max_input_tokens"), limit_rows[name].get("max_output_tokens"))
        for name in _EXPECTED_LIMITS
        if name in limit_rows
    }


@pytest.mark.parametrize(
    ("path", "name_key", "row_limits"),
    (
        ("/v1/model/info", "model_name", "model_info"),
        ("/model_group/info", "model_group", None),
        ("/v1/models", "id", None),
    ),
)
def test_openrouter_catalog_limits_reach_the_model_info_routes_with_no_cost_map_row(
    catalog_proxy: _CatalogProxy, path: str, name_key: str, row_limits: str | None
) -> None:
    limits: Final = eventually(
        lambda: _listed_limits(catalog_proxy.gateway, path, name_key, row_limits),
        lambda observed: observed == _EXPECTED_LIMITS,
        seconds=30,
    )

    assert limits == _EXPECTED_LIMITS
    catalog_calls: Final = catalog_proxy.wire.drain()
    assert {(call.method, call.target, call.headers["authorization"]) for call in catalog_calls} <= {
        ("GET", _CATALOG_ROUTE, f"Bearer {_API_KEY}")
    }

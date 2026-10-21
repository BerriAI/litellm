import json
from collections.abc import Iterator, Mapping
from typing import Final

import httpx
import pytest

import litellm
from litellm.litellm_core_utils import get_model_cost_map as get_model_cost_map_module
from litellm.litellm_core_utils.get_model_cost_map import (
    GetModelCostMap,
    ModelCostMapAdopted,
    areload_model_cost_map,
)
from litellm.litellm_core_utils.model_cost_store import ModelCostOverridesRejected

FEED: Final = "reviewed-feed"
GATEWAY: Final = "gateway"
PRICE_FIELDS: Final = frozenset(
    {
        "input_cost_per_token",
        "output_cost_per_token",
        "cache_read_input_token_cost",
        "cache_creation_input_token_cost",
    }
)
OWN_ROW: Final = {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 3e-06}


@pytest.fixture(autouse=True)
def withdraw_owners() -> Iterator[None]:
    yield
    for owner in (FEED, GATEWAY):
        assert litellm.set_model_cost_overrides(owner, {}) is None


@pytest.fixture
def restores_bundled_catalog() -> Iterator[None]:
    yield
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        litellm.reload_model_cost_map()


def _loaded_chat_keys(*fields: str) -> tuple[str, ...]:
    return tuple(
        key
        for key, row in GetModelCostMap.loaded_model_cost_map().items()
        if "/" not in key and row.get("mode") == "chat" and all(field in row for field in fields)
    )


def _loaded_chat_key(*fields: str) -> str:
    return _loaded_chat_keys(*fields)[0]


def _loaded_row(key: str) -> Mapping[str, object]:
    return GetModelCostMap.loaded_model_cost_map()[key]


def _input_price(key: str) -> object:
    return litellm.get_model_info(model=key, custom_llm_provider=str(_loaded_row(key)["litellm_provider"]))[
        "input_cost_per_token"
    ]


def test_an_override_is_served_and_withdrawing_it_brings_the_loaded_row_back() -> None:
    key: Final = _loaded_chat_key("input_cost_per_token", "litellm_provider")
    upstream_price: Final = _input_price(key)

    assert litellm.set_model_cost_overrides(FEED, {key: {"input_cost_per_token": 7e-06}}) is None
    assert _input_price(key) == 7e-06, "get_model_info must not serve the cached pre-override row"

    assert litellm.set_model_cost_overrides(FEED, {}) is None
    assert litellm.model_cost[key] == dict(_loaded_row(key))
    assert _input_price(key) == upstream_price


def test_replace_fields_drops_owned_fields_the_entry_leaves_out() -> None:
    key: Final = _loaded_chat_key("input_cost_per_token", "cache_read_input_token_cost")
    entry: Final = {
        "input_cost_per_token": 2e-06,
        "output_cost_per_token": 8e-06,
        "preloop_price_provenance": {"feed_id": "f-1"},
    }

    assert litellm.set_model_cost_overrides(FEED, {key: entry}, replace_fields=PRICE_FIELDS) is None

    unowned: Final = {field: value for field, value in _loaded_row(key).items() if field not in PRICE_FIELDS}
    assert litellm.model_cost[key] == {**unowned, **entry}


def test_without_replace_fields_the_entry_merges_over_the_loaded_row() -> None:
    key: Final = _loaded_chat_key("input_cost_per_token", "cache_read_input_token_cost")

    assert litellm.set_model_cost_overrides(FEED, {key: {"input_cost_per_token": 2e-06}}) is None

    assert litellm.model_cost[key] == {**_loaded_row(key), "input_cost_per_token": 2e-06}


def test_owners_compose_in_the_order_they_first_appeared() -> None:
    key: Final = _loaded_chat_key("input_cost_per_token")

    assert litellm.set_model_cost_overrides(FEED, {key: {"input_cost_per_token": 1e-06}}) is None
    assert litellm.set_model_cost_overrides(GATEWAY, {key: {"input_cost_per_token": 2e-06}}) is None
    assert litellm.set_model_cost_overrides(FEED, {key: {"input_cost_per_token": 3e-06}}) is None
    assert litellm.model_cost[key]["input_cost_per_token"] == 2e-06

    assert litellm.set_model_cost_overrides(GATEWAY, {}) is None
    assert litellm.model_cost[key]["input_cost_per_token"] == 3e-06


def test_a_key_only_an_owner_adds_is_removed_when_the_owner_withdraws() -> None:
    assert litellm.set_model_cost_overrides(GATEWAY, {"gateway-only-model": OWN_ROW}) is None
    assert litellm.get_model_info("gateway-only-model")["input_cost_per_token"] == 3e-06

    assert litellm.set_model_cost_overrides(GATEWAY, {}) is None
    assert "gateway-only-model" not in litellm.model_cost


def test_a_direct_write_on_a_key_no_owner_touches_survives_an_override() -> None:
    litellm.model_cost["directly-written-model"] = {**OWN_ROW, "input_cost_per_token": 9e-06}

    assert litellm.set_model_cost_overrides(GATEWAY, {"gateway-only-model": OWN_ROW}) is None

    assert litellm.model_cost["directly-written-model"] == {**OWN_ROW, "input_cost_per_token": 9e-06}


@pytest.mark.parametrize(
    ("bad_entry", "field"),
    [
        ({"input_cost_per_token": 1e-06}, "litellm_provider"),
        ({**OWN_ROW, "mode": "hologram"}, "mode"),
        ({**OWN_ROW, "input_cost_per_token": -1.0}, "input_cost_per_token"),
    ],
)
def test_an_invalid_composed_row_rejects_the_whole_call(bad_entry: Mapping[str, object], field: str) -> None:
    assert litellm.set_model_cost_overrides(GATEWAY, {"kept-model": OWN_ROW}) is None

    result: Final = litellm.set_model_cost_overrides(GATEWAY, {"ok-model": OWN_ROW, "bad-model": bad_entry})

    assert isinstance(result, ModelCostOverridesRejected)
    assert result.owner == GATEWAY
    assert len(result.problems) == 1
    assert result.problems[0].startswith(f"bad-model.{field}"), result.problems
    assert "ok-model" not in litellm.model_cost
    assert litellm.model_cost["kept-model"] == OWN_ROW, "a rejected call must leave the owner's previous set served"


def test_validation_reads_the_composed_row_not_the_entry() -> None:
    key: Final = _loaded_chat_key("input_cost_per_token", "litellm_provider")

    assert litellm.set_model_cost_overrides(FEED, {key: {"input_cost_per_token": 1e-06}}) is None
    result: Final = litellm.set_model_cost_overrides(
        GATEWAY, {key: {"input_cost_per_token": 1e-06}}, replace_fields=frozenset({"litellm_provider"})
    )

    assert isinstance(result, ModelCostOverridesRejected)
    assert tuple(problem.startswith(f"{key}.litellm_provider") for problem in result.problems) == (True,)


async def test_reload_replays_owners_after_register_model_recordings() -> None:
    key, directly_written_key = _loaded_chat_keys("input_cost_per_token")[:2]
    litellm.register_model({key: {"input_cost_per_token": 5e-06}})
    litellm.register_model({"registered-only-model": {**OWN_ROW, "max_input_tokens": 1234}})
    assert litellm.set_model_cost_overrides(FEED, {key: {"input_cost_per_token": 7e-06}}) is None
    litellm.model_cost[directly_written_key] = {**OWN_ROW, "input_cost_per_token": 9e-06}

    result: Final = await areload_model_cost_map()

    assert result == ModelCostMapAdopted(models_count=len(GetModelCostMap.loaded_model_cost_map()))
    assert _input_price(key) == 7e-06
    assert litellm.model_cost["registered-only-model"]["max_input_tokens"] == 1234
    assert litellm.model_cost[directly_written_key] == dict(_loaded_row(directly_written_key)), "reload drops it"


async def test_reload_skips_an_override_whose_row_no_longer_validates(
    monkeypatch: pytest.MonkeyPatch, restores_bundled_catalog: None
) -> None:
    key: Final = _loaded_chat_key("input_cost_per_token", "litellm_provider")
    assert litellm.set_model_cost_overrides(FEED, {key: {"input_cost_per_token": 7e-06}}) is None
    upstream: Final = json.loads(GetModelCostMap.read_local_model_cost_map_bytes())
    without_provider: Final = {field: value for field, value in upstream[key].items() if field != "litellm_provider"}
    served: Final = httpx.Response(200, json={**upstream, key: without_provider})
    remote_url: Final = "https://cost-map.test/map.json"
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    monkeypatch.setattr(
        get_model_cost_map_module,
        "_default_reload_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(lambda request: served)),
    )

    assert isinstance(await areload_model_cost_map(url=remote_url), ModelCostMapAdopted)
    assert litellm.model_cost[key] == dict(_loaded_row(key))
    assert "litellm_provider" not in litellm.model_cost[key]

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    assert isinstance(await areload_model_cost_map(url=remote_url), ModelCostMapAdopted)
    assert _input_price(key) == 7e-06, "the skipped override stays recorded for the next reload"


async def test_blocking_reload_works_inside_a_running_event_loop() -> None:
    key: Final = _loaded_chat_key("input_cost_per_token")
    assert litellm.set_model_cost_overrides(FEED, {key: {"input_cost_per_token": 7e-06}}) is None
    litellm.model_cost = {}

    assert isinstance(litellm.reload_model_cost_map(), ModelCostMapAdopted)

    assert litellm.model_cost[key]["input_cost_per_token"] == 7e-06


def test_invalidate_model_cost_caches_makes_a_direct_write_visible() -> None:
    litellm.model_cost["gateway-direct-model"] = OWN_ROW
    assert litellm.get_model_info("gateway-direct-model")["input_cost_per_token"] == 3e-06
    litellm.model_cost["gateway-direct-model"] = {**OWN_ROW, "input_cost_per_token": 4e-06}
    assert litellm.get_model_info("gateway-direct-model")["input_cost_per_token"] == 3e-06, "precondition: cached"

    litellm.invalidate_model_cost_caches()

    assert litellm.get_model_info("gateway-direct-model")["input_cost_per_token"] == 4e-06

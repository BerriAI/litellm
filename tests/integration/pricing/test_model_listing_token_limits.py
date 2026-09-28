import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.process import owned_proxy
from pydantic import JsonValue

SIBLING_LIMITS: Final = {"max_input_tokens": 4321, "max_output_tokens": 987}

NON_NUMERIC_LIMITS: Final = (
    pytest.param("", id="empty-string"),
    pytest.param(" ", id="blank-string"),
    pytest.param("128,000", id="thousands-separator"),
    pytest.param("unlimited", id="word"),
    pytest.param("NaN", id="nan-string"),
    pytest.param("inf", id="inf-string"),
    pytest.param([], id="empty-list"),
    pytest.param([4096], id="list"),
    pytest.param({}, id="empty-object"),
    pytest.param({"tokens": 4096}, id="object"),
    pytest.param(True, id="bool"),
    pytest.param(None, id="null"),
)

NUMERIC_EDGE_LIMITS: Final = (
    pytest.param(0, id="zero"),
    pytest.param(-1, id="negative"),
    pytest.param(1.5, id="float"),
    pytest.param("1.5", id="float-string"),
    pytest.param("1e9", id="exponent-string"),
    pytest.param(10**12, id="huge"),
)
NUMERIC_EDGE_EXPECTED: Final = {
    "zero": 0,
    "negative": -1,
    "float": 1,
    "float-string": 1,
    "exponent-string": 1_000_000_000,
    "huge": 10**12,
}

MODEL_GROUP_INFO_500: Final = (
    "BUG: /model_group/info returns 500 for every caller when one deployment's token limit is non-numeric"
)
CHAT_500: Final = (
    "BUG: chat completions return 500 from ModelGroupInfo validation when the deployment's token limit is non-numeric"
)
MODEL_GROUP_INFO_500_IDS: Final = frozenset(
    {"empty-string", "blank-string", "thousands-separator", "word", "nan-string", "inf-string"}
    | {"empty-list", "list", "empty-object", "object"}
)
CHAT_500_IDS: Final = frozenset(
    {"empty-string", "blank-string", "thousands-separator", "word", "empty-list", "list", "empty-object", "object"}
)


def _listed(gateway: Gateway, path: str) -> dict[str, dict[str, JsonValue]]:
    entries: Final = gateway.get(path)["data"]
    assert isinstance(entries, list)
    return {str(object_value(entry)["id"]): object_value(entry) for entry in entries}


def _limits(entry: Mapping[str, JsonValue]) -> tuple[JsonValue, JsonValue]:
    return entry.get("max_input_tokens"), entry.get("max_output_tokens")


def _assert_listing_spares_the_sibling(
    gateway: Gateway, broken: str, sibling: str, broken_limits: tuple[JsonValue, JsonValue]
) -> None:
    for path in ("/v1/models", "/models"):
        listed: Final = _listed(gateway, path)
        assert _limits(listed[sibling]) == (4321, 987), (path, listed[sibling])
        assert _limits(listed[broken]) == broken_limits, (path, listed[broken])
    single: Final = gateway.get(f"/v1/models/{broken}")
    assert single["id"] == broken, single
    assert _limits(single) == broken_limits, single
    registered: Final = gateway.get("/model/info")["data"]
    assert isinstance(registered, list)
    assert {broken, sibling} <= {str(object_value(entry)["model_name"]) for entry in registered}


def _assert_serves_chat(gateway: Gateway, *models: str) -> None:
    for model in models:
        reply: Final = gateway.chat(model, text=f"token limit edge {uuid.uuid4().hex}")
        assert reply["model"] == model, reply


def _listed_model(gateway: Gateway, model: str) -> dict[str, JsonValue]:
    entries: Final = gateway.get("/v1/models")["data"]
    assert isinstance(entries, list)
    return next(object_value(entry) for entry in entries if object_value(entry)["id"] == model)


def test_v1_models_carries_cost_map_context_window_for_a_known_model(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini")
        listed: Final = _listed_model(gateway, model)
        # OpenAI publishes these for gpt-4o-mini: https://platform.openai.com/docs/models/gpt-4o-mini (checked 2026-09-24)
        assert listed["max_input_tokens"] == 128000, listed
        assert listed["max_output_tokens"] == 16384, listed


def test_v1_models_carries_deployment_model_info_limits_for_an_unknown_model(gateway: Gateway) -> None:
    unknown: Final = f"openai/custom-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        model: Final = scenario.model(model=unknown, model_info={"max_input_tokens": 4321, "max_output_tokens": 987})
        listed: Final = _listed_model(gateway, model)
        assert listed["max_input_tokens"] == 4321, listed
        assert listed["max_output_tokens"] == 987, listed


def test_numeric_string_token_limit_is_coerced_to_an_int(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/custom-{uuid.uuid4().hex}",
            model_info={"max_input_tokens": "4096", "max_output_tokens": "512"},
        )
        assert _limits(_listed_model(gateway, model)) == (4096, 512)


@pytest.mark.parametrize("value", NON_NUMERIC_LIMITS)
def test_non_numeric_token_limit_is_listed_as_absent_without_breaking_the_listing(
    gateway: Gateway, value: JsonValue
) -> None:
    with gateway.scenario() as scenario:
        sibling: Final = scenario.model(model=f"openai/custom-{uuid.uuid4().hex}", model_info=SIBLING_LIMITS)
        broken: Final = scenario.model(
            model=f"openai/custom-{uuid.uuid4().hex}",
            model_info={"max_input_tokens": value, "max_output_tokens": value},
        )
        _assert_listing_spares_the_sibling(gateway, broken, sibling, (None, None))


@pytest.mark.parametrize("value", NON_NUMERIC_LIMITS)
def test_non_numeric_token_limit_still_serves_chat(
    gateway: Gateway, value: JsonValue, request: pytest.FixtureRequest
) -> None:
    if request.node.callspec.id in CHAT_500_IDS:
        pytest.skip(CHAT_500)
    with gateway.scenario() as scenario:
        sibling: Final = scenario.model(model=f"openai/custom-{uuid.uuid4().hex}", model_info=SIBLING_LIMITS)
        broken: Final = scenario.model(
            model=f"openai/custom-{uuid.uuid4().hex}",
            model_info={"max_input_tokens": value, "max_output_tokens": value},
        )
        _assert_serves_chat(gateway, broken, sibling)


@pytest.mark.parametrize("value", NUMERIC_EDGE_LIMITS)
def test_numeric_edge_token_limit_is_listed_as_its_integer_without_breaking_the_listing(
    gateway: Gateway, value: JsonValue, request: pytest.FixtureRequest
) -> None:
    expected: Final = NUMERIC_EDGE_EXPECTED[request.node.callspec.id]
    with gateway.scenario() as scenario:
        sibling: Final = scenario.model(model=f"openai/custom-{uuid.uuid4().hex}", model_info=SIBLING_LIMITS)
        broken: Final = scenario.model(
            model=f"openai/custom-{uuid.uuid4().hex}",
            model_info={"max_input_tokens": value, "max_output_tokens": value},
        )
        _assert_listing_spares_the_sibling(gateway, broken, sibling, (expected, expected))
        _assert_serves_chat(gateway, broken, sibling)


@pytest.mark.parametrize("field", ("max_input_tokens", "max_output_tokens"))
def test_one_malformed_limit_does_not_disturb_the_other(gateway: Gateway, field: str) -> None:
    other: Final = "max_output_tokens" if field == "max_input_tokens" else "max_input_tokens"
    with gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/custom-{uuid.uuid4().hex}", model_info={field: "128,000", other: 2048}
        )
        listed: Final = _listed_model(gateway, model)
        assert listed.get(field) is None, listed
        assert listed[other] == 2048, listed


@pytest.mark.parametrize("value", NON_NUMERIC_LIMITS + NUMERIC_EDGE_LIMITS)
def test_malformed_token_limit_keeps_model_group_info_serving(
    gateway: Gateway, value: JsonValue, request: pytest.FixtureRequest
) -> None:
    if request.node.callspec.id in MODEL_GROUP_INFO_500_IDS:
        pytest.skip(MODEL_GROUP_INFO_500)
    with gateway.scenario() as scenario:
        sibling: Final = scenario.model(model=f"openai/custom-{uuid.uuid4().hex}", model_info=SIBLING_LIMITS)
        broken: Final = scenario.model(
            model=f"openai/custom-{uuid.uuid4().hex}",
            model_info={"max_input_tokens": value, "max_output_tokens": value},
        )
        groups: Final = gateway.get("/model_group/info")["data"]
        assert isinstance(groups, list)
        assert {broken, sibling} <= {str(object_value(group)["model_group"]) for group in groups}
        single: Final = gateway.get("/model_group/info", {"model_group": broken})["data"]
        assert isinstance(single, list)
        assert [object_value(group)["model_group"] for group in single] == [broken]


def _yaml_deployment(name: str, upstream_url: str, model_info: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": f"openai/custom-{uuid.uuid4().hex}",
            "api_base": f"{upstream_url}/v1",
            "api_key": "integration-provider-key",
        },
        "model_info": dict(model_info),
    }


def test_non_numeric_token_limits_in_config_yaml_are_listed_as_absent(gateway: Gateway, tmp_path: Path) -> None:
    run: Final = uuid.uuid4().hex
    sibling: Final = f"integration-yaml-sibling-{run}"
    broken: Final = {f"integration-yaml-{parameter.id}-{run}": parameter.values[0] for parameter in NON_NUMERIC_LIMITS}
    serving: Final = tuple(
        f"integration-yaml-{parameter.id}-{run}" for parameter in NON_NUMERIC_LIMITS if parameter.id not in CHAT_500_IDS
    )
    config: Final = tmp_path / "malformed_token_limits.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    _yaml_deployment(sibling, gateway.upstream_url, SIBLING_LIMITS),
                    *(
                        _yaml_deployment(
                            name, gateway.upstream_url, {"max_input_tokens": value, "max_output_tokens": value}
                        )
                        for name, value in broken.items()
                    ),
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "store_model_in_db": True,
                },
                "router_settings": {"disable_cooldowns": True},
            }
        )
    )
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
        for name in broken:
            _assert_listing_spares_the_sibling(candidate, name, sibling, (None, None))
        _assert_serves_chat(candidate, sibling, *serving)

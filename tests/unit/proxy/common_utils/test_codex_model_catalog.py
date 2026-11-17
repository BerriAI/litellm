import json
import logging
from collections.abc import Iterator, Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Final, Literal

import pytest
from packaging.version import Version

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.litellm_core_utils.llm_cost_calc.utils import get_service_tier_cost_key
from litellm.llms.bedrock.common_utils import bedrock_model_name_without_provider
from litellm.proxy.common_utils.codex_model_catalog import (
    CODEX_BEDROCK_MODELS_MINIMAL_CLIENT_VERSION,
    CODEX_CATALOG_BYTE_LIMIT,
    CodexCatalogRow,
    CodexStockModel,
    CodexStockUpgrade,
    bundled_bedrock_codex_models,
    bundled_codex_models,
    codex_catalog_etag,
    codex_catalog_rows,
    codex_entry_names_default_reasoning_level,
    codex_model_list_body,
    codex_models_response_json,
    configured_service_tiers,
    deployment_capabilities,
    merged_codex_catalog,
)
from litellm.proxy.common_utils.codex_model_translation import CodexDeploymentCapabilities, CodexServiceTier
from litellm.types.proxy.model_listing import ModelInfoResponse

_PROMPT = "base prompt"
_FAST = {"id": "priority", "name": "Fast", "description": "2x speed, increased usage"}


def _stock(slug, *, visibility="list", supported_in_api=True, upgrade=None, service_tiers=(_FAST,), default_tier=None):
    return CodexStockModel(
        slug=slug,
        display_name=slug.upper(),
        priority=99,
        visibility=visibility,
        supported_in_api=supported_in_api,
        upgrade=CodexStockUpgrade(model=upgrade) if upgrade else None,
        service_tiers=tuple(CodexServiceTier(**tier) for tier in service_tiers),
        additional_speed_tiers=("fast",) if service_tiers else (),
        default_service_tier=default_tier,
        model_messages={"instructions_template": f"{slug} prompt"},
    )


_STOCK = MappingProxyType(
    {
        model.slug: model
        for model in (
            _stock("gpt-5.5", upgrade="gpt-6-sol"),
            _stock("gpt-6-sol", default_tier="priority"),
            _stock("gpt-daybreak", visibility="hide", supported_in_api=False),
        )
    }
)


def _row(model_id, **overrides):
    """A one-deployment row; `service_tiers` is that deployment's raw value, `deployments` a per-deployment tuple."""
    per_deployment = (
        {"service_tiers": (overrides["service_tiers"],)}
        if "service_tiers" in overrides
        else {"service_tiers": overrides["deployments"]}
        if "deployments" in overrides
        else {}
    )
    listing = {key: value for key, value in overrides.items() if key not in ("service_tiers", "deployments")}
    return CodexCatalogRow(
        **{"id": model_id, "mode": "responses", "max_input_tokens": 272000, **listing, **per_deployment}
    )


def _body(*rows, **kwargs):
    return codex_models_response_json(rows, stock=_STOCK, instructions=_PROMPT, **kwargs)


def _models(*rows, **kwargs):
    return json.loads(_body(*rows, **kwargs).json)["models"]


def _tiers(entry):
    return [(tier["id"], tier["name"], tier["description"]) for tier in entry["service_tiers"]]


def test_body_is_exactly_codex_models_response():
    body = json.loads(_body(_row("gpt-6-astra")).json)

    assert list(body) == ["models"]
    assert [entry["slug"] for entry in body["models"]] == ["gpt-6-astra"]


def test_unknown_model_entry_takes_its_slug_window_and_prompt_from_the_row():
    """The fields every Codex release decodes are pinned in test_codex_model_translation.py."""
    (entry,) = _models(_row("gpt-6-astra"))

    assert (entry["slug"], entry["display_name"], entry["visibility"], entry["supported_in_api"]) == (
        "gpt-6-astra",
        "gpt-6-astra",
        "list",
        True,
    )
    assert (entry["context_window"], entry["base_instructions"], entry["service_tiers"]) == (272000, _PROMPT, [])


@pytest.mark.parametrize("efforts", [None, ()], ids=["unknown", "non-reasoning"])
@pytest.mark.parametrize("client_version", ["0.165.0", "0.120.0", None])
def test_unknown_or_no_reasoning_efforts_offer_no_reasoning_level(efforts, client_version):
    """Codex reads a missing default level as `none`, so an entry offering no level sends no effort."""
    (entry,) = _models(_row("mystery", supported_reasoning_efforts=efforts), client_version=client_version)

    assert entry["supported_reasoning_levels"] == []
    assert "default_reasoning_level" not in entry


_ABSENT = object()


@pytest.mark.parametrize(
    ("efforts", "client_version", "levels", "default"),
    [
        (("low", "high", "max"), "0.165.0", ["low", "high", "max"], "low"),
        (("high", "xhigh", "max"), "0.165.0", ["high", "xhigh", "max"], "high"),
        (
            ("none", "low", "medium", "high", "xhigh", "max"),
            "0.165.0",
            ["none", "low", "medium", "high", "xhigh", "max"],
            "none",
        ),
        (("low", "medium", "high"), "0.165.0", ["low", "medium", "high"], "medium"),
        (("none", "low", "max"), "0.137.0", ["none", "low"], "none"),
        (("none",), "0.165.0", ["none"], "none"),
        (("low", "turbo"), "0.165.0", ["low", "turbo"], "low"),
        (("low", "high", "max"), "0.138.0", ["low", "high", "max"], "low"),
        (("low", "high", "max"), "0.137.9", ["low", "high"], "low"),
        (("low", "high", "max"), "0.138.0-alpha.2", ["low", "high"], "low"),
        (("minimal", "max"), "not-a-version", ["minimal"], "minimal"),
        pytest.param(("minimal", "max"), "9" * 5000, ["minimal"], "minimal", id="version-past-int-digit-limit"),
        (("low", "turbo"), None, ["low"], "low"),
        (("max",), "0.137.0", [], _ABSENT),
    ],
)
def test_unknown_model_offers_the_group_efforts_its_codex_release_parses(efforts, client_version, levels, default):
    (entry,) = _models(_row("gpt-6-astra", supported_reasoning_efforts=efforts), client_version=client_version)

    assert [level["effort"] for level in entry["supported_reasoning_levels"]] == levels
    assert all(
        level["description"] == f"Sends reasoning_effort={level['effort']} upstream"
        for level in entry["supported_reasoning_levels"]
    )
    assert entry.get("default_reasoning_level", _ABSENT) == default


@pytest.mark.parametrize(
    "levels",
    [[{"effort": "medium", "description": "Codex's own medium"}], [{"effort": "medium"}]],
    ids=["described", "undescribed"],
)
def test_a_model_codex_knows_keeps_its_own_reasoning_levels(levels):
    stock = CodexStockModel.model_validate(
        {
            "slug": "gpt-5.5",
            "display_name": "GPT-5.5",
            "priority": 3,
            "visibility": "list",
            "default_reasoning_level": "medium",
            "supported_reasoning_levels": levels,
        }
    )

    (entry,) = json.loads(
        codex_models_response_json(
            (_row("gpt-5.5", supported_reasoning_efforts=("low", "max")),),
            stock={"gpt-5.5": stock},
            instructions=_PROMPT,
            client_version="0.165.0",
        ).json
    )["models"]

    assert (entry["default_reasoning_level"], entry["supported_reasoning_levels"]) == ("medium", levels)


@pytest.mark.parametrize("minimum", ["0.153.0", [0, 153, 0]], ids=["string", "parts"])
@pytest.mark.parametrize(
    ("client_version", "display_name", "levels"),
    [
        ("0.153.0", "GPT-6.1 Terra", ["low", "max"]),
        ("0.166.0", "GPT-6.1 Terra", ["low", "max"]),
        ("0.166.0-alpha.5.1", "GPT-6.1 Terra", ["low", "max"]),
        (None, "GPT-6.1 Terra", ["low", "max"]),
        ("0.152.9", "gpt-6.1-terra", ["low", "max"]),
        ("0.137.0", "gpt-6.1-terra", ["low"]),
        ("not-a-version", "gpt-6.1-terra", ["low"]),
    ],
)
def test_a_codex_older_than_an_entrys_minimal_version_gets_an_entry_it_can_decode(
    minimum, client_version, display_name, levels
):
    """Codex 0.137 rejects the whole catalog on an effort it has no name for, such as the `max` Codex's own
    gpt-6.1-terra entry offers, so a release older than the entry's `minimal_client_version` gets the translated
    entry, whose levels are trimmed to that release."""
    stock = CodexStockModel.model_validate(
        {
            "slug": "gpt-6.1-terra",
            "display_name": "GPT-6.1 Terra",
            "priority": 3,
            "visibility": "list",
            "minimal_client_version": minimum,
            "default_reasoning_level": "low",
            "supported_reasoning_levels": [{"effort": "low"}, {"effort": "max"}],
        }
    )

    (entry,) = json.loads(
        codex_models_response_json(
            (_row("gpt-6.1-terra", supported_reasoning_efforts=("low", "max")),),
            stock={"gpt-6.1-terra": stock},
            instructions=_PROMPT,
            client_version=client_version,
        ).json
    )["models"]

    assert (entry["display_name"], [level["effort"] for level in entry["supported_reasoning_levels"]]) == (
        display_name,
        levels,
    )


def test_known_model_keeps_codex_stock_entry_listed_under_the_served_id():
    (entry,) = _models(_row("gpt-daybreak"))

    assert entry["model_messages"] == {"instructions_template": "gpt-daybreak prompt"}
    assert "base_instructions" not in entry
    assert (entry["slug"], entry["display_name"], entry["visibility"], entry["supported_in_api"]) == (
        "gpt-daybreak",
        "GPT-DAYBREAK",
        "list",
        True,
    )
    assert _tiers(entry) == [("priority", "Fast", "2x speed, increased usage")]


@pytest.mark.parametrize("upstream", ["gpt-5.5", "openai/gpt-5.5"])
def test_alias_of_a_known_upstream_model_takes_stock_metadata_under_its_own_name(upstream):
    (entry,) = _models(_row("team-55", upstream_model=upstream))

    assert entry["model_messages"] == {"instructions_template": "gpt-5.5 prompt"}
    assert (entry["slug"], entry["display_name"]) == ("team-55", "team-55")


@pytest.mark.parametrize(("max_input_tokens", "expected"), [(100000, 100000), (922000, 272000), (None, 272000)])
def test_alias_with_deployment_defaults_caps_codexs_window_at_its_listed_max_input_tokens(max_input_tokens, expected):
    stock = MappingProxyType({"gpt-5.5": _stock("gpt-5.5").model_copy(update={"context_window": 272000})})
    alias = _row("team-55", upstream_model="openai/gpt-5.5", max_input_tokens=max_input_tokens)

    body = codex_models_response_json((alias,), stock=stock, instructions=_PROMPT, alias_defaults="deployment")

    assert json.loads(body.json)["models"][0]["context_window"] == expected


_WIDE_STOCK = MappingProxyType(
    {
        "gpt-6-sol": _stock("gpt-6-sol").model_copy(update={"context_window": 272000, "max_context_window": 872000}),
        "gpt-5.5": _stock("gpt-5.5").model_copy(update={"context_window": 272000}),
    }
)


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ((500000,), 500000),
        ((100000,), 100000),
        ((1050000,), 872000),
        ((600000, 400000), 400000),
        ((600000, None), 272000),
        ((100000, None), 100000),
        ((None,), 272000),
        ((), 272000),
        ((0,), 272000),
        ((-5, 400000), 272000),
        ((0, 100000), 100000),
    ],
)
@pytest.mark.parametrize("alias_defaults", ["codex", "deployment"])
def test_a_known_model_and_its_alias_take_the_smallest_window_any_deployment_takes(
    configured: tuple[int | None, ...], expected: int, alias_defaults: Literal["codex", "deployment"]
) -> None:
    rows: Final = (
        _row("gpt-6-sol", max_input_tokens=1050000, configured_max_input_tokens=configured),
        _row(
            "sol-1m",
            upstream_model="openai/gpt-6-sol",
            max_input_tokens=1050000,
            configured_max_input_tokens=configured,
        ),
    )

    body: Final = codex_models_response_json(
        rows, stock=_WIDE_STOCK, instructions=_PROMPT, alias_defaults=alias_defaults
    )

    assert {entry["slug"]: entry["context_window"] for entry in json.loads(body.json)["models"]} == {
        "gpt-6-sol": expected,
        "sol-1m": expected,
    }


def test_a_configured_window_goes_past_codexs_own_when_the_stock_entry_sets_no_ceiling() -> None:
    (entry,) = json.loads(
        codex_models_response_json(
            (_row("gpt-5.5", configured_max_input_tokens=(1050000,)),), stock=_WIDE_STOCK, instructions=_PROMPT
        ).json
    )["models"]

    assert entry["context_window"] == 1050000


def test_a_deployment_without_a_limit_keeps_a_windowless_stock_entry_as_codex_ships_it() -> None:
    windowless: Final = MappingProxyType({"gpt-5.5": _stock("gpt-5.5").model_copy(update={"context_window": None})})

    (entry,) = json.loads(
        codex_models_response_json(
            (_row("gpt-5.5", configured_max_input_tokens=(1050000, None)),), stock=windowless, instructions=_PROMPT
        ).json
    )["models"]

    assert entry["context_window"] is None


def test_model_list_body_sizes_a_known_models_window_off_the_deployments_the_key_team_routes_to() -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-6-astra",
                "litellm_params": {"model": "openai/gpt-6-astra"},
                "model_info": {"max_input_tokens": 922000},
            },
            {
                "model_name": "gpt-6-astra",
                "litellm_params": {"model": "openai/gpt-6-astra", "api_base": "https://team-2.example"},
                "model_info": {"team_id": "team-2", "max_input_tokens": 400000},
            },
            {
                "model_name": "gpt-6-astra",
                "litellm_params": {"model": "openai/gpt-6-astra", "api_base": "https://team-3.example"},
                "model_info": {"team_id": "team-3"},
            },
        ],
        model_group_alias={"astra-1m": "gpt-6-astra"},
    )
    listing: Final = tuple(
        {"id": name, "object": "model", "created": 0, "owned_by": "openai", "mode": "chat"}
        for name in ("gpt-6-astra", "astra-1m")
    )
    entries: Final = (("gpt-6-astra", "gpt-6-astra"), ("astra-1m", "astra-1m"))
    stock: Final = bundled_codex_models()["gpt-6-astra"]
    assert stock.context_window is not None and stock.max_context_window is not None
    assert stock.context_window < 400000 < stock.max_context_window < 922000

    def windows(team_id: str) -> dict[str, int | None]:
        models: Final = json.loads(codex_model_list_body(listing, entries, router, team_id))["models"]
        return {entry["slug"]: entry["context_window"] for entry in models}

    assert windows("team-1") == {"gpt-6-astra": stock.max_context_window, "astra-1m": stock.max_context_window}
    assert windows("team-2") == {"gpt-6-astra": 400000, "astra-1m": 400000}
    assert windows("team-3") == {"gpt-6-astra": stock.context_window, "astra-1m": stock.context_window}


def test_model_list_body_skips_a_limit_a_saved_model_info_response_copied_from_the_cost_map() -> None:
    catalog_limit: Final = GetModelCostMap.loaded_model_cost_map()["gpt-6-astra"]["max_input_tokens"]
    stock: Final = bundled_codex_models()["gpt-6-astra"]
    assert isinstance(catalog_limit, int) and stock.max_context_window is not None
    assert stock.context_window is not None and stock.context_window < 600000 < stock.max_context_window

    def window(model_info: Mapping[str, object]) -> int | None:
        router: Final = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-6-astra",
                    "litellm_params": {"model": "openai/gpt-6-astra"},
                    "model_info": dict(model_info),
                }
            ]
        )
        listing: Final = ({"id": "gpt-6-astra", "object": "model", "created": 0, "owned_by": "openai", "mode": "chat"},)
        (entry,) = json.loads(codex_model_list_body(listing, (("gpt-6-astra", "gpt-6-astra"),), router))["models"]
        return entry["context_window"]

    assert window({"key": "gpt-6-astra", "max_input_tokens": catalog_limit}) == stock.context_window
    assert window({"key": "gpt-6-astra", "max_input_tokens": 600000}) == 600000
    assert window({"max_input_tokens": catalog_limit}) == min(catalog_limit, stock.max_context_window)


def test_alias_with_deployment_defaults_drops_codexs_tier_and_upgrade_while_the_native_name_keeps_them():
    rows = (_row("gpt-6-sol"), _row("team-sol", upstream_model="gpt-6-sol"), _row("team-55", upstream_model="gpt-5.5"))
    codex = {entry["slug"]: entry for entry in _models(*rows)}
    deployment = {entry["slug"]: entry for entry in _models(*rows, alias_defaults="deployment")}

    assert deployment["gpt-6-sol"] == codex["gpt-6-sol"]
    assert (codex["team-sol"]["default_service_tier"], codex["team-55"]["upgrade"]) == (
        "priority",
        {"model": "gpt-6-sol"},
    )
    assert _tiers(codex["team-sol"]) == _tiers(codex["gpt-6-sol"]) != []
    assert [
        (deployment[slug]["service_tiers"], deployment[slug]["default_service_tier"])
        for slug in ("team-sol", "team-55")
    ] == [
        ([], None),
        ([], None),
    ]
    assert deployment["team-55"]["upgrade"] is None
    assert deployment["team-sol"]["model_messages"] == codex["team-sol"]["model_messages"]


@pytest.mark.parametrize(("pinned", "expected"), [("high", "high"), ("turbo", None), (None, None)])
def test_alias_with_deployment_defaults_starts_only_at_a_reasoning_effort_its_deployments_pin(pinned, expected):
    stock = CodexStockModel.model_validate(
        {
            "slug": "gpt-5.5",
            "display_name": "GPT-5.5",
            "priority": 1,
            "visibility": "list",
            "default_reasoning_level": "medium",
            "supported_reasoning_levels": [{"effort": e, "description": e} for e in ("low", "medium", "high")],
        }
    )
    alias = _row("team-55", upstream_model="gpt-5.5", reasoning_effort=pinned)

    body = codex_models_response_json(
        (alias,), stock=MappingProxyType({"gpt-5.5": stock}), instructions=_PROMPT, alias_defaults="deployment"
    )

    assert json.loads(body.json)["models"][0]["default_reasoning_level"] == expected


def test_alias_with_deployment_defaults_sends_no_verbosity_or_promo_of_codexs_while_the_native_name_keeps_them():
    stock = CodexStockModel.model_validate(
        {
            "slug": "gpt-6-sol",
            "display_name": "GPT-6 Sol",
            "priority": 1,
            "visibility": "list",
            "support_verbosity": True,
            "default_verbosity": "low",
            "availability_nux": {"message": "GPT-6 Sol is here"},
        }
    )
    rows = (_row("gpt-6-sol"), _row("team-sol", upstream_model="gpt-6-sol"))

    body = codex_models_response_json(
        rows, stock=MappingProxyType({"gpt-6-sol": stock}), instructions=_PROMPT, alias_defaults="deployment"
    )
    native, alias = json.loads(body.json)["models"]

    assert [(e["support_verbosity"], e["default_verbosity"], e["availability_nux"]) for e in (native, alias)] == [
        (True, "low", {"message": "GPT-6 Sol is here"}),
        (True, None, None),
    ]


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (_row("gpt-5.5", display_name="Our 5.5"), "Our 5.5"),
        (_row("gpt-5.5"), "GPT-5.5"),
        (_row("fast-55", upstream_model="gpt-5.5"), "fast-55"),
        (_row("mystery", display_name="Mystery"), "Mystery"),
        (_row("mystery"), "mystery"),
    ],
)
def test_display_name_is_configured_then_stock_for_its_own_slug_then_the_id(row, expected):
    (entry,) = _models(row)

    assert entry["display_name"] == expected


def test_priority_is_listing_order():
    entries = _models(_row("gpt-6-sol"), _row("mystery"), _row("gpt-5.5"))

    assert [(entry["slug"], entry["priority"]) for entry in entries] == [
        ("gpt-6-sol", 0),
        ("mystery", 1),
        ("gpt-5.5", 2),
    ]


def test_upgrade_nudge_survives_only_when_its_target_is_served():
    with_target, without_target = (
        _models(_row("gpt-5.5"), _row("gpt-6-sol"))[0],
        _models(_row("gpt-5.5"))[0],
    )

    assert with_target["upgrade"]["model"] == "gpt-6-sol"
    assert without_target["upgrade"] is None


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (
            _row("gpt-6-astra", service_tiers=["ultrafast"]),
            [("ultrafast", "Ultrafast", "Sends service_tier=ultrafast upstream")],
        ),
        (
            _row("gpt-6-astra", service_tiers=[{"id": "ultrafast", "name": "Ultra fast", "description": "Fastest"}]),
            [("ultrafast", "Ultra fast", "Fastest")],
        ),
        (
            _row("gpt-6-astra", service_tiers=[{"id": "ultrafast"}]),
            [("ultrafast", "Ultrafast", "Sends service_tier=ultrafast upstream")],
        ),
        (
            _row("gpt-5.5", service_tiers=["ultrafast"]),
            [("ultrafast", "Ultrafast", "Sends service_tier=ultrafast upstream")],
        ),
        (
            _row("gpt-5.5", service_tiers=["priority", "ultrafast"]),
            [
                ("priority", "Fast", "2x speed, increased usage"),
                ("ultrafast", "Ultrafast", "Sends service_tier=ultrafast upstream"),
            ],
        ),
        (_row("gpt-5.5", service_tiers=[]), []),
        (
            _row("gpt-5.5", service_tiers=["ultrafast", "priority", "ultrafast", {"id": "priority"}]),
            [
                ("ultrafast", "Ultrafast", "Sends service_tier=ultrafast upstream"),
                ("priority", "Fast", "2x speed, increased usage"),
            ],
        ),
    ],
)
def test_configured_service_tiers_replace_stock_tiers_and_a_known_id_keeps_its_codex_name(row, expected):
    (entry,) = _models(row)

    assert _tiers(entry) == expected


@pytest.mark.parametrize(
    ("rows", "alias_defaults"),
    [
        ((_row("gpt-5.5", service_tiers=[]),), "codex"),
        ((_row("gpt-5.5", service_tiers=["ultrafast"]),), "codex"),
        ((_row("gpt-5.5", service_tiers=["priority"]),), "codex"),
        ((_row("gpt-5.5"),), "codex"),
        ((_row("team-55", upstream_model="gpt-5.5"),), "deployment"),
        ((_row("team-55", upstream_model="gpt-5.5", service_tiers=["priority"]),), "deployment"),
    ],
)
def test_codexs_fast_toggle_is_on_exactly_when_the_entry_offers_the_priority_tier(rows, alias_defaults):
    """Codex shows `/fast` for a speed tier `fast` as well as for a `priority` service tier, but only sends
    the tier when `service_tiers` lists it (codex-rs `ModelPreset::supports_fast_mode` and
    `get_service_tier`, 0.166.0), so a leftover speed tier is a `/fast` toggle that does nothing."""
    (entry,) = _models(*rows, alias_defaults=alias_defaults)

    offers_priority = "priority" in [tier["id"] for tier in entry["service_tiers"]]
    assert ("fast" in entry["additional_speed_tiers"]) is offers_priority


@pytest.mark.parametrize(
    "invalid", ["ultrafast", [""], ["  "], [1], [{"name": "no id"}], [{"id": "x", "bogus": 1}], {"id": "x"}]
)
def test_invalid_service_tiers_offer_no_tier_even_on_a_model_codex_ships_tiers_for(invalid):
    """Keeping the stock tiers would offer Codex's `/fast` on a model whose operator declared something else."""
    stock_entry, fallback_entry = _models(
        _row("gpt-5.5", service_tiers=invalid), _row("mystery", service_tiers=invalid)
    )

    assert _tiers(stock_entry) == []
    assert fallback_entry["service_tiers"] == []


@pytest.mark.parametrize(
    ("deployments", "expected"),
    [
        ((["ultrafast"], ["ultrafast"]), [("ultrafast", "Ultrafast", "Sends service_tier=ultrafast upstream")]),
        (
            (["priority", "ultrafast"], ["ultrafast", "flex", "priority"]),
            [
                ("priority", "Fast", "2x speed, increased usage"),
                ("ultrafast", "Ultrafast", "Sends service_tier=ultrafast upstream"),
            ],
        ),
        (
            (["priority", "ultrafast"], ["ultrafast"]),
            [("ultrafast", "Ultrafast", "Sends service_tier=ultrafast upstream")],
        ),
        ((["ultrafast"], ["priority"]), []),
        ((["ultrafast"], None), []),
        ((None, ["ultrafast"], None), []),
    ],
)
def test_a_tier_is_offered_only_when_every_deployment_of_the_model_lists_it(deployments, expected):
    """Any deployment can serve the request, so the first deployment's order is kept but a tier one of them
    lacks, or a deployment that lists none, offers nothing Codex could send to the wrong deployment."""
    (entry,) = _models(_row("gpt-5.5", deployments=deployments))

    assert _tiers(entry) == expected


def test_deployments_that_all_leave_service_tiers_unset_keep_the_stock_tiers():
    stock_entry, fallback_entry = _models(
        _row("gpt-5.5", deployments=(None, None)), _row("mystery", deployments=(None, None))
    )

    assert _tiers(stock_entry) == [("priority", "Fast", "2x speed, increased usage")]
    assert fallback_entry["service_tiers"] == []


@pytest.mark.parametrize(
    "deployments", [(["ultrafast"], "not-a-list"), ("not-a-list", ["ultrafast"]), (["priority"], "not-a-list")]
)
def test_an_invalid_value_on_one_deployment_offers_no_tier_for_the_whole_model(deployments):
    """The deployment with the typo lists nothing Codex could send, and the stock `/fast` stays off too."""
    (entry,) = _models(_row("gpt-5.5", deployments=deployments))

    assert _tiers(entry) == []


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, "priority"), (["priority", "ultrafast"], "priority"), (["ultrafast"], None), ([], None)],
)
def test_default_service_tier_survives_only_while_still_offered(configured, expected):
    (entry,) = _models(_row("gpt-6-sol", service_tiers=configured))

    assert entry["default_service_tier"] == expected


@pytest.mark.parametrize(("alias_defaults", "expected"), [("codex", "priority"), ("deployment", None)])
def test_alias_with_offered_tiers_takes_codexs_default_tier_only_under_codex_defaults(alias_defaults, expected):
    (entry,) = _models(
        _row("team-sol", upstream_model="gpt-6-sol", service_tiers=["priority"]), alias_defaults=alias_defaults
    )

    assert (_tiers(entry), entry["default_service_tier"]) == (
        [("priority", "Fast", "2x speed, increased usage")],
        expected,
    )


def test_non_chat_and_wildcard_rows_are_left_out_of_the_catalog():
    body = _body(
        _row("gpt-6-astra"),
        CodexCatalogRow(id="text-embedding-3-small", mode="embedding"),
        CodexCatalogRow(id="openai/*"),
        CodexCatalogRow(id="no-mode-model"),
        _row("gpt-5.5", mode="chat"),
    )

    assert body.listed == ("gpt-6-astra", "no-mode-model", "gpt-5.5")
    assert body.left_out == ()


def test_body_leaves_out_the_entry_that_would_cross_the_byte_limit():
    rows = tuple(_row(f"model-{index}") for index in range(6))
    unlimited = _body(*rows, byte_limit=None)
    limit = len(unlimited.json.encode()) - 1

    body = _body(*rows, byte_limit=limit)
    one_more = _body(*rows[: len(body.listed) + 1], byte_limit=None)

    assert len(body.json.encode()) <= limit < len(one_more.json.encode())
    assert body.listed + body.left_out == tuple(row.id for row in rows)
    assert body.left_out == ("model-5",)
    assert [entry["slug"] for entry in json.loads(body.json)["models"]] == list(body.listed)


def test_entries_offering_a_tier_survive_the_byte_limit_ahead_of_the_rest_and_keep_their_listing_place():
    rows = (
        _row("plain-first"),
        _row("gpt-5.5"),
        _row("plain-middle"),
        _row("tiered-last", service_tiers=["ultrafast"]),
    )
    unlimited = _models(*rows, byte_limit=None)
    two_entries_only = len(_body(rows[1], rows[3], byte_limit=None).json.encode())

    body = _body(*rows, byte_limit=two_entries_only)
    models = json.loads(body.json)["models"]

    assert body.listed == ("gpt-5.5", "tiered-last") and body.left_out == ("plain-first", "plain-middle")
    assert [entry["slug"] for entry in models] == ["gpt-5.5", "tiered-last"]
    assert [entry["priority"] for entry in models] == [1, 3]
    assert models == [unlimited[1], unlimited[3]]
    assert len(body.json.encode()) <= two_entries_only


def test_an_entry_too_large_for_the_bytes_left_is_passed_over_and_the_smaller_ones_after_it_are_kept():
    """A tier description longer than the whole byte limit sorts first (tiered) and can never fit,
    and a long display name fits an empty body but not what is left after the entries ahead of it."""
    kept_rows = (
        _row("plain-first"),
        _row("tiered-small", service_tiers=["ultrafast"]),
        _row("plain-last"),
    )
    limit = len(_body(*kept_rows, byte_limit=None).json.encode())
    oversized_tier = {"id": "huge", "name": "Huge", "description": "x" * limit}
    rows = (
        kept_rows[0],
        _row("tiered-oversized", service_tiers=[oversized_tier]),
        kept_rows[1],
        _row("plain-wide", display_name="w" * 200),
        kept_rows[2],
    )
    unlimited = _models(*rows, byte_limit=None)
    assert len(_body(rows[3], byte_limit=None).json.encode()) < limit

    body = _body(*rows, byte_limit=limit)
    models = json.loads(body.json)["models"]

    assert body.listed == ("plain-first", "tiered-small", "plain-last")
    assert body.left_out == ("tiered-oversized", "plain-wide")
    assert models == [unlimited[0], unlimited[2], unlimited[4]]
    assert [entry["priority"] for entry in models] == [0, 2, 4]
    assert len(body.json.encode()) <= limit


def test_merged_catalog_keeps_upstream_whole_and_adds_only_the_entries_that_fit_the_byte_limit():
    upstream = {"models": [{"slug": "upstream-model", "priority": 4, "base_instructions": "u" * 500}]}
    upstream_only = len(json.dumps(upstream, separators=(",", ":")))
    rows = tuple(_row(f"model-{index}") for index in range(4))

    def merged(*merged_rows, byte_limit):
        return merged_codex_catalog(
            json.dumps(upstream).encode(), merged_rows, stock=_STOCK, instructions=_PROMPT, byte_limit=byte_limit
        )

    unlimited = json.loads(merged(*rows, byte_limit=upstream_only * 100))["models"]
    limit = len(merged(*rows[:2], byte_limit=upstream_only * 100).encode())

    body = merged(*rows, byte_limit=limit)

    assert len(body.encode()) <= limit
    assert json.loads(body)["models"] == unlimited[:3]
    assert merged(*rows, byte_limit=upstream_only) is None


def test_merged_catalog_gives_an_alias_no_service_tier_or_upgrade_of_codexs_own():
    upstream = json.dumps({"models": [{"slug": "upstream-model", "priority": 0}]}).encode()
    rows = (_row("team-sol", upstream_model="openai/gpt-6-sol"), _row("team-55", upstream_model="openai/gpt-5.5"))

    added = json.loads(merged_codex_catalog(upstream, rows, stock=_STOCK, instructions=_PROMPT))["models"][1:]

    assert [
        (entry["slug"], entry["service_tiers"], entry["default_service_tier"], entry["upgrade"]) for entry in added
    ] == [
        ("team-sol", [], None, None),
        ("team-55", [], None, None),
    ]


def test_merged_catalog_warns_once_with_how_many_models_the_byte_limit_left_out(caplog, monkeypatch):
    upstream = json.dumps({"models": [{"slug": "upstream-model", "priority": 0}]}).encode()
    rows = tuple(_row(f"model-{index}") for index in range(4))
    limit = len(merged_codex_catalog(upstream, rows[:1], stock=_STOCK, instructions=_PROMPT, byte_limit=10**6).encode())
    monkeypatch.setattr(verbose_proxy_logger, "propagate", True)

    with caplog.at_level(logging.WARNING, logger=verbose_proxy_logger.name):
        merged_codex_catalog(upstream, rows, stock=_STOCK, instructions=_PROMPT, byte_limit=limit)
        merged_codex_catalog(upstream, rows[:1], stock=_STOCK, instructions=_PROMPT, byte_limit=limit)

    assert [record.getMessage() for record in caplog.records] == [
        f"Codex native model catalog left out 3 of 4 LiteLLM models to stay within Codex's {limit} byte limit"
    ]


def test_unlimited_body_keeps_every_entry():
    rows = tuple(_row(f"model-{index}") for index in range(3))

    body = _body(*rows, byte_limit=None)

    assert (body.listed, body.left_out) == (tuple(row.id for row in rows), ())
    assert CODEX_CATALOG_BYTE_LIMIT == 1024 * 1024


_LIMITED_CODEX: Final = "a codex that drops a catalog over 1 MiB gets one cut to fit"
_UNLIMITED_CODEX: Final = "a codex that reads any size gets every model"


@pytest.mark.parametrize(
    ("client_version", "cut"),
    [
        pytest.param("0.166.0-alpha.1", False, id=f"0.166.0-alpha.1, {_UNLIMITED_CODEX}"),
        pytest.param("0.166.0", False, id=f"0.166.0, {_UNLIMITED_CODEX}"),
        pytest.param("0.169.0-alpha.7", False, id=f"0.169.0-alpha.7, {_UNLIMITED_CODEX}"),
        pytest.param("0.166.0-alpha.5.1", False, id=f"0.166.0-alpha.5.1, {_UNLIMITED_CODEX}"),
        pytest.param("0.165.0", True, id=f"0.165.0, {_LIMITED_CODEX}"),
        pytest.param("0.165.0-alpha.1.2", True, id=f"0.165.0-alpha.1.2, {_LIMITED_CODEX}"),
        pytest.param("0.164.0", True, id=f"0.164.0, {_LIMITED_CODEX}"),
        pytest.param(None, True, id=f"no version, {_LIMITED_CODEX}"),
        pytest.param("not-a-version", True, id=f"unparseable version, {_LIMITED_CODEX}"),
        pytest.param("9" * 5000 + ".0.0", True, id=f"absurdly long version, {_LIMITED_CODEX}"),
    ],
)
def test_model_list_body_is_cut_to_1_mib_only_for_a_codex_that_drops_a_larger_catalog(
    client_version: str | None, cut: bool
) -> None:
    names: Final = tuple(f"model-{index}" for index in range(60))
    listing: Final[tuple[ModelInfoResponse, ...]] = tuple(
        {"id": name, "object": "model", "created": 0, "owned_by": "openai", "mode": "chat"} for name in names
    )

    body: Final = codex_model_list_body(
        listing, tuple((name, name) for name in names), None, client_version=client_version, instructions="x" * 20_000
    )
    slugs: Final[list[str]] = [entry["slug"] for entry in json.loads(body)["models"]]

    assert slugs == list(names[: len(slugs)])
    assert (len(slugs) < len(names)) is cut
    assert (len(body.encode()) <= CODEX_CATALOG_BYTE_LIMIT) is cut


def test_vendored_codex_catalog_entries_decode_on_codex():
    """Every stock entry served keeps the `model_messages` Codex's `ModelsResponse` decoder
    needs in place of `base_instructions`, and the catalog is read from the vendored file."""
    stock = bundled_codex_models()
    rows = tuple(CodexCatalogRow(id=slug, mode="responses") for slug in stock)

    body = codex_models_response_json(rows, instructions=_PROMPT, byte_limit=None)
    entries = json.loads(body.json)["models"]

    assert len(entries) == len(stock) > 0
    assert all(entry["model_messages"]["instructions_template"] for entry in entries)
    assert all(entry["visibility"] == "list" and entry["supported_in_api"] is True for entry in entries)
    assert all("base_instructions" not in entry for entry in entries)


def test_configured_service_tiers_without_known_tiers_names_strings_after_their_id():
    assert configured_service_tiers((["ultrafast"],), "m") == (
        CodexServiceTier(id="ultrafast", name="Ultrafast", description="Sends service_tier=ultrafast upstream"),
    )
    assert configured_service_tiers((None,), "m") is None
    assert configured_service_tiers((), "m") is None


def test_catalog_rows_read_the_router_by_each_entry_lookup_id():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "model_name_team-1_c0ffee",
                "litellm_params": {"model": "openai/gpt-5.5"},
                "model_info": {"display_name": "Team 5.5", "service_tiers": ["ultrafast"]},
            },
            {
                "model_name": "model_name_team-1_c0ffee",
                "litellm_params": {"model": "openai/gpt-5.5", "api_base": "https://second.example"},
            },
            {"model_name": "plain", "litellm_params": {"model": "openai/some-unmapped-model"}},
        ]
    )
    listing = (
        {
            "id": "gpt-5.5-team",
            "object": "model",
            "created": 0,
            "owned_by": "openai",
            "mode": "chat",
            "max_input_tokens": 7,
        },
        {"id": "plain", "object": "model", "created": 0, "owned_by": "openai"},
    )

    rows = codex_catalog_rows(listing, (("gpt-5.5-team", "model_name_team-1_c0ffee"), ("plain", "plain")), router)

    assert tuple(replace(row, deployment_capabilities=()) for row in rows) == (
        CodexCatalogRow(
            id="gpt-5.5-team",
            mode="chat",
            max_input_tokens=7,
            upstream_model="openai/gpt-5.5",
            upstream_provider="openai",
            upstreams=(("openai/gpt-5.5", "openai"), ("openai/gpt-5.5", "openai")),
            display_name="Team 5.5",
            service_tiers=(["ultrafast"], None),
            supported_reasoning_efforts=router.get_model_group_info(
                "model_name_team-1_c0ffee"
            ).supported_reasoning_efforts,
            search_tool_support=(None, None),
            configured_max_input_tokens=(None, None),
        ),
        CodexCatalogRow(
            id="plain",
            upstream_model="openai/some-unmapped-model",
            upstream_provider="openai",
            upstreams=(("openai/some-unmapped-model", "openai"),),
            service_tiers=(None,),
            search_tool_support=(None,),
            configured_max_input_tokens=(None,),
        ),
    )
    catalog_window = GetModelCostMap.loaded_model_cost_map()["gpt-5.5"]["max_input_tokens"]
    assert [deployment.max_input_tokens for deployment in rows[0].deployment_capabilities] == [catalog_window] * 2
    assert rows[1].deployment_capabilities == (CodexDeploymentCapabilities(),)


def test_catalog_rows_read_an_alias_off_its_target_under_the_alias_name():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-6-astra",
                "litellm_params": {"model": "openai/gpt-6-astra"},
                "model_info": {"display_name": "Astra", "service_tiers": ["ultrafast"]},
            }
        ],
        model_group_alias={"gpt-6": "gpt-6-astra"},
    )
    listing = (
        {"id": "gpt-6", "object": "model", "created": 0, "owned_by": "openai", "mode": "chat", "max_input_tokens": 9},
    )

    (row,) = codex_catalog_rows(listing, (("gpt-6", "gpt-6"),), router)

    assert len(row.deployment_capabilities) == 1
    assert replace(row, deployment_capabilities=()) == CodexCatalogRow(
        id="gpt-6",
        mode="chat",
        max_input_tokens=9,
        upstream_model="openai/gpt-6-astra",
        upstream_provider="openai",
        upstreams=(("openai/gpt-6-astra", "openai"),),
        service_tiers=(["ultrafast"],),
        supported_reasoning_efforts=router.get_model_group_info("gpt-6-astra").supported_reasoning_efforts,
        search_tool_support=(None,),
        configured_max_input_tokens=(None,),
    )
    (entry,) = json.loads(codex_model_list_body(listing, (("gpt-6", "gpt-6"),), router))["models"]
    assert (entry["slug"], entry["display_name"], [tier["id"] for tier in entry["service_tiers"]]) == (
        "gpt-6",
        "gpt-6",
        ["ultrafast"],
    )
    assert entry["model_messages"] == bundled_codex_models()["gpt-6-astra"].model_dump(mode="json")["model_messages"]


def test_model_list_body_offers_a_tier_only_off_the_deployments_the_key_team_can_route_to():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-6-astra",
                "litellm_params": {"model": "openai/gpt-6-astra"},
                "model_info": {"service_tiers": ["ultrafast"]},
            },
            {
                "model_name": "gpt-6-astra",
                "litellm_params": {"model": "openai/gpt-6-astra", "api_base": "https://team-2.example"},
                "model_info": {"team_id": "team-2"},
            },
        ]
    )
    listing = ({"id": "gpt-6-astra", "object": "model", "created": 0, "owned_by": "openai", "mode": "chat"},)
    entries = (("gpt-6-astra", "gpt-6-astra"),)

    def offered(team_id):
        (entry,) = json.loads(codex_model_list_body(listing, entries, router, team_id))["models"]
        return [tier["id"] for tier in entry["service_tiers"]]

    assert offered("team-1") == ["ultrafast"]
    assert offered("team-2") == []
    assert offered(None) == ["ultrafast"]


def test_model_list_body_takes_stock_metadata_off_the_deployment_the_key_team_routes_to():
    """Two teams own a deployment of one name: Codex's stock gpt-5.5 entry goes only to the team whose
    requests reach gpt-5.5, under the name and under a `model_group_alias` of it."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "coding-model",
                "litellm_params": {"model": "openai/gpt-5.5"},
                "model_info": {"team_id": "team-1"},
            },
            {
                "model_name": "coding-model",
                "litellm_params": {"model": "openai/some-unmapped-model"},
                "model_info": {"team_id": "team-2", "service_tiers": ["flex"]},
            },
        ],
        model_group_alias={"coding": "coding-model"},
    )
    listing = tuple(
        {"id": name, "object": "model", "created": 0, "owned_by": "openai", "mode": "chat"}
        for name in ("coding-model", "coding")
    )
    entries = (("coding-model", "coding-model"), ("coding", "coding"))
    stock = bundled_codex_models()["gpt-5.5"].model_dump(mode="json")

    def served(team_id):
        return json.loads(codex_model_list_body(listing, entries, router, team_id))["models"]

    for entry in served("team-1"):
        assert entry["model_messages"] == stock["model_messages"]
        assert entry["supported_reasoning_levels"] == stock["supported_reasoning_levels"] != []
        assert entry["service_tiers"] == stock["service_tiers"] != []
    for entry in served("team-2"):
        assert "model_messages" not in entry and entry["base_instructions"]
        assert entry["supported_reasoning_levels"] == []
        assert [tier["id"] for tier in entry["service_tiers"]] == ["flex"]
    assert [entry["slug"] for entry in served("team-2")] == ["coding-model", "coding"]


def test_model_list_body_offers_what_model_group_info_reports_for_each_listed_model():
    """Declared, unknown and non-reasoning groups through a real router, a hidden alias read like its
    target, and the requesting Codex's version deciding which declared levels it is offered."""
    vllm = {"model": "hosted_vllm/qa-local-coder", "api_base": "https://vllm.example/v1"}
    router = litellm.Router(
        model_list=[
            {
                "model_name": "declared",
                "litellm_params": vllm,
                "model_info": {"supports_reasoning": True, "reasoning_effort_levels": ["low", "high", "max"]},
            },
            {"model_name": "unknown", "litellm_params": vllm},
            {"model_name": "non-reasoning", "litellm_params": vllm, "model_info": {"supports_reasoning": False}},
        ],
        model_group_alias={"hidden-declared": {"model": "declared", "hidden": True}},
    )
    names = ("declared", "unknown", "non-reasoning", "hidden-declared")
    listing = tuple(
        {"id": name, "object": "model", "created": 0, "owned_by": "openai", "mode": "chat"} for name in names
    )
    entries = tuple((name, name) for name in names)

    def offered(client_version):
        body = codex_model_list_body(listing, entries, router, client_version=client_version)
        return {
            entry["slug"]: (
                [level["effort"] for level in entry["supported_reasoning_levels"]],
                entry.get("default_reasoning_level"),
            )
            for entry in json.loads(body)["models"]
        }

    assert router.get_model_group_info("declared").supported_reasoning_efforts == ("low", "high", "max")
    assert offered("0.165.0") == {
        "declared": (["low", "high", "max"], "low"),
        "unknown": ([], None),
        "non-reasoning": ([], None),
        "hidden-declared": (["low", "high", "max"], "low"),
    }
    assert offered("0.137.0")["declared"] == (["low", "high"], "low")


def test_catalog_rows_without_a_router_carry_only_the_listing():
    listing = ({"id": "plain", "object": "model", "created": 0, "owned_by": "openai", "mode": "chat"},)

    assert codex_catalog_rows(listing, (("plain", "plain"),), None) == (CodexCatalogRow(id="plain", mode="chat"),)


@pytest.mark.parametrize("client_version", [None, "0.165.0"])
def test_entry_names_a_default_reasoning_level_exactly_when_the_native_entry_carries_one(client_version):
    router = litellm.Router(
        model_list=[
            {"model_name": "open-weights", "litellm_params": {"model": "fireworks_ai/open-weights"}},
            {"model_name": "coding-model", "litellm_params": {"model": "openai/gpt-5.5"}},
            {"model_name": "pinned-model", "litellm_params": {"model": "openai/gpt-5.5", "reasoning_effort": "high"}},
            {"model_name": "split-pin", "litellm_params": {"model": "openai/gpt-5.5", "reasoning_effort": "high"}},
            {"model_name": "split-pin", "litellm_params": {"model": "openai/gpt-5.5", "reasoning_effort": "low"}},
            {
                "model_name": "max-only",
                "litellm_params": {"model": "hosted_vllm/qa-local-coder", "api_base": "https://vllm.example/v1"},
                "model_info": {"supports_reasoning": True, "reasoning_effort_levels": ["max"]},
            },
        ]
    )
    names = ("open-weights", "coding-model", "pinned-model", "split-pin", "max-only")
    listing = tuple(
        {"id": name, "object": "model", "created": 0, "owned_by": "openai", "mode": "chat"} for name in names
    )
    upstream = json.dumps({"models": [{"slug": "upstream-model", "priority": 0}]}).encode()
    rows = codex_catalog_rows(listing, tuple((name, name) for name in names), router)
    served = json.loads(merged_codex_catalog(upstream, rows, client_version=client_version))["models"][1:]
    served_names_one = [entry.get("default_reasoning_level") is not None for entry in served]

    assert [
        codex_entry_names_default_reasoning_level(entry["slug"], router, None, client_version) for entry in served
    ] == served_names_one
    assert served_names_one == [False, False, True, False, client_version is not None]
    assert served[2]["default_reasoning_level"] == "high"


def test_codex_tool_search_is_offered_off_the_route_and_model_info_and_only_when_every_deployment_answers_it():
    claude = "anthropic/claude-sonnet-5-5"
    converse = "bedrock/us.anthropic.claude-sonnet-5-5"
    router = litellm.Router(
        model_list=[
            {"model_name": "claude", "litellm_params": {"model": claude}},
            {
                "model_name": "claude-off",
                "litellm_params": {"model": claude},
                "model_info": {"supports_search_tool": False},
            },
            {"model_name": "converse", "litellm_params": {"model": converse}},
            {
                "model_name": "converse-on",
                "litellm_params": {"model": converse},
                "model_info": {"supports_search_tool": True},
            },
            {"model_name": "gpt-5.5", "litellm_params": {"model": "openai/gpt-5.5"}},
            {
                "model_name": "gpt-5.5-off",
                "litellm_params": {"model": "openai/gpt-5.5"},
                "model_info": {"supports_search_tool": False},
            },
            {"model_name": "mixed", "litellm_params": {"model": claude}},
            {"model_name": "mixed", "litellm_params": {"model": converse}},
        ]
    )
    names = ("claude", "claude-off", "converse", "converse-on", "gpt-5.5", "gpt-5.5-off", "mixed")
    listing = tuple(
        {"id": name, "object": "model", "created": 0, "owned_by": "x", "mode": "chat", "max_input_tokens": 1000000}
        for name in names
    )

    models = json.loads(codex_model_list_body(listing, tuple((name, name) for name in names), router))["models"]

    stock_support = bundled_codex_models()["gpt-5.5"].model_dump()["supports_search_tool"]
    assert {entry["slug"]: entry["supports_search_tool"] for entry in models} == {
        "claude": True,
        "claude-off": False,
        "converse": False,
        "converse-on": True,
        "gpt-5.5": stock_support,
        "gpt-5.5-off": False,
        "mixed": False,
    }
    assert models[0]["context_window"] == 1000000


def test_native_merged_catalog_offers_codex_tool_search_as_the_proxy_listing_does():
    claude = "anthropic/claude-sonnet-5-5"
    router = litellm.Router(
        model_list=[
            {"model_name": "claude", "litellm_params": {"model": claude}},
            {
                "model_name": "claude-off",
                "litellm_params": {"model": claude},
                "model_info": {"supports_search_tool": False},
            },
            {"model_name": "coding-model", "litellm_params": {"model": "openai/gpt-5.5"}},
        ]
    )
    names = ("claude", "claude-off", "coding-model")
    listing = tuple({"id": name, "object": "model", "created": 0, "owned_by": "x", "mode": "chat"} for name in names)
    entries = tuple((name, name) for name in names)
    upstream = json.dumps({"models": [{"slug": "upstream-model", "priority": 0}]}).encode()

    listed = json.loads(codex_model_list_body(listing, entries, router))["models"]
    merged = json.loads(merged_codex_catalog(upstream, codex_catalog_rows(listing, entries, router)))["models"][1:]

    assert [entry["slug"] for entry in merged] == list(names)
    assert [entry.get("supports_search_tool") for entry in merged] == [
        entry.get("supports_search_tool") for entry in listed
    ]
    assert [entry["supports_search_tool"] for entry in merged[:2]] == [True, False]


_SOL_SLUG: Final = next(
    slug for slug, model in bundled_codex_models().items() if model.default_service_tier is not None
)


@pytest.mark.parametrize("upstream", ["openai/{slug}", "{slug}"])
@pytest.mark.parametrize("slug", sorted(bundled_codex_models()))
def test_a_model_codex_knows_keeps_codexs_entry_under_any_provider_prefix(upstream: str, slug: str) -> None:
    stock: Final = bundled_codex_models()[slug].model_dump(mode="json")

    (entry,) = json.loads(
        codex_models_response_json(
            (_row("team-model", upstream_model=upstream.format(slug=slug)),), byte_limit=None
        ).json
    )["models"]

    assert entry["model_messages"] == stock["model_messages"]
    assert entry["supported_reasoning_levels"] == stock["supported_reasoning_levels"]


_BEDROCK_UPSTREAMS: Final = MappingProxyType(
    {
        "bedrock": (
            "bedrock/us.openai.{name}",
            "bedrock/global.openai.{name}",
            "bedrock/eu.openai.{name}",
            "bedrock/converse/us.openai.{name}",
            "bedrock/openai.{name}",
        ),
        "bedrock_mantle": ("bedrock_mantle/openai.{name}", "bedrock_mantle/us-east-1/openai.{name}"),
    }
)
_LISTING_FIELDS: Final = frozenset({"slug", "display_name", "priority"})
_TIER_FIELDS: Final = frozenset({"service_tiers", "additional_speed_tiers", "default_service_tier"})


def _bedrock_names(endpoint: Literal["bedrock", "bedrock_mantle"]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(bedrock_model_name_without_provider(slug) for slug in bundled_bedrock_codex_models(endpoint))
    )


def _bedrock_stock(endpoint: Literal["bedrock", "bedrock_mantle"], name: str) -> dict[str, object]:
    model: Final = next(
        model
        for slug, model in bundled_bedrock_codex_models(endpoint).items()
        if bedrock_model_name_without_provider(slug) == name
    )
    return model.model_dump(mode="json", exclude_unset=True)


def _codex_fields(
    entry: Mapping[str, object], stock: Mapping[str, object], skipped: frozenset[str]
) -> dict[str, object]:
    return {field: entry.get(field, _ABSENT) for field in stock if field not in skipped}


def _without(stock: Mapping[str, object], skipped: frozenset[str]) -> dict[str, object]:
    return {field: value for field, value in stock.items() if field not in skipped}


def _only_entry(*rows: CodexCatalogRow, client_version: str | None = None) -> dict[str, object]:
    (entry,) = json.loads(
        codex_models_response_json(rows, instructions=_PROMPT, byte_limit=None, client_version=client_version).json
    )["models"]
    return entry


def _bedrock_cases() -> Iterator[tuple[str, str, str]]:
    for endpoint, upstreams in _BEDROCK_UPSTREAMS.items():
        for upstream in upstreams:
            for name in _bedrock_names(endpoint):
                yield endpoint, upstream, name


@pytest.mark.parametrize(("endpoint", "upstream", "name"), tuple(_bedrock_cases()))
def test_a_bedrock_deployment_takes_codexs_entry_for_its_bedrock_endpoint_not_openais(
    endpoint: Literal["bedrock", "bedrock_mantle"], upstream: str, name: str
) -> None:
    stock: Final = _bedrock_stock(endpoint, name)

    entry: Final = _only_entry(_row("team-model", upstream_model=upstream.format(name=name)))
    openai_entry: Final = _only_entry(_row("team-model", upstream_model=f"openai/{name}"))

    assert _codex_fields(entry, stock, _LISTING_FIELDS) == _without(stock, _LISTING_FIELDS)
    assert _codex_fields(openai_entry, stock, _LISTING_FIELDS) != _without(stock, _LISTING_FIELDS)


@pytest.mark.parametrize("model", ["us.openai.{name}", "converse/global.openai.{name}"])
def test_a_bedrock_deployment_naming_its_provider_apart_from_the_model_takes_codexs_bedrock_entry(model: str) -> None:
    name: Final = _bedrock_names("bedrock")[0]
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "team-model",
                "litellm_params": {
                    "model": model.format(name=name),
                    "custom_llm_provider": "bedrock",
                    "aws_region_name": "us-east-1",
                },
            }
        ]
    )
    stock: Final = _bedrock_stock("bedrock", name)

    entry: Final = _served(router, "team-model")["team-model"]

    assert _codex_fields(entry, stock, _LISTING_FIELDS | _TIER_FIELDS) == _without(
        stock, _LISTING_FIELDS | _TIER_FIELDS
    )


@pytest.mark.parametrize(
    "upstreams",
    [
        ("openai/{name}", "bedrock/us.openai.{name}"),
        ("bedrock/us.openai.{name}", "openai/{name}"),
        ("bedrock_mantle/openai.{name}", "bedrock/us.openai.{name}"),
        ("openai/{name}", "bedrock_mantle/openai.{name}", "bedrock/us.openai.{name}"),
    ],
)
def test_a_model_name_any_of_whose_deployments_is_on_bedrock_runtime_takes_codexs_runtime_entry(
    upstreams: tuple[str, ...],
) -> None:
    name: Final = next(name for name in _bedrock_names("bedrock") if name in bundled_codex_models())
    router: Final = litellm.Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": upstream.format(name=name)}} for upstream in upstreams
        ]
    )
    stock: Final = _bedrock_stock("bedrock", name)

    entry: Final = _served(router, name)[name]

    assert _codex_fields(entry, stock, _LISTING_FIELDS | _TIER_FIELDS) == _without(
        stock, _LISTING_FIELDS | _TIER_FIELDS
    )


@pytest.mark.parametrize(
    "upstreams", [("openai/{name}", "bedrock_mantle/openai.{name}"), ("bedrock_mantle/openai.{name}",)]
)
def test_a_model_name_on_bedrock_mantle_but_not_runtime_takes_codexs_mantle_entry(upstreams: tuple[str, ...]) -> None:
    name: Final = next(name for name in _bedrock_names("bedrock_mantle") if name in bundled_codex_models())
    router: Final = litellm.Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": upstream.format(name=name)}} for upstream in upstreams
        ]
    )
    stock: Final = _bedrock_stock("bedrock_mantle", name)

    entry: Final = _served(router, name)[name]

    assert _codex_fields(entry, stock, _LISTING_FIELDS | _TIER_FIELDS) == _without(
        stock, _LISTING_FIELDS | _TIER_FIELDS
    )


def _left_out_of_bedrock() -> Iterator[tuple[str, str]]:
    for endpoint, upstream in (
        ("bedrock", "bedrock/us.openai.{slug}"),
        ("bedrock_mantle", "bedrock_mantle/openai.{slug}"),
    ):
        for slug in sorted(frozenset(bundled_codex_models()) - frozenset(_bedrock_names(endpoint))):
            yield upstream, slug


@pytest.mark.parametrize(("upstream", "slug"), tuple(_left_out_of_bedrock()))
def test_a_bedrock_deployment_of_a_model_its_bedrock_catalog_leaves_out_gets_a_translated_entry(
    upstream: str, slug: str
) -> None:
    entry: Final = _only_entry(_row("team-model", upstream_model=upstream.format(slug=slug)))

    assert "model_messages" not in entry and entry["base_instructions"] == _PROMPT


def _bedrock_endpoint_names() -> Iterator[tuple[str, str]]:
    for endpoint in _BEDROCK_UPSTREAMS:
        for name in _bedrock_names(endpoint):
            yield endpoint, name


def _just_below(version: str) -> str:
    release: Final = Version(version).release
    major, minor, micro = (*release, 0, 0)[:3]
    return f"{major}.{minor}.{micro - 1}" if micro else f"{major}.{minor - 1}.999"


@pytest.mark.parametrize(("endpoint", "name"), tuple(_bedrock_endpoint_names()))
def test_no_bedrock_entry_reaches_a_codex_whose_decoder_needs_fields_debug_models_leaves_out(
    endpoint: Literal["bedrock", "bedrock_mantle"], name: str
) -> None:
    """Codex rust-v0.147.0's ModelInfo requires `supports_parallel_tool_calls`, which 0.166.0's
    `codex debug models` does not print (checked in openai/codex on 2026-11-09)."""
    entry: Final = _only_entry(
        _row("team-model", upstream_model=_BEDROCK_UPSTREAMS[endpoint][0].format(name=name)), client_version="0.147.0"
    )

    assert "model_messages" not in entry and entry["base_instructions"] == _PROMPT


@pytest.mark.parametrize(("endpoint", "name"), tuple(_bedrock_endpoint_names()))
def test_a_codex_too_old_to_decode_a_bedrock_entry_gets_a_translated_one_instead(
    endpoint: Literal["bedrock", "bedrock_mantle"], name: str
) -> None:
    built_from: Final = bundled_codex_models()[name].minimal_client_version
    assert built_from is None or isinstance(built_from, str)
    minimum: Final = max(
        (version for version in (built_from, CODEX_BEDROCK_MODELS_MINIMAL_CLIENT_VERSION) if version is not None),
        key=Version,
    )
    row: Final = _row("team-model", upstream_model=_BEDROCK_UPSTREAMS[endpoint][0].format(name=name))

    older: Final = _only_entry(row, client_version=_just_below(minimum))
    at_minimum: Final = _only_entry(row, client_version=minimum)

    assert "model_messages" not in older and older["base_instructions"] == _PROMPT
    assert at_minimum["model_messages"] == _bedrock_stock(endpoint, name)["model_messages"]


@pytest.mark.parametrize("listed_as", ["{name}", "us.openai.{name}", "eu.openai.{name}"])
def test_a_bedrock_model_listed_under_its_own_name_keeps_codexs_defaults_in_the_native_catalog(listed_as: str) -> None:
    name: Final = next(
        name for name in _bedrock_names("bedrock") if _bedrock_stock("bedrock", name).get("default_reasoning_level")
    )
    upstream_model: Final = f"bedrock/us.openai.{name}"
    upstream: Final = json.dumps({"models": [{"slug": "upstream-model", "priority": 0}]}).encode()

    merged: Final = merged_codex_catalog(
        upstream,
        (
            _row(listed_as.format(name=name), upstream_model=upstream_model),
            _row("team-model", upstream_model=upstream_model),
        ),
        instructions=_PROMPT,
        client_version="0.166.0",
    )

    own_name, alias = json.loads(merged)["models"][1:]
    assert own_name["default_reasoning_level"] == _bedrock_stock("bedrock", name)["default_reasoning_level"]
    assert alias["default_reasoning_level"] is None


def test_an_upstream_with_no_provider_prefix_matches_only_by_its_own_name() -> None:
    (unknown, known) = json.loads(
        codex_models_response_json(
            (
                _row("first", upstream_model=f"us.openai.{_SOL_SLUG}"),
                _row("second", upstream_model=_SOL_SLUG),
            ),
            instructions=_PROMPT,
        ).json
    )["models"]

    assert "model_messages" not in unknown and unknown["base_instructions"] == _PROMPT
    assert known["model_messages"] == bundled_codex_models()[_SOL_SLUG].model_dump(mode="json")["model_messages"]


@pytest.mark.parametrize(
    "upstream",
    ["bedrock/us.openai.{slug}-mini", "bedrock/us.meta.{slug}x", "bedrock_mantle/openai.{slug}.1", "{slug}-0"],
)
def test_a_model_whose_name_only_starts_like_one_codex_knows_gets_a_translated_entry(upstream: str) -> None:
    (entry,) = json.loads(
        codex_models_response_json(
            (_row("team-model", upstream_model=upstream.format(slug=_SOL_SLUG)),), instructions=_PROMPT
        ).json
    )["models"]

    assert "model_messages" not in entry and entry["base_instructions"] == _PROMPT


def _listing(*names: str, max_input_tokens: int | None = None) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "id": name,
            "object": "model",
            "created": 0,
            "owned_by": "x",
            "mode": "chat",
            "max_input_tokens": max_input_tokens,
        }
        for name in names
    )


def _served(
    router: litellm.Router, *names: str, team_id: str | None = None, **listing: int
) -> dict[str, Mapping[str, object]]:
    body: Final = codex_model_list_body(
        _listing(*names, **listing), tuple((name, name) for name in names), router, team_id, client_version="0.166.0"
    )
    return {entry["slug"]: entry for entry in json.loads(body)["models"]}


def _vllm(name: str, host: str, **model_info: object) -> dict[str, object]:
    return {
        "model_name": name,
        "litellm_params": {"model": f"hosted_vllm/coder-{host}", "api_base": f"https://{host}.example/v1"},
        "model_info": model_info,
    }


def test_images_are_offered_only_when_every_deployment_behind_the_name_says_it_reads_them() -> None:
    router: Final = litellm.Router(
        model_list=[
            _vllm("mixed", "h0", supports_vision=True),
            _vllm("mixed", "h1", supports_vision=False),
            _vllm("vision-and-silent", "h2", supports_vision=True),
            _vllm("vision-and-silent", "h3"),
            _vllm("all-vision", "h3a", supports_vision=True),
            _vllm("all-vision", "h3b", supports_vision=True),
            _vllm("undeclared", "h3c"),
        ]
    )

    served: Final = _served(router, "mixed", "vision-and-silent", "all-vision", "undeclared")

    assert served["mixed"]["input_modalities"] == ["text"]
    assert served["vision-and-silent"]["input_modalities"] == ["text"]
    assert served["all-vision"]["input_modalities"] == ["text", "image"]
    assert served["undeclared"]["input_modalities"] == ["text"]


def test_a_silent_deployment_does_not_vote_against_tool_calls_but_one_without_them_does() -> None:
    router: Final = litellm.Router(
        model_list=[
            _vllm("tools-and-silent", "h4", supports_function_calling=True),
            _vllm("tools-and-silent", "h5"),
            _vllm("tools-and-none", "h6", supports_function_calling=True),
            _vllm("tools-and-none", "h7", supports_function_calling=False),
        ]
    )

    served: Final = _served(router, "tools-and-silent", "tools-and-none")

    assert served["tools-and-silent"]["apply_patch_tool_type"] == "freeform"
    assert served["tools-and-none"]["apply_patch_tool_type"] is None


def test_a_mixed_group_takes_the_smallest_window_any_deployment_declares() -> None:
    router: Final = litellm.Router(
        model_list=[
            _vllm("mixed", "h8", max_input_tokens=200_000),
            _vllm("mixed", "h9", max_input_tokens=128_000, supports_vision=True),
            _vllm("mixed", "h10"),
            _vllm("undeclared", "h11"),
        ]
    )

    served: Final = _served(router, "mixed", "undeclared", max_input_tokens=200_000)

    assert served["mixed"]["context_window"] == 128_000
    assert served["undeclared"]["context_window"] == 200_000


def test_a_claude_like_deployment_gets_codexs_coding_setup() -> None:
    router: Final = litellm.Router(
        model_list=[
            _vllm(
                "claude-like",
                "h-claude",
                max_input_tokens=1_000_000,
                supports_vision=True,
                supports_function_calling=True,
                supports_reasoning=True,
            ),
            _vllm("plain", "h12"),
            _vllm("no-reasoning", "h12b", supports_reasoning=False),
        ]
    )

    served: Final = _served(router, "claude-like", "plain", "no-reasoning")

    claude_like: Final = served["claude-like"]
    assert claude_like["apply_patch_tool_type"] == "freeform"
    assert claude_like["truncation_policy"] == {"mode": "tokens", "limit": 10000}
    assert claude_like["supports_reasoning_summary_parameter"] is True
    assert claude_like["context_window"] == 1_000_000
    assert served["plain"]["apply_patch_tool_type"] is None
    assert "supports_reasoning_summary_parameter" not in served["plain"]
    assert served["no-reasoning"]["supports_reasoning_summary_parameter"] is False


def test_capabilities_are_read_only_off_the_deployments_the_key_team_routes_to() -> None:
    router: Final = litellm.Router(
        model_list=[
            _vllm("coder", "h13", supports_vision=True),
            _vllm("coder", "h14", supports_vision=False, team_id="team-2"),
        ]
    )

    assert _served(router, "coder", team_id="team-1")["coder"]["input_modalities"] == ["text", "image"]
    assert _served(router, "coder", team_id="team-2")["coder"]["input_modalities"] == ["text"]


def _tier_fields(entry: Mapping[str, object]) -> tuple[object, object, object]:
    return (entry["service_tiers"], entry.get("additional_speed_tiers"), entry.get("default_service_tier"))


def test_codexs_own_tiers_stay_only_when_every_deployment_behind_the_name_prices_them() -> None:
    stock: Final = bundled_codex_models()[_SOL_SLUG].model_dump(mode="json")
    assert stock["service_tiers"] and stock["additional_speed_tiers"] and stock["default_service_tier"]
    token_price: Final = {"input_cost_per_token": 1e-06}
    priority_price: Final = {**token_price, "input_cost_per_token_priority": 1e-05}

    def upstream(host: str) -> dict[str, str]:
        return {"model": f"hosted_vllm/{_SOL_SLUG}", "api_base": f"https://{host}.example/v1"}

    router: Final = litellm.Router(
        model_list=[
            {"model_name": "all-priced", "litellm_params": upstream("a"), "model_info": priority_price},
            {"model_name": "all-priced", "litellm_params": upstream("b"), "model_info": priority_price},
            {"model_name": "half-priced", "litellm_params": upstream("c"), "model_info": priority_price},
            {"model_name": "half-priced", "litellm_params": upstream("d"), "model_info": token_price},
            {"model_name": "priced-and-unknown", "litellm_params": upstream("f"), "model_info": priority_price},
            {"model_name": "priced-and-unknown", "litellm_params": {"model": "azure/prod-sol-eastus"}},
            {
                "model_name": "configured",
                "litellm_params": upstream("e"),
                "model_info": {"service_tiers": ["priority"]},
            },
        ]
    )

    served: Final = _served(router, "all-priced", "half-priced", "priced-and-unknown", "configured")

    assert _tier_fields(served["all-priced"]) == _tier_fields(stock)
    assert _tier_fields(served["priced-and-unknown"]) == _tier_fields(stock)
    assert _tier_fields(served["half-priced"]) == ([], [], None)
    assert served["half-priced"]["model_messages"] == stock["model_messages"]
    assert [tier["id"] for tier in served["configured"]["service_tiers"]] == ["priority"]


def _tiered_stock(default_service_tier: str) -> Mapping[str, CodexStockModel]:
    tiers: Final = (
        CodexServiceTier(id="priority", name="Fast", description="faster"),
        CodexServiceTier(id="flex", name="Flex", description="cheaper"),
    )
    return MappingProxyType(
        {
            "tiered": CodexStockModel(
                slug="tiered",
                display_name="Tiered",
                priority=0,
                visibility="list",
                service_tiers=tiers,
                default_service_tier=default_service_tier,
            )
        }
    )


@pytest.mark.parametrize(("default_tier", "kept_default"), [("priority", "priority"), ("flex", None)])
def test_dropping_one_stock_tier_keeps_the_default_only_while_it_is_still_offered(default_tier, kept_default) -> None:
    priority_and_flex: Final = CodexDeploymentCapabilities(
        prices_tokens=True,
        priced_service_tier_keys=frozenset(
            get_service_tier_cost_key("input_cost_per_token", tier) for tier in ("priority", "flex")
        ),
    )
    priority_only: Final = CodexDeploymentCapabilities(
        prices_tokens=True,
        priced_service_tier_keys=frozenset({get_service_tier_cost_key("input_cost_per_token", "priority")}),
    )
    row: Final = _row("tiered", deployment_capabilities=(priority_and_flex, priority_only))

    (entry,) = json.loads(
        codex_models_response_json((row,), stock=_tiered_stock(default_tier), instructions=_PROMPT).json
    )["models"]

    assert [tier["id"] for tier in entry["service_tiers"]] == ["priority"]
    assert entry["default_service_tier"] == kept_default


def test_model_info_that_does_not_validate_declares_no_capability_but_keeps_its_prices() -> None:
    priced: Final = get_service_tier_cost_key("input_cost_per_token", "priority")

    capabilities: Final = deployment_capabilities(
        {"supports_vision": "sometimes", "supports_function_calling": False, priced: 1e-05}, "odd"
    )

    assert capabilities == CodexDeploymentCapabilities(priced_service_tier_keys=frozenset({priced}))


def test_model_list_body_takes_the_asking_releases_catalog_and_prompt() -> None:
    router: Final = litellm.Router(
        model_list=[
            {"model_name": "gpt-release-only", "litellm_params": {"model": "openai/gpt-release-only"}},
            {"model_name": "in-house", "litellm_params": {"model": "openai/in-house-coder"}},
        ]
    )
    release_entry: Final = bundled_codex_models()["gpt-5.5"].model_copy(
        update={"slug": "gpt-release-only", "display_name": "Release Only"}
    )
    names: Final = ("gpt-release-only", "in-house")

    body: Final = codex_model_list_body(
        _listing(*names),
        tuple((name, name) for name in names),
        router,
        client_version="0.170.0",
        stock={"gpt-release-only": release_entry},
        instructions="release prompt",
    )

    served: Final = {entry["slug"]: entry for entry in json.loads(body)["models"]}
    assert (served["gpt-release-only"]["display_name"], served["in-house"]["base_instructions"]) == (
        "Release Only",
        "release prompt",
    ), served


class _CollidingName(str):
    def __hash__(self) -> int:
        return 1


def test_catalog_etag_ignores_the_order_a_process_iterates_priced_service_tiers_in() -> None:
    def row_with_tiers(*tiers: str) -> CodexCatalogRow:
        capabilities: Final = CodexDeploymentCapabilities.model_construct(
            priced_service_tier_keys=frozenset(_CollidingName(tier) for tier in tiers)
        )
        return replace(_row("gpt-6-astra"), deployment_capabilities=(capabilities,))

    assert codex_catalog_etag("upstream", [row_with_tiers("flex", "priority")]) == codex_catalog_etag(
        "upstream", [row_with_tiers("priority", "flex")]
    )

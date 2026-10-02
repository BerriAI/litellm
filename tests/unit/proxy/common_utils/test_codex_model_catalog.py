"""The Codex catalog `/v1/models?client_version=...` serves decodes on Codex and says what the operator set."""

import json
from types import MappingProxyType

import pytest

import litellm
from litellm.proxy.common_utils.codex_model_catalog import (
    CODEX_CATALOG_BYTE_LIMIT,
    CodexCatalogRow,
    CodexServiceTier,
    CodexStockModel,
    CodexStockUpgrade,
    bundled_codex_models,
    codex_catalog_rows,
    codex_models_response_json,
    configured_service_tiers,
)

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
        else {"service_tiers": overrides["deployments"]} if "deployments" in overrides else {}
    )
    listing = {key: value for key, value in overrides.items() if key not in ("service_tiers", "deployments")}
    return CodexCatalogRow(**{"id": model_id, "mode": "responses", "max_input_tokens": 272000, **listing, **per_deployment})


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


def test_unknown_model_entry_carries_every_field_codex_deserializes_without_a_default():
    """Codex 0.159.3 `codex-rs/protocol/src/openai_models.rs` ModelInfo, the fields with no
    `#[serde(default)]`, read on 2026-10-01; `base_instructions` because its `ModelsResponse`
    decoder rejects an entry with neither it nor `model_messages.instructions_template`."""
    (entry,) = _models(_row("gpt-6-astra"))

    assert entry.keys() >= {
        "slug",
        "display_name",
        "description",
        "supported_reasoning_levels",
        "shell_type",
        "visibility",
        "supported_in_api",
        "priority",
        "availability_nux",
        "upgrade",
        "support_verbosity",
        "default_verbosity",
        "apply_patch_tool_type",
        "truncation_policy",
        "experimental_supported_tools",
    }
    assert (entry["slug"], entry["display_name"], entry["visibility"], entry["supported_in_api"]) == (
        "gpt-6-astra",
        "gpt-6-astra",
        "list",
        True,
    )
    assert (entry["context_window"], entry["base_instructions"], entry["service_tiers"]) == (272000, _PROMPT, [])


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

    assert [(entry["slug"], entry["priority"]) for entry in entries] == [("gpt-6-sol", 0), ("mystery", 1), ("gpt-5.5", 2)]


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
        (_row("gpt-6-astra", service_tiers=["ultrafast"]), [("ultrafast", "Ultrafast", "Sends service_tier=ultrafast upstream")]),
        (
            _row("gpt-6-astra", service_tiers=[{"id": "ultrafast", "name": "Ultra fast", "description": "Fastest"}]),
            [("ultrafast", "Ultra fast", "Fastest")],
        ),
        (_row("gpt-6-astra", service_tiers=[{"id": "ultrafast"}]), [("ultrafast", "Ultrafast", "Sends service_tier=ultrafast upstream")]),
        (
            _row("gpt-5.5", service_tiers=["ultrafast"]),
            [("ultrafast", "Ultrafast", "Sends service_tier=ultrafast upstream")],
        ),
        (
            _row("gpt-5.5", service_tiers=["priority", "ultrafast"]),
            [("priority", "Fast", "2x speed, increased usage"), ("ultrafast", "Ultrafast", "Sends service_tier=ultrafast upstream")],
        ),
        (_row("gpt-5.5", service_tiers=[]), []),
        (_row("gpt-5.5", service_tiers=["ultrafast", "priority", "ultrafast", {"id": "priority"}]), [
            ("ultrafast", "Ultrafast", "Sends service_tier=ultrafast upstream"),
            ("priority", "Fast", "2x speed, increased usage"),
        ]),
    ],
)
def test_configured_service_tiers_replace_stock_tiers_and_a_known_id_keeps_its_codex_name(row, expected):
    (entry,) = _models(row)

    assert _tiers(entry) == expected


@pytest.mark.parametrize("invalid", ["ultrafast", [""], ["  "], [1], [{"name": "no id"}], [{"id": "x", "bogus": 1}], {"id": "x"}])
def test_invalid_service_tiers_are_ignored_and_the_stock_tiers_stay(invalid):
    stock_entry, fallback_entry = _models(_row("gpt-5.5", service_tiers=invalid), _row("mystery", service_tiers=invalid))

    assert _tiers(stock_entry) == [("priority", "Fast", "2x speed, increased usage")]
    assert fallback_entry["service_tiers"] == []


@pytest.mark.parametrize(
    ("deployments", "expected"),
    [
        ((["ultrafast"], ["ultrafast"]), [("ultrafast", "Ultrafast", "Sends service_tier=ultrafast upstream")]),
        (
            (["priority", "ultrafast"], ["ultrafast", "flex", "priority"]),
            [("priority", "Fast", "2x speed, increased usage"), ("ultrafast", "Ultrafast", "Sends service_tier=ultrafast upstream")],
        ),
        ((["priority", "ultrafast"], ["ultrafast"]), [("ultrafast", "Ultrafast", "Sends service_tier=ultrafast upstream")]),
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
    stock_entry, fallback_entry = _models(_row("gpt-5.5", deployments=(None, None)), _row("mystery", deployments=(None, None)))

    assert _tiers(stock_entry) == [("priority", "Fast", "2x speed, increased usage")]
    assert fallback_entry["service_tiers"] == []


def test_an_invalid_value_on_one_deployment_ignores_the_configuration_for_the_whole_model():
    (entry,) = _models(_row("gpt-5.5", deployments=(["ultrafast"], "not-a-list")))

    assert _tiers(entry) == [("priority", "Fast", "2x speed, increased usage")]


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, "priority"), (["priority", "ultrafast"], "priority"), (["ultrafast"], None), ([], None)],
)
def test_default_service_tier_survives_only_while_still_offered(configured, expected):
    (entry,) = _models(_row("gpt-6-sol", service_tiers=configured))

    assert entry["default_service_tier"] == expected


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


def test_body_stops_before_the_entry_that_would_cross_the_byte_limit():
    rows = tuple(_row(f"model-{index}") for index in range(6))
    unlimited = _body(*rows, byte_limit=None)
    limit = len(unlimited.json.encode()) - 1

    body = _body(*rows, byte_limit=limit)
    one_more = _body(*rows[: len(body.listed) + 1], byte_limit=None)

    assert len(body.json.encode()) <= limit < len(one_more.json.encode())
    assert body.listed + body.left_out == tuple(row.id for row in rows)
    assert body.left_out == ("model-5",)
    assert [entry["slug"] for entry in json.loads(body.json)["models"]] == list(body.listed)


def test_unlimited_body_keeps_every_entry():
    rows = tuple(_row(f"model-{index}") for index in range(3))

    body = _body(*rows, byte_limit=None)

    assert (body.listed, body.left_out) == (tuple(row.id for row in rows), ())
    assert CODEX_CATALOG_BYTE_LIMIT == 1024 * 1024


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
        {"id": "gpt-5.5-team", "object": "model", "created": 0, "owned_by": "openai", "mode": "chat", "max_input_tokens": 7},
        {"id": "plain", "object": "model", "created": 0, "owned_by": "openai"},
    )

    rows = codex_catalog_rows(listing, (("gpt-5.5-team", "model_name_team-1_c0ffee"), ("plain", "plain")), router)

    assert rows == (
        CodexCatalogRow(
            id="gpt-5.5-team",
            mode="chat",
            max_input_tokens=7,
            upstream_model="openai/gpt-5.5",
            display_name="Team 5.5",
            service_tiers=(["ultrafast"], None),
        ),
        CodexCatalogRow(id="plain", upstream_model="openai/some-unmapped-model", service_tiers=(None,)),
    )


def test_catalog_rows_without_a_router_carry_only_the_listing():
    listing = ({"id": "plain", "object": "model", "created": 0, "owned_by": "openai", "mode": "chat"},)

    assert codex_catalog_rows(listing, (("plain", "plain"),), None) == (CodexCatalogRow(id="plain", mode="chat"),)

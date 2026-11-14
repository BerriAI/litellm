"""
The Python host of the native model cost map: installing a document as ``litellm.model_cost``, the
``register_model`` overlay across reloads, and the lookups over the dict.
"""

import asyncio
import json
import logging
import re
import threading
from collections.abc import Callable, Mapping
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.fallback_generalizations import (
    get_fallback_generalization_rules,
    match_routing_generalization,
    set_fallback_generalizations,
)
from litellm.litellm_core_utils.get_model_cost_map import (
    AliasedCostEntry,
    GetModelCostMap,
    ModelCostMapReloaded,
    ModelCostMapReloadUnavailable,
    fetch_registration_document,
    get_model_cost_map_provenance,
    git_blob_id,
    install_model_cost_map,
    refetch_model_cost_map,
    registration_rows,
)
from litellm.rust_bridge._native import NativeInstalled, NativeModelCostMap, NativeRegistered
from litellm.utils import catalog_with_runtime_registrations, get_model_cost_key, runtime_model_cost_registrations

FALLBACK_GENERALIZATIONS_KEY: Final = "fallback_generalizations"
_BUNDLED: Final = GetModelCostMap.read_local_model_cost_map_bytes()


def _marked_utf8(text: str) -> bytes:
    return b"\xef\xbb\xbf" + text.encode()


def _utf16(text: str) -> bytes:
    return text.encode("utf-16")


def _document_with(**rows: Mapping[str, object]) -> bytes:
    """The bundled document with some rows replaced, so it passes the integrity limits."""
    return json.dumps({**json.loads(_BUNDLED), **rows}).encode()


@pytest.fixture
def own_model_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restores the binding of ``litellm.model_cost`` after a test that installs a new table."""
    monkeypatch.setattr(litellm, "model_cost", litellm.model_cost)


# ---------------------------------------------------------------------------
# The shipped document
# ---------------------------------------------------------------------------


def test_shipped_backup_carries_the_claude_routing_rules():
    """The bundled backup must ship the Claude routing rules so a fresh install
    (or an offline fallback) routes unknown Claude models without code changes.
    Bedrock-syntax ids must hit the bedrock rule before the bare-id Anthropic rule."""
    backup = GetModelCostMap.load_local_model_cost_map()
    rules = backup.get(FALLBACK_GENERALIZATIONS_KEY, {}).get("rules", [])
    names = [r.get("name") for r in rules]
    assert names.index("bedrock-claude-ids") < names.index("anthropic-claude-ids")

    previous = list(get_fallback_generalization_rules())
    try:
        set_fallback_generalizations(rules)
        assert match_routing_generalization("claude-opus-4-9") == "anthropic"
        assert match_routing_generalization("global.anthropic.claude-opus-4-9") == "bedrock"
    finally:
        set_fallback_generalizations(previous)


def test_shipped_routing_rules_never_match_through_an_unrecognized_namespace():
    """Routing rules decide ``litellm_provider`` for otherwise-unknown ids, and the
    proxy's wildcard access check (``can_key_call_model`` with a ``bedrock/*`` key)
    trusts that inference: it rebuilds ``{provider}/{model}`` and matches it against
    the key's patterns. A routing pattern that matches as a substring lets
    ``bedrockz/anthropic.claude-...`` resolve to bedrock and slip through a
    ``bedrock/*`` key, so every shipped routing rule must anchor to the start of
    the name and never match an id carrying an unrecognized namespace prefix."""
    backup = GetModelCostMap.load_local_model_cost_map()
    rules = backup[FALLBACK_GENERALIZATIONS_KEY]["rules"]

    routing_rules = [r for r in rules if "litellm_provider" in r["model_info"]]
    assert routing_rules
    assert all(r["pattern"].startswith("^") for r in routing_rules)

    previous = list(get_fallback_generalization_rules())
    try:
        set_fallback_generalizations(rules)
        for bedrock_id in [
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "anthropic.claude-v2:1",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "us-gov.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "global.anthropic.claude-fable-5-20260120-v1:0",
        ]:
            assert match_routing_generalization(bedrock_id) == "bedrock", bedrock_id
        for namespaced in [
            "bedrockz/anthropic.claude-3-5-sonnet-20240620",
            "bedrockz/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "bedrockz/claude-3-5-sonnet-20240620",
        ]:
            assert match_routing_generalization(namespaced) is None, namespaced
    finally:
        set_fallback_generalizations(previous)


def test_shipped_backup_marks_claude_4_6_plus_adaptive_not_4_0():
    """Adaptive thinking is data, not code. The bundled backup must carry
    supports_adaptive_thinking on genuine Claude >= 4.6 entries (every provider
    route) and on the version-gated anthropic-claude-adaptive-thinking rule for
    unmapped future Claudes, while leaving the dated Claude 4.0 names
    ("...-4-20250514") unflagged so a date can never be mistaken for a 4.6+ minor
    version. The version-neutral claude-family-baseline capability rule must not flag
    it, so an unmapped sub-4.6 name resolves but stays non-adaptive. The adaptive rule
    carries only its delta; capability unioning stacks it onto the baseline, so the
    baseline block is never duplicated across rules and no rule needs ``extends``."""
    backup = GetModelCostMap.load_local_model_cost_map()

    rules = backup[FALLBACK_GENERALIZATIONS_KEY]["rules"]
    baseline_rule = next(r for r in rules if r.get("name") == "claude-family-baseline")
    adaptive_rule = next(r for r in rules if r.get("name") == "claude-adaptive-thinking")
    assert "supports_adaptive_thinking" not in baseline_rule["model_info"]
    assert "litellm_provider" not in baseline_rule["model_info"]
    assert adaptive_rule["model_info"] == {"supports_adaptive_thinking": True}
    assert all("extends" not in r for r in rules)

    for adaptive in [
        "anthropic.claude-opus-4-8",
        "vertex_ai/claude-opus-4-6@default",
        "us.anthropic.claude-sonnet-4-6",
        "azure_ai/claude-opus-4-7",
    ]:
        assert backup[adaptive]["supports_adaptive_thinking"] is True, adaptive

    for non_adaptive in [
        "us.anthropic.claude-opus-4-20250514-v1:0",
        "claude-opus-4-5",
    ]:
        assert "supports_adaptive_thinking" not in backup[non_adaptive], non_adaptive


# ---------------------------------------------------------------------------
# Runtime reload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        ({"min_model_count": -1, "max_shrink_ratio": 0.5}, "min_model_count must be 0 or more, got -1"),
        ({"min_model_count": 0, "max_shrink_ratio": 1.5}, "max_shrink_ratio must be between 0 and 1, got 1.5"),
    ],
)
def test_integrity_limits_out_of_range_name_the_limit_and_value(limits: dict[str, float], message: str):
    with pytest.raises(ValueError, match=re.escape(message)):
        NativeModelCostMap.bundled(**limits)


# ---------------------------------------------------------------------------
# What litellm.model_cost promises
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("own_model_cost")
def test_model_cost_is_a_dict_that_reflects_each_install():
    first_body: Final = _document_with(**{"lit-first": {"litellm_provider": "openai"}})
    second_body: Final = _document_with(**{"lit-second": {"litellm_provider": "openai"}})

    install_model_cost_map(first_body)
    first: Final = litellm.model_cost
    install_model_cost_map(second_body)

    assert type(litellm.model_cost) is dict
    assert litellm.model_cost is not first
    assert "lit-second" in litellm.model_cost
    assert "lit-first" not in litellm.model_cost


@pytest.mark.usefixtures("own_model_cost")
def test_registrations_survive_a_reload_in_call_order():
    litellm.register_model(
        {"lit-registered": {"litellm_provider": "openai", "input_cost_per_token": 1, "mode": "chat"}}
    )
    litellm.register_model({"lit-registered": {"input_cost_per_token": 2, "output_cost_per_token": 3}})

    install_model_cost_map(_BUNDLED)

    assert litellm.model_cost["lit-registered"]["input_cost_per_token"] == 2
    assert litellm.model_cost["lit-registered"]["output_cost_per_token"] == 3
    assert litellm.model_cost["lit-registered"]["mode"] == "chat"
    assert litellm.get_model_info("lit-registered")["input_cost_per_token"] == 2


@pytest.mark.usefixtures("own_model_cost")
def test_the_latest_registration_of_a_row_wins_in_the_registered_catalog_and_after_a_reload():
    litellm.register_model({"gpt-4o": {"input_cost_per_token": 1.0}})
    litellm.register_model({"openai/gpt-4o": {"input_cost_per_token": 2.0}})
    assert catalog_with_runtime_registrations()["gpt-4o"]["input_cost_per_token"] == 2.0

    litellm.register_model({"gpt-4o": {"input_cost_per_token": 3.0}})
    install_model_cost_map(_BUNDLED)

    assert litellm.model_cost["gpt-4o"]["input_cost_per_token"] == 3.0
    assert catalog_with_runtime_registrations()["gpt-4o"]["input_cost_per_token"] == 3.0


@pytest.mark.usefixtures("own_model_cost")
def test_a_registration_moves_to_the_row_a_reload_adds_for_it():
    litellm.register_model({"openai/lit-arrives": {"litellm_provider": "openai", "input_cost_per_token": 5.0}})

    install_model_cost_map(_document_with(**{"lit-arrives": {"litellm_provider": "openai", "mode": "chat"}}))

    assert litellm.model_cost["lit-arrives"]["input_cost_per_token"] == 5.0
    assert catalog_with_runtime_registrations()["lit-arrives"]["input_cost_per_token"] == 5.0


@pytest.mark.usefixtures("own_model_cost")
def test_a_reload_counts_the_rows_it_serves_and_not_the_registrations():
    litellm.register_model({"lit-only-registered": {"litellm_provider": "openai", "mode": "chat"}})

    result: Final = install_model_cost_map(_BUNDLED)

    assert isinstance(result, ModelCostMapReloaded)
    assert result.model_count == len(GetModelCostMap.loaded_model_cost_map()) == len(litellm.model_cost) - 1


@pytest.mark.usefixtures("own_model_cost")
def test_a_reload_adds_the_models_it_serves_to_the_provider_sets():
    install_model_cost_map(_document_with(**{"lit-reloaded-chat": {"litellm_provider": "openai", "mode": "chat"}}))

    assert "lit-reloaded-chat" in litellm.models_by_provider["openai"]


@pytest.mark.usefixtures("own_model_cost")
def test_a_reload_repeats_the_cache_pricing_warning_for_a_registration_no_catalog_row_matches(
    caplog: pytest.LogCaptureFixture,
):
    litellm.register_model({"lit-unmatched": {"litellm_provider": "openai", "input_cost_per_token": 1e-06}})
    caplog.clear()

    with caplog.at_level(logging.WARNING, logger=litellm.verbose_logger.name):
        install_model_cost_map(_BUNDLED)

    assert any(
        "lit-unmatched" in record.message and "cache_creation_input_token_cost" in record.message
        for record in caplog.records
    ), caplog.text


@pytest.mark.usefixtures("own_model_cost")
def test_a_registration_that_does_not_persist_is_gone_after_a_reload():
    litellm.register_model(
        {"lit-request-scoped": {"litellm_provider": "openai", "input_cost_per_token": 1}}, persist_across_reloads=False
    )
    assert "lit-request-scoped" in litellm.model_cost

    install_model_cost_map(_BUNDLED)

    assert "lit-request-scoped" not in litellm.model_cost


@pytest.mark.usefixtures("own_model_cost")
def test_a_registration_wins_the_fields_it_shares_with_a_request_scoped_row_after_a_reload():
    litellm.register_model({"gpt-4o": {"input_cost_per_token": 7.0}})
    litellm.register_model(
        {"gpt-4o": {"input_cost_per_token": 9.0, "output_cost_per_token": 8.0}}, persist_across_reloads=False
    )

    install_model_cost_map(_BUNDLED)

    assert litellm.model_cost["gpt-4o"]["input_cost_per_token"] == 7.0
    assert (
        litellm.model_cost["gpt-4o"]["output_cost_per_token"] == json.loads(_BUNDLED)["gpt-4o"]["output_cost_per_token"]
    )


@pytest.mark.usefixtures("own_model_cost")
def test_a_rejected_document_changes_nothing():
    installed: Final = litellm.model_cost
    rows: Final = dict(installed)
    provenance: Final = get_model_cost_map_provenance()

    result: Final = install_model_cost_map(json.dumps({"only": {"litellm_provider": "openai"}}).encode())

    assert isinstance(result, ModelCostMapReloadUnavailable)
    assert litellm.model_cost is installed
    assert litellm.model_cost == rows
    assert get_model_cost_map_provenance() == provenance


@pytest.mark.usefixtures("own_model_cost")
def test_a_reader_iterating_the_old_table_finishes_while_another_thread_installs():
    old: Final = litellm.model_cost
    installer: Final = threading.Thread(target=install_model_cost_map, args=(_BUNDLED,))
    walked: Final = iter(old.items())
    first: Final = next(walked)
    installer.start()
    installer.join()

    assert len((first, *walked)) == len(old)
    assert litellm.model_cost is not old


@pytest.mark.usefixtures("own_model_cost")
def test_get_model_info_follows_a_reload_and_a_registration():
    litellm.get_model_info("gpt-4o")
    install_model_cost_map(_document_with(**{"gpt-4o": {"litellm_provider": "openai", "input_cost_per_token": 0.5}}))
    assert litellm.get_model_info("gpt-4o")["input_cost_per_token"] == 0.5

    litellm.register_model({"gpt-4o": {"input_cost_per_token": 0.25}})
    assert litellm.get_model_info("gpt-4o")["input_cost_per_token"] == 0.25


@pytest.mark.usefixtures("own_model_cost")
def test_the_registered_catalog_reads_the_document_under_registrations_and_never_request_scoped_rows():
    litellm.register_model({"lit-kept": {"litellm_provider": "openai", "input_cost_per_token": 1}})
    litellm.register_model({"lit-request-row": {"litellm_provider": "openai"}}, persist_across_reloads=False)

    for _ in range(2):
        catalog = catalog_with_runtime_registrations()
        assert catalog["lit-kept"]["input_cost_per_token"] == 1
        assert "lit-request-row" not in catalog
        assert catalog["gpt-4o"] == GetModelCostMap.loaded_model_cost_map()["gpt-4o"]
        install_model_cost_map(_BUNDLED)


@pytest.mark.usefixtures("own_model_cost")
def test_the_loaded_catalog_follows_installs_and_ignores_registrations():
    install_model_cost_map(_document_with(**{"lit-loaded": {"litellm_provider": "openai", "lit_field": 1}}))
    litellm.register_model({"lit-loaded": {"input_cost_per_token": 5}})

    assert GetModelCostMap.loaded_model_cost_map()["lit-loaded"] == {"litellm_provider": "openai", "lit_field": 1}
    assert "lit_field" in GetModelCostMap.loaded_model_cost_map_fields()

    install_model_cost_map(_document_with(**{"lit-loaded": {"litellm_provider": "openai", "lit_other_field": 2}}))

    assert GetModelCostMap.loaded_model_cost_map()["lit-loaded"] == {"litellm_provider": "openai", "lit_other_field": 2}
    assert "lit_field" not in GetModelCostMap.loaded_model_cost_map_fields()
    assert "lit_other_field" in GetModelCostMap.loaded_model_cost_map_fields()


@pytest.mark.usefixtures("own_model_cost")
@pytest.mark.parametrize("add_to_provider_model_sets", [True, False])
def test_register_model_adds_names_to_the_provider_sets_only_when_asked_and_a_reload_keeps_it_so(
    add_to_provider_model_sets: bool,
):
    chat_model: Final = f"lit-registered-chat-{add_to_provider_model_sets}".lower()
    litellm.register_model(
        {
            "openrouter/lit-vendor/lit-model": {"litellm_provider": "openrouter", "mode": "chat"},
            chat_model: {"litellm_provider": "openai", "mode": "chat"},
        },
        add_to_provider_model_sets=add_to_provider_model_sets,
    )
    install_model_cost_map(_BUNDLED)

    assert ("lit-vendor/lit-model" in litellm.openrouter_models) is add_to_provider_model_sets
    assert (chat_model in litellm.models_by_provider["openai"]) is add_to_provider_model_sets


@pytest.mark.usefixtures("own_model_cost")
def test_install_hands_the_fallback_rules_to_the_generalization_lookup():
    rules: Final = [{"name": "lit-rule", "pattern": "^lit-family-", "model_info": {"litellm_provider": "openai"}}]
    previous: Final = list(get_fallback_generalization_rules())
    try:
        install_model_cost_map(_document_with(**{FALLBACK_GENERALIZATIONS_KEY: {"rules": rules}}))
        assert match_routing_generalization("lit-family-1") == "openai"
        assert FALLBACK_GENERALIZATIONS_KEY not in litellm.model_cost
    finally:
        set_fallback_generalizations(previous)


@pytest.mark.usefixtures("own_model_cost")
@pytest.mark.parametrize("value", ["mirrored 2026-11-30", None, [1, 2], 5], ids=["string", "null", "list", "number"])
def test_a_document_with_a_top_level_value_that_is_not_a_row_changes_nothing(value: object):
    installed: Final = litellm.model_cost
    price: Final = installed["gpt-4o"]["input_cost_per_token"]
    provenance: Final = get_model_cost_map_provenance()
    body: Final = json.dumps(
        {
            **json.loads(_BUNDLED),
            "gpt-4o": {"litellm_provider": "openai", "input_cost_per_token": 0.5},
            "_comment": value,
        }
    ).encode()

    result: Final = install_model_cost_map(body)

    assert isinstance(result, ModelCostMapReloadUnavailable)
    assert "_comment" in result.reason
    assert litellm.model_cost is installed
    assert litellm.model_cost["gpt-4o"]["input_cost_per_token"] == price
    assert get_model_cost_map_provenance() == provenance


@pytest.mark.usefixtures("own_model_cost")
@pytest.mark.parametrize(
    "encode",
    [_marked_utf8, _utf16],
    ids=["utf-8-byte-order-mark", "utf-16"],
)
def test_a_document_that_starts_with_a_byte_order_mark_installs(encode: Callable[[str], bytes]):
    row: Final = {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 0.5}

    result: Final = install_model_cost_map(encode(_document_with(**{"lit-marked": row}).decode()))

    assert isinstance(result, ModelCostMapReloaded)
    assert litellm.model_cost["lit-marked"] == row


@pytest.mark.parametrize("encode", [str.encode, _marked_utf8, _utf16], ids=["utf-8", "utf-8-byte-order-mark", "utf-16"])
def test_a_registration_document_reads_as_its_rows_without_the_rules_block(encode: Callable[[str], bytes]):
    row: Final = {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 4e-06}

    rows: Final = registration_rows(encode(json.dumps({"lit-url-model": row, FALLBACK_GENERALIZATIONS_KEY: {}})))

    assert rows == {"lit-url-model": row}


@pytest.mark.parametrize(
    "body", [b"not json", b"[1, 2]", b'{"lit-url-model": "not a row"}'], ids=["text", "list", "row"]
)
def test_a_registration_document_that_is_not_an_object_of_rows_is_a_value_error(body: bytes):
    with pytest.raises(ValueError):  # noqa: PT011  # the message is json's or pydantic's, only the type is ours
        registration_rows(body)


def test_register_model_from_a_url_fetches_nothing_in_offline_mode(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setenv("LITELLM_OFFLINE_MODE", "true")

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        document: Final = fetch_registration_document("http://127.0.0.1:9/cost.json")

    assert document is None
    assert "did not fetch http://127.0.0.1:9/cost.json because LITELLM_OFFLINE_MODE is set" in caplog.text


@pytest.mark.usefixtures("own_model_cost")
def test_git_blob_id_is_the_revision_an_install_records():
    body: Final = _document_with(**{"lit-revision": {"litellm_provider": "openai", "mode": "chat"}})

    install_model_cost_map(body)

    assert git_blob_id(body) == get_model_cost_map_provenance()["source_revision"]


@pytest.mark.usefixtures("own_model_cost")
def test_alias_names_share_one_entry_after_an_install_and_a_registration():
    install_model_cost_map(
        _document_with(**{"lit-owner": {"litellm_provider": "openai", "mode": "chat", "aliases": ["lit-alias"]}})
    )
    assert litellm.model_cost["lit-alias"] is litellm.model_cost["lit-owner"]

    litellm.register_model({"lit-alias": {"input_cost_per_token": 3e-06}})

    entry: Final = litellm.model_cost["lit-owner"]
    assert litellm.model_cost["lit-alias"] is entry
    assert isinstance(entry, AliasedCostEntry)
    assert entry.registered_fields == frozenset({"input_cost_per_token"})
    assert runtime_model_cost_registrations("lit-owner", "openai") == (("lit-owner", {"input_cost_per_token": 3e-06}),)

    litellm.register_model({"lit-owner": {"output_cost_per_token": 4e-06}})

    assert runtime_model_cost_registrations("lit-owner", "openai") == (
        ("lit-owner", {"input_cost_per_token": 3e-06, "output_cost_per_token": 4e-06}),
    )


def test_without_registrations_the_registered_catalog_is_the_loaded_catalog():
    catalog: Final = catalog_with_runtime_registrations()

    assert {key: dict(row) for key, row in catalog.items()} == {
        key: dict(row) for key, row in GetModelCostMap.loaded_model_cost_map().items()
    }


# ---------------------------------------------------------------------------
# A second instance over a dict of its own
# ---------------------------------------------------------------------------


def _own_instance(document: Mapping[str, Mapping[str, object]]) -> tuple[NativeModelCostMap, dict]:
    instance: Final = NativeModelCostMap.bundled(0, 0.0)
    installed: Final = instance.install(json.dumps(document).encode())
    assert isinstance(installed, NativeInstalled)
    return instance, installed.rows


def test_alias_of_rows_are_served_as_their_target_and_dangling_ones_are_dropped():
    _, rows = _own_instance(
        {
            "real": {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 1e-6},
            "other": {"litellm_provider": "openai", "mode": "chat", "aliases": ["twin-latest"]},
            "real-latest": {"alias_of": "real"},
            "twin-latest": {"alias_of": "real"},
            "ghost-latest": {"alias_of": "ghost"},
            "chain-latest": {"alias_of": "real-latest"},
            "listed-latest": {"alias_of": "missing", "aliases": ["extra"]},
        }
    )

    assert rows["real-latest"] is rows["real"]
    assert "alias_of" not in rows["real"]
    assert set(rows) == {"real", "other", "real-latest", "twin-latest"}
    assert rows["twin-latest"] is rows["other"]


def test_an_instance_writes_only_the_published_dict_it_is_handed(monkeypatch: pytest.MonkeyPatch):
    instance, rows = _own_instance({"lit-own": {"litellm_provider": "openai", "mode": "chat"}})
    previous: Final = litellm.model_cost
    monkeypatch.setattr(litellm, "model_cost", rows)

    written: Final = instance.register(rows, {"lit-own-new": {"litellm_provider": "openai"}}, True)

    assert isinstance(written, NativeRegistered)
    assert "lit-own-new" in rows
    assert "lit-own-new" not in previous
    assert instance.find_key(rows, "LIT-OWN-NEW") == "lit-own-new"


def test_the_name_index_is_a_cache():
    instance, rows = _own_instance(
        {
            "lit-a": {"litellm_provider": "p1"},
            "LIT-B": {"litellm_provider": "p2"},
            "lit-c": {"litellm_provider": "p1"},
        }
    )

    def queries() -> tuple[list[str | None], list[list[str]]]:
        return (
            [instance.find_key(rows, name) for name in ("LIT-A", "lit-b", "missing")],
            [instance.keys_in(rows, provider) for provider in ("p1", "p2", "p3")],
        )

    before: Final = queries()
    instance.invalidate()

    assert queries() == before
    assert before == (["lit-a", "LIT-B", None], [["lit-a", "lit-c"], ["LIT-B"], []])


def test_a_key_written_straight_to_the_dict_is_found_until_the_next_install():
    instance, rows = _own_instance({"lit-a": {"litellm_provider": "p"}})
    rows.setdefault("Lit-Direct", {"litellm_provider": "p"})
    assert instance.find_key(rows, "lit-direct") == "Lit-Direct"

    reinstalled: Final = instance.install(json.dumps({"lit-a": {"litellm_provider": "p"}}).encode())
    assert isinstance(reinstalled, NativeInstalled)
    assert instance.find_key(reinstalled.rows, "lit-direct") is None


def test_a_key_swapped_in_for_a_popped_one_replaces_it_in_lookups():
    instance, rows = _own_instance({"lit-a": {"litellm_provider": "p"}, "Lit-Gone": {"litellm_provider": "p"}})
    assert instance.find_key(rows, "lit-gone") == "Lit-Gone"
    rows.pop("Lit-Gone")
    rows["Lit-New"] = {"litellm_provider": "p"}

    assert instance.find_key(rows, "lit-new") == "Lit-New"
    assert instance.find_key(rows, "lit-gone") is None


def test_a_rebound_dict_is_what_lookups_read_next():
    instance, rows = _own_instance({"lit-a": {"litellm_provider": "p"}})
    rebound: Final = {"Lit-Rebound": {"litellm_provider": "p"}}

    assert instance.find_key(rows, "lit-rebound") is None
    assert instance.find_key(rebound, "lit-rebound") == "Lit-Rebound"


def test_case_insensitive_lookups_go_through_the_live_dict(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "model_cost", {**litellm.model_cost, "Lit-Mixed-Case": {"litellm_provider": "p"}})

    assert get_model_cost_key("GPT-4O") == "gpt-4o"
    assert get_model_cost_key("lit-mixed-case") == "Lit-Mixed-Case"
    assert get_model_cost_key("lit-unknown") is None


def test_main_cost_map_names_downstream_code_calls_stay_callable_with_their_main_signatures():
    # Not an API we would design today. Downstream open source SDK code calls these names from main directly, so
    # removing one breaks it. Their behavior is pinned in tests/integration/sdk/test_model_cost_sdk.py
    import inspect

    from litellm.litellm_core_utils import get_model_cost_map as cost_map_module
    from litellm.litellm_core_utils.get_model_cost_map import AliasedCostEntry, GetModelCostMap, ModelCostMapReloaded

    assert litellm.utils._model_cost_lowercase_map is None
    generation: Final = litellm.utils.get_model_cost_mutation_generation()
    litellm.utils._invalidate_model_cost_lowercase_map()
    assert litellm.utils.get_model_cost_mutation_generation() > generation
    for cached in (
        litellm.utils._cached_get_model_info,
        litellm.utils._cached_get_model_info_helper,
        litellm.utils.cached_cost_map_key,
        litellm.get_model_info,
    ):
        cached.cache_clear()
        assert cached.cache_info().currsize == 0
    assert litellm.utils._cached_get_model_info("gpt-4o")["key"] == "gpt-4o"
    assert GetModelCostMap.get_backup_model_count() == GetModelCostMap._get_backup_model_count() > 0
    assert GetModelCostMap.validate_model_cost_map(fetched_map={}, backup_model_count=1) is False
    assert GetModelCostMap.validate_model_cost_map(
        fetched_map=GetModelCostMap.load_local_model_cost_map(),
        backup_model_count=GetModelCostMap.get_backup_model_count(),
    )
    for function, arguments in (
        (cost_map_module.refetch_model_cost_map, {"url": "u", "timeout": 5, "max_attempts": 1, "client": None}),
        (cost_map_module.get_model_cost_map, {"url": "u", "timeout": 5, "max_attempts": 1, "sleep": None, "rng": None}),
        (cost_map_module.adopt_model_cost_map, {"new_model_cost_map": {}}),
        (cost_map_module._finalize_model_cost_map, {"model_cost": {}}),
    ):
        inspect.signature(function).bind(**arguments)
    assert {"model_cost_map", "revision", "etag"} <= set(dir(ModelCostMapReloaded))
    assert issubclass(AliasedCostEntry, dict)


@pytest.mark.usefixtures("own_model_cost")
def test_main_cost_map_helpers_keep_their_side_effects(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_model_cost_map import (
        _finalize_model_cost_map,  # pyright: ignore[reportPrivateUsage]  # downstream code calls main's finalizer
    )

    document: Final = json.loads(_BUNDLED)
    assert FALLBACK_GENERALIZATIONS_KEY in document

    rows: Final = _finalize_model_cost_map(document)

    assert FALLBACK_GENERALIZATIONS_KEY not in document
    assert FALLBACK_GENERALIZATIONS_KEY not in rows
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    with pytest.warns(DeprecationWarning, match="ignored"):
        asyncio.run(refetch_model_cost_map(url="http://127.0.0.1:9/unused.json", sleep=asyncio.sleep))

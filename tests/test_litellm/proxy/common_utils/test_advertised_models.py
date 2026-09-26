"""Tests for `general_settings.advertised_models`, the catalog-only entries
appended to the model listing by `litellm.proxy.common_utils.advertised_models`.
"""

from typing import Final

import litellm
from litellm.constants import DEFAULT_MODEL_CREATED_AT_TIME
from litellm.proxy.common_utils.advertised_models import (
    advertised_model_rows,
    configured_advertised_models,
)

CATALOG_ENTRY: Final = {"id": "catalog-only-model", "owned_by": "example-provider"}


def _settings(*entries: object) -> dict[str, object]:
    return {"advertised_models": list(entries)}


def test_entry_becomes_a_row_carrying_its_configured_id_and_owner():
    rows: Final = advertised_model_rows(_settings(CATALOG_ENTRY), [])

    assert rows == (
        {
            "id": "catalog-only-model",
            "object": "model",
            "created": DEFAULT_MODEL_CREATED_AT_TIME,
            "owned_by": "example-provider",
            "catalog_only": True,
        },
    ), f"expected one row carrying exactly the configured id and owner, got {rows}"


def test_entry_naming_an_already_listed_model_is_dropped():
    """A routed model is never displaced by a catalog entry that shares its id."""
    rows: Final = advertised_model_rows(_settings({"id": "routed-model", "owned_by": "impostor"}), ["routed-model"])

    assert rows == (), f"catalog entry must not shadow the routed model already listed, got {rows}"


def test_repeated_id_is_listed_once_keeping_the_first_entry():
    rows: Final = advertised_model_rows(
        _settings(CATALOG_ENTRY, {"id": "catalog-only-model", "owned_by": "second-declaration"}),
        [],
    )

    assert [row["owned_by"] for row in rows] == ["example-provider"], (
        f"a repeated id should be listed once, keeping the first declaration, got {rows}"
    )


def test_no_setting_produces_no_rows():
    assert advertised_model_rows({}, []) == (), "an unset advertised_models must add nothing to the listing"


def test_malformed_setting_is_skipped_rather_than_raising():
    """Misconfiguration fails open: the listing is returned as if nothing was set."""
    missing_owner: Final = advertised_model_rows({"advertised_models": [{"id": "no-owner"}]}, [])
    not_a_list: Final = advertised_model_rows({"advertised_models": "catalog-only-model"}, [])

    assert missing_owner == (), f"an entry without owned_by must be skipped, got {missing_owner}"
    assert not_a_list == (), f"a non-list advertised_models must be skipped, got {not_a_list}"


def test_row_carries_only_declared_fields_even_for_a_model_the_cost_map_knows():
    """Catalog rows are exactly what the operator declared, never enriched.

    The id is taken from the live cost map rather than hard-coded, so this keeps
    testing the no-enrichment guarantee as the catalog changes.
    """
    known_model: Final = next(iter(litellm.model_cost))

    rows: Final = advertised_model_rows(_settings({"id": known_model, "owned_by": "example-provider"}), [])

    assert rows == (
        {
            "id": known_model,
            "object": "model",
            "created": DEFAULT_MODEL_CREATED_AT_TIME,
            "owned_by": "example-provider",
            "catalog_only": True,
        },
    ), f"a catalog row must not pick up cost map details for {known_model}, got {rows}"


def test_configured_entries_are_typed_and_keep_their_order():
    entries: Final = configured_advertised_models(
        _settings(CATALOG_ENTRY, {"id": "second", "owned_by": "other-provider"})
    )

    assert [(entry.id, entry.owned_by) for entry in entries] == [
        ("catalog-only-model", "example-provider"),
        ("second", "other-provider"),
    ], f"entries should be parsed in configured order, got {entries}"


class _RouterStub:
    """Minimal stand-in for the bits of Router this module reads."""

    def __init__(
        self,
        names: tuple[str, ...],
        groups: tuple[str, ...] = (),
        team_public: frozenset[str] = frozenset(),
        aliases: tuple[str, ...] = (),
    ) -> None:
        self._names: Final = names
        self._groups: Final = groups
        self.team_public_model_names: Final = team_public
        self.model_group_alias: Final = {alias: "some-target" for alias in aliases}

    def get_model_names(self) -> list[str]:
        """Mirrors the real method: team-scoped names and aliases are absent here."""
        return list(self._names)

    def get_model_access_groups(self) -> dict[str, list[str]]:
        return {group: [] for group in self._groups}


def test_entry_naming_a_routed_model_hidden_from_this_caller_is_dropped():
    """A catalog entry must not re-expose a deployment the listing filtered out.

    `listed_ids` carries only what this caller may see, so an id that was scoped
    away, paused or health-filtered would otherwise reappear under any owner the
    config names.
    """
    rows: Final = advertised_model_rows(
        _settings({"id": "hidden-deployment", "owned_by": "impostor"}),
        [],
        _RouterStub(("hidden-deployment",)),
    )

    assert rows == (), f"a routed model absent from this caller's listing must stay absent, got {rows}"


def test_entry_naming_an_access_group_is_dropped():
    rows: Final = advertised_model_rows(
        _settings({"id": "beta-models", "owned_by": "impostor"}),
        [],
        _RouterStub((), ("beta-models",)),
    )

    assert rows == (), f"a catalog entry must not shadow an access group name, got {rows}"


def test_entry_is_listed_when_the_router_knows_nothing_about_it():
    rows: Final = advertised_model_rows(
        _settings(CATALOG_ENTRY),
        ["routed-model"],
        _RouterStub(("routed-model", "another-model")),
    )

    assert [row["id"] for row in rows] == ["catalog-only-model"], (
        f"an id no deployment claims should still be listed, got {rows}"
    )


def test_every_row_is_marked_catalog_only():
    rows: Final = advertised_model_rows(_settings(CATALOG_ENTRY, {"id": "second", "owned_by": "other"}), [])

    assert [row.get("catalog_only") for row in rows] == [
        True,
        True,
    ], f"every catalog row must be marked so clients can tell it from a routable model, got {rows}"


def test_entry_naming_a_team_scoped_public_name_is_dropped():
    """`get_model_names()` omits team-scoped deployments when given no team id."""
    rows: Final = advertised_model_rows(
        _settings({"id": "team-public-gpt", "owned_by": "impostor"}),
        [],
        _RouterStub((), team_public=frozenset({"team-public-gpt"})),
    )

    assert rows == (), f"a team's public model name must stay reserved, got {rows}"


def test_entry_naming_a_model_group_alias_is_dropped():
    """`get_model_names()` does not return alias keys despite its docstring."""
    rows: Final = advertised_model_rows(
        _settings({"id": "gpt-alias", "owned_by": "impostor"}),
        [],
        _RouterStub((), aliases=("gpt-alias",)),
    )

    assert rows == (), f"a routable alias must stay reserved, got {rows}"

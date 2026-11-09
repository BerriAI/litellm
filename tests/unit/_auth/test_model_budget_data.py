from math import inf, isnan, nan
from typing import Final

import pytest

from litellm._auth import model_budget_data as native
from litellm._auth.model_budget_data import (
    ModelBudgetEntry,
    ModelBudgetQueryFacts,
    ModelBudgetResolveFacts,
    ModelBudgetSnapshot,
)

from .budget_reference import model as reference


@pytest.mark.parametrize("limit", (None, -1.0, 0.0, 1.0, inf, nan))
@pytest.mark.parametrize("local", (None, -1.0, 0.0, 1.0, 2.0, inf, nan))
@pytest.mark.parametrize("shared", (None, 0.0, 2.0, inf, nan))
@pytest.mark.parametrize("now", (99, 100))
def test_raw_counter_and_shared_window_case_table(
    limit: float | None, local: float | None, shared: float | None, now: float
) -> None:
    facts: Final = ModelBudgetSnapshot(limit, local, 0, shared, 40, 60, now)
    result: Final = native.check(facts)
    expected: Final = reference(facts)
    assert result.denied == expected.denied
    assert result.read is None
    assert (
        isnan(result.current_spend) and isnan(expected.current_spend)
    ) or result.current_spend == expected.current_spend


@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (ModelBudgetQueryFacts("Key", ModelBudgetSnapshot(1)), "Local"),
        (ModelBudgetQueryFacts("Key", ModelBudgetSnapshot(-1)), None),
        (ModelBudgetQueryFacts("Key", ModelBudgetSnapshot(None)), None),
        (ModelBudgetQueryFacts("Key", ModelBudgetSnapshot(1, 0), 60, 59), "Legacy"),
        (ModelBudgetQueryFacts("EndUser", ModelBudgetSnapshot(1, 0), 60, 59), "Legacy"),
        (ModelBudgetQueryFacts("EndUser", ModelBudgetSnapshot(1, 0), 60, 60), None),
        (ModelBudgetQueryFacts("User", ModelBudgetSnapshot(1, 0), 60, 1), "Shared"),
        (ModelBudgetQueryFacts("Team", ModelBudgetSnapshot(1, 0), 60, 1, redis_present=True), None),
        (ModelBudgetQueryFacts("TeamMember", ModelBudgetSnapshot(1, 1), 60), None),
        (ModelBudgetQueryFacts("Key", ModelBudgetSnapshot(1, 0), 60, same_counter=True), "Shared"),
        (ModelBudgetQueryFacts("Key", ModelBudgetSnapshot(1, 0, 0), 60, shared_read=True), None),
    ),
)
def test_native_selects_the_next_io_read(facts: ModelBudgetQueryFacts, expected: str | None) -> None:
    assert native.query(facts).read == expected


def entry(name: str, limit: float | None = 1, duration: str | None = "1h") -> ModelBudgetEntry:
    return ModelBudgetEntry(name, limit, duration, 3600 if duration is not None else None)


@pytest.mark.parametrize("limit", (None, -1.0, 0.0, 1.0, inf, nan))
@pytest.mark.parametrize("overrides", (("alias",), ("target",), ("alias", "target")))
def test_team_exact_entries_are_removed_across_all_alias_names(limit: float | None, overrides: tuple[str, ...]) -> None:
    key: Final = tuple(entry(name, limit) for name in overrides)
    facts: Final = tuple(ModelBudgetResolveFacts(model, model, "", (), key, True) for model in ("alias", "target"))
    excluded: Final = native.team_exclusions(facts)
    assert excluded == (tuple(sorted(overrides)) if limit is not None and limit >= 0 else ())
    resolution: Final = native.resolve(
        ModelBudgetResolveFacts(
            "alias",
            "alias",
            "",
            (entry("alias"), entry("target"), entry("*")),
            access_groups=("alias", "target"),
            excluded_entries=excluded,
        )
    )
    expected: Final = "alias" if "alias" not in excluded else "target" if "target" not in excluded else "*"
    assert resolution.entry is not None and resolution.entry.name == expected


@pytest.mark.parametrize(
    ("entries", "groups", "expected"),
    (
        ((entry("*"), entry("m*"), entry("model")), ("group",), "model"),
        ((entry("*"), entry("m*"), entry("group")), ("group",), "group"),
        ((entry("*"), entry("m*"), entry("model", duration=None)), (), "m*"),
        ((entry("provider/model"), entry("model")), (), "provider/model"),
        ((entry("b"), entry("a"), entry("*")), ("b", "a"), "a"),
        ((entry("model", None), entry("*")), (), "model"),
        ((entry("m?del"),), (), None),
        ((entry("*a*"), entry("mo*"), entry("mod*l")), (), "mod*l"),
    ),
)
def test_resolution_prioritizes_exact_then_groups_then_specific_globs(
    entries: tuple[ModelBudgetEntry, ...],
    groups: tuple[str, ...],
    expected: str | None,
) -> None:
    result: Final = native.resolve(ModelBudgetResolveFacts("provider/model", "", "", entries, access_groups=groups))
    assert (result.entry.name if result.entry is not None else None) == expected
    assert result.read_groups is False


@pytest.mark.parametrize(
    ("key_limit", "expected"), ((0, "*"), (1, "*"), (-1, "model"), (None, "model"), (nan, "model"))
)
def test_only_enforced_named_key_entry_replaces_the_team_entry(key_limit: float | None, expected: str) -> None:
    facts: Final = ModelBudgetResolveFacts(
        "model", "", "", (entry("model"), entry("*")), (entry("model", key_limit),), team=True, access_groups=()
    )
    result: Final = native.resolve(facts)
    assert result.entry is not None
    assert result.entry.name == expected

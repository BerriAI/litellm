import pytest

import litellm
from litellm.proxy.management_endpoints.what_if_endpoints import (
    _GIT_REF,
    _already_in,
    _at_todays_price,
    _patch_drop,
    _recall_row,
    _swap_row,
    _swap_team,
)

MODEL = "gpt-4.1"


def _day(day: str, spend: float, prompt_tokens: int = 1_000_000, completion_tokens: int = 100_000) -> dict[str, object]:
    return {
        "day": day,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "requests": 10,
        "spend": spend,
        "max_budget": 100.0,
    }


def test_recall_lists_only_the_days_logged_at_the_wrong_price():
    right = _at_todays_price(MODEL, 1_000_000, 100_000)
    row = _recall_row("search", MODEL, (_day("2026-01-01", right), _day("2026-01-02", 0.0), _day("2026-01-03", 0.0)))

    assert row is not None
    assert (row.first_day, row.last_day, row.days) == ("2026-01-02", "2026-01-03", 2)
    assert row.logged == 0.0
    assert row.corrected == 2 * right
    assert row.delta == row.corrected


def test_recall_skips_a_model_whose_days_all_match_the_cost_map():
    right = _at_todays_price(MODEL, 1_000_000, 100_000)

    assert _recall_row("search", MODEL, (_day("2026-01-01", right), _day("2026-01-02", right * 1.005))) is None


def test_swap_prices_the_same_tokens_on_the_successor_the_cost_map_names():
    successor = litellm.model_cost[MODEL]["successor_model"]
    row = _swap_row("search", MODEL, (_day("2026-01-01", 1.0), _day("2026-01-02", 1.0)))

    assert row is not None
    assert row.successor == successor
    assert row.now == _at_todays_price(MODEL, 2_000_000, 200_000)
    assert row.after == _at_todays_price(successor, 2_000_000, 200_000)
    assert row.delta == row.after - row.now


def test_swap_skips_a_model_with_no_successor():
    assert _swap_row("search", "not-a-real-model", (_day("2026-01-01", 1.0),)) is None


def test_swap_team_budget_runs_out_sooner_when_the_successor_costs_more():
    days = (_day("2026-01-01", 30.0), _day("2026-01-02", 30.0))
    row = _swap_row("search", MODEL, days)
    assert row is not None
    pricier = row.model_copy(update={"delta": 60.0})

    team = _swap_team("search", days, (pricier, pricier.model_copy(update={"team": "other"})), window_days=30)

    assert (team.now, team.after) == (60.0, 120.0)
    assert team.budget_lasts_days_now == 50.0
    assert team.budget_lasts_days_after == 25.0


BASE = "def handler():\n    return 1\n"
UPSTREAM = {
    "v1.0.0": {"lts_fix.py": BASE, "main_fix.py": BASE, "ours.py": BASE},
    "lts": {"lts_fix.py": BASE + "def backported():\n    return 2\n", "main_fix.py": BASE, "ours.py": BASE},
    "main": {"lts_fix.py": BASE, "main_fix.py": BASE + "def main_only():\n    return 3\n", "ours.py": BASE},
}
LOCAL_PATCHES = {
    "lts_fix.py": "+++ b/lts_fix.py\n+def backported():\n+    return 2\n",
    "main_fix.py": "+++ b/main_fix.py\n+def main_only():\n+    return 3\n",
    "ours.py": "+++ b/ours.py\n+def only_ours():\n+    return 4\n",
}


def _customer_tree(tree: str, *args: str) -> str | None:
    match args:
        case ("describe", *_):
            return "v1.0.0\n"
        case ("tag", *_):
            return "v1.0.0\n"
        case ("diff", "--name-only", _):
            return "\n".join(LOCAL_PATCHES)
        case ("diff", _, "--", path):
            return LOCAL_PATCHES[path]
        case ("show", spec):
            ref, _, path = spec.partition(":")
            return UPSTREAM[ref].get(path)
    return None


def test_patch_drop_sorts_local_patches_by_where_upstream_has_them():
    report = _patch_drop("/srv/litellm", lts_branch="lts", main_branch="main", git=_customer_tree)

    assert report.installed == "v1.0.0"
    assert {patch.path: patch.found_in for patch in report.patches} == {
        "lts_fix.py": "lts",
        "main_fix.py": "main",
        "ours.py": None,
    }
    assert report.droppable == 1


def test_a_patch_is_not_upstream_until_every_added_line_is():
    half = "+def backported():\n+    return 2\n+def never_merged():\n"

    assert _already_in(_customer_tree, "/srv/litellm", "lts", "lts_fix.py", "+def backported():\n+    return 2\n")
    assert not _already_in(_customer_tree, "/srv/litellm", "lts", "lts_fix.py", half)


@pytest.mark.parametrize("ref", ["--output=/tmp/probe", "-c", "main --output=x", "main;id", ""])
def test_a_branch_that_git_would_read_as_an_option_is_not_a_ref(ref: str):
    assert _GIT_REF.fullmatch(ref) is None


@pytest.mark.parametrize("ref", ["origin/main", "origin/lts/1.108.x", "v1.108.15", "litellm_whatif_demos"])
def test_branch_and_tag_names_are_refs(ref: str):
    assert _GIT_REF.fullmatch(ref) is not None

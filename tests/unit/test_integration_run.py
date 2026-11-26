from pathlib import Path
from typing import Final

import pytest

from tests.integration.run import GROUPS, _write_assignment, assign_shards, select, uncollected, worker_settings

_GROUP: Final = (
    "tests/integration/cost_calculation/test_cost_tracking.py",
    "tests/integration/cost_calculation/test_rollups.py",
)
_CELL: Final = (
    "tests/integration/cost_calculation/test_cost_tracking.py"
    "::test_case_bills_expected_cost[perplexity/pplx-decider-v1-27b-decisions]"
)


def test_a_node_id_inside_a_group_file_is_selected_as_written() -> None:
    selection: Final = select((_CELL,), _GROUP)
    assert selection.nodes == (_CELL,)
    assert selection.foreign == ()


def test_a_node_id_outside_the_group_is_foreign_by_its_file() -> None:
    foreign: Final = "tests/integration/providers/test_decisions_wire.py::test_key_checks_match_chat"
    assert select((foreign, _CELL), _GROUP).foreign == (foreign,)


def test_no_request_selects_every_group_file() -> None:
    assert select((), _GROUP).nodes == _GROUP


def test_a_node_id_whose_file_collected_tests_is_not_empty() -> None:
    collected: Final = frozenset({_CELL, "tests/integration/cost_calculation/test_cost_tracking.py::test_other"})
    assert uncollected((_CELL,), collected) == ()


def test_a_selected_file_that_collected_nothing_is_reported() -> None:
    assert uncollected(_GROUP, frozenset({_CELL})) == ("tests/integration/cost_calculation/test_rollups.py",)


@pytest.mark.parametrize("group", tuple(GROUPS))
def test_every_group_run_by_several_workers_gives_each_worker_its_own_stack(group: str) -> None:
    assert worker_settings(group, 2)["INTEGRATION_WORKER_ISOLATION"] == "1"


@pytest.mark.parametrize("group", tuple(GROUPS))
def test_a_single_worker_run_keeps_the_supplied_deployment(group: str) -> None:
    assert worker_settings(group, 1) == {}


def test_a_file_heavier_than_the_rest_combined_gets_a_shard_to_itself() -> None:
    durations: Final = {"heavy.py": 100.0, "a.py": 10.0, "b.py": 10.0, "c.py": 10.0}
    assert assign_shards(("a.py", "b.py", "c.py", "heavy.py"), durations, 2) == (("heavy.py",), ("a.py", "b.py", "c.py"))


@pytest.mark.parametrize("count", (1, 2, 3, 7))
def test_every_file_lands_in_exactly_one_shard(count: int) -> None:
    files: Final = tuple(f"test_{index}.py" for index in range(10))
    shards: Final = assign_shards(files, {f"test_{index}.py": float(index % 4) for index in range(10)}, count)
    assert len(shards) == count
    assert sorted(file for shard in shards for file in shard) == sorted(files)


def test_files_missing_from_the_durations_are_spread_across_shards() -> None:
    files: Final = ("known_a.py", "known_b.py", "new_1.py", "new_2.py", "new_3.py", "new_4.py")
    shards: Final = assign_shards(files, {"known_a.py": 10.0, "known_b.py": 10.0}, 2)
    assert [len(shard) for shard in shards] == [3, 3], shards


def test_more_shards_than_files_leaves_the_extra_shards_empty() -> None:
    assert assign_shards(("only.py",), {}, 3) == (("only.py",), (), ())


def test_the_written_assignments_of_every_index_cover_the_selection_once(tmp_path: Path) -> None:
    nodes: Final = (*_GROUP, f"{_GROUP[0]}::test_cell")
    written: Final = tuple(
        _write_assignment(tmp_path / f"shard-{index}.txt", nodes, 2, index).read_text().splitlines()
        for index in range(2)
    )
    assert sorted(file for files in written for file in files) == sorted(_GROUP)

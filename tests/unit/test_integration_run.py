from typing import Final

from tests.integration.run import select, uncollected

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

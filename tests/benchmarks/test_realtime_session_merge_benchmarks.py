from typing import Final

import pytest

from litellm.litellm_core_utils.realtime_streaming import _merge_session_fields

NESTED_LEVELS: Final = 8
LEAVES: Final = 2000


def _nested_session(depth: int) -> dict[str, object]:
    return {"child": _nested_session(depth - 1)} if depth else {f"k{index}": index for index in range(LEAVES)}


@pytest.mark.benchmark
def test_merging_a_session_update_that_repeats_every_nested_level() -> None:
    session: Final = _nested_session(NESTED_LEVELS)

    merged: Final = _merge_session_fields(session, session)

    assert merged == session

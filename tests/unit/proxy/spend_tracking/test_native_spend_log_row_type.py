"""``NativeSpendLogRow`` against the rows the Rust builder must produce byte for byte."""

import json
from pathlib import Path
from typing import Final

from litellm.types.proxy.spend_log_rows import NativeSpendLogRow

_GOLDEN_ROWS: Final = Path(__file__).parents[4] / "litellm/rust/crates/callbacks-spend-log/tests/goldens/rows.jsonl"


def test_every_golden_row_has_the_typed_columns_in_their_order() -> None:
    columns: Final = list(NativeSpendLogRow.__annotations__)
    rows: Final = [json.loads(json.loads(line)["output"]) for line in _GOLDEN_ROWS.read_text().splitlines()]

    assert rows
    assert {tuple(row) for row in rows} == {tuple(columns)}

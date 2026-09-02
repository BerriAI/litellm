from __future__ import annotations

from typing import Final

from hypothesis import strategies as st
from pydantic import BaseModel, ConfigDict

from tests.route_parity.fixtures.inputs import generate_case_inputs


class _Input(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    identifier: str


def test_generate_case_inputs_is_deterministic() -> None:
    strategy: Final = st.builds(_Input, identifier=st.integers().map(str))

    assert generate_case_inputs(strategy, examples=4) == generate_case_inputs(strategy, examples=4)

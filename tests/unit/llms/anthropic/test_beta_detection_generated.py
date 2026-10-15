import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm.llms.anthropic.beta_detection_generated import detect_anthropic_betas


class Case(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    request: Mapping[str, JsonValue]
    expected: frozenset[str]


CASES: Final = TypeAdapter(tuple[Case, ...]).validate_json(
    (Path(__file__).resolve().parents[3] / "fixtures/anthropic_beta_detection.json").read_text()
)


@pytest.mark.parametrize("case", CASES, ids=tuple(case.name for case in CASES))
def test_shared_requests_select_the_expected_betas(case: Case) -> None:
    before: Final = json.dumps(case.request, sort_keys=True)

    assert detect_anthropic_betas(case.request) == case.expected
    assert json.dumps(case.request, sort_keys=True) == before

from __future__ import annotations

from typing import Final

from tests.test_litellm.parity.compare import parity_comparison
from tests.test_litellm.parity.models import ParityTrace, SDKOutput


def test_parity_comparison_reports_first_nested_output_difference() -> None:
    python: Final = ParityTrace(
        outputs=(SDKOutput(response_type="Chunk", response_json={"choices": [{"delta": {"content": "a"}}]}),),
        exception=None,
    )
    rust: Final = ParityTrace(
        outputs=(SDKOutput(response_type="Chunk", response_json={"choices": [{"delta": {"content": "b"}}]}),),
        exception=None,
    )

    explanation: Final = parity_comparison(rust, python)

    assert explanation is not None
    assert explanation == [
        "Comparing ParityTrace values:",
        "  $.outputs[0].response_json.choices[0].delta.content: 'b' != 'a'",
    ]

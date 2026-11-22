from typing import Final

import pytest

from litellm.litellm_core_utils.redact_messages import request_texts, scrub_request_text

REQUEST_TEXTS: Final = request_texts("hi there")


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("error", "scrubbed"),
    [
        ("." * 100_000, "." * 100_000),
        (" ...'x" * 20_000, " ...'x" * 20_000),
        ("input_value=" * 10_000, "input_value=redacted-by-litellm"),
        ("input_value=x, input_type=y\n" * 4_000, "input_value=redacted-by-litellm, input_type=y\n" * 4_000),
    ],
    ids=["dots", "quoted_cuts", "input_value_labels_on_one_line", "input_values_on_many_lines"],
)
def test_scrubbing_a_large_error(error: str, scrubbed: str) -> None:
    assert scrub_request_text(error, REQUEST_TEXTS) == scrubbed


NEAR_MISS_TEXTS: Final = frozenset("a" * (1_190 - shift) + "b" + "a" * (9 + shift) for shift in range(4_000))


@pytest.mark.benchmark
def test_scrubbing_an_error_against_many_near_miss_texts() -> None:
    error: Final = "x" * 60 + "a" * 2_300 + "y" * 77

    assert scrub_request_text(error, NEAR_MISS_TEXTS) == error


@pytest.mark.benchmark
def test_scrubbing_an_error_that_quotes_a_long_repetitive_text_over_and_over() -> None:
    error: Final = "Invalid message " + "a" * 500_000 + " end"

    assert scrub_request_text(error, request_texts("a" * 250_000, "a" * 500_000)) == "redacted-by-litellm"

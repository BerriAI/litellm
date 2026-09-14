from typing import Final

import pytest

import litellm
from litellm.rust_bridge.lifecycle import check_limits


@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize(
    "cap, attempted_retries, refused",
    [(5, 5, True), (5, 4, False), (0, 0, False), (0, 1, True)],
    ids=[
        "cap-above-four-reached",
        "cap-above-four-not-reached",
        "first-attempt-passes-cap-of-zero",
        "cap-of-zero-refuses-first-retry",
    ],
)
def test_check_limits_reads_attempted_retries(
    monkeypatch: pytest.MonkeyPatch, metadata_key: str, cap: int, attempted_retries: int, refused: bool
) -> None:
    monkeypatch.setattr(litellm, "num_retries_per_request", cap)
    monkeypatch.setattr(litellm, "max_budget", None)
    kwargs: Final = {"model": "mistral/mistral-ocr-latest", metadata_key: {"attempted_retries": attempted_retries}}
    if refused:
        with pytest.raises(RuntimeError, match="Max retries per request hit!"):
            check_limits(kwargs)
    else:
        check_limits(kwargs)

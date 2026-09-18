import datetime
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.rust_bridge.legacy_callbacks import check_limits, setup

_OCR_KWARGS: Final = MappingProxyType(
    {
        "model": "mistral/mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
    }
)


@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize(
    "cap, request_retry_count, refused",
    [(5, 5, True), (5, 4, False), (0, 0, False), (0, 1, True)],
    ids=[
        "cap-above-four-reached",
        "cap-above-four-not-reached",
        "first-attempt-passes-cap-of-zero",
        "cap-of-zero-refuses-first-retry",
    ],
)
def test_check_limits_reads_request_retry_count(
    monkeypatch: pytest.MonkeyPatch, metadata_key: str, cap: int, request_retry_count: int, refused: bool
) -> None:
    monkeypatch.setattr(litellm, "num_retries_per_request", cap)
    monkeypatch.setattr(litellm, "max_budget", None)
    kwargs: Final = {
        "model": "mistral/mistral-ocr-latest",
        metadata_key: {"request_retry_count": request_retry_count},
    }
    if refused:
        with pytest.raises(RuntimeError, match="Max retries per request hit!"):
            check_limits(kwargs)
    else:
        check_limits(kwargs)


def _supplied_logger() -> Logging:
    return Logging(
        model="mistral/mistral-ocr-latest",
        messages=[],
        stream=False,
        call_type="aocr",
        start_time=datetime.datetime.now(),
        litellm_call_id="supplied",
        function_id="supplied",
    )


def test_setup_adopts_a_supplied_logger_as_caller_owned() -> None:
    supplied: Final = _supplied_logger()
    result: Final = setup(
        "aocr", (), {**_OCR_KWARGS, "litellm_logging_obj": supplied}, datetime.datetime.now(), asynchronous=True
    )
    assert result.logger is supplied
    assert result.bridge_owned is False


@pytest.mark.parametrize(
    "call_type, kwargs",
    [
        ("aocr", _OCR_KWARGS),
        ("aembedding", MappingProxyType({"model": "text-embedding-3-large", "input": ["hi"]})),
    ],
    ids=["ocr", "embedding"],
)
def test_setup_owns_every_logger_it_builds(call_type: str, kwargs: Mapping[str, object]) -> None:
    result: Final = setup(call_type, (), kwargs, datetime.datetime.now(), asynchronous=True)
    assert result.bridge_owned is True
    assert result.logger.litellm_call_id == result.kwargs["litellm_call_id"]

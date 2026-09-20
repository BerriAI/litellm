import datetime
import inspect
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.rust_bridge import callbacks_legacy_python as legacy
from litellm.rust_bridge.callbacks_legacy_python import setup

_OCR_KWARGS: Final = MappingProxyType(
    {
        "model": "mistral/mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
    }
)


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


def test_setup_reuses_a_supplied_logger() -> None:
    supplied: Final = _supplied_logger()
    result: Final = setup(
        "aocr", (), {**_OCR_KWARGS, "litellm_logging_obj": supplied}, datetime.datetime.now(), asynchronous=True
    )
    assert result.logger is supplied


@pytest.mark.parametrize(
    "call_type, kwargs",
    [
        ("aocr", _OCR_KWARGS),
        ("aembedding", MappingProxyType({"model": "text-embedding-3-large", "input": ["hi"]})),
    ],
    ids=["ocr", "embedding"],
)
def test_setup_builds_a_logger_when_none_is_supplied(call_type: str, kwargs: Mapping[str, object]) -> None:
    result: Final = setup(call_type, (), kwargs, datetime.datetime.now(), asynchronous=True)
    assert result.logger.litellm_call_id == result.kwargs["litellm_call_id"]


CONTRACT_PATH: Final = (
    Path(__file__).parents[3] / "litellm-rust/crates/callbacks-legacy-python/python_contract.json"
)


def test_the_rust_contract_matches_the_shim_signatures() -> None:
    contract: Final = TypeAdapter(dict[str, list[str]]).validate_json(CONTRACT_PATH.read_text())

    assert contract == {name: list(inspect.signature(getattr(legacy, name)).parameters) for name in contract}

import inspect
from pathlib import Path
from typing import Final

import pytest
from pydantic import BaseModel, TypeAdapter

from litellm.rust_bridge import callbacks_v1 as callbacks

ROOT: Final = Path(__file__).parents[3]
CONTRACT_PATH: Final = ROOT / "litellm-rust/crates/callbacks-v1-python/python_contract.json"


def test_the_rust_contract_matches_the_shim_signatures() -> None:
    contract: Final = TypeAdapter(dict[str, list[str]]).validate_json(CONTRACT_PATH.read_text())
    assert contract == {name: list(inspect.signature(getattr(callbacks, name)).parameters) for name in contract}


class ResponseModel(BaseModel):
    value: int


def test_project_response_supports_pydantic_and_json_values() -> None:
    assert callbacks.project_response(ResponseModel(value=3)) == {"value": 3}
    original: Final = {"nested": [1, True, None]}
    assert callbacks.project_response(original) is original


def test_project_response_rejects_unsupported_objects() -> None:
    with pytest.raises(TypeError):
        callbacks.project_response(object())

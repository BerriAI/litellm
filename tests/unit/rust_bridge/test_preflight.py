import inspect
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.rust_bridge import preflight
from litellm.rust_bridge.preflight import check_limits


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


def test_check_limits_refuses_a_call_over_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "num_retries_per_request", None)
    monkeypatch.setattr(litellm, "max_budget", 1.0)
    monkeypatch.setattr(litellm, "_current_cost", 1.5)
    with pytest.raises(litellm.BudgetExceededError):
        check_limits({"model": "mistral/mistral-ocr-latest"})


CONTRACT_PATH: Final = Path(__file__).parents[3] / "litellm-rust/crates/python-bridge/preflight_contract.json"
_SHIMS: Final[MappingProxyType[str, Callable[..., object]]] = MappingProxyType(
    {
        "credential_list": preflight.credential_list,
        "warn_unknown_credential": preflight.warn_unknown_credential,
        "check_limits": preflight.check_limits,
    }
)


def test_the_rust_contract_matches_the_shim_signatures() -> None:
    contract: Final = TypeAdapter(dict[str, list[str]]).validate_json(CONTRACT_PATH.read_text())

    assert contract == {name: list(inspect.signature(_SHIMS[name]).parameters) for name in contract}

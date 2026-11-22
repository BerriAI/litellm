import itertools
from collections.abc import Iterator
from typing import Final

import pytest
from _pytest.mark.structures import ParameterSet
from integration._support.parameter_matrix import Cell, Endpoint, Group, Model, cells, groups, unroutable
from integration._support.parameter_matrix_known_gaps import (
    KNOWN_GAPS,
    KNOWN_PLAIN_FAILURES,
    KNOWN_UNROUTABLE,
    KnownGap,
    KnownPlainFailure,
    candidates,
    known_gap,
)
from integration._support.parameter_matrix_rig import (
    SILENT,
    Rig,
    Unbuildable,
    Unobservable,
    fixed_host,
    gap,
    isolate,
    matrix_rig,
    plain,
    verdict,
)

from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map_source_info

_FIXED_HOST: Final = "LiteLLM sends this provider's requests to a fixed host that no injected peer can observe"
_NO_PLAIN_REQUEST: Final = (
    "No plain request for this group reaches the provider, which "
    "test_a_plain_request_for_every_listed_model_reaches_the_provider reports"
)
_LOCAL_MAP: Final = (
    "The matrix is derived from the model cost map in this checkout, run with LITELLM_LOCAL_MODEL_COST_MAP=True "
    "PYTHONPATH=.:tests python -m pytest tests/integration/providers/test_parameter_support_matrix_real_io.py"
)
_LEDGER: Final = "tests/integration/_support/parameter_matrix_known_gaps.json"
_FIXED: Final = f"is fixed, remove it from {_LEDGER}"
_NARROW: Final = (
    f"no longer shows for every model group it covers. In {_LEDGER}, replace its provider with `models` naming one "
    "model of each group that still shows it. It stopped showing for the groups of"
)


def _listed(endpoint: Endpoint) -> Iterator[ParameterSet]:
    for group in groups(endpoint):
        yield from (pytest.param(group, model, id=f"{endpoint.name.lower()}-{model.key}") for model in group.models)


def _shows(rig: Rig, known: KnownGap, cell: Cell) -> bool | None:
    outcome: Final = verdict(rig, cell)
    if isinstance(outcome, Unobservable | Unbuildable):
        return None
    found: Final = gap(cell, outcome)
    return found is not None and found.kind is known.kind


def _routed(endpoint: Endpoint, key: str) -> Iterator[tuple[Group, Model]]:
    for group in groups(endpoint):
        yield from ((group, model) for model in group.models if model.key == key)


@pytest.fixture(scope="session")
def wires(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    assert get_model_cost_map_source_info()["source"] == "local", _LOCAL_MAP
    with matrix_rig(tmp_path_factory.mktemp("parameter-matrix")) as active:
        yield active


@pytest.fixture
def rig(wires: Rig, monkeypatch: pytest.MonkeyPatch) -> Rig:
    isolate(monkeypatch, wires)
    return wires


@pytest.mark.parametrize("cell", [pytest.param(cell, id=cell.id) for cell in cells()])
def test_litellm_sends_drops_or_refuses_a_request_parameter_as_its_supported_list_says(rig: Rig, cell: Cell) -> None:
    outcome: Final = verdict(rig, cell)
    if isinstance(outcome, Unobservable):
        pytest.skip(_FIXED_HOST)
    if isinstance(outcome, Unbuildable):
        pytest.skip(_NO_PLAIN_REQUEST)

    problem: Final = gap(cell, outcome)
    if problem is not None and known_gap(cell, problem.kind):
        pytest.xfail(f"BUG: {problem.problem}")

    assert problem is None, problem.problem


@pytest.mark.parametrize("drop_params", [True, False], ids=["drop_params_on", "drop_params_off"])
@pytest.mark.parametrize(
    ("group", "model"), list(itertools.chain.from_iterable(_listed(endpoint) for endpoint in Endpoint))
)
def test_a_plain_request_for_every_listed_model_reaches_the_provider(
    rig: Rig, group: Group, model: Model, drop_params: bool
) -> None:
    sent: Final = plain(rig, group, model, drop_params)
    if sent.strayed and not sent.requests and fixed_host(group, model):
        pytest.skip(_FIXED_HOST)

    reason: Final = sent.failure.reason if sent.failure is not None else SILENT
    setting: Final = "on" if drop_params else "off"
    elsewhere: Final = f" after contacting {', '.join(sorted(sent.strayed))}" if sent.strayed else ""
    problem: Final = (
        f"LiteLLM lists {model.key}, but with drop_params {setting} a plain {group.endpoint.value} request for it "
        f"never reaches the api_base it was given for {group.provider}{elsewhere}: {reason}"
    )
    if not sent.requests and KnownPlainFailure(group.endpoint, drop_params, model.key) in KNOWN_PLAIN_FAILURES:
        pytest.xfail(f"BUG: {problem}")

    assert sent.requests, problem


def test_every_listed_chat_and_responses_model_resolves_to_a_provider() -> None:
    lost: Final = frozenset(unroutable())
    new: Final = sorted(lost - KNOWN_UNROUTABLE)
    if lost and not new:
        pytest.xfail(
            f"BUG: LiteLLM lists {len(lost)} models it cannot resolve to a provider: {', '.join(sorted(lost))}"
        )

    assert not new, f"LiteLLM lists {len(new)} models it cannot resolve to a provider: {', '.join(new)}"


@pytest.mark.parametrize("known", [pytest.param(known, id=known.id) for known in sorted(KNOWN_GAPS, key=str)])
def test_every_known_parameter_gap_still_reproduces(rig: Rig, known: KnownGap) -> None:
    seen: Final = tuple((cell.group, _shows(rig, known, cell)) for cell in candidates(known))
    showing: Final = frozenset(group for group, shows in seen if shows)
    stopped: Final = sorted({group.models[0].key for group, shows in seen if shows is False and group not in showing})

    assert showing, f"The known gap {known.id} {_FIXED}"
    assert not stopped, f"The known gap {known.id} {_NARROW} {', '.join(stopped)}"


@pytest.mark.parametrize("known", [pytest.param(known, id=known.id) for known in sorted(KNOWN_PLAIN_FAILURES, key=str)])
def test_every_known_plain_request_failure_still_reproduces(rig: Rig, known: KnownPlainFailure) -> None:
    reached: Final = [
        plain(rig, group, model, known.drop_params).requests for group, model in _routed(known.endpoint, known.model)
    ]

    assert reached and not any(reached), f"The known plain request failure {known.id} {_FIXED}"


def test_every_known_unroutable_model_is_still_unroutable() -> None:
    routed: Final = sorted(KNOWN_UNROUTABLE - frozenset(unroutable()))

    assert not routed, f"These models now resolve to a provider, so each {_FIXED}: {', '.join(routed)}"

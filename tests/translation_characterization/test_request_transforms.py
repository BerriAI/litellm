"""Pin v1 request transforms: corpus cases -> provider wire-format bodies."""

import pytest

from . import _corpus
from ._helpers import CASES_DIR, canonical_json, load_json
from ._helpers import assert_snapshot
from ._seams import PROVIDERS, run_request_transform

CASES = _corpus.build_request_cases()


def test_corpus_files_in_sync(snapshot_update: bool) -> None:
    """cases/*.json must be exactly what the builder emits (stable ids)."""
    if snapshot_update:
        _corpus.write_case_files()
    on_disk = sorted(p.stem for p in CASES_DIR.glob("*.json"))
    assert on_disk == sorted(CASES), "corpus builder and cases/ dir diverged"
    for case_id, case in CASES.items():
        assert canonical_json(case) == (CASES_DIR / f"{case_id}.json").read_text()


@pytest.mark.parametrize("case_id", sorted(CASES))
@pytest.mark.parametrize("provider_key", sorted(PROVIDERS))
def test_request_transform(
    provider_key: str, case_id: str, snapshot_update: bool
) -> None:
    case = load_json(CASES_DIR / f"{case_id}.json")
    if provider_key in case["skip"]:
        pytest.skip(case["skip"][provider_key])
    body = run_request_transform(provider_key, case)
    assert_snapshot(f"requests/{provider_key}/{case_id}.json", body, snapshot_update)

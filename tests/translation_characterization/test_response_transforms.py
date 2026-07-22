"""Pin v1 response transforms: recorded provider JSON -> OpenAI ModelResponse."""

import pytest

from ._helpers import FIXTURES_DIR, assert_snapshot, load_json
from ._seams import PROVIDERS, run_response_transform

_MESSAGES = [{"role": "user", "content": "What is the capital of France?"}]


def _fixture_ids(provider_key: str) -> list:
    return sorted(p.stem for p in (FIXTURES_DIR / "responses" / provider_key).glob("*.json"))


@pytest.mark.parametrize(
    "provider_key,fixture_id",
    [
        (p, f)
        for p in sorted(PROVIDERS)
        for f in _fixture_ids(p)
    ],
)
def test_response_transform(
    provider_key: str, fixture_id: str, snapshot_update: bool
) -> None:
    payload = load_json(FIXTURES_DIR / "responses" / provider_key / f"{fixture_id}.json")
    result = run_response_transform(provider_key, payload, _MESSAGES)
    assert_snapshot(
        f"responses/{provider_key}/{fixture_id}.json", result, snapshot_update
    )

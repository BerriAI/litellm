from collections.abc import Callable

import pytest

import litellm
from litellm.litellm_core_utils.core_helpers import is_client_caused_failure


def _complete_maritalk_without_a_key() -> object:
    return litellm.completion(model="maritalk/sabia-3", messages=[{"role": "user", "content": "hi"}], api_key=None)


def _embed_openai_like_without_an_api_base() -> object:
    return litellm.embedding(model="openai_like/qa-embedder", input=["hi"], api_key="qa-key")


@pytest.mark.parametrize(
    ("call", "expected_message"),
    [
        pytest.param(_complete_maritalk_without_a_key, "Missing API Key", id="missing_api_key"),
        pytest.param(_embed_openai_like_without_an_api_base, "Missing API Base", id="missing_api_base"),
    ],
)
def test_a_deployment_missing_its_key_or_base_keeps_its_400_but_is_not_the_callers_fault(
    monkeypatch: pytest.MonkeyPatch, call: Callable[[], object], expected_message: str
) -> None:
    for name in ("MARITALK_API_KEY", "MARITALK_API_BASE", "OPENAI_LIKE_API_BASE", "OPENAI_LIKE_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(litellm.BadRequestError, match=expected_message) as caught:
        call()

    assert caught.value.status_code == 400
    assert is_client_caused_failure(caught.value) is False

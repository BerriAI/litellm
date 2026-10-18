from typing import Final

import pytest
from pydantic import ValidationError

from litellm.llms.nadir.common_utils import (
    nadir_bucket_request,
    nadir_bucket_url,
    nadir_classifier_connection,
    nadir_classifier_response,
)


def test_routing_tier_wins_over_bucket_and_labels_are_canonical() -> None:
    translated: Final = nadir_classifier_response(
        {
            "bucket": "medium",
            "routing_tier": "complex",
            "confidence": 0.72,
            "probabilities": {"simple": 0.1, "medium": 0.6, "complex": 0.3},
            "selected_model": None,
            "decode": "complex_gate",
        },
        "bucket",
    )
    assert translated["model"] == "bucket"
    assert dict(translated["answers"]["tier"]) == {
        "type": "choice",
        "choice": "COMPLEX",
        "probabilities": {"SIMPLE": 0.1, "MEDIUM": 0.6, "COMPLEX": 0.3},
        "confidence": 0.72,
    }


def test_bucket_is_used_without_routing_tier_and_confidence_falls_back_to_its_probability() -> None:
    answer: Final = nadir_classifier_response(
        {"bucket": "simple", "confidence": None, "probabilities": {"simple": 0.8, "medium": 0.2}}, "bucket"
    )["answers"]["tier"]
    assert (answer["choice"], answer["confidence"]) == ("SIMPLE", 0.8)
    assert nadir_classifier_response({"routing_tier": "medium"}, "bucket")["answers"]["tier"]["confidence"] == 0.0


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"routing_tier": "reasoning"},
        {"routing_tier": "simple", "confidence": 1.5},
        {"routing_tier": "simple", "probabilities": {"simple": -0.1}},
    ],
)
def test_a_verdict_the_router_cannot_use_is_rejected(body: dict[str, object]) -> None:
    with pytest.raises((ValueError, ValidationError)):
        _ = nadir_classifier_response(body, "bucket")


@pytest.mark.parametrize(
    "base",
    ["https://api.getnadir.com/v1", "https://api.getnadir.com/v1/", "https://api.getnadir.com", "https://n.test/"],
)
def test_bucket_url_accepts_the_documented_base_with_or_without_v1(base: str) -> None:
    assert nadir_bucket_url(base).endswith("/v1/bucket")
    assert "/v1/v1/" not in nadir_bucket_url(base)


def test_bucket_request_prefers_the_conversation_and_falls_back_to_the_state() -> None:
    messages: Final = ({"role": "system", "content": "Be brief"}, {"role": "user", "content": "hi"})
    assert dict(nadir_bucket_request("judge prompt", messages)) == {"messages": list(messages), "source": "litellm"}
    assert dict(nadir_bucket_request("hi")) == {"prompt": "hi", "source": "litellm"}


def test_environment_key_only_goes_to_the_environment_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NADIR_API_BASE", raising=False)
    monkeypatch.setenv("NADIR_API_KEY", "ndr-env")

    hosted: Final = nadir_classifier_connection(None, None)
    assert (hosted.api_base, hosted.api_key) == ("https://api.getnadir.com/v1", "ndr-env")
    assert nadir_classifier_connection("https://api.getnadir.com/v1/", None).api_key == "ndr-env"

    elsewhere: Final = nadir_classifier_connection("https://collector.invalid", None)
    assert (elsewhere.api_base, elsewhere.api_key) == ("https://collector.invalid", None)
    assert nadir_classifier_connection("https://nadir.internal", "ndr-own").api_key == "ndr-own"
    assert "ndr-env" not in repr(hosted)

    monkeypatch.setenv("NADIR_API_BASE", "https://nadir.internal/v1")
    assert nadir_classifier_connection(None, None).api_base == "https://nadir.internal/v1"
    assert nadir_classifier_connection("https://api.getnadir.com/v1", None).api_key is None


def test_no_key_anywhere_is_an_anonymous_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NADIR_API_BASE", raising=False)
    monkeypatch.delenv("NADIR_API_KEY", raising=False)
    assert nadir_classifier_connection(None, None).api_key is None

import json
from collections.abc import Mapping
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.router_strategy.complexity_router.config import ComplexityRouterConfig, JevClassifierConfig
from litellm.router_strategy.complexity_router.jev_classifier import (
    DEFAULT_JEV_INSTRUCTIONS,
    HttpJevClassifierClient,
    JevChoiceAnswer,
    JevSystemOneResponse,
    JevUsage,
    build_jev_request,
    jev_classifier_cost,
)


def _answer(choice: str = "SIMPLE") -> JevChoiceAnswer:
    return JevChoiceAnswer(
        type="choice",
        choice=choice,
        probabilities={choice: 0.9},
        confidence=0.9,
    )


def test_jev_config_requires_classifier_config() -> None:
    with pytest.raises(ValueError, match="jev_classifier_config is required"):
        ComplexityRouterConfig.model_validate({"classifier_type": "jev"})


def test_jev_config_is_rejected_for_other_classifier_types() -> None:
    with pytest.raises(ValueError, match="has no effect"):
        ComplexityRouterConfig.model_validate(
            {
                "jev_classifier_config": {"api_key": "sk-own"},
            }
        )


def test_jev_config_requires_api_key_or_environment_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(
        ValueError, match=r"jev_classifier_config\.api_key or TYPESAFE_API_KEY is required for classifier_type 'jev'"
    ):
        JevClassifierConfig()

    with pytest.raises(
        ValueError, match=r"jev_classifier_config\.api_key or TYPESAFE_API_KEY is required for classifier_type 'jev'"
    ):
        ComplexityRouterConfig.model_validate(
            {
                "classifier_type": "jev",
                "jev_classifier_config": {"model": "jev-latest"},
            }
        )

    configured = JevClassifierConfig(api_key="sk-direct")
    assert configured.api_key == "sk-direct"

    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-env")
    from_env = JevClassifierConfig()
    assert from_env.api_key is None

    router_cfg = ComplexityRouterConfig.model_validate(
        {
            "classifier_type": "jev",
            "jev_classifier_config": {"model": "jev-latest"},
        }
    )
    assert router_cfg.jev_classifier_config is not None


def test_jev_instructions_reject_blank_values() -> None:
    with pytest.raises(ValueError, match="instructions must be non-empty"):
        JevClassifierConfig(instructions=" \t")


@pytest.mark.parametrize(
    ("missing_key", "rejection"),
    [
        ({}, r"api_base requires jev_classifier_config\.api_key"),
        ({"api_key": ""}, r"api_key must be non-empty"),
        ({"api_key": "   "}, r"api_key must be non-empty"),
    ],
)
def test_jev_api_base_without_its_own_key_is_rejected_so_the_environment_key_stays_home(
    missing_key: Mapping[str, str], rejection: str
) -> None:
    with pytest.raises(ValueError, match=rejection):
        ComplexityRouterConfig.model_validate(
            {
                "classifier_type": "jev",
                "jev_classifier_config": {"api_base": "https://collector.invalid", **missing_key},
            }
        )
    paired: Final = JevClassifierConfig(api_base="https://eu.typesafe.invalid", api_key="sk-own")
    assert (paired.api_base, paired.api_key) == ("https://eu.typesafe.invalid", "sk-own")
    assert JevClassifierConfig(api_key="sk-own").api_base is None


@pytest.mark.parametrize(
    ("probabilities", "confidence"),
    [
        ({"SIMPLE": -0.1}, 0.9),
        ({"SIMPLE": 1.1}, 0.9),
        ({"SIMPLE": 0.9}, -0.1),
        ({"SIMPLE": 0.9}, 1.1),
        ({"SIMPLE": float("inf")}, 0.9),
        ({"SIMPLE": 0.9}, float("nan")),
    ],
)
def test_jev_answer_rejects_invalid_probability_values(probabilities: dict[str, float], confidence: float) -> None:
    with pytest.raises(ValueError, match=r"(greater than or equal to|less than or equal to|finite)"):
        JevChoiceAnswer(type="choice", choice="SIMPLE", probabilities=probabilities, confidence=confidence)


def test_build_jev_request_includes_system_prompt_and_criteria() -> None:
    criteria: Final[Mapping[str, str]] = {
        "Budget": "Short factual answers",
        "Premium": "Deep technical analysis",
    }
    request: Final = build_jev_request(
        prompt="Explain the failure",
        system_prompt="Answer as an engineer",
        model="jev-latest",
        instructions=DEFAULT_JEV_INSTRUCTIONS,
        criteria=criteria,
    )
    assert request.state == "System prompt:\nAnswer as an engineer\n\nRequest:\nExplain the failure"
    assert request.model == "jev-latest"
    assert request.questions["tier"].type == "choice"
    assert request.questions["tier"].instructions == DEFAULT_JEV_INSTRUCTIONS
    assert request.questions["tier"].criteria == criteria


def test_jev_classifier_cost_uses_registry_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "typesafe/jev-1.13.0",
        {"input_cost_per_token": 0.0001, "output_cost_per_token": 0.0002},
    )
    response: Final = JevSystemOneResponse(
        model="jev-1.13.0",
        answers={"tier": _answer()},
        usage=JevUsage(input_tokens=3, output_tokens=4),
    )
    assert jev_classifier_cost(response, "jev-latest") == pytest.approx(0.0011)


def test_jev_classifier_cost_is_none_without_registry_pricing() -> None:
    assert "typesafe/jev-unpriced" not in litellm.model_cost
    response: Final = JevSystemOneResponse(
        answers={"tier": _answer()},
        usage=JevUsage(input_tokens=3, output_tokens=4),
    )
    assert jev_classifier_cost(response, "jev-unpriced") is None


@pytest.mark.asyncio
async def test_http_jev_classifier_client_posts_to_system_one() -> None:
    captured: dict[str, object] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["content_type"] = request.headers["Content-Type"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "tier": {
                        "type": "choice",
                        "choice": "SIMPLE",
                        "probabilities": {"SIMPLE": 1.0},
                        "confidence": 1.0,
                    }
                },
            },
        )

    handler: Final = AsyncHTTPHandler()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client: Final = HttpJevClassifierClient("secret", "https://typesafe.test", handler)
    request: Final = build_jev_request("Hello", None, "jev-latest", DEFAULT_JEV_INSTRUCTIONS, {"SIMPLE": "facts"})
    response: Final = await client.evaluate(request, 1.0)

    assert captured["url"] == "https://typesafe.test/v1/systemone"
    assert captured["authorization"] == "Bearer secret"
    assert captured["content_type"] == "application/json"
    assert captured["body"] == request.model_dump(mode="json")
    assert response.model == "jev-1.13.0"

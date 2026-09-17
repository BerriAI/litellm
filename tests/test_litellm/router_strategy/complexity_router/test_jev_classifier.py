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
                "jev_classifier_config": {},
            }
        )


def test_jev_instructions_reject_blank_values() -> None:
    with pytest.raises(ValueError, match="instructions must be non-empty"):
        JevClassifierConfig(instructions=" \t")


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
    response: Final = JevSystemOneResponse(
        answers={"tier": _answer()},
        usage=JevUsage(input_tokens=3, output_tokens=4),
    )
    assert jev_classifier_cost(response, "jev-latest") is None


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

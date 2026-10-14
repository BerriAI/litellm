import json
from typing import Final

import httpx
import pytest

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.decisions.transformation import systemone_request_to_ir
from litellm.types.decisions import DecisionsRequestBody, DecisionsIRScoreAnswer
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_score_array_legend_preserves_labels_probabilities_and_metadata() -> None:
    config: Final = ProviderConfigManager.get_provider_decisions_config("drex-v2.0", LlmProviders("nace"))
    assert config is not None
    request: Final = systemone_request_to_ir(
        DecisionsRequestBody.model_validate(
            {
                "state": {"text": "The order arrived damaged"},
                "questions": {"severity": {"type": "score", "criteria": ["low", "high"]}},
            }
        )
    )
    payload: Final = {
        "model": "drex-v2.0",
        "answers": {
            "severity": {
                "type": "score",
                "score": 0.75,
                "confidence": 0.8,
                "legend": ["low", "high"],
                "probabilities": {"0": 0.25, "1": 0.75},
            }
        },
        "usage": {"input_tokens": 20, "output_tokens": 30},
        "request_id": "test-request",
        "evaluation_time_ms": 8.5,
    }
    original: Final = json.dumps(payload)
    response: Final = config.parse_response(payload, request)
    answer: Final = response.answers[0]
    assert isinstance(answer, DecisionsIRScoreAnswer)
    assert (answer.score, answer.confidence) == (0.75, 0.8)
    assert tuple((item.value, item.label, item.probability) for item in answer.probabilities) == (
        (0, "low", 0.25),
        (1, "high", 0.75),
    )
    assert response.model == payload["model"]
    assert (response.usage.input_tokens, response.usage.output_tokens) == (20, 30)
    assert response.extra == {"request_id": "test-request", "evaluation_time_ms": 8.5}
    assert json.dumps(payload) == original


@pytest.mark.parametrize("legend", ({"0": "low", "1": "high"}, ["low", "high"]))
def test_score_legend_formats_share_the_systemone_contract(legend: object) -> None:
    config: Final = ProviderConfigManager.get_provider_decisions_config("drex-v2.0", LlmProviders("nace"))
    assert config is not None
    normalized: Final = config.unwrap_response(
        {
            "answers": {
                "a": {
                    "type": "score",
                    "legend": legend,
                    "score": 1,
                    "confidence": 1,
                    "probabilities": {"0": 0, "1": 1},
                }
            }
        }
    )
    assert normalized == {
        "answers": {
            "a": {
                "type": "score",
                "legend": {"0": "low", "1": "high"},
                "score": 1,
                "confidence": 1,
                "probabilities": {"0": 0, "1": 1},
            }
        }
    }


@pytest.mark.parametrize(
    "payload", ([], {"answers": []}, {"answers": {"a": None}}, {"answers": {"a": {"type": "score", "legend": [1]}}})
)
def test_invalid_provider_response_is_an_api_error(payload: object) -> None:
    config: Final = ProviderConfigManager.get_provider_decisions_config("drex-v2.0", LlmProviders("nace"))
    assert config is not None
    request: Final = systemone_request_to_ir(
        DecisionsRequestBody.model_validate(
            {
                "state": "text",
                "questions": {"a": {"type": "score", "criteria": ["low", "high"]}},
            }
        )
    )
    with pytest.raises(BaseLLMException, match="unexpected response") as error:
        config.transform_decisions_response("drex-v2.0", "nace", httpx.Response(200, json=payload), request)
    assert error.value.status_code == 500

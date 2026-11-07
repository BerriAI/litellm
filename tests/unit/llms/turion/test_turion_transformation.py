import json
from collections.abc import Mapping
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.base_llm.decisions.transformation import systemone_request_to_ir
from litellm.llms.base_llm.extraction.transformation import BaseExtractionConfig
from litellm.types.decisions import DecisionsIRChoiceAnswer, DecisionsRequestBody
from litellm.types.extraction import ExtractionRequestBody
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

# Bodies as https://api.turion.ai answered them (POST /v1/systemone and POST /v1/extract).
TURION_SYSTEMONE_RESPONSE: Final = {
    "model": "turion-jev-31b",
    "answers": {
        "refund": {"type": "noul", "noul": 0.892483},
        "team": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.991577, "technical": 0.008423},
            "confidence": 0.991577,
        },
        "urgency": {
            "type": "score",
            "score": 1.128746,
            "legend": {"0": "Can wait", "1": "This week", "2": "Right now"},
            "probabilities": {"0": 0.17257, "1": 0.526114, "2": 0.301316},
            "confidence": 0.526114,
        },
    },
    "usage": {"input_tokens": 409, "output_tokens": 0},
}
TURION_EXTRACT_RESPONSE: Final = {
    "id": "01a193fd-bf10-742e-a4a4-8f1865b3b83d",
    "model": "turion-extract-alpha",
    "answers": {
        "invoice_number": {
            "type": "text",
            "value": "INV-2291",
            "p_null": 0.00012,
            "confidence": 0.99988,
            "span": [8, 16],
        },
        "total": {"type": "number", "value": 1080, "p_null": 0.000008, "confidence": 0.999992},
    },
    "truncated": False,
    "usage": {"input_tokens": 106, "output_tokens": 0, "passes": 1},
}


def _extraction_config() -> BaseExtractionConfig:
    config: Final = ProviderConfigManager.get_provider_extraction_config("turion-extract-alpha", LlmProviders.TURION)
    assert config is not None
    return config


def test_turion_serves_decisions_and_extraction() -> None:
    assert isinstance(
        ProviderConfigManager.get_provider_decisions_config("turion-jev-31b", LlmProviders.TURION),
        litellm.TurionDecisionsConfig,
    )
    assert isinstance(_extraction_config(), litellm.TurionExtractionConfig)


@pytest.mark.parametrize("api_base", ("https://api.turion.ai", "https://api.turion.ai/", "https://api.turion.ai/v1"))
def test_routes_to_the_turion_api(monkeypatch: pytest.MonkeyPatch, api_base: str) -> None:
    monkeypatch.delenv("TURION_API_BASE", raising=False)
    decisions: Final = ProviderConfigManager.get_provider_decisions_config("turion-jev-31b", LlmProviders.TURION)
    assert decisions is not None
    assert decisions.resolve_api_base(None) == "https://api.turion.ai"
    assert _extraction_config().resolve_api_base(None) == "https://api.turion.ai"
    assert decisions.get_complete_url(api_base, "turion-jev-31b") == "https://api.turion.ai/v1/systemone"
    assert _extraction_config().get_complete_url(api_base, "turion-extract-alpha") == "https://api.turion.ai/v1/extract"


def test_reads_turion_api_key_then_the_turion_cli_token(monkeypatch: pytest.MonkeyPatch) -> None:
    config: Final = _extraction_config()
    monkeypatch.delenv("TURION_API_KEY", raising=False)
    monkeypatch.setenv("TURION_API_TOKEN", "tk_from_the_cli")
    assert config.resolve_api_key(None) == "tk_from_the_cli"
    monkeypatch.setenv("TURION_API_KEY", "tk_litellm")
    assert config.resolve_api_key(None) == "tk_litellm"
    assert config.resolve_api_key("tk_explicit") == "tk_explicit"


def test_extraction_request_keeps_the_clairic_body() -> None:
    body: Final = _extraction_config().transform_extraction_request(
        "turion-extract-alpha",
        ExtractionRequestBody.model_validate(
            {
                "state": "Invoice INV-2291 issued 2026-09-02. Total $1080.",
                "fields": {"invoice_number": {"type": "text"}, "total": {"type": "number"}},
                "reference_date": "2026-10-20",
            }
        ),
    )
    fields: Final = body["fields"]
    assert body["model"] == "turion-extract-alpha"
    assert body["reference_date"] == "2026-10-20"
    assert isinstance(fields, Mapping)
    assert list(fields) == ["invoice_number", "total"]


def test_extraction_response_parses_turion_answers() -> None:
    response: Final = _extraction_config().transform_extraction_response(
        "turion-extract-alpha",
        "turion",
        httpx.Response(status_code=200, content=json.dumps(TURION_EXTRACT_RESPONSE).encode()),
    )
    assert response.truncated is False
    assert response.answers["invoice_number"].value == "INV-2291"
    assert response.answers["invoice_number"].span == (8, 16)
    assert response.answers["total"].value == 1080
    assert response.usage.input_tokens == 106


def test_systemone_response_parses_turion_answers() -> None:
    config: Final = ProviderConfigManager.get_provider_decisions_config("turion-jev-31b", LlmProviders.TURION)
    assert config is not None
    request: Final = systemone_request_to_ir(
        DecisionsRequestBody.model_validate(
            {
                "state": "I was charged twice for my subscription.",
                "questions": {
                    "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"},
                    "team": {
                        "type": "choice",
                        "instructions": "Which team should handle this?",
                        "criteria": {"billing": "Charges and refunds", "technical": "Bugs and outages"},
                    },
                    "urgency": {
                        "type": "score",
                        "instructions": "How urgent is this?",
                        "criteria": ["Can wait", "This week", "Right now"],
                    },
                },
            }
        )
    )
    response: Final = config.parse_response(TURION_SYSTEMONE_RESPONSE, request)
    assert len(response.answers) == 3
    team: Final = next(answer for answer in response.answers if isinstance(answer, DecisionsIRChoiceAnswer))
    assert team.choice == "billing"
    assert response.usage is not None
    assert response.usage.input_tokens == 409

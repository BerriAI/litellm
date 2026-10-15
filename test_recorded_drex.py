import json
from pathlib import Path

import pytest

from litellm.llms.base_llm.decisions.transformation import ir_to_systemone_response, systemone_request_to_ir
from litellm.types.decisions import DecisionsRequestBody
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


@pytest.mark.parametrize("provider", ("nace", "openrouter", "fireworks_ai", "typesafe"))
@pytest.mark.parametrize("format", ("captured_array", "mapping_control"))
def test_recorded_drex_response(provider: str, format: str) -> None:
    directory = Path(__file__).parent
    payload = json.loads((directory / "upstream-response.json").read_text())
    body = json.loads((directory / "upstream-request.json").read_text())
    request = systemone_request_to_ir(DecisionsRequestBody.model_validate(body))
    score = payload["answers"]["urgency"]
    assert isinstance(score["legend"], list)
    normalized = {
        **payload,
        "answers": {
            **payload["answers"],
            "urgency": {**score, "legend": {str(index): label for index, label in enumerate(score["legend"])}},
        },
    }
    config = ProviderConfigManager.get_provider_decisions_config(body["model"], LlmProviders(provider))
    assert config is not None
    response = config.parse_response(payload if format == "captured_array" else normalized, request)
    assert ir_to_systemone_response(response, request).model_dump(mode="json") == normalized

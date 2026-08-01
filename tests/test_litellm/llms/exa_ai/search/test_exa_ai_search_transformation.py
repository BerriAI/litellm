from unittest.mock import Mock

import pytest

from litellm.llms.exa_ai.search.transformation import ExaAISearchConfig

OUTPUT_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
EXA_OUTPUT = {
    "content": {"answer": "Exa is an applied AI lab"},
    "grounding": [{"field": "answer", "citations": [{"url": "https://exa.ai/about"}], "confidence": "high"}],
}


@pytest.mark.parametrize("exa_output", [EXA_OUTPUT, None], ids=["exa_returns_output", "exa_omits_output"])
def test_output_schema_round_trip(exa_output):
    config = ExaAISearchConfig()

    assert config.transform_search_request("q", {"outputSchema": OUTPUT_SCHEMA})["outputSchema"] == OUTPUT_SCHEMA

    raw_response = Mock()
    raw_response.json.return_value = {"results": [], **({"output": exa_output} if exa_output else {})}
    dumped = config.transform_search_response(raw_response, logging_obj=Mock()).model_dump()

    assert ("output" in dumped) is (exa_output is not None)
    assert dumped.get("output") == exa_output

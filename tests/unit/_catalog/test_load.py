import json
import logging
from typing import Final

import litellm
import pytest

from litellm._catalog._load import CatalogUnusable, LoadedCatalog, load, served_rows
from litellm.litellm_core_utils.fallback_generalizations import get_fallback_generalization_rules
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map_source_info


def test_load_preserves_live_state_and_serves_aliases_with_shared_rows(caplog: pytest.LogCaptureFixture) -> None:
    before_rows: Final = dict(litellm.model_cost)
    before_source: Final = get_model_cost_map_source_info()
    before_rules: Final = tuple(get_fallback_generalization_rules())
    body: Final = json.dumps(
        {
            "sample_spec": {"mode": "chat"},
            "canonical": {"litellm_provider": "openai", "mode": "chat", "aliases": ["alias"], "custom": [1]},
            "bad": {"litellm_provider": "openai", "input_cost_per_token": -1},
            "fallback_generalizations": {"rules": []},
        }
    ).encode()
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        catalog: Final = load(body)
    assert isinstance(catalog, LoadedCatalog)
    rows: Final = served_rows(catalog)
    assert rows == {
        "canonical": {"litellm_provider": "openai", "mode": "chat", "custom": [1]},
        "alias": {"litellm_provider": "openai", "mode": "chat", "custom": [1]},
        "bad": {"litellm_provider": "openai", "input_cost_per_token": -1},
    }
    assert rows["canonical"] is rows["alias"]
    assert rows["canonical"].names == ("canonical", "alias")
    rows["alias"]["custom"] = [2]
    assert rows["canonical"]["custom"] == [2]
    assert served_rows(catalog)["canonical"]["custom"] == [1]
    assert len(caplog.records) == 1
    assert "bad.input_cost_per_token" in caplog.text
    assert litellm.model_cost == before_rows
    assert get_model_cost_map_source_info() == before_source
    assert tuple(get_fallback_generalization_rules()) == before_rules


@pytest.mark.parametrize("body", [b"null", b"[]", b"{}", b"not-json", b'{"sample_spec": {}}'])
def test_unusable_document_is_a_failure_value(body: bytes) -> None:
    result: Final = load(body)
    assert isinstance(result, CatalogUnusable)
    assert result.reason

from datetime import datetime, timezone
from typing import Final

import pytest

from litellm.proxy.memory.content import fuzzy_memories, redact_memory
from litellm.types.memory_v2 import MemoryEntry


@pytest.mark.parametrize("query", ("autorouter clasifier rationle", "rout clasif", "clasifier"))
def test_fuzzy_recall_matches_standalone_misspelling_and_partial_name_examples(query: str) -> None:
    routing: Final = MemoryEntry(
        memory_id="routing",
        key="routing",
        title="Auto Router classifier rationale",
        when_to_use="Investigating classifier decisions",
        scope="Auto Router",
        content="Use route diagnostics first.",
        evidence="Observed a routing investigation",
        updated_at=datetime(2026, 9, 12, tzinfo=timezone.utc),
    )
    billing: Final = routing.model_copy(
        update={
            "memory_id": "billing",
            "title": "Customer billing",
            "content": "Invoices and payment collection.",
            "when_to_use": "Collecting subscription payments.",
            "scope": "Finance",
        }
    )
    result: Final = fuzzy_memories(query, (billing, routing))
    assert result[0][0].memory_id == "routing"
    assert result[0][1] > 0
    assert fuzzy_memories("zzxxyyqq", (routing, billing)) == ()


def test_recognizable_credentials_are_redacted_without_removing_the_observation() -> None:
    text: Final = (
        "The integration failed with sk-abcdefghijklmnopqrstuvwxyz and ghp_abcdefghijklmnopqrstuvwxyz. "
        "Authorization: Bearer abcdefghijklmnopqrst. "
        "-----BEGIN RSA PRIVATE KEY-----\nprivate material\n-----END RSA PRIVATE KEY-----"
    )
    redacted: Final = redact_memory(text)
    assert "The integration failed" in redacted
    assert "abcdefghijklmnopqrst" not in redacted
    assert "private material" not in redacted
    assert "[REDACTED PRIVATE KEY]" in redacted

import json
from typing import Final

from litellm.proxy.lens.models import Evidence, TracePart
from litellm.proxy.lens.trace_store import trace_store


def test_trace_store_pages_large_payloads_and_recovers_exact_evidence() -> None:
    with trace_store() as store:
        for index in range(1001):
            store.add(
                (
                    TracePart(
                        execution_id="run",
                        span_id=f"{index:04}",
                        parent_span_id="root",
                        name="tool",
                        kind="tool",
                        content="x" * 8000,
                    ),
                )
            )
        assert store.count() == 1001
        catalogs: Final = tuple(store.catalogs(1))
        assert len(catalogs) > 1
        assert all(len(json.dumps(page)) < 25000 for page in catalogs)
        assert sum(len(page) for page in catalogs) == 1001
        assert store.previous("1000") == "0999"
        assert store.previous("0000") == ""
        assert store.get("missing") is None
        original: Final = store.get("1000")
        assert original is not None and original.content == "x" * 8000
        later: Final = TracePart(
            execution_id="run", span_id="1000", name="tool", kind="tool", content="verified failure"
        )
        store.add_reads((later,))
        assert store.evidence(Evidence(execution_id="run", span_id="1000", quote="verified failure")) == later
        assert store.evidence(Evidence(execution_id="other", span_id="1000", quote="verified failure")) is None
        assert store.evidence(Evidence(execution_id="run", span_id="1000", quote="fabricated")) is None

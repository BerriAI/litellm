import base64
import json
from typing import Final

import pytest

from litellm.proxy.engine.models import Scope, MetadataFilter
from litellm.proxy.engine.sources import SourceReader
from tests.unit.proxy.engine.test_state import engine

from litellm.proxy.engine.sources import execution_id, parse_execution


def test_same_trace_id_from_different_keys_is_a_distinct_execution() -> None:
    assert execution_id("traces", "team", "trace", "key-one-ref") != execution_id(
        "traces", "team", "trace", "key-two-ref"
    )
    assert parse_execution(execution_id("traces", "team", "trace", "key-one-ref")) == (
        "traces",
        "team",
        "trace",
        "key-one-ref",
    )


def test_previous_saved_findings_keep_their_execution_links() -> None:
    assert parse_execution(base64.urlsafe_b64encode(json.dumps(("traces", "team", "trace")).encode()).decode()) == (
        "traces",
        "team",
        "trace",
        "",
    )


@pytest.mark.asyncio
async def test_sample_never_returns_authentication_attributes() -> None:
    class StorageResponse:
        async def lens_sample(self, parameters):
            assert parameters["team"] == "alpha"
            return [
                {
                    "source": "traces",
                    "trace_id": "trace",
                    "team_id": "alpha",
                    "name": "run",
                    "start_time": "",
                    "span_count": 1,
                    "root_seen": 1,
                    "eligible": 1,
                    "attributes": [
                        ["litellm.api_key_hash", "opaque-oauth-bearer"],
                        ["environment", "production"],
                        ["", "invalid"],
                        ["oversized", "x" * 501],
                    ],
                }
            ]

    reader: Final = SourceReader(StorageResponse())
    sample: Final = await reader.sample(Scope(team_id="alpha"), engine().settings, 1, 2)
    assert sample.executions[0].metadata == (MetadataFilter(key="environment", value="production"),)
    assert "opaque-oauth-bearer" not in sample.model_dump_json()
    assert sample.eligible == 1

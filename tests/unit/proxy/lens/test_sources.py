import base64
import json
from typing import Final

import pytest

from litellm.proxy.lens.models import MetadataFilter, Scope
from litellm.proxy.lens.sources import SourceReader, execution_id, parse_execution
from litellm.rust_bridge.trace_queries import ActivityAvailability, AgentRow, ExecutionRow
from tests.unit.proxy.lens.test_state import lens


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
            assert parameters.team == "alpha"
            return [
                ExecutionRow(
                    source="traces",
                    trace_id="trace",
                    team_id="alpha",
                    name="run",
                    start_time="",
                    span_count=1,
                    root_seen=1,
                    eligible=1,
                    attributes=(
                        ("litellm.api_key_hash", "opaque-oauth-bearer"),
                        ("environment", "production"),
                        ("", "invalid"),
                        ("oversized", "x" * 501),
                    ),
                )
            ]

    reader: Final = SourceReader(StorageResponse())
    sample: Final = await reader.sample(Scope(team_id="alpha"), lens().settings, 1, 2)
    assert sample.executions[0].metadata == (MetadataFilter(key="environment", value="production"),)
    assert "opaque-oauth-bearer" not in sample.model_dump_json()
    assert sample.eligible == 1


@pytest.mark.asyncio
async def test_agents_use_the_same_team_and_key_scope_as_samples() -> None:
    class AgentStorage:
        async def lens_agents(self, parameters):
            assert parameters.all_teams == 0
            assert parameters.team == "alpha"
            assert parameters.key_hash == "key-hash"
            return (AgentRow(agent_name="research_agent"), AgentRow(agent_name="support_agent"))

    names: Final = await SourceReader(AgentStorage()).agents(Scope(team_id="alpha", api_key_hash="key-hash"))
    assert names == ("research_agent", "support_agent")


@pytest.mark.asyncio
async def test_request_only_storage_is_available_for_investigation() -> None:
    class RequestStorage:
        async def lens_availability(self, parameters):
            assert parameters.team == "alpha"
            return (ActivityAvailability(traces=False, requests=True),)

    available: Final = await SourceReader(RequestStorage()).availability(Scope(team_id="alpha"))
    assert available.requests
    assert not available.traces


@pytest.mark.asyncio
async def test_agent_filter_is_independent_of_service_and_metadata() -> None:
    class SampleStorage:
        async def lens_sample(self, parameters):
            assert parameters.agent_name == "research_agent"
            assert parameters.service == "shared-app"
            assert parameters.filter_keys == ("enduser.id",)
            assert parameters.filter_values == ("user-42",)
            return []

    settings: Final = lens().settings.model_copy(
        update={
            "agent_name": "research_agent",
            "service": "shared-app",
            "filters": (MetadataFilter(key="enduser.id", value="user-42"),),
        }
    )
    assert not (await SourceReader(SampleStorage()).sample(Scope(all_teams=True), settings, 1, 2)).executions

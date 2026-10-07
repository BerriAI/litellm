import base64
import json
from typing import Final, Literal

import pytest

from litellm.proxy.lens.agent_workspace import EvidenceRequest, PythonRequest, load_workspace
from litellm.proxy.lens.models import Evidence, Execution, ExecutionContent, MetadataFilter, Sample, Scope, TracePart
from litellm.proxy.lens.sources import SourceReader, execution_id, parse_execution
from litellm.rust_bridge.trace.generated.models import (
    ActivityAvailability,
    AgentRow,
    CountRow,
    ExecutionRow,
    LensContentParams,
    LensEvidenceParams,
    PartRow,
)
from tests.unit.proxy.lens.test_agent_workspace import python_data
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
    assert sample.executions[0].metadata == (
        MetadataFilter(key="environment", value="production"),
        MetadataFilter(key="oversized", value="x" * 501),
    )
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


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ("traces", "requests"))
async def test_recorded_times_survive_source_catalog_reads_search_and_python(
    source: Literal["traces", "requests"],
) -> None:
    run: Final = Execution(
        id=execution_id(source, "team", "run"),
        source=source,
        trace_id="run",
        team_id="team",
        name="run",
        start_time="2026-10-03 10:00:00.123456789",
        span_count=3 if source == "traces" else 1,
        root_seen=True,
    )
    rows: Final = (
        (
            PartRow(
                span_id="a-child",
                parent_span_id="z-root",
                name="child",
                kind="agent",
                start_time="2026-10-03 10:00:00.200000001",
                end_time="2026-10-03 10:00:00.300000002",
                content="Input: delegated task\nOutput: child result\nStatus: OK ",
                truncated=0,
            ),
            PartRow(
                span_id="m-tool",
                parent_span_id="a-child",
                name="tool",
                kind="tool",
                start_time="2026-10-03 10:00:00.200000009",
                end_time="2026-10-03 10:00:00.200000019",
                content="Input: child action\nOutput: tool result\nStatus: OK ",
                truncated=0,
            ),
            PartRow(
                span_id="z-root",
                parent_span_id="",
                name="root",
                kind="agent",
                start_time=run.start_time,
                end_time="2026-10-03 10:00:00.323456789",
                content="Input: task\nOutput: final result\nStatus: OK ",
                truncated=0,
            ),
        )
        if source == "traces"
        else (
            PartRow(
                span_id="request",
                parent_span_id="",
                name="model",
                kind="llm",
                start_time="2026-10-03 10:00:00.123",
                end_time="2026-10-03 10:00:00.987",
                content="Input: task\nOutput: request result\nError: ",
                truncated=0,
            ),
        )
    )

    class ContentStorage:
        async def lens_content(self, parameters: LensContentParams) -> tuple[PartRow, ...]:
            assert (
                parameters.source == source
                and parameters.record_team == "team"
                and parameters.start_time == run.start_time
            )
            return rows

        async def lens_evidence(self, parameters: LensEvidenceParams) -> tuple[CountRow, ...]:
            assert parameters.start_time == run.start_time
            return (CountRow(count=1),)

    reader: Final = SourceReader(ContentStorage())

    async def read(identity: str, cursor: str, offset: int) -> ExecutionContent:
        assert identity == run.id
        return await reader.content(Scope(team_id="team"), run, cursor, offset)

    expected: Final = tuple(
        TracePart(
            execution_id=run.id,
            span_id=row.span_id,
            parent_span_id=row.parent_span_id,
            name=row.name,
            kind=row.kind,
            content=row.content,
            start_time=row.start_time,
            end_time=row.end_time,
        )
        for row in rows
    )
    workspace: Final = await load_workspace(Sample(executions=(run,), eligible=1), read, 1)
    catalog: Final = await workspace.respond(EvidenceRequest(action="catalog", execution_id=run.id))
    assert catalog.catalog[0].spans == tuple(
        (row.span_id, row.parent_span_id, row.name, row.kind, len(row.content), row.start_time, row.end_time)
        for row in rows
    )
    assert (await workspace.respond(EvidenceRequest(action="read", execution_id=run.id))).parts == expected
    assert (await workspace.respond(EvidenceRequest(action="search", query="result"))).parts == expected
    computed: Final = await python_data(workspace, PythonRequest(action="python", code="print(data)"))
    assert computed.sessions[0].parts == expected
    assert min(computed.sessions[0].parts, key=lambda part: part.start_time).span_id == rows[-1].span_id
    assert await workspace.valid(Evidence(execution_id=run.id, span_id=rows[0].span_id, quote=rows[0].content))
    assert await reader.verify_evidence(
        Scope(team_id="team"),
        run,
        Evidence(execution_id=run.id, span_id=rows[0].span_id, quote=rows[0].content),
    )


@pytest.mark.asyncio
async def test_workspace_preserves_first_characters_and_quotes_across_gateway_pages() -> None:
    from tests.unit.proxy.lens.test_agent_workspace import execution

    run: Final = execution("trace").model_copy(update={"root_seen": True})
    text: Final = "Input: " + "x" * 7990 + "boundary evidence" + "tail" * 3000

    class PagedStorage:
        async def lens_content(self, parameters: LensContentParams) -> tuple[PartRow, ...]:
            start: Final = max(0, parameters.offset - 2)
            return (
                PartRow(
                    span_id="span",
                    parent_span_id="",
                    name="agent",
                    kind="agent",
                    start_time="",
                    end_time="",
                    content="excerpt of long content" if parameters.offset == 1 else text[start : start + 8000],
                    truncated=int(start + 8000 < len(text)),
                ),
            )

    reader: Final = SourceReader(PagedStorage())

    async def read(_identity: str, cursor: str, offset: int) -> ExecutionContent:
        return await reader.content(Scope(all_teams=True), run, cursor, offset)

    workspace: Final = await load_workspace(Sample(executions=(run,), eligible=1), read, 1)
    loaded: Final = await workspace.respond(EvidenceRequest(action="read", execution_id=run.id))
    assert loaded.parts[0].content == text
    assert await workspace.valid(Evidence(execution_id=run.id, span_id="span", quote="boundary evidence"))


@pytest.mark.asyncio
@pytest.mark.parametrize("position", (0, 3000, 7999, 8000, 12000, 19999))
async def test_long_span_fingerprint_detects_equal_length_edits_on_every_gateway_page(position: int) -> None:
    from tests.unit.proxy.lens.test_agent_workspace import execution

    run: Final = execution("trace").model_copy(update={"root_seen": True})
    original: Final = "x" * 20000

    class PagedStorage:
        def __init__(self, text: str) -> None:
            self.text: Final = text

        async def lens_content(self, parameters: LensContentParams) -> tuple[PartRow, ...]:
            start: Final = max(0, parameters.offset - 2)
            return (
                PartRow(
                    span_id="span",
                    parent_span_id="",
                    name="agent",
                    kind="agent",
                    start_time="",
                    end_time="",
                    content="unchanged excerpt" if parameters.offset == 1 else self.text[start : start + 8000],
                    truncated=int(start + 8000 < len(self.text)),
                ),
            )

    async def fingerprint(text: str) -> str:
        reader: Final = SourceReader(PagedStorage(text))

        async def read(_identity: str, cursor: str, offset: int) -> ExecutionContent:
            return await reader.content(Scope(all_teams=True), run, cursor, offset)

        workspace: Final = await load_workspace(Sample(executions=(run,), eligible=1), read, 1)
        return await workspace.fingerprint(run.id)

    baseline: Final = await fingerprint(original)
    assert await fingerprint(original) == baseline
    assert await fingerprint(original[:position] + "y" + original[position + 1 :]) != baseline

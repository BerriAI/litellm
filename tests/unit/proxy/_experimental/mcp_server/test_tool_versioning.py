from datetime import datetime, timezone
from typing import Final

import pytest

from litellm.proxy._experimental.mcp_server.tool_versioning import (
    PlannedToolVersion,
    classify_tool_change,
    plan_tool_versions,
)
from litellm.types.mcp_server.mcp_server_manager import (
    MCPToolChange,
    MCPToolChangeKind,
    MCPToolVersion,
    PinnedMCPTool,
)


def _tool(
    description: str = "",
    input_schema: dict[str, object] | None = None,
) -> PinnedMCPTool:
    return PinnedMCPTool(description=description, input_schema=input_schema if input_schema is not None else {})


def _version(
    tool_name: str,
    version: int,
    change_kind: MCPToolChangeKind,
    description: str = "",
    input_schema: dict[str, object] | None = None,
) -> MCPToolVersion:
    return MCPToolVersion(
        server_id="srv",
        tool_name=tool_name,
        version=version,
        description=description,
        input_schema=input_schema if input_schema is not None else {},
        change_kind=change_kind,
        created_at=datetime(2026, 10, 2, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        (
            _tool("Before"),
            _tool("After"),
            (MCPToolChange(breaking=False, summary="Description changed"),),
        ),
        (
            _tool(input_schema={"properties": {"a": {"type": "string"}}}),
            _tool(input_schema={"properties": {"a": {"type": "string"}}, "required": ["a"]}),
            (MCPToolChange(breaking=True, summary='Parameter "a" is now required'),),
        ),
        (
            _tool(input_schema={"type": "object"}),
            _tool(input_schema={"type": "object", "required": ["token"]}),
            (MCPToolChange(breaking=True, summary='Parameter "token" is now required'),),
        ),
        (
            _tool(input_schema={"type": "object", "required": ["token"]}),
            _tool(input_schema={"type": "object"}),
            (MCPToolChange(breaking=False, summary='Parameter "token" is now optional'),),
        ),
        (
            _tool(input_schema={"properties": {"a": {"type": "string"}}}),
            _tool(input_schema={"properties": {"b": {"type": "string"}}}),
            (
                MCPToolChange(breaking=True, summary='Removed parameter "a"'),
                MCPToolChange(breaking=False, summary='Added optional parameter "b"'),
            ),
        ),
        (
            _tool(input_schema={"properties": {}}),
            _tool(input_schema={"properties": {"a": {"type": "string"}}, "required": ["a"]}),
            (MCPToolChange(breaking=True, summary='Added required parameter "a"'),),
        ),
        (
            _tool(input_schema={"properties": {"a": {"type": "string"}}}),
            _tool(input_schema={"properties": {"a": {"type": "integer"}}}),
            (
                MCPToolChange(
                    breaking=True,
                    summary='Parameter "a" type changed from "string" to "integer"',
                ),
            ),
        ),
        (
            _tool(input_schema={"properties": {"a": {}}}),
            _tool(input_schema={"properties": {"a": {"type": "string"}}}),
            (MCPToolChange(breaking=True, summary='Parameter "a" type changed from any to "string"'),),
        ),
        (
            _tool(input_schema={"properties": {"a": {"type": "string", "enum": ["a"]}}}),
            _tool(input_schema={"properties": {"a": {"type": "string", "enum": ["a", "b"]}}}),
            (MCPToolChange(breaking=True, summary='Parameter "a" schema changed'),),
        ),
        (
            _tool(input_schema={"properties": {"a": {"type": "string", "description": "Before"}}}),
            _tool(input_schema={"properties": {"a": {"type": "string", "description": "After"}}}),
            (MCPToolChange(breaking=False, summary='Parameter "a" description changed'),),
        ),
        (
            _tool(input_schema={"properties": {"a": {"type": "string"}}, "required": ["a"]}),
            _tool(input_schema={"properties": {"a": {"type": "string"}}}),
            (MCPToolChange(breaking=False, summary='Parameter "a" is now optional'),),
        ),
        (
            _tool(input_schema={"properties": {"a": {"type": "string", "title": "Before"}}}),
            _tool(input_schema={"properties": {"a": {"type": "string", "title": "After"}}}),
            (MCPToolChange(breaking=False, summary='Parameter "a" description changed'),),
        ),
        (
            _tool(
                input_schema={
                    "properties": {
                        "filter": {
                            "type": "object",
                            "properties": {"term": {"type": "string", "description": "Before"}},
                        }
                    }
                }
            ),
            _tool(
                input_schema={
                    "properties": {
                        "filter": {"type": "object", "properties": {"term": {"type": "string", "description": "After"}}}
                    }
                }
            ),
            (MCPToolChange(breaking=False, summary='Parameter "filter" description changed'),),
        ),
        (
            _tool(
                input_schema={"properties": {"filter": {"type": "object", "properties": {"term": {"type": "string"}}}}}
            ),
            _tool(
                input_schema={"properties": {"filter": {"type": "object", "properties": {"term": {"type": "integer"}}}}}
            ),
            (MCPToolChange(breaking=True, summary='Parameter "filter" schema changed'),),
        ),
        (
            _tool(
                input_schema={"properties": {"filter": {"type": "object", "properties": {"title": {"type": "string"}}}}}
            ),
            _tool(input_schema={"properties": {"filter": {"type": "object", "properties": {}}}}),
            (MCPToolChange(breaking=True, summary='Parameter "filter" schema changed'),),
        ),
        (
            _tool(
                input_schema={
                    "properties": {"items": {"type": "array", "items": {"type": "string", "description": "Before"}}}
                }
            ),
            _tool(
                input_schema={
                    "properties": {"items": {"type": "array", "items": {"type": "string", "description": "After"}}}
                }
            ),
            (MCPToolChange(breaking=False, summary='Parameter "items" description changed'),),
        ),
        (
            _tool(input_schema={"properties": None}),
            _tool(input_schema={"properties": ["invalid"]}),
            (),
        ),
        (
            _tool(input_schema={"type": "object"}),
            _tool(input_schema={"type": "array"}),
            (MCPToolChange(breaking=True, summary="Input schema changed"),),
        ),
        (
            _tool(input_schema={"type": "object", "description": "Before"}),
            _tool(input_schema={"type": "object", "description": "After"}),
            (MCPToolChange(breaking=False, summary="Input schema description changed"),),
        ),
        (
            _tool(input_schema={"type": "object", "title": "Before"}),
            _tool(input_schema={"type": "object", "title": "After"}),
            (MCPToolChange(breaking=False, summary="Input schema title changed"),),
        ),
        (
            _tool(input_schema={"type": "object", "$defs": {"filter": {"type": "string", "description": "Before"}}}),
            _tool(input_schema={"type": "object", "$defs": {"filter": {"type": "string", "description": "After"}}}),
            (MCPToolChange(breaking=False, summary="Input schema documentation changed"),),
        ),
        (
            _tool(input_schema={"type": "object", "$defs": {"filter": {"type": "string"}}}),
            _tool(input_schema={"type": "object", "$defs": {"filter": {"type": "integer"}}}),
            (MCPToolChange(breaking=True, summary="Input schema changed"),),
        ),
    ],
)
def test_classify_tool_change(
    previous: PinnedMCPTool,
    current: PinnedMCPTool,
    expected: tuple[MCPToolChange, ...],
) -> None:
    assert classify_tool_change(previous, current) == expected


def test_schema_draft_change_is_breaking_and_recorded() -> None:
    previous_schema: Final = {"$schema": "http://json-schema.org/draft-07/schema#"}
    current: Final = _tool(input_schema={"$schema": "https://json-schema.org/draft/2020-12/schema"})
    expected_change: Final = MCPToolChange(breaking=True, summary="Input schema changed")
    previous: Final = _tool(input_schema=previous_schema)
    latest: Final = {"tool": _version("tool", 3, "initial", input_schema=previous_schema)}

    assert classify_tool_change(previous, current) == (expected_change,)
    assert plan_tool_versions(latest, {"tool": current}) == (
        PlannedToolVersion("tool", 4, current, "breaking", (expected_change,)),
    )


def test_classify_tool_change_reports_type_changes_before_schema_changes():
    previous: Final = _tool(
        description="Before",
        input_schema={
            "properties": {
                "z": {"type": "string"},
                "a": {"type": "string", "enum": ["a"]},
            },
            "type": "object",
        },
    )
    current: Final = _tool(
        description="After",
        input_schema={
            "properties": {
                "z": {"type": "integer"},
                "a": {"type": "integer", "enum": ["b"]},
            },
            "type": "array",
        },
    )

    assert classify_tool_change(previous, current) == (
        MCPToolChange(breaking=False, summary="Description changed"),
        MCPToolChange(
            breaking=True,
            summary='Parameter "a" type changed from "string" to "integer"',
        ),
        MCPToolChange(
            breaking=True,
            summary='Parameter "z" type changed from "string" to "integer"',
        ),
        MCPToolChange(breaking=True, summary="Input schema changed"),
    )


def test_plan_tool_versions_creates_initial_versions_and_sorts_by_name():
    snapshot: Final = {"z": _tool("Z"), "a": _tool("A")}

    assert plan_tool_versions({}, snapshot) == (
        PlannedToolVersion("a", 1, _tool("A"), "initial", ()),
        PlannedToolVersion("z", 1, _tool("Z"), "initial", ()),
    )


def test_plan_tool_versions_skips_unchanged_tools():
    latest: Final = {"a": _version("a", 3, "non_breaking", "A")}

    assert plan_tool_versions(latest, {"a": _tool("A")}) == ()


@pytest.mark.parametrize(
    ("current", "expected_kind", "expected_change"),
    [
        (
            _tool(input_schema={"properties": {"a": {"type": "integer"}}}),
            "breaking",
            MCPToolChange(
                breaking=True,
                summary='Parameter "a" type changed from "string" to "integer"',
            ),
        ),
        (
            _tool("After", input_schema={"properties": {"a": {"type": "string"}}}),
            "non_breaking",
            MCPToolChange(breaking=False, summary="Description changed"),
        ),
    ],
)
def test_plan_tool_versions_bumps_only_changed_tools(
    current: PinnedMCPTool,
    expected_kind: MCPToolChangeKind,
    expected_change: MCPToolChange,
) -> None:
    latest: Final = {
        "a": _version("a", 4, "initial", input_schema={"properties": {"a": {"type": "string"}}}),
        "stable": _version("stable", 2, "initial"),
    }
    snapshot: Final = {"a": current, "stable": _tool()}

    assert plan_tool_versions(latest, snapshot) == (
        PlannedToolVersion("a", 5, current, expected_kind, (expected_change,)),
    )


def test_plan_tool_versions_marks_required_only_parameter_change_as_breaking():
    latest: Final = {
        "a": _version("a", 1, "initial", input_schema={"type": "object"}),
    }
    snapshot: Final = {
        "a": _tool(input_schema={"type": "object", "required": ["token"]}),
    }

    planned: Final = plan_tool_versions(latest, snapshot)

    assert planned[0].change_kind == "breaking"


def test_plan_tool_versions_marks_removed_tools_only_once_and_restores_changed_tools():
    latest: Final = {
        "gone": _version(
            "gone",
            1,
            "initial",
            input_schema={"properties": {"a": {"type": "string"}}},
        )
    }

    assert plan_tool_versions(latest, {}) == (
        PlannedToolVersion(
            "gone",
            2,
            _tool(input_schema={"properties": {"a": {"type": "string"}}}),
            "removed",
            (MCPToolChange(breaking=True, summary="Tool removed"),),
        ),
    )
    assert plan_tool_versions({"gone": _version("gone", 2, "removed", "Before")}, {}) == ()
    assert plan_tool_versions(
        {
            "gone": _version(
                "gone",
                2,
                "removed",
                input_schema={"properties": {"a": {"type": "string"}}},
            )
        },
        {"gone": _tool(input_schema={"properties": {"a": {"type": "integer"}}})},
    ) == (
        PlannedToolVersion(
            "gone",
            3,
            _tool(input_schema={"properties": {"a": {"type": "integer"}}}),
            "breaking",
            (
                MCPToolChange(breaking=False, summary="Tool restored"),
                MCPToolChange(
                    breaking=True,
                    summary='Parameter "a" type changed from "string" to "integer"',
                ),
            ),
        ),
    )

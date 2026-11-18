from typing import Final

import pytest

from litellm.llms.litellm_proxy.skills.prompt_injection import SkillPromptInjectionHandler
from litellm.proxy._types import LiteLLM_SkillsTable

NO_ARGUMENTS_SCHEMA: Final = {"type": "object", "properties": {}, "required": []}
SQL_ARGUMENTS_SCHEMA: Final = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}


@pytest.mark.parametrize(
    ("skill", "expected"),
    [
        pytest.param(
            LiteLLM_SkillsTable(
                skill_id="translate-file v2",
                display_title="Document Translator",
                description="Converts files between languages",
                instructions="Translate the uploaded document",
            ),
            {
                "name": "translate_file_v2",
                "description": "Translate the uploaded document",
                "input_schema": NO_ARGUMENTS_SCHEMA,
            },
            id="instructions-describe-the-tool-and-the-id-becomes-a-function-name",
        ),
        pytest.param(
            LiteLLM_SkillsTable(
                skill_id="warehouse_sql",
                display_title="Warehouse SQL Analyst",
                description="Runs SQL against the inventory database",
                metadata={"parameters": SQL_ARGUMENTS_SCHEMA},
            ),
            {
                "name": "warehouse_sql",
                "description": "Runs SQL against the inventory database",
                "input_schema": SQL_ARGUMENTS_SCHEMA,
            },
            id="metadata-parameters-become-the-input-schema",
        ),
        pytest.param(
            LiteLLM_SkillsTable(
                skill_id="trip-planner",
                display_title="Trip Planner",
                metadata={"parameters": "not a schema"},
            ),
            {"name": "trip_planner", "description": "Trip Planner", "input_schema": NO_ARGUMENTS_SCHEMA},
            id="non-dict-parameters-keep-the-no-arguments-schema",
        ),
        pytest.param(
            LiteLLM_SkillsTable(skill_id="bare"),
            {"name": "bare", "description": "Skill: bare", "input_schema": NO_ARGUMENTS_SCHEMA},
            id="a-skill-with-no-text-is-described-by-its-id",
        ),
    ],
)
def test_convert_skill_to_anthropic_tool_builds_the_messages_api_tool(
    skill: LiteLLM_SkillsTable, expected: dict[str, object]
) -> None:
    assert SkillPromptInjectionHandler().convert_skill_to_anthropic_tool(skill) == expected


@pytest.mark.parametrize(
    ("instructions", "expected_description"),
    [
        ("x" * 1024, "x" * 1024),
        ("x" * 1025, "x" * 1021 + "..."),
    ],
)
def test_convert_skill_to_anthropic_tool_caps_the_description_at_1024_characters(
    instructions: str, expected_description: str
) -> None:
    tool = SkillPromptInjectionHandler().convert_skill_to_anthropic_tool(
        LiteLLM_SkillsTable(skill_id="long-instructions", instructions=instructions)
    )

    assert tool["description"] == expected_description

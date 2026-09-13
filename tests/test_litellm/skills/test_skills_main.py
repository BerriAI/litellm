from unittest.mock import MagicMock

import litellm.skills.main as skills_main
from litellm.types.utils import LlmProviders


def test_create_skill_forwards_description_and_instructions_from_top_level_kwargs(
    monkeypatch,
) -> None:
    """The REST /v1/skills form endpoint passes description/instructions as top-level
    kwargs (not extra_body). Regression for a bug where the litellm_proxy dispatch
    branch of create_skill() dropped both, so every LiteLLM-hosted skill was created
    with description=None and instructions=None regardless of what the caller sent."""
    handler = MagicMock()
    monkeypatch.setattr(skills_main, "_get_litellm_skills_handler", lambda: handler)

    skills_main.create_skill(
        display_title="Document Translator",
        description="Converts files from one language into another",
        instructions="Take an uploaded document and produce it in the target language",
        custom_llm_provider=LlmProviders.LITELLM_PROXY.value,
    )

    assert handler.create_skill_handler.call_args.kwargs["description"] == (
        "Converts files from one language into another"
    )
    assert handler.create_skill_handler.call_args.kwargs["instructions"] == (
        "Take an uploaded document and produce it in the target language"
    )


def test_create_skill_forwards_description_and_instructions_from_extra_body(monkeypatch) -> None:
    """The SDK convention (see tests/proxy_unit_tests/test_skills_db.py) nests them under
    extra_body instead of passing them as top-level kwargs; both paths must reach the DB."""
    handler = MagicMock()
    monkeypatch.setattr(skills_main, "_get_litellm_skills_handler", lambda: handler)

    skills_main.create_skill(
        display_title="Warehouse SQL Analyst",
        extra_body={"description": "Runs SQL against the inventory database", "instructions": "Summarize results"},
        custom_llm_provider=LlmProviders.LITELLM_PROXY.value,
    )

    assert handler.create_skill_handler.call_args.kwargs["description"] == (
        "Runs SQL against the inventory database"
    )
    assert handler.create_skill_handler.call_args.kwargs["instructions"] == "Summarize results"


def test_create_skill_without_description_or_instructions_passes_none(monkeypatch) -> None:
    handler = MagicMock()
    monkeypatch.setattr(skills_main, "_get_litellm_skills_handler", lambda: handler)

    skills_main.create_skill(display_title="Bare Skill", custom_llm_provider=LlmProviders.LITELLM_PROXY.value)

    assert handler.create_skill_handler.call_args.kwargs["description"] is None
    assert handler.create_skill_handler.call_args.kwargs["instructions"] is None

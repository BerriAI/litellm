"""Tests for skill injection into chat completion bodies (S2-05)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy.skill_endpoints.injection import (
    _merge_tools,
    _render_prompt,
    _split_skill_ref,
    inject_skills_into_chat_request,
)


def test_split_skill_ref_with_and_without_version():
    assert _split_skill_ref("fact-check@v3") == ("fact-check", "v3")
    assert _split_skill_ref("summarize") == ("summarize", None)


def test_render_prompt_uses_inputs():
    out = _render_prompt("Hello {{ name }}!", {"name": "javen"})
    assert "javen" in out


def test_render_prompt_missing_var_falls_back_safely():
    out = _render_prompt("Hi {name}, ok?", {})
    # Either Jinja2 raises and we fall through (returning template), or
    # SafeDict substitutes empty string. Both end states are acceptable;
    # the key is: no exception escapes.
    assert isinstance(out, str)


def test_merge_tools_dedupes_by_function_name():
    existing = [{"function": {"name": "get_weather"}}]
    additions = [
        {"function": {"name": "get_weather"}},  # dup, dropped
        {"function": {"name": "search"}},
    ]
    merged = _merge_tools(existing, additions)
    names = [t["function"]["name"] for t in merged]
    assert names == ["get_weather", "search"]


def _mock_skill(
    skill_id="sk-1",
    system_prompt_template="You are X.",
    tool_schema=None,
    source="custom",
):
    row = MagicMock()
    row.skill_id = skill_id
    row.system_prompt_template = system_prompt_template
    row.tool_schema = tool_schema or []
    row.source = source
    return row


@pytest.mark.asyncio
async def test_inject_no_skills_field_is_noop():
    data = {"messages": [{"role": "user", "content": "hi"}]}
    await inject_skills_into_chat_request(data)
    assert "skills" not in data
    assert data["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_inject_prepends_system_message_and_merges_tools():
    prisma = MagicMock()
    prisma.db.litellm_skillstable.find_unique = AsyncMock(
        return_value=_mock_skill(
            system_prompt_template="Always cite sources.",
            tool_schema=[{"function": {"name": "cite"}}],
        )
    )
    data = {
        "skills": ["cite"],
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"function": {"name": "weather"}}],
    }
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        await inject_skills_into_chat_request(data)
    # skills/skill_inputs were stripped
    assert "skills" not in data and "skill_inputs" not in data
    # system message inserted at index 0
    assert data["messages"][0]["role"] == "system"
    assert "Always cite sources." in data["messages"][0]["content"]
    # tools merged (existing weather + new cite)
    assert {t["function"]["name"] for t in data["tools"]} == {"weather", "cite"}


@pytest.mark.asyncio
async def test_inject_unknown_skill_raises_400():
    prisma = MagicMock()
    prisma.db.litellm_skillstable.find_unique = AsyncMock(return_value=None)
    data = {"skills": ["nope"]}
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        with pytest.raises(HTTPException) as exc:
            await inject_skills_into_chat_request(data)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_inject_versioned_miss_raises_404():
    prisma = MagicMock()
    prisma.db.litellm_skillstable.find_many = AsyncMock(return_value=[])
    data = {"skills": ["fact-check@v99"]}
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        with pytest.raises(HTTPException) as exc:
            await inject_skills_into_chat_request(data)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_inject_concatenates_with_existing_system_message():
    prisma = MagicMock()
    prisma.db.litellm_skillstable.find_unique = AsyncMock(
        return_value=_mock_skill(system_prompt_template="From skill.")
    )
    data = {
        "skills": ["sk-1"],
        "messages": [
            {"role": "system", "content": "Existing rule."},
            {"role": "user", "content": "hi"},
        ],
    }
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        await inject_skills_into_chat_request(data)
    # Skill prompt prepended; existing system message preserved.
    assert data["messages"][0]["role"] == "system"
    assert "From skill." in data["messages"][0]["content"]
    assert "Existing rule." in data["messages"][0]["content"]
    # And the user message is still at index 1.
    assert data["messages"][1]["role"] == "user"
